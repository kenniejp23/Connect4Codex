# Development guide

## Build and tools

From the repository root:

```sh
mise trust
mise install
mise run build
```

[`mise.toml`](../mise.toml) pins Python 3.11.13, uv 0.11.25, CMake 3.28.3, and Ninja 1.13.0.
Provide Git and a C/C++20 compiler separately. CMake fetches pinned nlohmann/json, FTXUI, SQLite,
and Catch2 (for tests); Release builds require interprocedural optimization support.

The package uses scikit-build-core and nanobind. `mise run install` runs `uv sync --frozen`;
`mise run build` forces package reinstallation to rebuild the native extension. Development
dependencies are declared in `dependency-groups.dev` in [`pyproject.toml`](../pyproject.toml).

```sh
uv run python -c 'import c4a0, c4a0_cpp; print(c4a0_cpp.N_ROWS, c4a0_cpp.N_COLS)'
```

## Validation tasks

| Task | What it runs |
| --- | --- |
| `mise run lint` | Ruff and clang-format checks |
| `mise run typecheck` | Pyright |
| `mise run test:cpp` | Debug CMake build and CTest |
| `mise run test:python` | Package rebuild and pytest |
| `mise run check` | Lint, typecheck, native tests, Python tests |
| `mise run test:wheel` | Wheel build and imports in a clean environment |
| `mise run train:smoke` | Bounded asynchronous CPU training integration test |
| `mise run ci` | Check tasks, wheel check, and training smoke test |
| `mise run test:sanitize` | AddressSanitizer and UndefinedBehaviorSanitizer native tests |
| `mise run test:thread` | ThreadSanitizer native tests |

[GitHub Actions](../.github/workflows/ci.yaml) runs `ci` and separate sanitizer jobs on
`ubuntu-latest`. The TSan preset uses `/usr/bin/setarch x86_64 -R`, requiring a host that permits
that personality change. See [`CMakePresets.json`](../CMakePresets.json) for all native presets.

The smoke task selects
`tests/c4a0_tests/training_v2_test.py::test_async_worker_completes_after_one_candidate_decision`.
It exercises a temporary V2 run through replay generation, training, and one arena decision.

## CLI

Use `uv run c4a0 --help` or `uv run c4a0 <command> --help` for the full option list.
The entry point is [`src/c4a0/main.py`](../src/c4a0/main.py).

| Command | Purpose |
| --- | --- |
| `gui` | Launch PySide6/Qt Quick desktop UI |
| `train` | Create/resume asynchronous V2 training |
| `train-legacy` | Run synchronous generation-based Lightning training |
| `training-status` | Print V2 manifest and replay status as JSON |
| `play` | Interactive terminal play, including human/human and AI/AI modes |
| `minimax-test` | Evaluate a model against a random/minimax ladder |
| `benchmark-training` | Write a fixed-work neural benchmark report |
| `compare-training-benchmarks` | Compare V2 and frozen Rust benchmark reports |
| `nn-sweep` | Optuna network sweep over legacy training data |
| `mcts-sweep` | Optuna MCTS sweep using the legacy training workflow and solver |
| `score` | Score legacy self-play policies with the external solver |

## Desktop application

```sh
mise run gui
```

[`gui/app.py`](../src/c4a0/gui/app.py) exposes application state to QML.
[`gui/jobs.py`](../src/c4a0/gui/jobs.py) queues compute jobs and launches
[`worker.py`](../src/c4a0/worker.py) subprocesses that emit structured events.
A single queued job runs at a time; a V2 training job itself has actor and learner subprocesses.

The UI provides play, training and sweep configuration, model/data inspection, tournaments,
solver scoring, settings, and validation. Model inspection distinguishes V2 attempts from legacy
generations. Solver scoring and the sweeps use legacy data/workflows. Settings persist via
`QSettings`; run artifacts and Optuna databases are stored separately on disk.

## V2 training

```sh
uv run c4a0 train --base-dir training-v2 --max-gens 10
uv run c4a0 training-status --base-dir training-v2
```

Configuration is validated by `TrainingV2Config` in [`config.py`](../src/c4a0/config.py).
Device selection prefers CUDA, then MPS on macOS, otherwise CPU; the macOS detection path raises
if MPS is unavailable. `--device` provides an explicit override.

On resume, persisted training/search configuration takes precedence over new CLI values. The
invocation can change `base_dir`, `device`, `max_gens`, and `max_candidate_attempts`. A run that has
already reached its accepted-promotion limit returns immediately when no candidate is pending.

Selected CLI defaults:

| Setting | Default / behavior |
| --- | --- |
| `--run-seed` | `1337` |
| `--n-mcts-iterations` | `1400` |
| `--mcts-value-scale`, `--value-loss-weight` | Both `0.0`: policy/terminal search and policy-only loss |
| `--self-play-batch-games` | `512`, a whole multiple of the shard size |
| `--self-play-shard-games` | `256` |
| `--replay-capacity-games`, `--replay-warmup-games` | `20000`, `2048` |
| `--replay-ratio` | `4.0` |
| `--training-batch-size`, `--inference-batch-size` | `512`, `128` |
| `--mcts-worker-threads` | `0`, native automatic selection |
| `--precision` | `auto`; also accepts `16-mixed` and `32-true` |
| `--inference-amp-min-batch-size` | `96` |
| `--arena-min-games`, `--arena-max-games` | `40`, `800`, both even |
| `--arena-pair-batch-size` | `100` color-swapped pairs per batch |
| `--max-gens` | Unbounded unless set; limits run-wide accepted promotions |
| `--max-candidate-attempts` | Unbounded unless set; bounds decisions in this invocation |

The opponent weights default to 65% champion, 30% accepted archive, 2.5% uniform, and 2.5% random.
Search uses root Dirichlet noise and temperature-controlled sampling for self-play. The arena
uses persisted openings and paired colors with SPRT thresholds. A rejection can retain a learner
incumbent whose arena score meets `--learner-incumbent-min-score` (default `0.5`); otherwise the
learner falls back to the champion or a retained qualifying incumbent.

Actor pause/resume watermarks (`--debt-pause-shards 2`, `--debt-resume-shards 1`) limit outstanding
replay training work. CUDA inference can reduce its effective batch size after an out-of-memory
error. These defaults are fixed configuration values, not automatic GPU-capacity tuning.

Artifacts:

```text
training-v2/
  run.sqlite3                     # manifest, configuration, attempts, replay and progress
  arena_openings.json             # reusable arena openings
  replay/<first-game-id>.cbor      # durable game shards
  attempts/<attempt>/model.pt      # versioned model checkpoints
  learner.pt                      # resumable learner and optimizer state
  tensorboard/events.out.tfevents.*
```

The coordinator writes the manifest; shards are atomically saved before registration. Replay
retirement operates on whole shards after release by the learner. Accepted checkpoints remain;
rejected checkpoint pruning keeps the configured recent count (default three) and any preserved
learner incumbent. Attempt zero initializes the champion. V2 refuses legacy generation directories.

```sh
uv run tensorboard --logdir training-v2/tensorboard --port 6006
uv run c4a0 play --base-dir training-v2 --model best
```

## Legacy training and scoring

```sh
uv run c4a0 train-legacy --base-dir training --max-gens 3 \
  --max-epochs 20 --early-stopping-patience 5
```

Legacy generations store `metadata.json`, `games.pkl`, and `model.pkl`. Lightning uses the run's
base directory as `default_root_dir`. Play and tournament loaders support legacy models as well
as V2 checkpoints. V2 does not convert or modify legacy generations.

Given a built Pascal Pons Connect Four solver and its opening book:

```sh
uv run c4a0 score solver/c4solver solver/7x6.book --base-dir training
uv run c4a0 train-legacy --base-dir training \
  --solver-path solver/c4solver --book-path solver/7x6.book
```

The solver scores generated policies; its results are not training targets. The default cache is
`./solutions.db`. `train` has no solver options. Sweep databases can be inspected with:

```sh
uv run optuna-dashboard sqlite:///optuna.db
```

For measurement commands and their scope, see [native performance](native-performance.md) and
[training benchmarks](training-benchmark.md).
