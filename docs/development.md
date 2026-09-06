# Development Guide

This project has two main parts:

- `cpp/`: C++20 Connect Four rules, MCTS, self-play, solver interface, TUI, and nanobind bindings.
- `src/c4a0/`: PyTorch/PyTorch Lightning model, training loop, sweeps, and CLI commands.

The CLI entrypoint is `src/c4a0/main.py`.

## Tooling model

Use [`mise`](https://mise.jdx.dev/) as the project entrypoint. `mise.toml` pins and bootstraps the required tools:

- `uv` for Python dependency management
- CMake and Ninja for the C++ build
- a C++20 compiler (GCC 13+ or Clang 18+ on the supported Linux target)

Install mise once, then from the repo root run:

```sh
mise trust
mise install
```

The CMake build fetches pinned C++ dependencies and compiles SQLite into the extension. A C++20
compiler, CMake, and Ninja are the only native build prerequisites.

## Common tasks

List tasks:

```sh
mise tasks
```

Install dependencies and build the editable package:

```sh
mise run install
mise run build
```

Run local validation:

```sh
mise run lint
mise run typecheck
mise run test:cpp
mise run test:python
mise run check
```

The Linux TSan preset runs discovered tests through `setarch x86_64 -R`. Disabling ASLR for the
test process avoids GCC TSan's documented early-runtime `unexpected memory mapping` failure on the
supported Pop!_OS/Ubuntu hosts; it does not alter production builds.

Run the full CI suite locally, including smoke training:

```sh
mise run ci
```

CI uses the same task:

```sh
mise run ci
```

## Packaging/import check

This is a mixed scikit-build-core Python/C++ package. After `mise run build`, both imports should work without `PYTHONPATH` hacks:

```sh
mise exec -- uv run python - <<'PY'
import c4a0
import c4a0_cpp

print(c4a0.__file__)
print(c4a0_cpp.N_ROWS, c4a0_cpp.N_COLS)
PY
```

`c4a0_cpp` re-exports the native `c4a0_cpp._native` nanobind extension.

## CLI commands

Show available commands:

```sh
mise exec -- uv run python src/c4a0/main.py --help
```

Current commands:

- `gui`: launch the native PySide6/Qt Quick desktop application
- `train`: train via self-play
- `play`: open the terminal UI and play against a model/random/uniform player
- `score`: score generated policies with an external Connect Four solver
- `nn-sweep`: Optuna sweep over NN hyperparameters using existing training data
- `mcts-sweep`: Optuna sweep over self-play/MCTS hyperparameters

## Desktop UI

Launch the Linux-first desktop interface from a source checkout:

```sh
mise run gui
```

The interface keeps one compute-heavy job active at a time and queues additional training,
evaluation, sweep, scoring, or validation work. Worker processes emit structured events so the Qt
event loop stays responsive. Application preferences are stored with Qt's user settings; training
artifacts and Optuna databases remain the durable experiment records.

## Smoke train a model

Run the checked-in smoke task:

```sh
mise run train:smoke
```

The task runs the bounded process-level integration test. It creates a temporary V2 run, generates
an atomic CBOR shard, trains from replay, submits a candidate, completes an arena decision, reloads
the champion, and exits after that decision. No solver path is present in the V2 worker.

V2 artifacts use this layout:

```text
training-v2/run.sqlite3
training-v2/arena_openings.json
training-v2/replay/*.cbor
training-v2/attempts/<attempt>/model.pt
training-v2/learner.pt
training-v2/tensorboard/events.out.tfevents.*
```

The coordinator is the only manifest writer. A staged shard is fsynced and atomically renamed
before registration; whole old shards are retired only after the trainer releases them. Use
`uv run c4a0 training-status --base-dir training-v2` for the champion, pending candidate, component
progress, replay occupancy, and globally unique next game ID.

Legacy timestamped `metadata.json`, `games.pkl`, and `model.pkl` generations are still supported by
`train-legacy` and all playback/evaluation loaders.

## Inspect legacy self-play stats

Use this only for a `train-legacy` directory; V2 inspection uses `training-status`:

```sh
mise exec -- uv run python - <<'PY'
from c4a0.training import TrainingGen

base = "training/ci-smoke"
for gen in reversed(TrainingGen.load_all(base)):
    games = gen.get_games(base)
    if games is None:
        print(f"gen={gen.gen_n}: root/no games, val_loss={gen.val_loss}, solver_score={gen.solver_score}")
        continue
    lengths = [len(g.samples) for g in games.results]
    scores = [g.player0_score() for g in games.results]
    print(
        f"gen={gen.gen_n}: games={len(games.results)}, "
        f"unique_positions={games.unique_positions()}, "
        f"samples={sum(lengths)}, min_len={min(lengths)}, max_len={max(lengths)}, "
        f"avg_len={sum(lengths)/len(lengths):.2f}, "
        f"p0_score_avg={sum(scores)/len(scores):.3f}, "
        f"val_loss={gen.val_loss}, solver_score={gen.solver_score}"
    )
PY
```

## Full/default training

The defaults target the detected CUDA device and a 4 GB-class GPU:

```sh
uv run c4a0 train --base-dir training-v2 --max-gens 10
```

Useful knobs:

- `--device cpu|cuda|mps`
- `--self-play-shard-games`
- `--self-play-batch-games` (native concurrency; must be a whole multiple of the durable shard size)
- `--n-mcts-iterations`
- `--inference-batch-size`
- `--mcts-worker-threads` (`0` uses the benchmarked native automatic setting)
- `--precision auto|16-mixed|32-true`
- `--inference-amp-min-batch-size`
- `--training-batch-size`
- `--replay-capacity-games`
- `--replay-ratio`
- `--arena-pair-batch-size`
- `--max-candidate-attempts` (a bounded experiment/smoke safeguard)
- `--base-dir`
- `--max-gens` (accepted champions, not rejected attempts)

The CUDA defaults were selected from the utilization and throughput sweeps documented in the
[end-to-end training benchmark](training-benchmark.md). They intentionally optimize completed
games and positions per second rather than maximizing the GPU percentage shown by `nvidia-smi`.

## Play against a model

After the V2 run has its attempt-0 champion:

```sh
uv run c4a0 play --base-dir training-v2 --model best
```

Other model options:

```sh
uv run c4a0 play --model random
uv run c4a0 play --model uniform
```

This opens a terminal UI, so run it in an interactive terminal.

## TensorBoard / dev server

There is no web app dev server in this repo. The useful local servers are for experiment inspection:

```sh
uv run tensorboard --logdir training-v2/tensorboard --port 6006
```

For Optuna sweeps:

```sh
mise exec -- uv run optuna-dashboard sqlite:///optuna.db
```

## Optional solver scoring

The solver is optional and is not used for training. It scores generated policies against objective Connect Four solutions.

For replacement decisions, use the reproducible [end-to-end training benchmark](training-benchmark.md).
It compares V2 with the frozen sequential Rust workflow and deliberately fails closed when solver
results or another required metric are absent.

```sh
git clone https://github.com/PascalPons/connect4.git solver
cd solver
make
wget https://github.com/PascalPons/connect4/releases/download/book/7x6.book
cd ..
```

Score an existing training directory:

```sh
mise exec -- uv run python src/c4a0/main.py score solver/c4solver solver/7x6.book --base-dir training/ci-smoke
```

Or score during training:

```sh
mise exec -- uv run python src/c4a0/main.py train \
  --solver-path solver/c4solver \
  --book-path solver/7x6.book
```

Scores are cached in `solutions.db` by default.

## Verified checks

The release gate covers:

- `mise run build`: builds and installs the mixed Python/C++ package
- `mise run lint`: Ruff passed
- `mise run typecheck`: Pyright passed with 0 errors
- `mise run test:cpp`: runs native CTest unit and property tests
- `mise run test:python`: Python API, training, tournament, and native-boundary tests
- `mise run train:smoke`: runs bounded asynchronous shard, replay, candidate, and arena training

The terminal UI suite creates its own pseudo-terminal and verifies terminal restoration on both
normal exit and evaluator failure.
