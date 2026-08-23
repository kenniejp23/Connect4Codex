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

The task runs a tiny CPU job equivalent to:

```sh
rm -rf training/ci-smoke
uv run python src/c4a0/main.py train \
  --base-dir training/ci-smoke \
  --device cpu \
  --n-self-play-games 4 \
  --n-mcts-iterations 4 \
  --self-play-batch-size 16 \
  --training-batch-size 16 \
  --n-residual-blocks 1 \
  --conv-filter-size 8 \
  --n-policy-layers 1 \
  --n-value-layers 1 \
  --lr-schedule 0 \
  --lr-schedule 0.001 \
  --l2-reg 0 \
  --max-gens 1
```

Verified smoke runs in this environment:

- generated 4 games
- generated a generation 0 root model and a generation 1 trained model
- produced self-play samples and unique-position counts in the training metadata/artifacts
- saved `metadata.json`, `games.pkl`, and `model.pkl` for each generation
- left `solver_score` as `null` because no external solver was configured

Training artifacts are stored as timestamped generation directories:

```text
training/<run-name>/<timestamp>/metadata.json
training/<run-name>/<timestamp>/games.pkl
training/<run-name>/<timestamp>/model.pkl
```

PyTorch Lightning logs are written to `lightning_logs/`.

## Inspect self-play stats

Use this after a training run:

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

The README default is much larger and intended for a GPU-class machine:

```sh
mise exec -- uv run python src/c4a0/main.py train --max-gens 10
```

Useful knobs:

- `--device cpu|cuda|mps`
- `--n-self-play-games`
- `--n-mcts-iterations`
- `--self-play-batch-size`
- `--training-batch-size`
- `--base-dir`
- `--max-gens`

## Play against a model

After training at least one generation:

```sh
mise exec -- uv run python src/c4a0/main.py play --base-dir training/ci-smoke --model best
```

Other model options:

```sh
mise exec -- uv run python src/c4a0/main.py play --model random
mise exec -- uv run python src/c4a0/main.py play --model uniform
```

This opens a terminal UI, so run it in an interactive terminal.

## TensorBoard / dev server

There is no web app dev server in this repo. The useful local servers are for experiment inspection:

```sh
mise exec -- uv run tensorboard --logdir lightning_logs --port 6006
```

For Optuna sweeps:

```sh
mise exec -- uv run optuna-dashboard sqlite:///optuna.db
```

## Optional solver scoring

The solver is optional and is not used for training. It scores generated policies against objective Connect Four solutions.

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
- `mise run train:smoke`: runs end-to-end self-play + model training

The terminal UI suite creates its own pseudo-terminal and verifies terminal restoration on both
normal exit and evaluator failure.
