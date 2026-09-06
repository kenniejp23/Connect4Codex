# c4a0: Connect Four AlphaZero

A Connect Four application with neural-guided Monte Carlo Tree Search (MCTS), self-play training,
a PySide6/Qt Quick desktop interface, and a terminal interface.

The C++20 engine implements game rules, search, batched self-play, serialization, the external
solver interface, and the terminal UI. Python provides the PyTorch network, training coordinator,
evaluation tools, and desktop UI through the `c4a0_cpp` nanobind extension.

## Setup

Install mise, Git, and a C/C++ toolchain supporting C++20, then run from the repository root:

```sh
mise trust
mise install
mise run build
mise run gui
```

[`mise.toml`](mise.toml) pins Python, uv, CMake, and Ninja. The compiler is supplied by the host.
The build downloads pinned native dependencies and installs the editable Python package.
The checked-in CI targets Linux.

## Train and inspect a run

```sh
uv run c4a0 train --base-dir training-v2 --max-gens 10
uv run c4a0 training-status --base-dir training-v2
uv run tensorboard --logdir training-v2/tensorboard --port 6006
```

`train` creates or resumes a V2 run. Separate actor and learner processes generate replay data and
train candidates; a paired SPRT arena evaluates candidates against the accepted champion.
`--max-gens` sets the run-wide accepted-promotion limit, excluding the initial champion. Use
`--max-candidate-attempts` to bound candidate decisions, including rejections.

The opponent mix uses the champion, accepted historical models, and random/uniform evaluators.
The default `--mcts-value-scale 0.0` and `--value-loss-weight 0.0` disable neural value contributions
in V2 search and value losses in training, respectively. Policy learning and terminal game outcomes
remain active. The V2 trainer does not call the external solver.

The synchronous Lightning workflow remains available as `train-legacy`, with its default data
root at `training/`. See the [development guide](docs/development.md) for configuration and artifacts.

## Play

```sh
uv run c4a0 play --base-dir training-v2 --model best
uv run c4a0 play --model random
uv run c4a0 play --model uniform --mode human-human
uv run c4a0 play --model best --mode human-ai --human-side blue
uv run c4a0 play --model best --mode ai-ai
```

The terminal UI defaults to human Red versus AI Blue. Keys `1`–`7` select columns; AI turns move
automatically when the search limit is reached. `B` plays the best searched move and `R` samples
from the search policy. `best` loads the V2 champion or the latest legacy generation in `--base-dir`;
it requires an existing model. `random` and `uniform` supply MCTS evaluators and need no checkpoint.

The desktop UI includes play and live search analysis, V2 training controls, model inspection,
tournaments, legacy-data solver scoring and sweeps, and developer validation. Preferences and
paths are stored in Qt user settings.

## Evaluate

```sh
uv run c4a0 minimax-test --base-dir training-v2 --max-depth 3
```

The minimax ladder evaluates against random and successively deeper minimax evaluators with
color-swapped games. It advances after scoring more than half the configured games in points
(a draw is half a point), and stops at the first failed level. Both sides use the native search
pipeline. `--model-path` selects a checkpoint directly. The default maximum depth is 42; the
example above bounds it to three.

The external Pascal Pons solver is optional. Given its executable and opening book, score legacy
self-play policies with:

```sh
uv run c4a0 score solver/c4solver solver/7x6.book --base-dir training
```

See [training benchmarks](docs/training-benchmark.md) for fixed-work neural benchmarks and
[native performance](docs/native-performance.md) for engine-only measurements.

## Code map

| Component | Source |
| --- | --- |
| CLI and validated configuration | [`main.py`](src/c4a0/main.py), [`config.py`](src/c4a0/config.py) |
| Desktop UI and job queue | [`gui/`](src/c4a0/gui/), [`worker.py`](src/c4a0/worker.py) |
| Residual CNN with policy and two value outputs | [`nn.py`](src/c4a0/nn.py) |
| Asynchronous replay training and champion gate | [`training_v2.py`](src/c4a0/training_v2.py) |
| Legacy generation training | [`training.py`](src/c4a0/training.py) |
| Bitboard rules, MCTS, and self-play | [`cpp/src/`](cpp/src/) |
| Native Python API and type stubs | [`src/c4a0_cpp/`](src/c4a0_cpp/) |

## Validation and license

`mise run check` runs lint, type checks, native tests, and Python tests. `mise run ci` also checks
wheel installation and bounded V2 training. These commands describe available checks, not the
status of a particular checkout.

See [LICENSE.md](LICENSE.md) for the license and [implementation notes](docs/improvements.md)
for remaining limitations.
