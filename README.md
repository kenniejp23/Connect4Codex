# c4a0: Connect Four Alpha-Zero

An Alpha-Zero-style Connect Four engine trained entirely via self play.

The game logic, Monte Carlo Tree Search, multi-threaded self-play engine, solver cache, and terminal
UI are implemented in C++20 under [`cpp/`](cpp/).

The neural network is written in Python/PyTorch under [`src/c4a0/`](src/c4a0/) and calls the C++
engine through the `c4a0_cpp` nanobind extension.

![Terminal UI](https://raw.githubusercontent.com/advait/c4a0/refs/heads/master/images/tui.png)

## Usage

1. Install [mise](https://mise.jdx.dev/getting-started.html). The checked-in tool configuration
   installs the pinned Python, uv, CMake, and Ninja versions.

```sh
curl https://mise.run | sh
```

2. Install the toolchain, dependencies, and editable package:

```sh
mise trust
mise install
mise run build
```

3. Run the complete validation suite:

```sh
mise run check
```

4. Launch the native desktop application:

```sh
mise run gui
# or, after installation
uv run c4a0 gui
```

The desktop UI provides mouse and keyboard play, live MCTS analysis, training and sweep
configuration, model/data inspection, tournaments, solver scoring, and developer validation. The
existing terminal commands remain available.

5. Train a network with the asynchronous, neural-only V2 pipeline:

```sh
uv run c4a0 train --base-dir training-v2 --max-gens 10
uv run c4a0 training-status --base-dir training-v2
```

`train` creates or resumes only a V2 run. It overlaps self-play and replay training, evaluates
candidates against the accepted champion with an SPRT arena gate, and uses accepted historical
networks plus small random/uniform allocations for diversity. `--max-gens` counts newly accepted
champions. The previous synchronous implementation remains available as `train-legacy`.

6. Play against the network:

```sh
uv run src/c4a0/main.py play --model=best
```

The default game mode is human-versus-AI: the human plays Red and moves first. Choose a column
with keys `1` through `7`; after the MCTS search reaches its configured iteration limit, Blue
makes its move automatically. The TUI labels both colors as Human or AI.

Use `--human-side blue` to let the AI move first, `--mode human-human` for local two-player play,
or `--mode ai-ai` to watch the selected model play both colors:

```sh
uv run src/c4a0/main.py play --model best --mode human-ai --human-side blue
uv run src/c4a0/main.py play --model best --mode human-human
uv run src/c4a0/main.py play --model best --mode ai-ai
```

`--model` selects the evaluator used by MCTS: `best` loads the accepted champion, `random`
uses random policy logits, and `uniform` gives every legal move equal policy weight. `B` plays the
current best searched move immediately, while `R` samples a move from the current search policy.

7. (Optional) Download a [connect four solver](https://github.com/PascalPons/connect4?ts=2) to
   objectively measure training progress:

```sh
git clone https://github.com/PascalPons/connect4.git solver
cd solver
make
# Download opening book to speed up solutions
wget https://github.com/PascalPons/connect4/releases/download/book/7x6.book
```

The V2 trainer never imports or invokes the solver. Standalone legacy scoring remains available:

```sh
uv run python src/c4a0/main.py score solver/c4solver solver/7x6.book
```

V2 replacement decisions use the fixed-workload [end-to-end training benchmark](docs/training-benchmark.md),
including a fail-closed comparison with the frozen sequential Rust implementation.

## Data compatibility

V2 checkpoints are versioned PyTorch dictionaries under `training-v2/attempts/`; immutable replay
shards use native CBOR and the run manifest is SQLite in WAL mode. Accepted checkpoints are kept,
while old rejected weights are compacted. Legacy `model.pkl` generations remain readable by play,
tournaments, and `train-legacy`; V2 never modifies a legacy directory.

## Results

After 9 generations of training (approx ~15 min on an RTX 3090) we achieve the following results:

![Training Results](https://raw.githubusercontent.com/advait/c4a0/refs/heads/master/images/learning.png)

## Architecture

### PyTorch NN [`src/c4a0/nn.py`](https://github.com/advait/c4a0/blob/master/src/c4a0/nn.py?ts=2)

A ResNet-style CNN takes a board position and outputs a policy (a probability distribution over
moves) and Q values (predicted win/loss values in `[-1, 1]`).

Neural-network hyperparameters can be swept via the `nn-sweep` command.

### Connect Four Game Logic [`cpp/src/position.cpp`](cpp/src/position.cpp)

Implements the compact `Position` bitboard representation and all Connect Four rules
and game logic.

### Monte Carlo Tree Search (MCTS) [`cpp/src/mcts.cpp`](cpp/src/mcts.cpp)

Implements Monte Carlo Tree Search—the core search algorithm behind AlphaZero. It probabilistically
explores potential game pathways and optimally hones in on the optimal move to play from any
position.

MCTS relies on outputs from the NN. The output of MCTS helps train the next generation's NN.

### Self Play [`cpp/src/self_play.cpp`](cpp/src/self_play.cpp)

Uses C++20 worker threads and batched Python/PyTorch callbacks to parallelize training-data
generation.

### Solver [`cpp/src/solver.cpp`](cpp/src/solver.cpp)

Connect Four is a perfectly solved game. See Pascal Pons's [great
writeup](http://blog.gamesolver.org/) on how to build a perfect solver. We can use these solutions
to objectively measure our NN's performance. Importantly we **never train on these solutions**,
instead only using our self-play data to improve the NN's performance.

`solver.cpp` contains the stdin/stdout interface used to obtain objective solutions for training
positions. Because solutions are expensive to compute, they are cached in a versioned SQLite
database (`solutions.db`). We then measure how often generated policies recommend optimal moves.
