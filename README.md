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

5. Train a network:

```sh
uv run src/c4a0/main.py train --max-gens=10
```

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

`--model` selects the evaluator used by MCTS: `best` loads the latest trained network, `random`
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

Now pass the solver paths to `train`, `score` and other commands:

```sh
uv run python src/c4a0/main.py score solver/c4solver solver/7x6.book
```

## Data compatibility

Neural-network checkpoints remain ordinary Python `model.pkl` files. Native game records now use a
versioned CBOR schema, and solver results use a versioned SQLite `solutions.db` cache. Game data or
RocksDB caches created by the former backend are intentionally unsupported; begin with fresh
self-play data and a fresh solver cache after this migration. Unknown data or cache versions fail
with a clear error instead of being interpreted silently.

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
