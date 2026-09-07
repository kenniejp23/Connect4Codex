# Native performance benchmarks

[`benchmarks/native_benchmarks.cpp`](../benchmarks/native_benchmarks.cpp) measures the C++ engine
with an in-process uniform evaluator. It excludes Python, neural inference, accelerator transfers,
and optimization. Use the [training benchmark](training-benchmark.md) for those components.

## Build and run

From the repository root with the mise toolchain installed:

```sh
mise exec -- cmake --preset release
mise exec -- cmake --build --preset release --target c4a0_benchmarks
build/release/c4a0_benchmarks
```

Run one warm-up, then collect several measured runs and compare medians:

```sh
for benchmark_run in 1 2 3 4 5; do
  build/release/c4a0_benchmarks
done
```

Measure self-play separately, including peak process memory if GNU time is available:

```sh
/usr/bin/time -v build/release/c4a0_benchmarks --self-play-only
```

## Workloads and output

| Output | Work timed |
| --- | --- |
| `position_ops_per_second` | 10,000,000 iterations, each checking a legal move and terminal state and flipping a fixed position horizontally |
| `mcts_iterations_per_second` | 20 fresh trees of 50,000 uniform-evaluator iterations each; includes tree construction/destruction |
| `self_play_games_per_second` | 512 games, evaluation batch limit 32, 64 MCTS iterations, exploration 4.0, ply penalty 0.01 |
| `checksum` | Accumulated work counter; expected `11000512` for the full workload and `512` for self-play only |

Self-play uses the engine's automatic worker selection. Release builds enable interprocedural
optimization. Record the revision, compiler, CPU, worker behavior, host load, and any affinity or
power-policy settings alongside timings; changes to scheduling or game trajectories affect results.
Peak RSS from a single run is not a long-duration memory test.

The repository does not contain raw reports or the temporary Rust harness for the old native
migration measurements. Their previous tables have been removed because they cannot establish
performance of the current engine. Run the checked-in harness to obtain a baseline for this revision.

## Interactive release checks

The interactive engine now calls evaluators outside its board mutex and discards results after a
board revision changes. Native regression tests cover responsive snapshots/reset during a delayed
callback and successful retry after failure. These correctness checks do not establish a general
throughput improvement. Historical measurements above remain tied to their original revisions.
