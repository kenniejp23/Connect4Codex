# End-to-end training benchmark

The training cutover benchmark measures a fixed workload through the real neural evaluator and
optimizer:

- self-play games per second;
- optimizer positions per second;
- latency from starting self-play until fixed-work candidate weights are ready;
- paired arena games per second; and
- solver score and solver score per total wall-clock hour.

The last metric normalizes the achieved solver score by the complete benchmark duration. It is a
cutover throughput metric for this fixed workload, not a claim about long-run Elo improvement.
Run the benchmark on an otherwise idle host, with the same device, workload arguments, solver,
opening book, and cache state for both workflows. Reports with different fixed-work fields are
rejected by the comparison command. Use at least three repeats; medians are reported.

Every V2 run also samples process and system CPU load, active logical CPUs, RSS, thread count,
NVIDIA utilization, memory, power, clocks, temperature, and estimated GPU energy. The report keeps
separate summaries for self-play, optimization, arena evaluation, and solver scoring. Utilization
is diagnostic: the replacement gate remains based on completed work, latency, and solver strength.

## V2 report

```sh
mise run build
uv run c4a0 benchmark-training benchmark-results/v2.json \
  --workflow v2 \
  --device cuda \
  --games 64 \
  --arena-games 32 \
  --mcts-iterations 64 \
  --inference-batch-size 128 \
  --mcts-worker-threads 0 \
  --precision auto \
  --repeats 3 \
  --solver-path solver/c4solver \
  --book-path solver/7x6.book \
  --solver-cache-path benchmark-results/v2-solutions.sqlite3
```

## Frozen sequential Rust report

The comparison baseline is the repository immediately before the C++ migration,
`496a1f16cd5c7b742a573c6445cb6555ab246bb0`. Build it in a separate worktree so the active V2
tree and its environment are not modified:

```sh
git worktree add /tmp/c4a0-legacy-rust 496a1f16cd5c7b742a573c6445cb6555ab246bb0
cd /tmp/c4a0-legacy-rust
uv sync --frozen
uv run python /path/to/Connect4Codex/src/c4a0/training_benchmark.py \
  --workflow legacy-rust \
  --output /path/to/Connect4Codex/benchmark-results/legacy-rust.json \
  --device cuda \
  --games 64 \
  --arena-games 32 \
  --mcts-iterations 64 \
  --repeats 3 \
  --solver-path /path/to/Connect4Codex/solver/c4solver \
  --book-path /path/to/Connect4Codex/solver/7x6.book \
  --solver-cache-path /path/to/Connect4Codex/benchmark-results/rust-solutions.sqlite3
```

This invokes the deleted `c4a0_rust.play_games` engine and the original sequential arrangement:
self-play completes, then optimization starts, then evaluation runs. The same standalone harness is
used so stage boundaries and arithmetic are identical.

## Replacement gate

```sh
uv run c4a0 compare-training-benchmarks \
  benchmark-results/v2.json \
  benchmark-results/legacy-rust.json \
  --output benchmark-results/comparison.json
```

The command exits with status 2 and keeps the legacy workflow as the baseline unless all gates
pass. By default, each V2 throughput metric—including solver score per wall-clock hour—must reach
90% of Rust, and candidate latency may be at most 110% of Rust. Missing solver results fail the
gate. Thresholds can be changed explicitly on the command line, and the chosen values are retained
in the comparison report.

## Recorded comparison

The first end-to-end comparison was recorded on 2026-08-24 on the same Intel i7-13620H host used
for the native migration measurements. Both workflows ran on CPU with 32 self-play games, 16 arena
games, 32 MCTS iterations, inference batch 32, training batch 64, 10 optimizer steps, and three
repeats. Solver scoring used Pascal Pons' solver and the `7x6.book`; each workflow used an isolated
cache. Values below are medians.

| Metric | V2 | Frozen Rust | V2 / Rust | Gate |
| --- | ---: | ---: | ---: | --- |
| Self-play games/s | 16.282 | 13.313 | 122.3% | Pass |
| Train positions/s | 1,609.330 | 1,595.927 | 100.8% | Pass |
| Candidate latency | 2.363 s | 2.874 s | 82.2% | Pass (lower is better) |
| Arena games/s | 5.813 | 5.754 | 101.0% | Pass |
| Solver score | 0.3341 | 0.5231 | 63.9% | Informational |
| Solver score / wall-clock hour | 222.906 | 317.860 | 70.1% | **Fail** |

That initial fail-closed result did not approve V2. The post-tuning rerun below supersedes the
cutover decision while retaining these values as historical evidence.

## RTX 2050 throughput tuning

On 2026-08-24, the CUDA workflow was profiled on an Intel i7-13620H (16 logical CPUs), an RTX 2050
4 GB, and 62 GiB RAM. More than 50 fixed-work trials covered inference batch size, native MCTS
workers, concurrent games, concurrent actors, arena batch size, training batch size, precision, and
optimizer implementation. The production defaults selected from those trials are:

- 512 native games per self-play call, split into durable 256-game replay shards;
- inference batch 128;
- automatic native MCTS workers (96 on the measured host);
- FP32 for inference batches below 96 and FP16 autocast for batches at or above 96;
- fused CUDA Adam without changing the effective training batch or update count; and
- up to 100 paired arena openings (200 games) per native call; and
- actor pause/resume watermarks of two/one replay shards, giving the trainer exclusive GPU bursts
  before its queue grows inefficiently.

The 512-game call is a throughput/latency knee: increasing from 256 to 512 games improved the
400-MCTS screen from 13.677 to 15.329 games/s and reduced measured GPU energy per game from 1.845 J
to 1.602 J. At 768 and 1,024 games, throughput was statistically flat while individual-call latency
continued to grow. In three-repeat worker trials, 96 workers gave the best median throughput:

| Native workers | Median games/s | Range | Process CPU cores | GPU utilization | GPU J/game |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 48 | 13.205 | 13.081–13.269 | 1.90 | 70.4% | 1.835 |
| 64 | 13.408 | 13.262–13.475 | 1.89 | 69.6% | 1.808 |
| 96 | **13.604** | 13.443–13.628 | 2.03 | 66.4% | 1.761 |
| 128 | 13.508 | 13.386–13.634 | 2.09 | 64.7% | **1.748** |

The GPU percentage falling as useful throughput rises is expected for this MCTS workload: the GPU
waits for tree-search dependencies. Two concurrent actors drove the GPU to 90.6%, but their
aggregate 13.49 games/s was slower than one tuned 512-game actor at 15.33 games/s. Therefore, high
GPU utilization alone is not a valid optimization target.

Arena concurrency had much more headroom. With 1,400 MCTS iterations, increasing a native call
from 16 to 200 games improved arena throughput from 0.413 to 3.069 games/s. The SPRT still evaluates
color-swapped pairs and stops only at a valid paired boundary; the larger batch can merely overshoot
a statistical boundary by more games.

For isolated optimizer work, fused CUDA Adam improved a 512-position fixed workload from 39,393 to
59,159 positions/s (+50.2%) without altering training semantics. Training batch sizes above 512
were faster mechanically, but were not adopted because they reduce the number of optimizer updates
and therefore change the learning experiment.

A production-process test then exercised actor, trainer, candidate handoff, and a fixed 40-game
arena together at 400 MCTS iterations. The original configuration took 229.12 s, averaged 1.63 CPU
cores, and overshot the 1,024-game warmup target to 1,280 replay games. The final configuration took
132.34 s, averaged 2.27 CPU cores, and stopped exactly at 1,024 games: a 42.2% wall-time reduction
without changing the warmup target, replay ratio, training batch, MCTS work, or arena game count.
Both candidates performed the same 34 optimizer steps over 17,408 positions. Mean self-play batch
throughput increased from 7.08 to 9.11 games/s (+28.6%), candidate latency fell from 148.69 to
112.90 s (-24.1%), and the paired arena increased from 1.015 to 3.007 games/s (+196.2%).

The detailed machine-readable evidence is under `benchmark-results/`, including
`util-baseline-amp.json`, `util-tuned-256.json`, `util-tuned-512.json`, and the parameter-sweep
subdirectories. These CUDA artifacts measure systems throughput; replacement approval comes only
from the solver-scored, same-workload comparison.

## Post-tuning Rust replacement decision

The same three-repeat CPU workload and frozen Rust report were compared again after the native
worker and batching improvements. The V2 solver score itself was unchanged, but the useful-work
rate improved enough for solver score per complete wall-clock hour to clear the fail-closed gate.

| Metric | Tuned V2 | Frozen Rust | V2 / Rust | Gate |
| --- | ---: | ---: | ---: | --- |
| Self-play games/s | 29.007 | 13.313 | 217.9% | Pass |
| Train positions/s | 1,779.159 | 1,595.927 | 111.5% | Pass |
| Candidate latency | 1.494 s | 2.874 s | 52.0% | Pass (lower is better) |
| Arena games/s | 8.799 | 5.754 | 152.9% | Pass |
| Solver score | 0.3341 | 0.5231 | 63.9% | Informational |
| Solver score / wall-clock hour | 298.367 | 317.860 | 93.9% | **Pass** |

The generated `benchmark-results/comparison-current.json` sets `replacement_approved` to `true`:
**V2 clears the defined sequential Rust replacement gate.** This is approval under the documented
fixed-workload criterion, not evidence that its raw candidate is stronger in an equal-game budget;
longer CUDA solver-scored runs should continue to track that distinction.
