# Fixed-work training benchmark

[`training_benchmark.py`](../src/c4a0/training_benchmark.py) compares the C++ V2 self-play API with
the frozen Rust API using real neural inference and a fixed optimizer workload. Both workflows
run self-play, training, arena games, and optional solver scoring **sequentially**. This harness
does not run the asynchronous production coordinator in `training_v2.py`.

Each repeat initializes seeded champion and candidate networks, generates self-play games, trains
the candidate, and plays color-swapped candidate/champion arena games. Optional solver scoring
applies to the original self-play policies, not the trained candidate or arena results.

## Metrics

| Metric | Definition |
| --- | --- |
| `games_per_second` | Requested self-play games / self-play time |
| `train_positions_per_second` | Optimizer positions processed / training time |
| `candidate_latency_seconds` | Self-play time + training time |
| `arena_games_per_second` | Completed arena games / arena time |
| `solver_score` | External solver score of generated self-play policies; null without a solver |
| `solver_score_per_wall_clock_hour` | Solver score × 3600 / total measured repeat duration |

Reports contain per-repeat measurements, medians, configuration, Git revision, and environment
metadata. Total duration covers the timed stages and monitoring overhead, after initial model and
router setup. The normalized solver metric is a workload metric; it does not measure Elo or
learning improvement per hour. Arena throughput is measured without the production SPRT gate.

The utilization monitor collects process/system CPU, memory and thread information, plus NVIDIA
metrics when available, with summaries by phase. Missing hardware telemetry is not a throughput
measurement. See [`performance.py`](../src/c4a0/performance.py) for collection details.

## Generate a V2 report

```sh
mise run build
uv run c4a0 benchmark-training benchmark-results/v2.json \
  --workflow v2 --device cpu \
  --games 64 --arena-games 32 --mcts-iterations 64 \
  --inference-batch-size 64 --training-batch-size 128 --training-steps 25 \
  --repeats 3 \
  --solver-path solver/c4solver --book-path solver/7x6.book \
  --solver-cache-path benchmark-results/v2-solutions.sqlite3
```

The solver executable and opening book must already exist. Omit both solver options to measure
throughput only; that report cannot pass the comparison gate. `mise run benchmark:training` runs
the default CPU benchmark without solver arguments.

For CUDA, select `--device cuda`. `--precision`, `--inference-amp-min-batch-size`, and
`--mcts-worker-threads` control inference and native scheduling. Benchmark defaults are distinct
from production training defaults; inspect `uv run c4a0 benchmark-training --help`.

## Frozen Rust workflow

The baseline revision is `496a1f16cd5c7b742a573c6445cb6555ab246bb0`. It requires its own Rust
extension and environment; the current package does not provide `c4a0_rust`.

The module exposes a standalone `--workflow legacy-rust --output <path>` entry point for a
historical environment. However, the current harness imports `c4a0.performance.UtilizationMonitor`,
a module absent from that frozen revision. A fresh historical worktree therefore needs a compatible
harness environment before it can run this comparison; `uv sync --frozen` alone is insufficient.
The frozen Rust build also requires its historical native dependencies. The checked-in comparison
below should not be presented as immediately reproducible from the old checkout alone.

## Comparison gate

From the current checkout:

```sh
uv run c4a0 compare-training-benchmarks \
  benchmark-results/v2.json benchmark-results/legacy-rust.json \
  --output benchmark-results/comparison.json
```

The comparison validates report format and workflow, and requires equal seed, game counts, MCTS
iterations, inference batch size, training batch size, and optimizer steps. It does not enforce
matching device, precision, solver/book contents, cache state, or host load; keep these comparable
when collecting reports.

By default, self-play, training, arena, and normalized solver throughput must each reach 90% of the
Rust value. Candidate latency must be at most 110%. Missing required metrics or zero baseline
metrics fail the gate. `--throughput-ratio` and `--latency-ratio` override these thresholds.
A failed gate exits with status 2. The command writes a decision report; it does not switch the
application's training implementation.

## Checked-in evidence

[`comparison-current.json`](../benchmark-results/comparison-current.json) contains this recorded
comparison (rounded):

| Metric | V2 | Rust | V2 / Rust |
| --- | ---: | ---: | ---: |
| Self-play games/s | 29.007 | 13.313 | 2.179 |
| Training positions/s | 1779.159 | 1595.927 | 1.115 |
| Arena games/s | 8.799 | 5.754 | 1.529 |
| Solver score / wall-clock hour | 298.367 | 317.860 | 0.939 |
| Candidate latency, seconds | 1.494 | 2.874 | 0.520 |

That artifact sets `replacement_approved` to `true` and records V2 revision `aa0da6f` and the frozen
Rust revision above. It is historical evidence, not a benchmark of the current checkout.
[`v2-current-e2e.json`](../benchmark-results/v2-current-e2e.json) contains the V2 report; the paired
full Rust report is not checked in, so the comparison cannot be regenerated from repository
artifacts alone.

[`benchmark-results/`](../benchmark-results/) also holds utilization reports and sweeps for batch
sizes, native workers, game concurrency, and arena concurrency. These are measurements of their
recorded configurations. Use the current [`TrainingV2Config`](../src/c4a0/config.py) and CLI for
production defaults, and rerun measurements before drawing conclusions about current throughput.
Generated checkpoints, replay files, databases, and TensorBoard logs are excluded from Git.
