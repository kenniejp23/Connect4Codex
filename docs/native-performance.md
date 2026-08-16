# Native performance baselines

This page records reproducible C++ Release and frozen legacy Rust Release baselines for regression
testing. These are native-engine microbenchmarks, not end-to-end training benchmarks: self-play
uses a uniform in-process evaluator and does not include Python, PyTorch, accelerator transfers, or
neural-network inference.

## Recorded environment

Measurements were taken on 2026-08-16 from the migration working tree based on Git revision
`496a1f16cd5c`.

- Pop!_OS 24.04 LTS, Linux `7.0.11-76070011-generic`, x86-64
- Intel Core i7-13620H, 10 physical cores / 16 hardware threads
- 62 GiB RAM
- GCC 13.3.0, CMake 3.28.3, Ninja 1.13.0
- C++20 Release flags: `-O3 -DNDEBUG`
- CPU frequency governor: `performance`
- No CPU affinity or system isolation; the machine was otherwise idle, so scheduler and hybrid-core
  placement remain sources of run-to-run variation

The calibrated benchmark runs three workloads:

- `position_ops_per_second`: 10,000,000 composite position-loop iterations. Each iteration calls
  `legal_moves()`, `terminal_state()`, and `flip_horizontal()`; the reported count is composite loop
  iterations, not the sum of the three method calls.
- `mcts_iterations_per_second`: 1,000,000 uniform-evaluator MCTS iterations, split into 20 trees of
  50,000 iterations to bound peak memory. Tree construction and destruction are included.
- `self_play_games_per_second`: 512 games, maximum evaluation batch size 32, 64 root MCTS
  iterations, exploration coefficient 4.0, and ply penalty 0.01.

## C++ results

One warm-up execution was discarded, followed by five consecutive measured executions. Every full
run produced checksum `11000512`.

| Run | Position composite ops/s | MCTS iterations/s | Self-play games/s |
| ---: | ---: | ---: | ---: |
| 1 | 15,824,611.100 | 3,332,630.015 | 895.897 |
| 2 | 15,773,298.457 | 3,278,857.606 | 887.225 |
| 3 | 15,931,939.621 | 3,336,645.833 | 911.844 |
| 4 | 15,695,821.385 | 3,339,681.511 | 879.036 |
| 5 | 15,652,690.539 | 3,286,016.049 | 869.515 |
| **Median** | **15,773,298.457** | **3,332,630.015** | **887.225** |

These final measurements include behavior-preserving engine and build optimizations: Release
interprocedural optimization (IPO), hoisting the parent-visit logarithm out of child-selection
loops, an equal-logit softmax fast path, lazy materialization of `std::unique_ptr` MCTS children,
and bounded ring-buffer queues with bulk transfers.

Self-play is intentionally multi-threaded and showed the widest spread in this non-isolated desktop
environment. Use multiple runs and compare medians rather than treating one execution as a gate.

For a representative self-play-only run, GNU `time -v` reported:

- 917.489 games/s
- maximum resident set size: 8,064 KiB (approximately 7.9 MiB)
- checksum: `512`

The RSS figure is a single-process peak for this bounded workload, not a long-duration leak test.

## Reproduce

From the repository root after installing the checked-in mise toolchain:

```sh
mise trust
mise install
mise exec -- cmake --preset release
mise exec -- cmake --build --preset release --target c4a0_benchmarks
```

Run one unrecorded warm-up and then at least five measured executions:

```sh
build/release/c4a0_benchmarks
for benchmark_run in 1 2 3 4 5; do
  echo "run=${benchmark_run}"
  build/release/c4a0_benchmarks
done
```

Capture self-play peak RSS separately so the standalone MCTS workload does not affect it:

```sh
/usr/bin/time -v build/release/c4a0_benchmarks --self-play-only
```

For lower-noise comparisons, keep the governor and background load constant and optionally pin the
process to the same CPU set. Record any such pinning alongside the result because it differs from
the baseline above.

## Frozen Rust results

The legacy implementation was measured before deletion with an ephemeral Release binary compiled
inside the original Cargo package. The harness called the actual Rust `Pos`, `MctsGame`,
`self_play`, and record types; it was not a standalone reimplementation. Its position and MCTS loops
were operation-for-operation copies of the calibrated C++ workloads. The temporary harness and its
`rlib` build setting were removed after measurement.

The same host, governor, lack of CPU affinity, and otherwise-idle conditions described above were
used. The Rust toolchain was `rustc 1.89.0` and `cargo 1.89.0`; Cargo's standard Release profile was
used. The optimized build included the package's unconditional PyO3 and RocksDB dependency graph.

One full execution was discarded as warm-up. Five subsequent executions all produced checksum
`11000512` and the same self-play unique-position count (`7,138`).

| Run | Position composite ops/s | MCTS iterations/s | Self-play games/s | Full-run peak RSS (KiB) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6,755,318.444 | 943,679.948 | 975.832 | 60,176 |
| 2 | 6,699,205.759 | 932,863.801 | 931.862 | 59,816 |
| 3 | 6,711,459.511 | 954,158.592 | 931.789 | 58,544 |
| 4 | 6,708,818.023 | 940,370.199 | 929.493 | 59,968 |
| 5 | 6,723,740.655 | 926,887.572 | 933.315 | 59,044 |
| **Median** | **6,711,459.511** | **940,370.199** | **931.862** | **59,816** |

A separate self-play-only run reported 950.808 games/s and maximum resident set size 41,072 KiB
(approximately 40.1 MiB). As with the C++ RSS figure, this is a single-process peak rather than a
long-duration leak measurement.

## Comparable ratios and limitations

The position loop and standalone MCTS loop have genuinely equivalent inputs, operation counts,
coefficients, chunking, timing boundaries, and checksums. Their median ratios are therefore suitable
for the migration performance gate:

| Equivalent workload | Rust median | C++ median | C++ / Rust | 90% gate |
| --- | ---: | ---: | ---: | --- |
| Position composite loop | 6,711,459.511 ops/s | 15,773,298.457 ops/s | 235.020% | Pass |
| MCTS, 20 trees x 50,000 iterations | 940,370.199 iter/s | 3,332,630.015 iter/s | 354.396% | Pass |

The self-play calls have the same public workload parameters—512 games, batch size 32, 64 root
iterations, uniform evaluation, exploration 4.0, and ply penalty 0.01—but are not exact internal
workloads:

- Rust samples moves with its legacy `StdRng`; C++ intentionally uses the migration's portable PCG
  contract, so game trajectories and lengths are not identical.
- Rust's timed public `self_play` call constructs a `HashSet` of every returned sample position and
  prints its unique-position summary before returning. The C++ call does not perform that post-pass.
- Rust iterates deduplicated leaves in `HashSet` order, while C++ sorts them deterministically, so
  batch scheduling can differ even with the same uniform evaluator.

The raw medians were 931.862 games/s for Rust and 887.225 games/s for C++ (an observed C++/Rust
ratio of 95.210%), so the parameter-equivalent comparison also exceeds 90%. It remains an
end-to-end comparison rather than an operation-identical cutover ratio. The raw self-play-only
results were 950.808 games/s and 41,072 KiB for Rust versus 917.489 games/s and 8,064 KiB for C++
(96.496% of Rust throughput); they carry the same limitation. A C++ TTY diagnostic counted 470,121
MCTS iterations in its 512-game run, which rules out a trivially empty workload but does not make
the two trajectories identical.

The operation-equivalent position and standalone MCTS workloads therefore both satisfy the “at
least 90% of Rust” cutover gate. Raw self-play throughput also exceeds 90%, while remaining
supporting parameter-equivalent evidence rather than an exact operation-count gate.
