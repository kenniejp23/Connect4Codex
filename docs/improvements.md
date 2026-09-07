# Implementation notes and remaining limitations

This page records concrete limitations in the current implementation. It is not a ranked roadmap.

| Area | Current behavior | Source |
| --- | --- | --- |
| Legacy artifacts | The synchronous trainer still stores pickled models and games. V2 has versioned checkpoints, CBOR shards, and a SQLite manifest; it does not migrate legacy runs. | [`training.py`](../src/c4a0/training.py), [`training_v2.py`](../src/c4a0/training_v2.py) |
| Run provenance | V2 records configuration and resumable state. Benchmark reports additionally record Git revision, Python/PyTorch versions, platform, and device; regular run creation does not collect that same environment report. | [`training_v2.py`](../src/c4a0/training_v2.py), [`training_benchmark.py`](../src/c4a0/training_benchmark.py) |
| Solver setup | Commands accept an external executable and opening book. There is no solver installation task in mise. | [`main.py`](../src/c4a0/main.py), [`mise.toml`](../mise.toml) |
| Benchmark coverage | Component benchmarks remain sequential. The production benchmark exercises replay, candidate submission, arena decisions and persistence, with separate saved-candidate quality metrics. Extended production qualification is tracked separately. | [`production_benchmark.py`](../src/c4a0/production_benchmark.py), [release gates](release-readiness.md) |
| Platform validation | Checked-in CI targets Ubuntu; the TSan preset includes an x86-64 Linux `setarch` command. | [CI workflow](../.github/workflows/ci.yaml), [`CMakePresets.json`](../CMakePresets.json) |

Training duration controls, run-local Lightning logging, V2 seed controls, native type stubs,
and modern Python development dependency groups are already implemented. Their usage belongs in
the [development guide](development.md), rather than an outstanding-work list.

Release remediation and its current evidence are tracked in [release-readiness.md](release-readiness.md).
Further replay vectorization, native worker-pool changes, and smaller projected heads require
profiling and controlled strength comparisons. Existing checkpoint architecture and learning
defaults are preserved. Direct native API users must supply callbacks that return: native close
waits for an in-flight callback. The desktop isolates inference in a terminable child process,
so a blocked model does not prevent game shutdown. Worker jobs have process-group escalation
and solver deadlines.
