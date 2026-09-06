# Implementation notes and remaining limitations

This page records concrete limitations in the current implementation. It is not a ranked roadmap.

| Area | Current behavior | Source |
| --- | --- | --- |
| Legacy artifacts | The synchronous trainer still stores pickled models and games. V2 has versioned checkpoints, CBOR shards, and a SQLite manifest; it does not migrate legacy runs. | [`training.py`](../src/c4a0/training.py), [`training_v2.py`](../src/c4a0/training_v2.py) |
| Run provenance | V2 records configuration and resumable state. Benchmark reports additionally record Git revision, Python/PyTorch versions, platform, and device; regular run creation does not collect that same environment report. | [`training_v2.py`](../src/c4a0/training_v2.py), [`training_benchmark.py`](../src/c4a0/training_benchmark.py) |
| Solver setup | Commands accept an external executable and opening book. There is no solver installation task in mise. | [`main.py`](../src/c4a0/main.py), [`mise.toml`](../mise.toml) |
| Benchmark coverage | The fixed-work benchmark runs sequential stages for both engines. It does not exercise the production asynchronous coordinator, replay retention, or SPRT promotion lifecycle. | [`training_benchmark.py`](../src/c4a0/training_benchmark.py) |
| Platform validation | Checked-in CI targets Ubuntu; the TSan preset includes an x86-64 Linux `setarch` command. | [CI workflow](../.github/workflows/ci.yaml), [`CMakePresets.json`](../CMakePresets.json) |

Training duration controls, run-local Lightning logging, V2 seed controls, native type stubs,
and modern Python development dependency groups are already implemented. Their usage belongs in
the [development guide](development.md), rather than an outstanding-work list.
