# Release qualification

The remediation implementation is covered by the evidence below. **A public release is not
approved yet.** Remaining qualification gates are listed explicitly; passing these regression
checks does not establish long-run training stability or playing strength.

## Implemented changes

- Native interactive callbacks release the board mutex. Board revisions invalidate stale results;
  repeatable error diagnostics and `retry_evaluation()` preserve the board. Desktop model loading
  and inference run in a child process with a 30-second deadline and bounded shutdown.
- Checkpoints own CPU tensor storage, including optimizer state. Candidate counters identify the
  saved validation snapshot. Value-loss coefficients are independent of sample eligibility.
- Versioned inference settings are shared across play, training and evaluation; old V2 runs recover
  them from their manifest. Compact `export-model` artifacts exclude optimizer/RNG state and use
  restricted loading. Standalone full learner checkpoints require `--trusted-checkpoint`; missing
  old inference settings additionally require `--mcts-value-scale`.
- Minimax uses 69 legal winning masks, bounded depth, a deadline, and cancellation checks. Sweeps
  use generation-zero optimizer scheduling and cancellation inside train/validation batches.
- Jobs finalize startup failures, malformed events and missing terminal events as failures.
  Cancellation timers belong to one process; queued work waits for worker descendants to exit.
  Shutdown prevents queued starts, and external solvers have deadlines.
- Coordinators use Linux advisory locks; children receive a parent-death signal. Readers use
  read-only SQLite connections. Artifact paths migrate to run-relative storage, and atomic
  renames synchronize their directories. Resume settings are visible and immutable in the GUI.
- Model discovery/statistics are asynchronous. Forms reflow at minimum size, primary actions
  have explicit contrast, and board columns expose keyboard/accessibility actions and status
  announcements. Complete presets come from Python. Logs and job history are bounded.
- Component reports validate workload identity and finite metrics. Production benchmarking runs
  the real coordinator and measures saved-candidate quality on a separate held-out opening bank.
  Paired multinomial GSPRT simulation coverage is distinct from production calibration.
- Research dependencies are optional. Runtime constraints, CPU/NVIDIA installation, launcher,
  wheel qualification, pull-request CI, sanitizer jobs, audited pins, release-draft packaging,
  and controlled NVIDIA qualification are provided.

## Evidence collected on 7 September 2026

These are development-worktree results, not qualification of a published release artifact.
The retained benchmark reports identify their original revision and dirty-tree state.

| Check | Latest completed result |
|---|---|
| Python suite | 115 passed |
| Native suite | 33 passed |
| ASan/UBSan | 33 passed |
| TSan | 33 passed |
| Ruff / Pyright | Passed; zero type errors or warnings |
| Runtime dependency audit | No known vulnerabilities reported for updated pins; [audit](qualification/runtime-audit.json) |
| Persistence recovery | Forced termination before/after shard registration, candidate saving and promotion, plus during arena evaluation; consistent resume passed |
| Ownership / cancellation | Second writer rejected, readers usable, restart after owner death, automatic child exit, malformed/startup failures, queue isolation and descendant cleanup passed |
| Blocked interactive model | Closing interrupted the waiting evaluator and reaped its child; saved-model inference consistency passed |
| X11 (Xvfb and desktop XWayland) | 4 GUI tests passed on each backend |
| Wayland desktop | 4 GUI tests passed on the active COSMIC compositor |
| GUI rendering | Seven pages, both themes, three sizes (1024×720, 1366×768, 1920×1080), including 150% scaling; inspection led to layout/contrast fixes |
| CUDA | PyTorch 2.13.0+cu130 on RTX 2050, 4 GB: forced acceptance/rejection, resume, injected OOM recovery, and 301-second natural-decision soak passed; [report](qualification/cuda-bounded.json) |
| Production smoke | Three actual candidate decisions with held-out saved-candidate evaluation and persistence passed; [report](qualification/production-smoke.json) |
| Packaging | Source archive → wheel → clean installation passed with locked PyTorch 2.13.0 and unconstrained 2.14.0; CPU installer and `pip check` passed |

The bounded CUDA workload peaked at 780 MiB device-wide GPU memory and roughly 4.48 GiB summed
process-tree RSS (shared pages may be counted repeatedly). Other validation work ran concurrently;
these numbers are diagnostics, not a performance acceptance baseline. OOM was injected, and
accept/reject lifecycle decisions were forced before the natural-decision soak. The production
smoke used a tiny network/workload; its scores are not evidence of release playing strength.

## Required before publication

- Run the extended CUDA qualification on the release revision. The NVIDIA workflow defaults to
  eight hours. Exercise physical memory pressure in addition to synthetic OOM handler coverage.
- Complete keyboard-only and screen-reader workflow qualification on X11 and Wayland, including
  every evaluation/training workflow, populated/error states, both themes and increased scaling.
  The four GUI tests establish a narrower set of interactions.
- Collect repeated, isolated production measurements, inspect memory growth and GUI response
  latency, and evaluate strength on controlled held-out data. Calibrate statistical error rates
  for production openings; synthetic distributions alone do not establish those rates.
- Qualify the exact wheel on a clean Ubuntu 24.04 machine through both CPU and NVIDIA installers.
  Local virtual environments are useful checks but are not clean-machine certification.
- Attach this evidence to a release draft for the tested revision before publishing it.

## Commands

```sh
mise run check
mise run test:sanitize
mise run test:thread
mise exec -- bash scripts/check-wheel.sh locked
mise exec -- bash scripts/check-wheel.sh resolver
mise exec -- uvx pip-audit==2.10.1 --disable-pip --no-deps -r constraints/linux-py311.txt
QT_QPA_PLATFORM=offscreen uv run python scripts/qualify-gui.py qualification/gui
QT_SCALE_FACTOR=1.5 QT_QPA_PLATFORM=offscreen uv run python scripts/qualify-gui.py qualification/gui-scale150
uv run python scripts/qualify-training.py --device cuda --inject-oom --seconds 28800 --output qualification/cuda.json
uv run c4a0 benchmark-production config.json qualification/production.json --repeats 3
```

Learning defaults and checkpoint architecture remain unchanged. Policy-only inference skips the
value head when disabled. Replay vectorization, smaller heads and native thread-pool changes
remain experiments requiring profiles and strength evidence before adoption.
