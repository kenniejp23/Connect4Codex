# Ranked Improvement Backlog

Ranking is ordered for **highest impact with lowest difficulty first**. Impact and difficulty use a 1-5 scale where 5 is highest. Difficulty 1 means quick/easy; difficulty 5 means large/risky.

| Rank | Improvement | Impact | Difficulty | Why this order / expected payoff |
|---:|---|---:|---:|---|
| 4 | Fix Python packaging so `import c4a0` works after install | 5 | 3 | Done: scikit-build-core installs both `c4a0` and the `c4a0_cpp` native module without `PYTHONPATH` hacks. |
| 5 | Add a first-class self-play stats CLI command | 4 | 2 | Today stats require ad-hoc pickle loading. A command like `stats --base-dir training` should report games, samples, unique positions, lengths, win rates, losses, and solver scores. |
| 6 | Make training duration configurable | 4 | 2 | `max_epochs=100`, early stopping patience, and logging cadence are hard-coded. CLI/config knobs would make smoke runs, CPU runs, and serious GPU runs less awkward. |
| 7 | Move Lightning logs under each training run directory | 4 | 2 | `lightning_logs/` currently lands at repo root. Keeping logs with generation artifacts improves reproducibility and cleanup. |
| 9 | Replace or wrap pickle artifacts with versioned formats | 4 | 3 | `model.pkl` and `games.pkl` are fragile across code changes. Prefer `state_dict` + JSON config and a versioned CBOR/NPZ/parquet format for self-play samples. |
| 10 | Save full experiment config and environment metadata | 4 | 2 | Store CLI args, model config, git SHA, package versions, device, seed, and command in each run for reproducibility. |
| 11 | Add deterministic seed controls | 4 | 3 | Training currently mixes C++/Python/PyTorch randomness. Explicit seeds enable reproducible smoke tests and more reliable comparisons. |
| 12 | Modernize `pyproject.toml` dependency groups | 3 | 1 | `tool.uv.dev-dependencies` is deprecated. Move to `dependency-groups.dev` and pin/organize dev tools. |
| 14 | Generate or maintain native type stubs | 3 | 2 | `c4a0_cpp/__init__.pyi` is checked against the extension API to prevent drift. |
| 15 | Add solver setup automation | 3 | 2 | Provide a script or documented task to fetch/build Pascal Pons solver and opening book, with cache paths managed consistently. |
| 17 | Improve training metric logging | 3 | 2 | Log policy KL, value MSEs, unique positions, game lengths, win/draw rates, and solver score in a consistent dashboard-friendly format. |
