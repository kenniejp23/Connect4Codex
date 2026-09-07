# Linux release candidate

Target: Linux x86-64, Ubuntu 24.04, Python 3.11. A published release requires the
qualification gates in [release-readiness.md](release-readiness.md); a release draft
or passing unit tests is not release approval.

Download the wheel, `linux-py311.txt`, `linux-runtime-packages.txt`, `install-linux.sh`,
and `SHA256SUMS` from the same GitHub Release. Verify downloaded files with
`sha256sum --ignore-missing -c SHA256SUMS`. Install Python 3.11 with its venv module,
then install the shared Ubuntu runtime list used by CI:

```sh
sudo apt-get update
xargs -r -a linux-runtime-packages.txt sudo apt-get install -y
```

```sh
bash install-linux.sh cpu /path/to/c4a0.whl /path/to/linux-py311.txt
# Or, with a compatible NVIDIA driver:
bash install-linux.sh nvidia /path/to/c4a0.whl /path/to/linux-py311.txt
```

The installer creates an isolated environment under `~/.local/share/c4a0/venv` and a
user desktop launcher. It installs the CPU build of PyTorch from PyTorch's CPU index,
or the constrained CUDA-enabled build from PyPI. See the [PyTorch installation
instructions](https://pytorch.org/get-started/locally/) for driver/build selection.
The NVIDIA path checks `torch.cuda.is_available()` before installing the app.
`C4A0_PYTHON`, `C4A0_INSTALL_DIR`, and `C4A0_DESKTOP_DIR` override the interpreter,
installation destination, and desktop-entry directory.
Launch from any directory with `~/.local/share/c4a0/venv/bin/c4a0 gui`.

`constraints/linux-py311.txt` pins runtime dependencies from `uv.lock`. CI separately
tests that environment and the current unconstrained resolver result. Notebook,
plotting, TensorBoardX, and Optuna dashboard tools are available through the `research`
extra rather than mandatory app dependencies.
The lock uses PyTorch 2.13.0 and updated audited transitive packages. CI audits these
runtime pins separately from functional tests; audit results depend on the current advisory database.

The release workflow builds the wheel and source archive on Ubuntu 24.04, adds the
constraints, installer, and SHA-256 checksums, and creates a GitHub Release **draft**.
Publish only after attaching the completed GUI, recovery, CUDA soak, and performance
evidence for that exact revision. NVIDIA qualification runs only on an explicitly
provisioned self-hosted runner; it is not triggered by pull requests.
