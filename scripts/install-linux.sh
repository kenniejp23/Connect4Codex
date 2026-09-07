#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 || ( "$1" != cpu && "$1" != nvidia ) ]]; then
  echo "Usage: $0 cpu|nvidia /path/to/c4a0.whl /path/to/linux-py311.txt" >&2
  exit 2
fi
mode="$1"
wheel_path="$(realpath "$2")"
constraints_path="$(realpath "$3")"
install_dir="${C4A0_INSTALL_DIR:-$HOME/.local/share/c4a0}"
python_binary="${C4A0_PYTHON:-python3.11}"
"$python_binary" -c 'import platform,sys; assert sys.version_info[:2] == (3,11); assert platform.system() == "Linux" and platform.machine() == "x86_64"'
"$python_binary" -m venv "$install_dir/venv"
runner="$install_dir/venv/bin/python"
"$runner" -m pip install --upgrade pip
if [[ "$mode" == cpu ]]; then
  "$runner" -m pip install --no-deps --constraint "$constraints_path" torch --index-url https://download.pytorch.org/whl/cpu
else
  "$runner" -m pip install --constraint "$constraints_path" torch
  "$runner" -c 'import torch; assert torch.cuda.is_available(), "NVIDIA driver is unavailable or incompatible with this PyTorch build"'
fi
"$runner" -m pip install --constraint "$constraints_path" "$wheel_path"
"$runner" -m pip check
desktop_dir="${C4A0_DESKTOP_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/applications}"
mkdir -p "$desktop_dir"
C4A0_DESKTOP_DIR="$desktop_dir" C4A0_LAUNCHER_PATH="$install_dir/venv/bin/c4a0" "$runner" - <<'PY'
import os
from pathlib import Path
executable = os.environ['C4A0_LAUNCHER_PATH']
# Desktop Entry quoting is separate from shell quoting.
quoted = executable.replace('\\', '\\\\').replace('"', '\\"').replace('`', '\\`').replace('$', '\\$').replace('%', '%%')
entry = '[Desktop Entry]\nType=Application\nName=c4a0 Connect Four\nExec="' + quoted + '" gui\nTerminal=false\nCategories=Game;Education;\n'
(Path(os.environ['C4A0_DESKTOP_DIR']) / 'c4a0.desktop').write_text(entry)
PY
printf 'Installed. Launch with: %s gui\n' "$install_dir/venv/bin/c4a0"
