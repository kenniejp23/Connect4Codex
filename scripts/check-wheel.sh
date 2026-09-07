#!/usr/bin/env bash
set -euo pipefail

wheel_dir="$(mktemp -d)"
venv_dir="$(mktemp -d)"
cleanup() {
  rm -rf -- "$wheel_dir" "$venv_dir"
}
trap cleanup EXIT

uv build --sdist --out-dir "$wheel_dir"
sdist_path="$(find "$wheel_dir" -maxdepth 1 -type f -name '*.tar.gz' -print -quit)"
if [[ -z "$sdist_path" ]]; then
  echo "source build did not produce an sdist" >&2
  exit 1
fi
if tar -tzf "$sdist_path" | grep -Eq '/(rust|c4a0_rust)(/|$)|find-libclang\.py'; then
  echo "sdist contains a removed native-backend artifact" >&2
  exit 1
fi

uv build --wheel --out-dir "$wheel_dir" "$sdist_path"
uv venv --python 3.11 "$venv_dir"
wheel_path="$(find "$wheel_dir" -maxdepth 1 -type f -name '*.whl' -print -quit)"
if [[ -z "$wheel_path" ]]; then
  echo "wheel build did not produce a wheel" >&2
  exit 1
fi
constraint_args=()
if [[ "${1:-resolver}" == "locked" ]]; then
  constraint_args=(--constraint "$(pwd)/constraints/linux-py311.txt")
fi
UV_LINK_MODE=hardlink uv pip install --python "$venv_dir/bin/python" "${constraint_args[@]}" "$wheel_path"
(
cd "$venv_dir"
env -u PYTHONPATH QT_QPA_PLATFORM=offscreen "$venv_dir/bin/python" - <<'PY'
import importlib.util
from importlib.resources import files

import numpy as np

import c4a0
import c4a0_cpp
import c4a0.gui.app

assert c4a0_cpp.N_ROWS == 6
assert c4a0_cpp.N_COLS == 7
assert importlib.util.find_spec("c4a0_rust") is None
assert files("c4a0.gui").joinpath("qml", "Main.qml").is_file()

def evaluate(_model_id, positions):
    batch_size = positions.shape[0]
    return (
        np.zeros((batch_size, 7), dtype=np.float32),
        np.zeros((batch_size,), dtype=np.float32),
        np.zeros((batch_size,), dtype=np.float32),
    )

games = c4a0_cpp.play_games(
    [c4a0_cpp.GameMetadata(0, 0, 0)], 8, 1, 1.4, 0.01, evaluate
)
assert len(games.results) == 1
assert games.results[0].samples[-1].to_numpy()[0].shape == (2, 6, 7)
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QUrl
from c4a0.gui.app import AppController, GameController
from c4a0.gui.jobs import JobManager
application = QGuiApplication([])
controller = AppController()
game = GameController(controller)
jobs = JobManager()
engine = QQmlApplicationEngine()
engine.rootContext().setContextProperty("App", controller)
engine.rootContext().setContextProperty("Game", game)
engine.rootContext().setContextProperty("Jobs", jobs)
engine.load(QUrl.fromLocalFile(str(files("c4a0.gui").joinpath("qml", "Main.qml"))))
assert engine.rootObjects()
application.processEvents()
game.shutdown()
import shiboken6
shiboken6.delete(engine)
application.processEvents()
print(c4a0.__file__)
print(c4a0_cpp.__file__)
PY
env -u PYTHONPATH "$venv_dir/bin/python" -m c4a0.main --help >/dev/null
)
