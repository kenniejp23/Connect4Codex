#!/usr/bin/env python3
"""Capture every desktop page at supported sizes and themes on the selected Qt backend."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import shiboken6

from PySide6.QtCore import QSettings, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle

from c4a0.gui.app import AppController, GameController
from c4a0.gui.jobs import JobManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    application = QGuiApplication([])
    QQuickStyle.setStyle("Material")
    with tempfile.TemporaryDirectory(prefix="c4a0-gui-qualification-") as temporary:
        QSettings.setPath(
            QSettings.Format.NativeFormat, QSettings.Scope.UserScope, temporary
        )
        app = AppController()
        app.setTrainingDir(str(Path(temporary) / "run"))
        game = GameController(app)
        jobs = JobManager()
        engine = QQmlApplicationEngine()
        for name, controller in [("App", app), ("Game", game), ("Jobs", jobs)]:
            engine.rootContext().setContextProperty(name, controller)
        qml = (
            Path(__import__("c4a0.gui.app", fromlist=["__file__"]).__file__).parent
            / "qml/Main.qml"
        )
        engine.load(QUrl.fromLocalFile(str(qml)))
        if not engine.rootObjects():
            raise RuntimeError("GUI failed to load")
        window = engine.rootObjects()[0]
        captures = []
        for theme in ["dark", "light"]:
            app.setTheme(theme)
            for width, height in [(1024, 720), (1366, 768), (1920, 1080)]:
                window.setWidth(width)
                window.setHeight(height)
                for page, name in enumerate(
                    [
                        "home",
                        "play",
                        "training",
                        "evaluation",
                        "models",
                        "validation",
                        "settings",
                    ]
                ):
                    app.setPage(page)
                    deadline = time.monotonic() + 0.12
                    while time.monotonic() < deadline:
                        application.processEvents()
                        time.sleep(0.005)
                    image = window.grabWindow()
                    captures.append(
                        {
                            "theme": theme,
                            "page": name,
                            "requested_size": [width, height],
                            "actual_size": [window.width(), window.height()],
                            "pixel_size": [image.width(), image.height()],
                        }
                    )
                    if image.isNull() or not image.save(
                        str(args.output / f"{theme}-{width}x{height}-{name}.png")
                    ):
                        raise RuntimeError("GUI screenshot failed")
        (args.output / "environment.json").write_text(
            json.dumps(
                {
                    "backend": application.platformName(),
                    "scale_factor": os.environ.get("QT_SCALE_FACTOR", "1"),
                    "device_pixel_ratio": window.devicePixelRatio(),
                    "screenshots": 42,
                    "captures": captures,
                    "all_requested_sizes_matched": all(
                        item["requested_size"] == item["actual_size"]
                        for item in captures
                    ),
                },
                indent=2,
            )
        )
        game.shutdown()
        window.close()
        shiboken6.delete(engine)
        application.processEvents()


if __name__ == "__main__":
    main()
