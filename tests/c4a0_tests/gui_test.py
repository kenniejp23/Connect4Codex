import os
import json
from pathlib import Path
import time
from typing import Any, cast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, QPoint, QPointF, QSettings, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtQml import QQmlApplicationEngine  # noqa: E402
from PySide6.QtQuickControls2 import QQuickStyle  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from c4a0.gui.app import (  # noqa: E402
    AppController,
    GameController,
    _configured_training_dir,
)
from c4a0.gui.jobs import JobManager  # noqa: E402


def _application(tmp_path: Path) -> QGuiApplication:
    QSettings.setPath(
        QSettings.Format.NativeFormat,
        QSettings.Scope.UserScope,
        str(tmp_path),
    )
    application = QGuiApplication.instance()
    if application is None:
        application = QGuiApplication([])
        QQuickStyle.setStyle("Material")
    return cast(QGuiApplication, application)


def _wait_until(application: QGuiApplication, predicate, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        application.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def test_gui_migrates_the_old_default_training_directory_once(tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    settings.setValue("paths/training", "training")

    assert _configured_training_dir(settings) == "training-v2"
    assert settings.value("paths/training") == "training-v2"

    settings.setValue("paths/training", "training")
    assert _configured_training_dir(settings) == "training"


def test_qml_shell_loads_and_game_controller_accepts_moves(tmp_path):
    application = _application(tmp_path)
    app_controller = AppController()
    game_controller = GameController(app_controller)
    jobs = JobManager()
    engine = QQmlApplicationEngine()
    context = engine.rootContext()
    context.setContextProperty("App", app_controller)
    context.setContextProperty("Game", game_controller)
    context.setContextProperty("Jobs", jobs)
    qml = Path(__file__).parents[2] / "src/c4a0/gui/qml/Main.qml"
    engine.load(QUrl.fromLocalFile(str(qml)))
    assert engine.rootObjects()
    window = cast(Any, engine.rootObjects()[0])
    window.setWidth(1024)
    window.setHeight(720)
    app_controller.setPage(1)
    application.processEvents()

    game_controller.startGame("Human", "Human", 1, 1.0, 0.01)
    _wait_until(application, lambda: game_controller._iterations >= 1, timeout=10)
    assert game_controller._iterations >= 1
    game_board = cast(Any, window.findChild(QObject, "gameBoard"))
    analysis = cast(Any, window.findChild(QObject, "analysisPanel"))
    board_origin = game_board.mapToItem(window.contentItem(), QPointF(0, 0))
    analysis_origin = analysis.mapToItem(window.contentItem(), QPointF(0, 0))
    assert board_origin.x() + game_board.width() < analysis_origin.x()

    click_position = QPoint(
        round(board_origin.x() + game_board.width() * 3.5 / 7),
        round(board_origin.y() + game_board.height() / 2),
    )
    QTest.mouseClick(
        window,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        click_position,
    )
    _wait_until(application, lambda: game_controller._move_count == 1)
    assert game_controller._board[-4] == 1
    assert game_controller._side == "gold"
    game_controller.undo()
    _wait_until(application, lambda: game_controller._move_count == 0)
    _wait_until(application, lambda: game_controller._iterations >= 1)
    window.requestActivate()
    assert QTest.qWaitForWindowActive(window, 2000)
    QTest.keyClick(window, Qt.Key.Key_4)
    _wait_until(application, lambda: game_controller._move_count == 1)
    assert game_controller._board[-4] == 1
    game_controller.endGame()
    assert game_controller._terminal == "ongoing"
    assert game_controller._board == [0] * 42
    game_controller.shutdown()
    for root in engine.rootObjects():
        cast(Any, root).close()
    import shiboken6

    shiboken6.delete(engine)


def test_job_manager_rejects_invalid_configuration_before_starting():
    jobs = JobManager()
    job_id = jobs.submit(
        "tournament",
        json.dumps({"players": ["latest"], "games_per_match": 3}),
        "Invalid tournament",
    )
    assert job_id == ""
    assert not jobs.active
    assert "Invalid configuration" in jobs._result


def test_primary_action_visible_in_both_themes_at_minimum_size(tmp_path):
    application = _application(tmp_path)
    controller = AppController()
    game = GameController(controller)
    jobs = JobManager()
    engine = QQmlApplicationEngine()
    for name, value in [("App", controller), ("Game", game), ("Jobs", jobs)]:
        engine.rootContext().setContextProperty(name, value)
    engine.load(
        QUrl.fromLocalFile(str(Path(__file__).parents[2] / "src/c4a0/gui/qml/Main.qml"))
    )
    window = cast(Any, engine.rootObjects()[0])
    window.setWidth(1024)
    window.setHeight(720)
    controller.setPage(1)
    for theme in ("light", "dark"):
        controller.setTheme(theme)
        application.processEvents()
        button = cast(Any, window.findChild(QObject, "startGameButton"))
        assert button.isVisible() and button.width() >= 100 and button.height() >= 40
        origin = button.mapToItem(window.contentItem(), QPointF(0, 0))
        assert 0 <= origin.x() <= 1024 - button.width()
        assert 0 <= origin.y() <= 720 - button.height()
        # Primary action uses explicit, contrasting colors in both themes.
        label = cast(Any, button.property("contentItem"))
        background = cast(Any, button.property("background"))
        assert label.property("color") != background.property("color")
    game.shutdown()
    window.close()
    import shiboken6

    shiboken6.delete(engine)


def test_rapid_game_starts_cancel_pending_loads(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    application = _application(tmp_path)
    controller = AppController()
    game = GameController(controller)
    game._executor.shutdown(wait=True)
    game._executor = ThreadPoolExecutor(max_workers=1)
    release = threading.Event()
    blocker = game._executor.submit(release.wait, 5)
    obsolete = []
    try:
        for _ in range(30):
            game.startGame("Human", "Human", 1, 1.0, 0.01)
            obsolete.append(game._load_future)
        assert all(
            future is not None and future.cancelled() for future in obsolete[:-1]
        )
        assert obsolete[-1] is not None and not obsolete[-1].cancelled()
        game.shutdown()
        assert obsolete[-1].cancelled()
    finally:
        release.set()
        blocker.result(timeout=3)
        game._executor.shutdown(wait=True)
        application.processEvents()
