"""Application entrypoint and Qt-facing controllers."""

from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import sys
from typing import Any

from PySide6.QtCore import Property, QSettings, QTimer, QObject, QUrl, Signal, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle
import torch
import shiboken6

import c4a0_cpp
from c4a0.config import TrainingV2Config
from c4a0.training import TrainingGen
from c4a0.gui.inference import ProcessEvaluator, build_players
from c4a0.utils import get_torch_device
from c4a0.gui.jobs import JobManager
from c4a0.training_v2 import (
    effective_training_config,
    is_v2_run,
    list_attempts,
    training_status,
)


def _configured_training_dir(settings: QSettings) -> str:
    """Migrate the former GUI default once, without blocking explicit legacy use."""
    migration_key = "migrations/trainingV2Default"
    configured = str(settings.value("paths/training", "training-v2"))
    if not settings.value(migration_key, False, type=bool):
        if configured.strip() in {"", "training", "./training"}:
            configured = "training-v2"
            settings.setValue("paths/training", configured)
        settings.setValue(migration_key, True)
    return configured


class AppController(QObject):
    changed = Signal()
    statsReady = Signal(str)
    modelsChanged = Signal()
    _modelsReady = Signal(int, object)
    _statsFinished = Signal(int, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._settings = QSettings("c4a0", "c4a0")
        self._page = int(str(self._settings.value("navigation/page", 0)))
        self._training_dir = _configured_training_dir(self._settings)
        self._solver_path = str(self._settings.value("paths/solver", ""))
        self._book_path = str(self._settings.value("paths/book", ""))
        self._solutions_path = str(
            self._settings.value("paths/solutions", "./solutions.db")
        )
        self._theme = str(self._settings.value("appearance/theme", "dark"))
        self._preferred_play_player = "Latest Model"
        self._reduced_motion = (
            str(self._settings.value("appearance/reducedMotion", "false")).lower()
            == "true"
        )
        self._device = str(get_torch_device())
        self._generations: list[dict[str, Any]] = []
        self._model_error = ""
        self._training_v2 = True
        self._refresh_revision = 0
        self._stats_revision = 0
        self._statsFinished.connect(self._apply_stats)
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="c4a0-data"
        )
        self._modelsReady.connect(self._apply_models)
        self.refreshGenerations()

    @Property(int, notify=changed)
    def page(self) -> int:
        return self._page

    @Property(str, notify=changed)
    def trainingDir(self) -> str:
        return self._training_dir

    @Property(bool, notify=changed)
    def trainingV2(self) -> bool:
        return self._training_v2

    @Property(str, notify=changed)
    def modelError(self) -> str:
        return self._model_error

    @Slot(str, result=str)
    def trainingPreset(self, name: str) -> str:
        return TrainingV2Config.preset(name).model_dump_json()

    @Slot(str, result=str)
    def trainingConfiguration(self, directory: str) -> str:
        try:
            config = effective_training_config(TrainingV2Config(base_dir=directory))
            return json.dumps(
                {
                    "resume": is_v2_run(directory),
                    "config": config.model_dump(mode="json"),
                }
            )
        except Exception as error:
            return json.dumps({"error": str(error)})

    @Property(str, notify=changed)
    def modelCollectionName(self) -> str:
        return "attempts" if self.trainingV2 else "generations"

    @Property(str, notify=changed)
    def solverPath(self) -> str:
        return self._solver_path

    @Property(str, notify=changed)
    def bookPath(self) -> str:
        return self._book_path

    @Property(str, notify=changed)
    def solutionsPath(self) -> str:
        return self._solutions_path

    @Property(str, notify=changed)
    def theme(self) -> str:
        return self._theme

    @Property(bool, notify=changed)
    def reducedMotion(self) -> bool:
        return self._reduced_motion

    @Property(str, constant=True)
    def device(self) -> str:
        return self._device

    @Property(str, notify=changed)
    def interactiveDevice(self) -> str:
        return str(self._settings.value("compute/interactiveDevice", "cpu"))

    @Slot(str)
    def setInteractiveDevice(self, device: str) -> None:
        if device not in {"cpu", "cuda"} or (
            device == "cuda" and not torch.cuda.is_available()
        ):
            return
        self._settings.setValue("compute/interactiveDevice", device)
        self.changed.emit()

    @Property(bool, constant=True)
    def cudaAvailable(self) -> bool:
        return torch.cuda.is_available()

    @Property(bool, constant=True)
    def sourceCheckout(self) -> bool:
        return (Path.cwd() / "mise.toml").is_file()

    @Property(list, notify=modelsChanged)
    def generations(self) -> list[dict[str, Any]]:
        return self._generations

    @Property(str, notify=changed)
    def latestGeneration(self) -> str:
        if not self._generations:
            return "No trained model"
        champion = next(
            (item for item in self._generations if item.get("isChampion")),
            self._generations[0],
        )
        return f"Champion {champion['generation']}"

    @Property(list, notify=modelsChanged)
    def playerOptions(self) -> list[str]:
        options = ["Human", "Latest Model"]
        label = "Attempt" if self.trainingV2 else "Generation"
        options.extend(f"{label} {item['generation']}" for item in self._generations)
        options.extend(["Random", "Uniform"])
        return options

    @Property(str, notify=changed)
    def preferredPlayPlayer(self) -> str:
        return self._preferred_play_player

    @Slot(int)
    def setPage(self, page: int) -> None:
        self._page = page
        self._settings.setValue("navigation/page", page)
        self.changed.emit()

    @Slot(str)
    def setTrainingDir(self, path: str) -> None:
        normalized = path.strip() or "training-v2"
        if normalized == self._training_dir:
            return
        self._training_dir = normalized
        self._stats_revision += 1
        self._generations = []
        self.modelsChanged.emit()
        self._settings.setValue("paths/training", normalized)
        self.refreshGenerations()

    @Slot(str, str, str)
    def setSolverPaths(self, solver: str, book: str, solutions: str) -> None:
        self._solver_path = solver.strip()
        self._book_path = book.strip()
        self._solutions_path = solutions.strip() or "./solutions.db"
        self._settings.setValue("paths/solver", self._solver_path)
        self._settings.setValue("paths/book", self._book_path)
        self._settings.setValue("paths/solutions", self._solutions_path)
        self.changed.emit()

    @Slot(str)
    def setTheme(self, theme: str) -> None:
        self._theme = "light" if theme == "light" else "dark"
        self._settings.setValue("appearance/theme", self._theme)
        self.changed.emit()

    @Slot(bool)
    def setReducedMotion(self, reduced: bool) -> None:
        self._reduced_motion = reduced
        self._settings.setValue("appearance/reducedMotion", reduced)
        self.changed.emit()

    @Slot()
    def refreshGenerations(self) -> None:
        self._refresh_revision += 1
        revision, directory = self._refresh_revision, self._training_dir
        self._model_error = "Loading models…"
        self.changed.emit()

        def collect():
            try:
                v2 = is_v2_run(directory)
                if v2:
                    config = effective_training_config(
                        TrainingV2Config(base_dir=directory)
                    )
                    models = [
                        {
                            "generation": int(item["attempt_n"]),
                            "created": str(item["created_at"])[:16].replace("T", " "),
                            "valLoss": item["val_loss"],
                            "solverScore": None,
                            "mctsIterations": config.n_mcts_iterations,
                            "exploration": config.c_exploration,
                            "status": item["status"],
                            "isChampion": item["is_champion"],
                        }
                        for item in list_attempts(directory)
                        if Path(item["checkpoint_path"]).is_file()
                    ]
                else:
                    generations = TrainingGen.load_all(directory)
                    v2 = not generations
                    models = [
                        {
                            "generation": generation.gen_n,
                            "created": generation.created_at.strftime("%Y-%m-%d %H:%M"),
                            "valLoss": generation.val_loss,
                            "solverScore": generation.solver_score,
                            "mctsIterations": generation.n_mcts_iterations,
                            "exploration": generation.c_exploration,
                        }
                        for generation in generations
                    ]
                return models, v2, ""
            except Exception as error:
                return (
                    [],
                    is_v2_run(directory),
                    f"Cannot read models in {directory}: {error}",
                )

        future = self._executor.submit(collect)
        future.add_done_callback(
            lambda result: self._modelsReady.emit(revision, result.result())
        )

    @Slot(int, object)
    def _apply_models(self, revision, result):
        if revision != self._refresh_revision:
            return
        models, self._training_v2, self._model_error = result
        if not self._model_error:
            if self._generations != models:
                self._generations = models
                self.modelsChanged.emit()
        self.changed.emit()

    @Slot(int)
    def useGenerationInPlay(self, generation_number: int) -> None:
        label = "Attempt" if self.trainingV2 else "Generation"
        option = f"{label} {generation_number}"
        if not any(
            item["generation"] == generation_number for item in self._generations
        ):
            return
        self._preferred_play_player = option
        self.setPage(1)

    @Slot(int, result=str)
    def generationStats(self, generation_number: int) -> str:
        directory = self._training_dir
        self._stats_revision += 1
        revision = self._stats_revision
        future = self._executor.submit(
            self._generation_stats, directory, generation_number
        )
        future.add_done_callback(
            lambda result: self._statsFinished.emit(revision, result.result())
        )
        return "Loading model statistics…"

    @Slot(int, str)
    def _apply_stats(self, revision: int, value: str) -> None:
        if revision == self._stats_revision:
            self.statsReady.emit(value)

    @staticmethod
    def _generation_stats(directory: str, generation_number: int) -> str:
        try:
            if is_v2_run(directory):
                attempt = next(
                    item
                    for item in list_attempts(directory)
                    if int(item["attempt_n"]) == generation_number
                )
                return json.dumps(
                    {"attempt": attempt, "run": training_status(directory)},
                    indent=2,
                    default=str,
                )
            generation = next(
                item
                for item in TrainingGen.load_all(directory)
                if item.gen_n == generation_number
            )
            games = generation.get_games(directory)
            payload = {
                "generation": generation.gen_n,
                "artifact": generation.gen_folder(directory),
                "mctsIterations": generation.n_mcts_iterations,
                "exploration": generation.c_exploration,
                "plyPenalty": generation.c_ply_penalty,
                "networkConfig": (
                    generation.network_config.model_dump()
                    if generation.network_config is not None
                    else {}
                ),
            }
            if games is None:
                payload["message"] = "Root generation has no self-play games"
            else:
                lengths = [len(result.samples) for result in games.results]
                scores = [result.player0_score() for result in games.results]
                payload.update(
                    {
                        "games": len(games.results),
                        "samples": sum(lengths),
                        "uniquePositions": games.unique_positions(),
                        "averageMoves": (sum(lengths) / len(lengths) if lengths else 0),
                        "redWins": sum(score == 1.0 for score in scores),
                        "draws": sum(score == 0.5 for score in scores),
                        "goldWins": sum(score == 0.0 for score in scores),
                    }
                )
            return json.dumps(payload, indent=2)
        except Exception as error:
            return json.dumps({"error": str(error)}, indent=2)


class GameController(QObject):
    changed = Signal()
    _loaded = Signal(int, object)

    def __init__(self, app: AppController, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._app = app
        self._game = None
        self._router: ProcessEvaluator | None = None
        self._board = [0] * 42
        self._legal = [True] * 7
        self._winning: set[int] = set()
        self._policy = [0.0] * 7
        self._side = "red"
        self._terminal = "ongoing"
        self._status = "Choose players, then start a game"
        self._error = ""
        self._q_penalty = 0.0
        self._q_no_penalty = 0.0
        self._iterations = 0
        self._max_iterations = 1400
        self._exploration = 6.6
        self._ply_penalty = 0.01
        self._background_running = False
        self._move_count = 0
        self._revision = 0
        self._red_spec = "Human"
        self._gold_spec = "Latest Model"
        self._auto = {"red": False, "gold": True}
        self._load_revision = 0
        self._load_future: Future | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="c4a0-play"
        )
        self._loaded.connect(self._apply_loaded)
        self._timer = QTimer(self)
        self._timer.setInterval(75)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    @Property(bool, notify=changed)
    def active(self) -> bool:
        return self._game is not None

    @Property(list, notify=changed)
    def board(self) -> list[int]:
        return self._board

    @Property(list, notify=changed)
    def legalMoves(self) -> list[bool]:
        return self._legal

    @Property(list, notify=changed)
    def policy(self) -> list[float]:
        return self._policy

    @Property(str, notify=changed)
    def sideToMove(self) -> str:
        return self._side

    @Property(str, notify=changed)
    def terminalState(self) -> str:
        return self._terminal

    @Property(str, notify=changed)
    def status(self) -> str:
        return self._status

    @Property(str, notify=changed)
    def error(self) -> str:
        return self._error

    @Property(float, notify=changed)
    def qPenalty(self) -> float:
        return self._q_penalty

    @Property(float, notify=changed)
    def qNoPenalty(self) -> float:
        return self._q_no_penalty

    @Property(int, notify=changed)
    def iterations(self) -> int:
        return self._iterations

    @Property(int, notify=changed)
    def maxIterations(self) -> int:
        return self._max_iterations

    @Property(bool, notify=changed)
    def searching(self) -> bool:
        return self._background_running

    @Property(int, notify=changed)
    def moveCount(self) -> int:
        return self._move_count

    @Property(int, notify=changed)
    def revision(self) -> int:
        return self._revision

    @Property(str, notify=changed)
    def redPlayer(self) -> str:
        return self._red_spec

    @Property(str, notify=changed)
    def goldPlayer(self) -> str:
        return self._gold_spec

    @Property(bool, notify=changed)
    def humanTurn(self) -> bool:
        return (
            self.active
            and self._terminal == "ongoing"
            and not self._auto[self._side]
            and self._iterations > 0
        )

    @Slot(int, result=bool)
    def isWinningCell(self, index: int) -> bool:
        return index in self._winning

    @Slot(str, str, int, float, float)
    def startGame(
        self,
        red_spec: str,
        gold_spec: str,
        iterations: int,
        exploration: float,
        ply_penalty: float,
    ) -> None:
        self.shutdown()
        revision = self._load_revision
        directory = self._app._training_dir
        # Interactive CPU inference avoids competing with CUDA training by default.
        device = str(self._app._settings.value("compute/interactiveDevice", "cpu"))
        self._red_spec, self._gold_spec = red_spec, gold_spec
        self._exploration, self._ply_penalty = exploration, ply_penalty
        self._auto = {"red": red_spec != "Human", "gold": gold_spec != "Human"}
        self._status, self._error = "Loading players…", ""
        self.changed.emit()

        def build():
            if revision != self._load_revision:
                return None, None, None
            try:
                router = ProcessEvaluator(
                    build_players, (red_spec, gold_spec, directory, device)
                )
                game = c4a0_cpp.InteractivePlay(
                    router.forward_numpy,
                    max(1, iterations),
                    max(0.0, exploration),
                    max(0.0, ply_penalty),
                    0,
                    1,
                )
                return game, router, None
            except Exception as error:
                return None, None, str(error)

        future = self._executor.submit(build)
        self._load_future = future
        future.add_done_callback(
            lambda result: (
                self._loaded.emit(revision, result.result())
                if not result.cancelled()
                else None
            )
        )

    @Slot(int, object)
    def _apply_loaded(self, revision, result):
        game, router, error = result
        if revision != self._load_revision:
            if game is not None:
                self._executor.submit(self._close_game, game, router)
            return
        self._game, self._router = game, router
        if error is not None:
            self._error, self._status = error, "Could not start game"
            self.changed.emit()
        else:
            self._poll()

    @Slot(int)
    def makeMove(self, column: int) -> None:
        if not self.humanTurn or self._game is None:
            return
        try:
            self._game.make_move(column)
            self._poll()
        except Exception as error:
            self._set_error(error)

    @Slot()
    def bestMove(self) -> None:
        if self._game is None or self._terminal != "ongoing" or self._iterations == 0:
            return
        try:
            self._game.make_best_move()
            self._poll()
        except Exception as error:
            self._set_error(error)

    @Slot()
    def randomMove(self) -> None:
        if self._game is None or self._terminal != "ongoing" or self._iterations == 0:
            return
        try:
            self._game.make_random_move(1.0)
            self._poll()
        except Exception as error:
            self._set_error(error)

    @Slot(int)
    def addIterations(self, count: int) -> None:
        if self._game is None:
            return
        try:
            self._game.increase_mcts_iterations(max(1, count))
            self._poll()
        except Exception as error:
            self._set_error(error)

    @Slot()
    def undo(self) -> None:
        if self._game is not None:
            try:
                self._game.undo()
                self._poll()
            except Exception as error:
                self._set_error(error)

    @Slot()
    def rematch(self) -> None:
        if self._game is not None:
            try:
                self._game.reset()
                self._poll()
            except Exception as error:
                self._set_error(error)

    @Slot()
    def swapSides(self) -> None:
        if self._game is None:
            return
        self.startGame(
            self._gold_spec,
            self._red_spec,
            self._max_iterations,
            self._exploration,
            self._ply_penalty,
        )

    @Slot()
    def endGame(self) -> None:
        self.shutdown()
        self._status = "Choose players, then start a game"
        self._error = ""
        self._board = [0] * 42
        self._legal = [True] * 7
        self._winning.clear()
        self._policy = [0.0] * 7
        self._terminal = "ongoing"
        self._move_count = 0
        self._revision += 1
        self.changed.emit()

    @Slot()
    def retry(self) -> None:
        if self._game is None or not self._error:
            return
        self._error = ""
        self._game.retry_evaluation()
        self._status = "Retrying evaluation"
        self._poll()

    def _set_error(self, error: Exception) -> None:
        self._error = str(error)
        self._status = "Game paused after an evaluation error"
        self.changed.emit()

    def _poll(self) -> None:
        if self._game is None or self._error:
            return
        try:
            snapshot = self._game.snapshot()
            self._board = [cell for row in snapshot.board for cell in row]
            self._legal = list(snapshot.legal_moves)
            self._winning = {
                int(row) * 7 + int(column) for row, column in snapshot.winning_cells
            }
            self._policy = list(snapshot.policy)
            self._side = snapshot.side_to_move
            self._terminal = snapshot.terminal_state
            self._q_penalty = snapshot.q_penalty
            self._q_no_penalty = snapshot.q_no_penalty
            self._iterations = snapshot.n_mcts_iterations
            self._max_iterations = snapshot.max_mcts_iterations
            self._background_running = snapshot.background_running
            self._move_count = len(snapshot.move_history)
            if self._terminal == "red_win":
                self._status = "Red wins"
            elif self._terminal == "gold_win":
                self._status = "Gold wins"
            elif self._terminal == "draw":
                self._status = "Draw"
            elif self._auto[self._side]:
                self._status = f"{self._side.title()} is thinking"
            else:
                self._status = f"{self._side.title()} to move"
            self._revision += 1
            self.changed.emit()
            if (
                self._terminal == "ongoing"
                and self._auto[self._side]
                and self._iterations >= self._max_iterations
            ):
                self._game.make_best_move_if_ready()
        except Exception as error:
            self._set_error(error)

    @staticmethod
    def _close_game(game, router):
        if router is not None:
            router.close()
        game.close()

    def shutdown(self) -> None:
        self._load_revision += 1
        if self._load_future is not None:
            self._load_future.cancel()
            self._load_future = None
        if self._game is not None:
            self._executor.submit(self._close_game, self._game, self._router)
        self._game = None
        self._router = None


def run_gui() -> int:
    application = QGuiApplication.instance()
    owns_application = application is None
    if application is None:
        application = QGuiApplication(sys.argv)
    application.setOrganizationName("c4a0")
    application.setApplicationName("c4a0")
    QQuickStyle.setStyle("Material")

    app_controller = AppController()
    game_controller = GameController(app_controller)
    jobs = JobManager()
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("App", app_controller)
    engine.rootContext().setContextProperty("Game", game_controller)
    engine.rootContext().setContextProperty("Jobs", jobs)
    qml_path = Path(__file__).resolve().parent / "qml" / "Main.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    if not engine.rootObjects():
        game_controller.shutdown()
        return 1
    if not owns_application:
        return 0
    exit_code = application.exec()
    game_controller.shutdown()
    shiboken6.delete(engine)
    return exit_code
