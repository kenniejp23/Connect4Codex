"""QProcess-backed single-heavy-job queue."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import json
import sys
from typing import Any
from uuid import uuid4

from PySide6.QtCore import Property, QProcess, QTimer, QObject, Signal, Slot

from c4a0.config import (
    MCTSSweepConfig,
    NNSweepConfig,
    SolverScoreConfig,
    TournamentConfig,
    TrainingConfig,
    ValidationConfig,
)


JOB_CONFIG_TYPES = {
    "training": TrainingConfig,
    "solver_score": SolverScoreConfig,
    "tournament": TournamentConfig,
    "nn_sweep": NNSweepConfig,
    "mcts_sweep": MCTSSweepConfig,
    "validation": ValidationConfig,
}


@dataclass
class QueuedJob:
    kind: str
    title: str
    config: dict[str, Any]
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class JobManager(QObject):
    stateChanged = Signal()
    eventReceived = Signal(str, dict)
    notification = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._queue: deque[QueuedJob] = deque()
        self._current: QueuedJob | None = None
        self._process: QProcess | None = None
        self._stdout_buffer = ""
        self._logs: list[str] = []
        self._recent: list[dict[str, Any]] = []
        self._phase = "Idle"
        self._progress = 0.0
        self._result = ""
        self._result_data: dict[str, Any] = {}
        self._saw_terminal_event = False
        self._stop_after_generation_requested = False

    @Property(bool, notify=stateChanged)
    def active(self) -> bool:
        return self._current is not None

    @Property(str, notify=stateChanged)
    def currentTitle(self) -> str:
        return self._current.title if self._current else "No active job"

    @Property(str, notify=stateChanged)
    def currentKind(self) -> str:
        return self._current.kind if self._current else ""

    @Property(str, notify=stateChanged)
    def phase(self) -> str:
        return self._phase

    @Property(float, notify=stateChanged)
    def progress(self) -> float:
        return self._progress

    @Property(int, notify=stateChanged)
    def queueDepth(self) -> int:
        return len(self._queue)

    @Property(str, notify=stateChanged)
    def logs(self) -> str:
        return "\n".join(self._logs[-1000:])

    @Property(str, notify=stateChanged)
    def result(self) -> str:
        return self._result

    @Property(dict, notify=stateChanged)
    def resultData(self) -> dict[str, Any]:
        return self._result_data

    @Property(bool, notify=stateChanged)
    def stopAfterGenerationRequested(self) -> bool:
        return self._stop_after_generation_requested

    @Property(list, notify=stateChanged)
    def recentJobs(self) -> list[dict[str, Any]]:
        return self._recent[:20]

    @Slot(str, str, str, result=str)
    def submit(self, kind: str, config_json: str, title: str) -> str:
        try:
            config = json.loads(config_json)
            if not isinstance(config, dict):
                raise ValueError("job configuration must be an object")
            config_type = JOB_CONFIG_TYPES.get(kind)
            if config_type is None:
                raise ValueError(f"unsupported job type: {kind}")
            config = config_type.model_validate(config).model_dump(mode="json")
        except (TypeError, json.JSONDecodeError, ValueError) as error:
            self._result = f"Invalid configuration: {error}"
            self.stateChanged.emit()
            self.notification.emit(self._result)
            return ""
        job = QueuedJob(kind=kind, title=title, config=config)
        self._queue.append(job)
        self.stateChanged.emit()
        self._start_next()
        return job.id

    @Slot()
    def cancel(self) -> None:
        if self._process is None:
            return
        self._phase = "Cancelling"
        self.stateChanged.emit()
        self._process.terminate()
        QTimer.singleShot(5000, self._kill_if_running)

    @Slot()
    def stopAfterGeneration(self) -> None:
        if (
            self._process is None
            or self._current is None
            or self._current.kind != "training"
            or self._stop_after_generation_requested
        ):
            return
        self._stop_after_generation_requested = True
        self._process.write(b'{"command":"stop_after_generation"}\n')
        self.stateChanged.emit()

    @Slot()
    def clearLogs(self) -> None:
        self._logs.clear()
        self.stateChanged.emit()

    def _kill_if_running(self) -> None:
        if (
            self._process is not None
            and self._process.state() != QProcess.ProcessState.NotRunning
        ):
            self._process.kill()

    def _start_next(self) -> None:
        if self._current is not None or not self._queue:
            return
        self._current = self._queue.popleft()
        self._logs = []
        self._phase = "Starting"
        self._progress = 0.0
        self._result = ""
        self._result_data = {}
        self._stdout_buffer = ""
        self._saw_terminal_event = False
        self._stop_after_generation_requested = False
        process = QProcess(self)
        self._process = process
        process.setProgram(sys.executable)
        process.setArguments(["-m", "c4a0.worker"])
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.started.connect(self._send_request)
        process.finished.connect(self._finished)
        process.start()
        self.stateChanged.emit()

    def _send_request(self) -> None:
        if self._process is None or self._current is None:
            return
        request = {
            "version": 1,
            "kind": self._current.kind,
            "config": self._current.config,
        }
        self._process.write((json.dumps(request) + "\n").encode())

    def _read_stdout(self) -> None:
        if self._process is None:
            return
        self._stdout_buffer += bytes(
            self._process.readAllStandardOutput().data()
        ).decode(errors="replace")
        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                self._logs.append(line)
                continue
            self._handle_event(event)

    def _read_stderr(self) -> None:
        if self._process is None:
            return
        content = bytes(self._process.readAllStandardError().data()).decode(
            errors="replace"
        )
        self._logs.extend(line for line in content.splitlines() if line)
        self.stateChanged.emit()

    def _handle_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type", "")
        if event_type == "started":
            self._phase = "Running"
        elif event_type == "phase":
            self._phase = str(event.get("name", "Running")).replace("_", " ").title()
        elif event_type == "progress":
            self._progress = max(0.0, min(1.0, float(event.get("fraction", 0.0))))
        elif event_type == "log":
            self._logs.append(str(event.get("message", "")))
        elif event_type == "metric":
            self._logs.append(f"{event.get('name')}: {event.get('value')}")
        elif event_type in {"completed", "cancelled", "failed"}:
            self._saw_terminal_event = True
            if event_type == "completed":
                self._phase = "Completed"
                self._progress = 1.0
                result = event.get("result", {})
                self._result_data = result if isinstance(result, dict) else {}
                self._result = json.dumps(result, indent=2)
                self.notification.emit(f"{self.currentTitle} completed")
            elif event_type == "cancelled":
                self._phase = "Cancelled"
                self._result_data = {}
                self._result = str(event.get("message", "Job cancelled"))
                self.notification.emit(f"{self.currentTitle} was cancelled")
            else:
                self._phase = "Failed"
                self._result_data = {}
                self._result = str(event.get("message", "Job failed"))
                self.notification.emit(f"{self.currentTitle} failed: {self._result}")
                details = event.get("details")
                if details:
                    self._logs.extend(str(details).splitlines())
        self.eventReceived.emit(event_type, event)
        self.stateChanged.emit()

    def _finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        if self._current is None:
            return
        if self._stdout_buffer.strip():
            self._logs.append(self._stdout_buffer.strip())
        if not self._saw_terminal_event:
            self._phase = "Failed" if exit_code else "Completed"
            self._result = (
                f"Worker exited with status {exit_code} before reporting a result"
            )
        self._recent.insert(
            0,
            {
                "id": self._current.id,
                "title": self._current.title,
                "kind": self._current.kind,
                "status": self._phase,
                "createdAt": self._current.created_at,
            },
        )
        self._process.deleteLater() if self._process is not None else None
        self._process = None
        self._current = None
        self.stateChanged.emit()
        QTimer.singleShot(0, self._start_next)
