"""QProcess-backed single-heavy-job queue."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import sys
from typing import Any
from uuid import uuid4

from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    Qt,
    Property,
    QProcess,
    QTimer,
    QObject,
    Signal,
    Slot,
)

from c4a0.config import (
    MCTSSweepConfig,
    NNSweepConfig,
    SolverScoreConfig,
    TournamentConfig,
    TrainingV2Config,
    ValidationConfig,
)


JOB_CONFIG_TYPES = {
    "training": TrainingV2Config,
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


class LogModel(QAbstractListModel):
    def __init__(self, records: deque[str], parent: QObject):
        super().__init__(parent)
        self.records = records

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.records)

    def roleNames(self):
        return {int(Qt.ItemDataRole.UserRole): b"line"}

    def data(self, index, role=int(Qt.ItemDataRole.DisplayRole)):
        if index.isValid() and 0 <= index.row() < len(self.records):
            return self.records[index.row()]
        return None

    def append(self, line: str):
        if len(self.records) == self.records.maxlen:
            self.beginRemoveRows(QModelIndex(), 0, 0)
            self.records.popleft()
            self.endRemoveRows()
        row = len(self.records)
        self.beginInsertRows(QModelIndex(), row, row)
        self.records.append(line)
        self.endInsertRows()

    def clear(self):
        self.beginResetModel()
        self.records.clear()
        self.endResetModel()


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
        self._logs: deque[str] = deque(maxlen=1000)
        self._log_model = LogModel(self._logs, self)
        self._recent: deque[dict[str, Any]] = deque(maxlen=20)
        self._shutting_down = False
        self._cancel_timer = QTimer(self)
        self._cancel_timer.setSingleShot(True)
        self._cancel_callback = None
        self._worker_group: int | None = None
        self._log_cache = ""
        self._log_dirty = True
        self._notification_timer = QTimer(self)
        self._notification_timer.setSingleShot(True)
        self._notification_timer.setInterval(50)
        self._notification_timer.timeout.connect(self.stateChanged.emit)
        self._phase = "Idle"
        self._progress = 0.0
        self._result = ""
        self._result_data: dict[str, Any] = {}
        self._live_training: dict[str, Any] = {
            "actor": "Idle",
            "trainer": "Idle",
            "arena": "Idle",
            "replay": "0 games",
            "champion": "0",
            "candidate": "None",
        }
        self._saw_terminal_event = False
        self._stop_after_generation_requested = False

    def _notify(self) -> None:
        if not self._notification_timer.isActive():
            self._notification_timer.start()

    def _append_log(self, line: str) -> None:
        self._log_model.append(
            line[:16384] + ("… [truncated]" if len(line) > 16384 else "")
        )
        self._log_dirty = True

    def _extend_logs(self, lines) -> None:
        for line in lines:
            self._append_log(line)

    @Property(QObject, constant=True)
    def logModel(self):
        return self._log_model

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
        if self._log_dirty:
            self._log_cache = "\n".join(self._logs)
            self._log_dirty = False
        return self._log_cache

    @Property(str, notify=stateChanged)
    def result(self) -> str:
        return self._result

    @Property(dict, notify=stateChanged)
    def resultData(self) -> dict[str, Any]:
        return self._result_data

    @Property(dict, notify=stateChanged)
    def liveTraining(self) -> dict[str, Any]:
        return self._live_training

    @Property(bool, notify=stateChanged)
    def stopAfterGenerationRequested(self) -> bool:
        return self._stop_after_generation_requested

    @Property(list, notify=stateChanged)
    def recentJobs(self) -> list[dict[str, Any]]:
        return list(self._recent)

    @Slot(str, str, str, result=str)
    def submit(self, kind: str, config_json: str, title: str) -> str:
        if self._shutting_down:
            return ""
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
            self._notify()
            self.notification.emit(self._result)
            return ""
        job = QueuedJob(kind=kind, title=title, config=config)
        self._queue.append(job)
        self._notify()
        self._start_next()
        return job.id

    @Slot()
    def cancel(self) -> None:
        if self._process is None:
            return
        if not self._saw_terminal_event:
            self._phase = "Cancelling"
        self._notify()
        self._process.terminate()
        process = self._process
        self._cancel_timer.stop()
        if self._cancel_callback is not None:
            self._cancel_timer.timeout.disconnect(self._cancel_callback)
        self._cancel_callback = lambda: self._kill_if_running(process)
        self._cancel_timer.timeout.connect(self._cancel_callback)
        self._cancel_timer.start(6000)

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
        self._notify()

    @Slot()
    def clearLogs(self) -> None:
        self._log_model.clear()
        self._log_dirty = True
        self._notify()

    @Slot()
    def shutdown(self) -> None:
        self._shutting_down = True
        self._queue.clear()
        self.cancel()
        self._notify()

    def _kill_if_running(self, process: QProcess) -> None:
        if (
            self._process is process
            and process.state() != QProcess.ProcessState.NotRunning
        ):
            self._kill_worker_group()
            process.kill()

    def _kill_worker_group(self) -> None:
        if self._worker_group is not None:
            try:
                os.killpg(self._worker_group, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _worker_group_running(self) -> bool:
        if self._worker_group is None:
            return False
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
                if int(fields[2]) == self._worker_group and fields[0] not in {"Z", "X"}:
                    return True
            except (OSError, ValueError, IndexError):
                continue
        return False

    def _process_error(self, process: QProcess, error: QProcess.ProcessError) -> None:
        if process is not self._process:
            return
        if error == QProcess.ProcessError.FailedToStart:
            self._handle_event({"type": "failed", "message": process.errorString()})
            self._finished(-1, QProcess.ExitStatus.CrashExit)

    def _start_next(self) -> None:
        if self._shutting_down or self._current is not None or not self._queue:
            return
        self._current = self._queue.popleft()
        self._log_model.clear()
        self._log_dirty = True
        self._worker_group = None
        self._phase = "Starting"
        self._progress = 0.0
        self._result = ""
        self._result_data = {}
        self._live_training = {
            "actor": "Starting",
            "trainer": "Starting",
            "arena": "Idle",
            "replay": "0 games",
            "champion": "0",
            "candidate": "None",
        }
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
        process.finished.connect(
            lambda code, status: (
                self._finished(code, status) if self._process is process else None
            )
        )
        process.errorOccurred.connect(lambda error: self._process_error(process, error))
        process.start()
        self._notify()

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
        if len(self._stdout_buffer) > 1_048_576 and "\n" not in self._stdout_buffer:
            self._append_log(self._stdout_buffer)
            self._stdout_buffer = ""
        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                self._append_log(line)
                continue
            if isinstance(event, dict) and event.get("version") == 1:
                try:
                    self._handle_event(event)
                except (ValueError, TypeError, KeyError) as error:
                    self._handle_event(
                        {"type": "failed", "message": f"Invalid worker event: {error}"}
                    )
                    self.cancel()
            else:
                self._append_log(line)

    def _read_stderr(self) -> None:
        if self._process is None:
            return
        content = bytes(self._process.readAllStandardError().data()).decode(
            errors="replace"
        )
        self._extend_logs(line for line in content.splitlines() if line)
        self._notify()

    def _handle_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type", "")
        if self._saw_terminal_event:
            return
        if event_type == "started":
            self._phase = "Running"
            if self._process is not None:
                pid = int(self._process.processId())
                if event.get("process_group") == pid and pid > 0:
                    self._worker_group = pid
        elif event_type == "phase":
            phase_name = str(event.get("name", "Running"))
            self._phase = phase_name.replace("_", " ").title()
            component = str(event.get("component", ""))
            if component in {"actor", "trainer", "arena"}:
                if phase_name == "backpressure":
                    self._live_training[component] = (
                        "Paused for training debt"
                        if event.get("paused")
                        else "Self-play"
                    )
                elif phase_name == "self_play_progress":
                    self._live_training[component] = (
                        f"{event.get('completed_games', 0)}/"
                        f"{event.get('total_games', '?')} games · "
                        f"MCTS {event.get('mcts_iterations', 0)} · "
                        f"q {event.get('neural_queue_depth', 0)}/"
                        f"{event.get('mcts_queue_depth', 0)}"
                    )
                elif phase_name == "arena_progress":
                    self._live_training[component] = (
                        f"{event.get('completed_games', 0)}/"
                        f"{event.get('total_games', '?')} · "
                        f"{event.get('wins', 0)}W/"
                        f"{event.get('draws', 0)}D/"
                        f"{event.get('losses', 0)}L · "
                        f"q {event.get('neural_queue_depth', 0)}/"
                        f"{event.get('mcts_queue_depth', 0)}"
                    )
                elif phase_name == "training" and "sample_budget" in event:
                    self._live_training[component] = (
                        f"{event.get('positions_per_second', 0):.0f} pos/s · "
                        f"queue {event.get('sample_budget', 0)} pos"
                    )
                else:
                    self._live_training[component] = self._phase
            if "replay_games" in event:
                self._live_training["replay"] = (
                    f"{event['replay_games']} games / "
                    f"{event.get('replay_shards', 0)} shards"
                )
            if "champion" in event:
                self._live_training["champion"] = str(event["champion"])
            if "pending_candidate" in event:
                pending = event["pending_candidate"]
                self._live_training["candidate"] = (
                    "None" if pending is None else str(pending)
                )
        elif event_type == "progress":
            self._progress = max(0.0, min(1.0, float(event.get("fraction", 0.0))))
        elif event_type == "log":
            self._append_log(str(event.get("message", "")))
        elif event_type == "metric":
            self._append_log(f"{event.get('name')}: {event.get('value')}")
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
                    self._extend_logs(str(details).splitlines())
        self.eventReceived.emit(event_type, event)
        self._notify()

    def _finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        if self._current is None:
            return
        self._cancel_timer.stop()
        self._read_stdout()
        self._read_stderr()
        self._kill_worker_group()
        if self._worker_group_running():
            # Keep the queue occupied until killed descendants have actually exited.
            process = self._process
            QTimer.singleShot(
                25,
                lambda: (
                    self._finished(exit_code, _exit_status)
                    if self._process is process
                    else None
                ),
            )
            return
        self._worker_group = None
        if self._stdout_buffer.strip():
            self._append_log(self._stdout_buffer.strip())
        if not self._saw_terminal_event:
            self._phase = "Failed"
            self._result = (
                f"Worker exited with status {exit_code} before reporting a result"
            )
        if self._phase == "Completed" and (
            exit_code != 0 or _exit_status != QProcess.ExitStatus.NormalExit
        ):
            self._phase = "Failed"
            self._result = (
                f"Worker crashed after reporting completion (status {exit_code})"
            )
        self._recent.appendleft(
            {
                "id": self._current.id,
                "title": self._current.title,
                "kind": self._current.kind,
                "status": self._phase,
                "createdAt": self._current.created_at,
                "result": self._result_data,
                "message": self._result,
            },
        )
        self._process.deleteLater() if self._process is not None else None
        self._process = None
        self._current = None
        self._notify()
        QTimer.singleShot(0, self._start_next)
