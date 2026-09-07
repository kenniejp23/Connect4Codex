"""Asynchronous, neural-only AlphaZero training with replay and arena gating."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
import copy
import ctypes
import signal
import fcntl
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import random
import sqlite3
import tempfile
import threading
import time
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

import c4a0_cpp  # type: ignore
from c4a0.arena_statistics import PairedSprtGate
from c4a0.config import TrainingV2Config
from c4a0.nn import ConnectFourNet, ModelConfig
from c4a0.training_common import TrainingCancelled, parse_lr_schedule


RUN_FORMAT = "c4a0-async-training"
RUN_VERSION = 1
CHECKPOINT_FORMAT = "c4a0-neural-checkpoint"
CHECKPOINT_VERSION = 1
UNIFORM_MODEL_ID = (1 << 64) - 1
RANDOM_MODEL_ID = (1 << 64) - 2


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


def _baseline_model_id(kind: int, game_id: int) -> int:
    """Encode the baseline kind and game ID without colliding with attempts."""
    if kind == UNIFORM_MODEL_ID:
        return UNIFORM_MODEL_ID - 2 * game_id
    if kind == RANDOM_MODEL_ID:
        return RANDOM_MODEL_ID - 2 * game_id
    raise ValueError("unknown baseline model kind")


def _baseline_kind_and_game(model_id: int) -> tuple[int, int] | None:
    if model_id < (1 << 63):
        return None
    delta = UNIFORM_MODEL_ID - model_id
    if delta < 0:
        return None
    if delta % 2 == 0:
        return UNIFORM_MODEL_ID, delta // 2
    return RANDOM_MODEL_ID, (delta - 1) // 2


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def training_run_owner(base_dir: Path):
    """Own the complete coordinator lifecycle; readers never take this lock."""
    base_dir.mkdir(parents=True, exist_ok=True)
    with (base_dir / ".trainer.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another trainer owns this run: {base_dir}") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".staging", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".staging", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _owned_cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _owned_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_owned_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_owned_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _checkpoint_payload(
    model: ConnectFourNet,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    **state: Any,
) -> dict[str, Any]:
    model_device = next(model.parameters()).device
    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "model_config": model.config.model_dump(mode="json"),
        "inference": {"version": 1, "mcts_value_scale": model.mcts_value_scale},
        "value_loss_weight": model.value_loss_weight,
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "optimizer_state": (
            _owned_cpu_tree(optimizer.state_dict()) if optimizer is not None else None
        ),
        "scaler_state": copy.deepcopy(scaler.state_dict())
        if scaler is not None
        else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (
            torch.cuda.get_rng_state_all() if model_device.type == "cuda" else None
        ),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        **state,
    }


def _load_checkpoint(
    path: str | Path, *, trusted: bool = True
) -> tuple[ConnectFourNet, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=not trusted)
    if (
        payload.get("format") != CHECKPOINT_FORMAT
        or payload.get("version") != CHECKPOINT_VERSION
    ):
        raise ValueError(f"unsupported neural checkpoint: {path}")
    config = ModelConfig.model_validate(payload["model_config"])
    model = ConnectFourNet(config)
    state_dict = payload.get("state_dict", payload.get("model_state"))
    if state_dict is None:
        raise ValueError(f"neural checkpoint has no state_dict: {path}")
    model.load_state_dict(state_dict)
    inference = payload.get("inference")
    if inference is None:
        # Version-one checkpoints predate inference metadata. Recover it from
        # their enclosing run, including runs moved to a different directory.
        for parent in Path(path).resolve().parents:
            database = parent / "run.sqlite3"
            if database.is_file():
                with sqlite3.connect(
                    f"{database.as_uri()}?mode=ro", uri=True
                ) as connection:
                    row = connection.execute(
                        "SELECT value FROM run_state WHERE key='config'"
                    ).fetchone()
                connection.close()
                if row:
                    saved_config = json.loads(row[0])
                    inference = {
                        "version": 1,
                        "mcts_value_scale": saved_config["mcts_value_scale"],
                    }
                    model.value_loss_weight = saved_config["value_loss_weight"]
                break
    if inference is not None:
        if inference.get("version") != 1 or not math.isfinite(
            float(inference["mcts_value_scale"])
        ):
            raise ValueError("unsupported checkpoint inference settings")
        model.mcts_value_scale = float(inference["mcts_value_scale"])
        payload["inference"] = inference
    model.value_loss_weight = float(
        payload.get("value_loss_weight", model.value_loss_weight)
    )
    return model, payload


def load_champion_model(base_dir: str) -> ConnectFourNet:
    with RunManifest.open_existing(Path(base_dir)) as manifest:
        model, _ = _load_checkpoint(manifest.champion()["checkpoint_path"])
        return model


def load_model_checkpoint(
    checkpoint_path: str | Path,
    *,
    mcts_value_scale: float | None = None,
    trusted: bool = False,
) -> ConnectFourNet:
    """Load a standalone V2 model checkpoint for evaluation tools."""
    model, payload = _load_checkpoint(checkpoint_path, trusted=trusted)
    if mcts_value_scale is not None:
        if not math.isfinite(mcts_value_scale):
            raise ValueError("mcts_value_scale must be finite")
        model.mcts_value_scale = mcts_value_scale
    elif "inference" not in payload:
        raise ValueError(
            "Checkpoint inference settings unavailable; provide mcts_value_scale explicitly"
        )
    return model


def export_inference_model(
    base_dir: str, output: str | Path, attempt_n: int | None = None
) -> Path:
    """Export trusted run weights to a compact artifact loadable with weights_only=True."""
    model = (
        load_champion_model(base_dir)
        if attempt_n is None
        else load_attempt_model(base_dir, attempt_n)
    )
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "artifact": "inference",
        "inference_version": 1,
        "model_config": model.config.model_dump(mode="json"),
        "state_dict": _owned_cpu_tree(model.state_dict()),
        "inference": {"version": 1, "mcts_value_scale": model.mcts_value_scale},
    }
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Inference export already exists: {output}")
    _atomic_torch_save(output, payload)
    return output


def is_v2_run(base_dir: str) -> bool:
    return (Path(base_dir) / "run.sqlite3").is_file()


def load_attempt_model(base_dir: str, attempt_n: int) -> ConnectFourNet:
    with RunManifest.open_existing(Path(base_dir)) as manifest:
        row = manifest.connection.execute(
            "SELECT checkpoint_path FROM attempts WHERE attempt_n = ?", (attempt_n,)
        ).fetchone()
        if (
            row is None
            or not manifest.resolve_artifact(row["checkpoint_path"]).is_file()
        ):
            raise FileNotFoundError(
                f"V2 attempt {attempt_n} does not have retained weights"
            )
        model, _ = _load_checkpoint(manifest.resolve_artifact(row["checkpoint_path"]))
        return model


def list_attempts(base_dir: str) -> list[dict[str, Any]]:
    with RunManifest.open_existing(Path(base_dir)) as manifest:
        champion = int(manifest.champion()["attempt_n"])
        rows = manifest.connection.execute(
            "SELECT * FROM attempts ORDER BY attempt_n DESC"
        ).fetchall()
        return [
            {
                **manifest.artifact_row(row),
                "is_champion": int(row["attempt_n"]) == champion,
            }
            for row in rows
        ]


class RunManifest:
    """Single-writer manifest for an asynchronous training run."""

    def __init__(self, base_dir: Path, connection: sqlite3.Connection):
        self.base_dir = base_dir.resolve()
        self.connection = connection
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def relative_artifact(self, path: str | Path) -> str:
        path = Path(path)
        if ".." in path.parts:
            raise ValueError(f"Artifact contains parent traversal: {path}")
        try:
            return path.resolve().relative_to(self.base_dir).as_posix()
        except ValueError:
            # Existing manifests may contain paths rooted at a former run location.
            for component in ("attempts", "replay"):
                if component in path.parts:
                    index = len(path.parts) - 1 - path.parts[::-1].index(component)
                    return Path(*path.parts[index:]).as_posix()
            if not path.is_absolute():
                return path.as_posix()
            raise ValueError(f"Artifact is outside the run: {path}")

    def resolve_artifact(self, path: str | Path) -> Path:
        return self.base_dir / self.relative_artifact(path)

    def artifact_row(self, row) -> dict[str, Any]:
        result = dict(row)
        for key in ("checkpoint_path", "path"):
            if key in result:
                result[key] = str(self.resolve_artifact(result[key]))
        return result

    def _migrate_artifacts(self) -> None:
        with self.connection:
            for table, column, key in (
                ("attempts", "checkpoint_path", "attempt_n"),
                ("replay_shards", "path", "shard_id"),
            ):
                for row in self.connection.execute(
                    f"SELECT {key}, {column} FROM {table}"
                ).fetchall():
                    self.connection.execute(
                        f"UPDATE {table} SET {column} = ? WHERE {key} = ?",
                        (self.relative_artifact(row[column]), row[key]),
                    )
            self.connection.execute(
                "INSERT OR REPLACE INTO run_state VALUES('artifact_paths_version', '2')"
            )

    @classmethod
    def create(cls, base_dir: Path, config: TrainingV2Config) -> "RunManifest":
        database_path = base_dir / "run.sqlite3"
        if (
            base_dir.exists()
            and any(item.name != ".trainer.lock" for item in base_dir.iterdir())
            and not database_path.exists()
        ):
            raise FileExistsError(
                f"V2 training requires an empty or V2 directory: {base_dir}"
            )
        base_dir.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(database_path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        manifest = cls(base_dir, connection)
        manifest._create_schema()
        if manifest.get_state("format") is None:
            with connection:
                connection.executemany(
                    "INSERT INTO run_state(key, value) VALUES(?, ?)",
                    [
                        (key, json.dumps(value))
                        for key, value in {
                            "format": RUN_FORMAT,
                            "version": RUN_VERSION,
                            "config": config.model_dump(mode="json"),
                            "next_game_id": 1,
                            "accepted_count": 0,
                        }.items()
                    ],
                )
        elif (
            manifest.get_state("format") != RUN_FORMAT
            or manifest.get_state("version") != RUN_VERSION
        ):
            raise ValueError("unsupported asynchronous training run format")
        manifest._migrate_artifacts()
        return manifest

    @classmethod
    def open_existing(cls, base_dir: Path) -> "RunManifest":
        database_path = base_dir / "run.sqlite3"
        if not database_path.is_file():
            raise FileNotFoundError(f"no V2 training run found in {base_dir}")
        connection = sqlite3.connect(
            f"{database_path.resolve().as_uri()}?mode=ro", uri=True
        )
        manifest = cls(base_dir, connection)
        if (
            manifest.get_state("format") != RUN_FORMAT
            or manifest.get_state("version") != RUN_VERSION
        ):
            connection.close()
            raise ValueError("unsupported asynchronous training run format")
        return manifest

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS run_state(
                key TEXT PRIMARY KEY NOT NULL,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS attempts(
                attempt_n INTEGER PRIMARY KEY,
                status TEXT NOT NULL,
                parent_attempt INTEGER,
                checkpoint_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                val_loss REAL,
                training_steps INTEGER NOT NULL DEFAULT 0,
                training_samples INTEGER NOT NULL DEFAULT 0,
                submitted_fresh_games INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS replay_shards(
                shard_id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                games INTEGER NOT NULL,
                samples INTEGER NOT NULL,
                train_samples INTEGER NOT NULL,
                champion_attempt INTEGER NOT NULL,
                first_game_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS arena_results(
                attempt_n INTEGER PRIMARY KEY,
                result_json TEXT NOT NULL,
                FOREIGN KEY(attempt_n) REFERENCES attempts(attempt_n)
            );
            """
        )
        self.connection.commit()

    def get_state(self, key: str) -> Any | None:
        row = self.connection.execute(
            "SELECT value FROM run_state WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else json.loads(row["value"])

    def set_state(self, key: str, value: Any) -> None:
        self.connection.execute(
            "INSERT INTO run_state(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )
        self.connection.commit()

    def add_root(self, checkpoint_path: Path) -> None:
        if self.connection.execute("SELECT 1 FROM attempts LIMIT 1").fetchone():
            return
        self.connection.execute(
            "INSERT INTO attempts(attempt_n, status, parent_attempt, checkpoint_path, "
            "created_at) VALUES(0, 'accepted', NULL, ?, ?)",
            (self.relative_artifact(checkpoint_path), datetime.now().isoformat()),
        )
        self.set_state("champion_attempt", 0)
        self.connection.commit()

    def next_attempt(self) -> int:
        row = self.connection.execute(
            "SELECT COALESCE(MAX(attempt_n), -1) + 1 AS value FROM attempts"
        ).fetchone()
        return int(row["value"])

    def add_pending_attempt(
        self,
        attempt_n: int,
        parent_attempt: int,
        checkpoint_path: Path,
        val_loss: float,
        training_steps: int,
        training_samples: int,
        submitted_fresh_games: int,
    ) -> None:
        self.connection.execute(
            "INSERT INTO attempts(attempt_n, status, parent_attempt, checkpoint_path, "
            "created_at, val_loss, training_steps, training_samples, "
            "submitted_fresh_games) VALUES(?, 'pending', ?, ?, ?, ?, ?, ?, ?)",
            (
                attempt_n,
                parent_attempt,
                self.relative_artifact(checkpoint_path),
                datetime.now().isoformat(),
                val_loss,
                training_steps,
                training_samples,
                submitted_fresh_games,
            ),
        )
        self.connection.commit()

    def add_interrupted_attempt(
        self, attempt_n: int, parent_attempt: int, checkpoint_path: Path
    ) -> None:
        self.connection.execute(
            "INSERT INTO attempts(attempt_n, status, parent_attempt, checkpoint_path, "
            "created_at) VALUES(?, 'interrupted', ?, ?, ?)",
            (
                attempt_n,
                parent_attempt,
                self.relative_artifact(checkpoint_path),
                datetime.now().isoformat(),
            ),
        )
        self.connection.commit()

    def pending_attempt(self) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM attempts WHERE status = 'pending' ORDER BY attempt_n LIMIT 1"
        ).fetchone()
        return None if row is None else self.artifact_row(row)

    def champion(self) -> dict[str, Any]:
        attempt = self.get_state("champion_attempt")
        if attempt is None:
            raise FileNotFoundError("V2 run does not have a champion")
        row = self.connection.execute(
            "SELECT * FROM attempts WHERE attempt_n = ?", (attempt,)
        ).fetchone()
        if row is None:
            raise RuntimeError("champion pointer references a missing attempt")
        return self.artifact_row(row)

    def accepted_models(self, limit: int) -> list[dict[str, Any]]:
        champion = int(self.champion()["attempt_n"])
        rows = self.connection.execute(
            "SELECT * FROM attempts WHERE status = 'accepted' AND attempt_n != ? "
            "ORDER BY attempt_n DESC LIMIT ?",
            (champion, limit),
        ).fetchall()
        return [self.artifact_row(row) for row in rows]

    def decide_attempt(self, attempt_n: int, result: dict[str, Any]) -> bool:
        accepted = result["decision"] == "accepted"
        with self.connection:
            self.connection.execute(
                "UPDATE attempts SET status = ? WHERE attempt_n = ?",
                ("accepted" if accepted else "rejected", attempt_n),
            )
            self.connection.execute(
                "INSERT OR REPLACE INTO arena_results(attempt_n, result_json) "
                "VALUES(?, ?)",
                (attempt_n, json.dumps(result)),
            )
            if accepted:
                self.connection.execute(
                    "INSERT INTO run_state(key, value) VALUES('champion_attempt', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (json.dumps(attempt_n),),
                )
                count = int(self.get_state("accepted_count") or 0) + 1
                self.connection.execute(
                    "INSERT INTO run_state(key, value) VALUES('accepted_count', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (json.dumps(count),),
                )
        return accepted

    def register_shard(self, event: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO replay_shards(path, games, samples, train_samples, "
                "champion_attempt, first_game_id, created_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
                (
                    self.relative_artifact(event["path"]),
                    event["games"],
                    event["samples"],
                    event["train_samples"],
                    event["champion_attempt"],
                    event["first_game_id"],
                    datetime.now().isoformat(),
                ),
            )
            self.connection.execute(
                "INSERT INTO run_state(key, value) VALUES('next_game_id', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (json.dumps(event["first_game_id"] + event["games"]),),
            )
        retired: list[dict[str, Any]] = []
        config = self.get_state("config")
        if not isinstance(config, dict):
            raise RuntimeError("V2 run is missing its configuration")
        capacity = int(config["replay_capacity_games"])
        active = self.active_shards()
        total = sum(int(item["games"]) for item in active)
        for item in active:
            if total <= capacity:
                break
            self.connection.execute(
                "UPDATE replay_shards SET active = 0 WHERE shard_id = ?",
                (item["shard_id"],),
            )
            retired.append(item)
            total -= int(item["games"])
        self.connection.commit()
        if cursor.lastrowid is None:
            raise RuntimeError("replay shard insert did not return an identifier")
        return int(cursor.lastrowid), retired

    def active_shards(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM replay_shards WHERE active = 1 ORDER BY shard_id"
        ).fetchall()
        return [self.artifact_row(row) for row in rows]

    def status(self) -> dict[str, Any]:
        champion = self.champion()
        shards = self.active_shards()
        pending = self.pending_attempt()
        return {
            "format": RUN_FORMAT,
            "version": RUN_VERSION,
            "champion": champion["attempt_n"],
            "accepted_champions": int(self.get_state("accepted_count") or 0),
            "pending_candidate": None if pending is None else pending["attempt_n"],
            "replay_shards": len(shards),
            "replay_games": sum(int(item["games"]) for item in shards),
            "replay_samples": sum(int(item["samples"]) for item in shards),
            "next_game_id": int(self.get_state("next_game_id") or 1),
            "components": {
                "actor": self.get_state("actor_progress") or {"state": "idle"},
                "trainer": self.get_state("trainer_progress") or {"state": "idle"},
                "arena": self.get_state("arena_progress") or {"state": "idle"},
            },
        }


def effective_training_config(config: TrainingV2Config) -> TrainingV2Config:
    """Inspect the exact configuration that a coordinator will use on resume."""
    if not is_v2_run(config.base_dir):
        return config.model_copy(deep=True)
    with RunManifest.open_existing(Path(config.base_dir)) as manifest:
        persisted = manifest.get_state("config")
    if not isinstance(persisted, dict):
        raise ValueError("Run is missing its configuration")
    for field in ("base_dir", "device", "max_gens", "max_candidate_attempts"):
        persisted[field] = getattr(config, field)
    return TrainingV2Config.model_validate(persisted)


def training_status(base_dir: str) -> dict[str, Any]:
    with RunManifest.open_existing(Path(base_dir)) as manifest:
        return manifest.status()


def _validation_game(seed: int, game_id: int, fraction: float) -> bool:
    digest = hashlib.blake2b(f"{seed}:{game_id}".encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "little") / float(1 << 64)
    return value < fraction


def _model_loss(model: ConnectFourNet, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
    from c4a0.losses import policy_value_losses

    policy, penalty, value = policy_value_losses(
        model(batch[0]), batch, model.value_loss_weight
    )
    return policy + penalty + value


@dataclass(frozen=True)
class _ReplayGame:
    game_id: int
    player0_id: int
    player1_id: int
    samples: tuple[Any, ...]


@dataclass(frozen=True)
class _ReplayShard:
    games: tuple[_ReplayGame, ...]
    game_weights: tuple[int, ...]


class ReplayPool:
    def __init__(self, config: TrainingV2Config, shards: Iterable[dict[str, Any]]):
        self.config = config
        self.shards = list(shards)
        self.cache: OrderedDict[str, _ReplayShard] = OrderedDict()
        self.cache_limit = (
            math.ceil(config.replay_capacity_games / config.self_play_shard_games) + 1
        )
        self.rng = random.Random(config.run_seed ^ 0xA17E)
        self.lock = threading.RLock()

    def add(self, shard: dict[str, Any]) -> None:
        with self.lock:
            self.shards.append(shard)

    def remove(self, shard_id: int) -> None:
        with self.lock:
            removed = [
                item for item in self.shards if int(item["shard_id"]) == shard_id
            ]
            self.shards = [
                item for item in self.shards if int(item["shard_id"]) != shard_id
            ]
            for item in removed:
                self.cache.pop(str(item["path"]), None)

    def rng_state(self) -> tuple[Any, ...]:
        with self.lock:
            return self.rng.getstate()

    def restore_rng_state(self, state: tuple[Any, ...]) -> None:
        with self.lock:
            self.rng.setstate(state)

    def resume_sample_budget(self) -> int:
        """Return the replay work remaining at the actor resume watermark.

        The coordinator keeps this much work queued so self-play and training can
        overlap.  A candidate must therefore be emitted at this watermark rather
        than only after the queue is completely empty.
        """
        with self.lock:
            if not self.shards:
                return 0
            average_train_samples = sum(
                int(shard["train_samples"]) for shard in self.shards
            ) / len(self.shards)
        return math.ceil(
            average_train_samples
            * self.config.replay_ratio
            * self.config.debt_resume_shards
        )

    def _load(self, path: str) -> _ReplayShard:
        if path in self.cache:
            value = self.cache.pop(path)
            self.cache[path] = value
            return value
        with open(path, "rb") as stream:
            native_result = c4a0_cpp.PlayGamesResult.from_cbor(stream.read())
        # Nanobind converts ``results`` and ``samples`` vectors into fresh Python
        # lists on every property access.  Materialize each vector once per cached
        # shard; doing it for every sampled position made production training
        # hundreds of times slower than the tensor/model benchmark.
        games = tuple(
            _ReplayGame(
                game_id=int(game.metadata.game_id),
                player0_id=int(game.metadata.player0_id),
                player1_id=int(game.metadata.player1_id),
                samples=tuple(game.samples),
            )
            for game in native_result.results
        )
        value = _ReplayShard(
            games=games,
            game_weights=tuple(len(game.samples) for game in games),
        )
        self.cache[path] = value
        # The sampler deliberately mixes positions across the whole active replay
        # window.  Retaining fewer than that window causes near-continuous CBOR
        # decoding and native-vector conversion as shards alternate within a batch.
        while len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)
        return value

    def _choose_shard(self) -> dict[str, Any]:
        if not self.shards:
            raise RuntimeError("replay buffer is empty")
        if self.rng.random() < 0.5:
            start = max(0, len(self.shards) - max(1, math.ceil(len(self.shards) / 4)))
            choices = self.shards[start:]
        else:
            choices = self.shards
        weights = [int(item["samples"]) for item in choices]
        return self.rng.choices(choices, weights=weights, k=1)[0]

    def batch(
        self, batch_size: int, device: torch.device, validation: bool
    ) -> tuple[torch.Tensor, ...]:
        with self.lock:
            values: list[tuple[torch.Tensor, ...]] = []
            attempts = 0
            while len(values) < batch_size:
                attempts += 1
                if attempts > batch_size * 500:
                    raise ReplayPartitionEmpty(
                        "could not sample the requested replay partition"
                    )
                shard = self._choose_shard()
                replay_shard = self._load(str(shard["path"]))
                if not replay_shard.games:
                    continue
                game = self.rng.choices(
                    replay_shard.games, weights=replay_shard.game_weights, k=1
                )[0]
                if (
                    _validation_game(
                        self.config.run_seed,
                        game.game_id,
                        self.config.validation_fraction,
                    )
                    != validation
                ):
                    continue
                sample_index = self.rng.randrange(len(game.samples))
                sample = game.samples[sample_index]
                if not validation and self.rng.random() < 0.5:
                    sample = sample.flip_h()
                pos, policy, q_penalty, q_no_penalty = sample.to_numpy()
                terminal = sample_index == len(game.samples) - 1
                player_id = (
                    game.player0_id if int(sample.ply) % 2 == 0 else game.player1_id
                )
                policy_weight = float(
                    not terminal and player_id == int(shard["champion_attempt"])
                )
                values.append(
                    (
                        torch.from_numpy(pos),
                        torch.from_numpy(policy),
                        torch.from_numpy(q_penalty),
                        torch.from_numpy(q_no_penalty),
                        torch.tensor(policy_weight, dtype=torch.float32),
                        torch.tensor(1.0, dtype=torch.float32),
                    )
                )
            pin = device.type == "cuda"
            return tuple(
                torch.stack([value[index] for value in values]).pin_memory()
                if pin
                else torch.stack([value[index] for value in values])
                for index in range(6)
            )


class ReplayPrefetcher:
    """Persistent background replay loader overlapping CPU fetch with training."""

    def __init__(self, pool: ReplayPool, workers: int):
        self.pool = pool
        self.executor = (
            ThreadPoolExecutor(max_workers=workers, thread_name_prefix="replay-loader")
            if workers > 0
            else None
        )
        self.future: Future[tuple[torch.Tensor, ...]] | None = None
        self.signature: tuple[int, str] | None = None

    def training_batch(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        signature = (batch_size, str(device))
        if self.future is not None and self.signature == signature:
            batch = self.future.result()
        else:
            batch = self.pool.batch(batch_size, device, False)
        if self.executor is not None:
            self.future = self.executor.submit(
                self.pool.batch, batch_size, device, False
            )
            self.signature = signature
        else:
            self.future = None
        return batch

    def invalidate(self) -> None:
        if self.future is not None and not self.future.cancel():
            # Wait for the RNG consumer before snapshotting or changing replay.
            self.future.result()
        self.future = None
        self.signature = None

    def close(self) -> None:
        self.invalidate()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)


class ReplayPartitionEmpty(RuntimeError):
    """Raised while a new replay buffer has no games in one stable partition."""


def _move_batch(batch: tuple[torch.Tensor, ...], device: torch.device):
    return tuple(
        value.to(device, non_blocking=device.type == "cuda") for value in batch
    )


def _learning_rate(config: TrainingV2Config, accepted_count: int) -> float:
    schedule = sorted(parse_lr_schedule(config.lr_schedule).items())
    rate = schedule[0][1]
    for threshold, candidate in schedule[1:]:
        if accepted_count < threshold:
            break
        rate = candidate
    return rate


def _restore_rng(payload: dict[str, Any]) -> None:
    if payload.get("torch_rng_state") is not None:
        torch.set_rng_state(payload["torch_rng_state"])
    if payload.get("numpy_rng_state") is not None:
        np.random.set_state(payload["numpy_rng_state"])
    if payload.get("python_rng_state") is not None:
        random.setstate(payload["python_rng_state"])
    if payload.get("cuda_rng_states") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng_states"])


def _restore_optimizer_state(
    optimizer: Adam, state: dict[str, Any], device: torch.device
) -> None:
    """Restore moments without letting an old checkpoint disable fused CUDA Adam."""
    if device.type == "cuda":
        # ``Optimizer.load_state_dict`` consults this flag while moving scalar
        # step tensors. Set it before loading so legacy CPU steps move to CUDA
        # along with the moment tensors.
        state = {
            **state,
            "param_groups": [
                {**group, "fused": True} for group in state["param_groups"]
            ],
        }
    optimizer.load_state_dict(state)


def _trainer_process(
    config_data: dict[str, Any],
    champion_path: str,
    champion_attempt: int,
    initial_shards: list[dict[str, Any]],
    initially_pending: bool,
    command_queue: Any,
    event_queue: Any,
    cancelled: Any,
) -> None:
    prefetcher: ReplayPrefetcher | None = None
    try:
        config = TrainingV2Config.model_validate(config_data)
        random.seed(config.run_seed)
        np.random.seed(config.run_seed % (1 << 32))
        torch.manual_seed(config.run_seed)
        device = torch.device(config.device)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.run_seed)
        model, payload = _load_checkpoint(champion_path)
        model.value_loss_weight = config.value_loss_weight
        progress_payload = payload
        model.to(device)
        optimizer = Adam(
            model.parameters(),
            lr=_learning_rate(config, 0),
            weight_decay=config.l2_reg,
            fused=device.type == "cuda",
        )
        use_amp = device.type == "cuda" and config.precision != "32-true"
        scaler = torch.GradScaler("cuda", enabled=use_amp)
        learner_path = Path(config.base_dir) / "learner.pt"
        fresh_games = sum(int(item["games"]) for item in initial_shards)
        sample_budget = sum(
            round(int(item["train_samples"]) * config.replay_ratio)
            for item in initial_shards
        )
        training_steps = 0
        training_samples = 0
        best_loss = float("inf")
        best_payload: dict[str, Any] | None = None
        pending = initially_pending
        accepted_count = 0
        pool = ReplayPool(config, initial_shards)
        prefetcher = ReplayPrefetcher(pool, config.data_loader_workers)
        batch_size = config.training_batch_size
        training_started = time.monotonic()
        candidate_cycle_started = training_started
        if learner_path.is_file():
            resumed_model, resumed = _load_checkpoint(learner_path)
            if int(resumed.get("champion_attempt", -1)) == champion_attempt:
                progress_payload = resumed
                model = resumed_model.to(device)
                model.value_loss_weight = config.value_loss_weight
                accepted_count = int(resumed.get("accepted_count", 0))
                optimizer = Adam(
                    model.parameters(),
                    lr=_learning_rate(config, accepted_count),
                    weight_decay=config.l2_reg,
                    fused=device.type == "cuda",
                )
                if resumed.get("optimizer_state") is not None:
                    _restore_optimizer_state(
                        optimizer, resumed["optimizer_state"], device
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = _learning_rate(config, accepted_count)
                if resumed.get("scaler_state") is not None:
                    scaler.load_state_dict(resumed["scaler_state"])
                _restore_rng(resumed)
                if resumed.get("replay_rng_state") is not None:
                    pool.restore_rng_state(resumed["replay_rng_state"])
                fresh_games = int(resumed.get("fresh_games", 0))
                sample_budget = int(resumed.get("sample_budget", 0))
                training_steps = int(resumed.get("training_steps", 0))
                training_samples = int(resumed.get("training_samples", 0))

        current_loss = float(progress_payload.get("val_loss", 0.0))
        rate_start_steps = training_steps
        rate_start_samples = training_samples
        event_queue.put(
            {
                "type": "trainer_progress",
                "steps": training_steps,
                "samples": training_samples,
                "champion_attempt": champion_attempt,
                "val_loss": current_loss,
                "sample_budget": sample_budget,
                "fresh_games": fresh_games,
                "batches_per_second": 0.0,
                "positions_per_second": 0.0,
            }
        )

        def save_learner() -> None:
            prefetcher.invalidate()
            _atomic_torch_save(
                learner_path,
                _checkpoint_payload(
                    model,
                    optimizer,
                    scaler,
                    champion_attempt=champion_attempt,
                    accepted_count=accepted_count,
                    fresh_games=fresh_games,
                    sample_budget=sample_budget,
                    training_steps=training_steps,
                    training_samples=training_samples,
                    effective_batch_size=batch_size,
                    replay_rng_state=pool.rng_state(),
                ),
            )

        while not cancelled.is_set():
            handled = False
            while True:
                try:
                    command = command_queue.get_nowait()
                except queue.Empty:
                    break
                handled = True
                kind = command["type"]
                if kind == "stop":
                    cancelled.set()
                    break
                if kind == "add_shard":
                    shard = command["shard"]
                    pool.add(shard)
                    fresh_games += int(shard["games"])
                    sample_budget += round(
                        int(shard["train_samples"]) * config.replay_ratio
                    )
                elif kind == "remove_shard":
                    pool.remove(int(command["shard_id"]))
                    event_queue.put(
                        {"type": "shard_removed", "shard_id": command["shard_id"]}
                    )
                elif kind == "decision":
                    pending = bool(command.get("pause_after_decision", False))
                    submitted_fresh = int(command["submitted_fresh_games"])
                    fresh_games = max(0, fresh_games - submitted_fresh)
                    champion_attempt = int(command["champion_attempt"])
                    accepted_count = int(command["accepted_count"])
                    # The actor always reloads the accepted champion. A rejected
                    # learner is only worth retaining when it is at least a
                    # break-even challenger; retaining a clearly losing model
                    # compounds regressions across every later candidate cycle.
                    model, decision_payload = _load_checkpoint(command["learner_path"])
                    model.value_loss_weight = config.value_loss_weight
                    model.to(device)
                    optimizer = Adam(
                        model.parameters(),
                        lr=_learning_rate(config, accepted_count),
                        weight_decay=config.l2_reg,
                        fused=device.type == "cuda",
                    )
                    if decision_payload.get("optimizer_state"):
                        _restore_optimizer_state(
                            optimizer, decision_payload["optimizer_state"], device
                        )
                        for group in optimizer.param_groups:
                            group["lr"] = _learning_rate(config, accepted_count)
                    if decision_payload.get("scaler_state"):
                        scaler.load_state_dict(decision_payload["scaler_state"])
                    best_loss = float("inf")
                    best_payload = None
                    candidate_cycle_started = time.monotonic()
                    save_learner()
                    event_queue.put(
                        {
                            "type": "decision_applied",
                            "champion_attempt": champion_attempt,
                            "fresh_games": fresh_games,
                        }
                    )
                elif kind == "pending":
                    pending = True
            if cancelled.is_set():
                break
            if pending or sample_budget < batch_size or not pool.shards:
                if not handled:
                    time.sleep(0.02)
                continue
            try:
                batch = _move_batch(
                    prefetcher.training_batch(batch_size, device), device
                )
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    loss = _model_loss(model, batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            except RuntimeError as error:
                if not _is_cuda_oom(error) or device.type != "cuda" or batch_size <= 2:
                    raise
                torch.cuda.empty_cache()
                batch_size = max(2, batch_size // 2)
                prefetcher.invalidate()
                event_queue.put(
                    {
                        "type": "effective_batch",
                        "component": "trainer",
                        "value": batch_size,
                    }
                )
                continue
            training_steps += 1
            training_samples += batch_size
            sample_budget = max(0, sample_budget - batch_size)
            if training_steps % config.validation_interval_steps == 0:
                model.eval()
                validation_losses: list[float] = []
                with torch.inference_mode():
                    for _ in range(config.validation_batches):
                        try:
                            validation = _move_batch(
                                pool.batch(batch_size, device, True), device
                            )
                        except ReplayPartitionEmpty:
                            break
                        with torch.autocast(
                            device_type=device.type,
                            dtype=torch.float16,
                            enabled=use_amp,
                        ):
                            validation_losses.append(
                                float(_model_loss(model, validation).detach().cpu())
                            )
                current_loss = (
                    sum(validation_losses) / len(validation_losses)
                    if validation_losses
                    else float(loss.detach().cpu())
                )
                if validation_losses and current_loss < best_loss:
                    best_loss = current_loss
                    prefetcher.invalidate()
                    best_payload = _checkpoint_payload(
                        model,
                        optimizer,
                        scaler,
                        champion_attempt=champion_attempt,
                        accepted_count=accepted_count,
                        fresh_games=fresh_games,
                        sample_budget=sample_budget,
                        replay_rng_state=pool.rng_state(),
                        val_loss=best_loss,
                        training_steps=training_steps,
                        training_samples=training_samples,
                    )
                save_learner()
            if training_steps % min(config.validation_interval_steps, 10) == 0:
                elapsed = max(time.monotonic() - training_started, 1e-9)
                event_queue.put(
                    {
                        "type": "trainer_progress",
                        "steps": training_steps,
                        "samples": training_samples,
                        "champion_attempt": champion_attempt,
                        "val_loss": current_loss,
                        "sample_budget": sample_budget,
                        "fresh_games": fresh_games,
                        "batches_per_second": (training_steps - rate_start_steps)
                        / elapsed,
                        "positions_per_second": (training_samples - rate_start_samples)
                        / elapsed,
                    }
                )
            if (
                fresh_games >= config.replay_warmup_games
                # The coordinator starts self-play again once debt falls below
                # this watermark.  Waiting for an empty queue races that refill
                # and makes candidate creation unreachable whenever the actor is
                # faster than the trainer.
                and sample_budget <= max(batch_size, pool.resume_sample_budget())
            ):
                if best_payload is None:
                    prefetcher.invalidate()
                    best_payload = _checkpoint_payload(
                        model,
                        optimizer,
                        scaler,
                        champion_attempt=champion_attempt,
                        accepted_count=accepted_count,
                        fresh_games=fresh_games,
                        sample_budget=sample_budget,
                        replay_rng_state=pool.rng_state(),
                        val_loss=float(loss.detach().cpu()),
                        training_steps=training_steps,
                        training_samples=training_samples,
                    )
                    best_loss = float(best_payload["val_loss"])
                candidate_path = Path(config.base_dir) / ".candidate.pt"
                _atomic_torch_save(candidate_path, best_payload)
                pending = True
                event_queue.put(
                    {
                        "type": "candidate",
                        "path": str(candidate_path),
                        "val_loss": best_loss,
                        "training_steps": best_payload["training_steps"],
                        "training_samples": best_payload["training_samples"],
                        "submitted_fresh_games": fresh_games,
                        "candidate_latency_seconds": time.monotonic()
                        - candidate_cycle_started,
                    }
                )
        save_learner()
        prefetcher.close()
        event_queue.put({"type": "trainer_stopped"})
    except BaseException as error:
        if prefetcher is not None:
            prefetcher.close()
        event_queue.put(
            {"type": "error", "component": "trainer", "message": repr(error)}
        )


class _ActorEvaluator:
    def __init__(
        self,
        device: torch.device,
        seed: int,
        precision: str = "auto",
        amp_min_batch_size: int = 96,
        value_scale: float = 1.0,
    ):
        self.device = device
        self.models: dict[int, ConnectFourNet] = {}
        self.paths: dict[int, str] = {}
        self.seed = seed
        self.precision = precision
        self.amp_min_batch_size = amp_min_batch_size
        self.value_scale = value_scale
        self.neural_evaluations = 0

    def configure(self, entries: Iterable[dict[str, Any]]) -> None:
        wanted = {
            int(entry["attempt_n"]): str(entry["checkpoint_path"]) for entry in entries
        }
        for model_id in list(self.models):
            if model_id not in wanted:
                del self.models[model_id]
                self.paths.pop(model_id, None)
        for model_id, path in wanted.items():
            if self.paths.get(model_id) == path:
                continue
            model, _ = _load_checkpoint(path)
            model.eval().to(self.device)
            self.models[model_id] = model
            self.paths[model_id] = path

    def __call__(
        self, model_id: int, positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size = positions.shape[0]
        self.neural_evaluations += batch_size
        baseline = _baseline_kind_and_game(model_id)
        if baseline is not None and baseline[0] == UNIFORM_MODEL_ID:
            return (
                np.zeros((batch_size, c4a0_cpp.N_COLS), dtype=np.float32),
                np.zeros(batch_size, dtype=np.float32),
                np.zeros(batch_size, dtype=np.float32),
            )
        if baseline is not None and baseline[0] == RANDOM_MODEL_ID:
            policy, q_penalty, q_no_penalty = _random_baseline_batch(
                self.seed ^ baseline[1], positions
            )
            return (
                policy,
                np.ascontiguousarray(q_penalty * self.value_scale),
                np.ascontiguousarray(q_no_penalty * self.value_scale),
            )
        model = self.models[int(model_id)]
        return model.infer_numpy(
            positions,
            value_scale=self.value_scale,
            precision=self.precision,
            amp_min_batch_size=self.amp_min_batch_size,
        )


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorized SplitMix64 output used by the stateless random baseline."""
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def _random_baseline_batch(
    seed: int, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate every random-opponent leaf in one NumPy kernel.

    The output is stateless and depends on the run seed and board, so batching and
    native worker scheduling cannot change it.  A shared model ID lets leaves from
    all random-opponent games occupy the same inference batch.
    """
    batch_size = int(positions.shape[0])
    flat = np.ascontiguousarray(positions > 0).reshape(batch_size, -1)
    bit_indices = np.arange(flat.shape[1], dtype=np.uint64)
    keys = np.full(batch_size, np.uint64(seed), dtype=np.uint64)
    # Fold all occupied channel/cell bits into a board key without Python loops.
    salts = _splitmix64(bit_indices + np.uint64(0xD1B54A32D192ED03))
    keys ^= np.bitwise_xor.reduce(np.where(flat, salts, np.uint64(0)), axis=1)
    streams = keys[:, None] + np.arange(c4a0_cpp.N_COLS + 2, dtype=np.uint64)
    random_bits = _splitmix64(streams)
    unit = ((random_bits >> np.uint64(40)).astype(np.float32)) * np.float32(
        1.0 / (1 << 24)
    )
    return (
        np.ascontiguousarray(unit[:, : c4a0_cpp.N_COLS]),
        np.ascontiguousarray(unit[:, c4a0_cpp.N_COLS] * 2.0 - 1.0),
        np.ascontiguousarray(unit[:, c4a0_cpp.N_COLS + 1] * 2.0 - 1.0),
    )


def _allocate_counts(total: int, weights: list[float]) -> list[int]:
    raw = [total * weight for weight in weights]
    counts = [math.floor(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(
        range(len(weights)), key=lambda index: raw[index] - counts[index], reverse=True
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def _league_requests(
    config: TrainingV2Config,
    champion_attempt: int,
    archive_attempts: list[int],
    first_game_id: int,
) -> list[Any]:
    weights = [
        config.champion_self_play_weight,
        config.archive_weight,
        config.uniform_weight,
        config.random_weight,
    ]
    counts = _allocate_counts(config.self_play_batch_games, weights)
    if not archive_attempts:
        counts[0] += counts[1]
        counts[1] = 0
    categories: list[int] = []
    for index, count in enumerate(counts):
        categories.extend([index] * count)
    allocation_rng = random.Random(config.run_seed ^ first_game_id)
    allocation_rng.shuffle(categories)
    archive_weights = [
        config.archive_recency**age for age in range(len(archive_attempts))
    ]
    requests = []
    category_offsets = [0, 0, 0, 0]
    for offset, category in enumerate(categories):
        game_id = first_game_id + offset
        if category == 0:
            red = gold = champion_attempt
        else:
            if category == 1:
                opponent_rng = random.Random((config.run_seed << 64) ^ game_id)
                opponent = opponent_rng.choices(
                    archive_attempts, weights=archive_weights, k=1
                )[0]
            elif category == 2:
                opponent = _baseline_model_id(UNIFORM_MODEL_ID, game_id)
            else:
                # One shared ID is intentional: the evaluator is stateless by
                # board and can now evaluate leaves from many random games in a
                # single vectorized callback.
                opponent = RANDOM_MODEL_ID
            if category_offsets[category] % 2 == 0:
                red, gold = champion_attempt, opponent
            else:
                red, gold = opponent, champion_attempt
            category_offsets[category] += 1
        requests.append(
            c4a0_cpp.GameRequest(c4a0_cpp.GameMetadata(game_id, red, gold), [])
        )
    return requests


def _self_play_options(config: TrainingV2Config, inference_batch: int):
    options = c4a0_cpp.SelfPlayOptions()
    options.max_nn_batch_size = inference_batch
    options.n_mcts_iterations = config.n_mcts_iterations
    options.c_exploration = config.c_exploration
    options.c_ply_penalty = config.c_ply_penalty
    options.root_dirichlet_alpha = config.root_dirichlet_alpha
    options.root_dirichlet_epsilon = config.root_dirichlet_epsilon
    options.temperature_midpoint_ply = config.temperature_cutoff_ply
    options.temperature_cutoff_ply = config.temperature_cutoff_ply
    options.early_temperature = config.early_temperature
    options.middle_temperature = config.early_temperature
    options.late_temperature = config.late_temperature
    options.seed = config.run_seed
    options.worker_threads = config.mcts_worker_threads
    return options


def _run_arena(
    config: TrainingV2Config,
    evaluator: _ActorEvaluator,
    champion: dict[str, Any],
    candidate: dict[str, Any],
    openings: list[list[int]],
    cancelled: Any,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    starting_evaluations = evaluator.neural_evaluations
    evaluator.configure([champion, candidate])
    options = _self_play_options(config, config.inference_batch_size)
    options.root_dirichlet_epsilon = 0.0
    options.root_dirichlet_alpha = 0.0
    options.temperature_midpoint_ply = 0
    options.temperature_cutoff_ply = 0
    options.early_temperature = 0.0
    options.middle_temperature = 0.0
    options.late_temperature = 0.0
    gate = PairedSprtGate(
        p0=config.arena_p0,
        p1=config.arena_p1,
        alpha=config.arena_alpha,
        beta=config.arena_beta,
        minimum_games=config.arena_min_games,
        maximum_games=config.arena_max_games,
    )
    wins = draws = losses = 0
    path: list[float] = []
    decision = "inconclusive"
    candidate_id = int(candidate["attempt_n"])
    champion_id = int(champion["attempt_n"])
    game_id = 1
    opening_index = 0
    while opening_index < len(openings) and gate.games < gate.maximum_games:
        remaining_pairs = (gate.maximum_games - gate.games) // 2
        pair_count = min(
            config.arena_pair_batch_size,
            remaining_pairs,
            len(openings) - opening_index,
        )
        batch_openings = openings[opening_index : opening_index + pair_count]
        opening_index += pair_count
        requests = []
        for opening in batch_openings:
            requests.extend(
                [
                    c4a0_cpp.GameRequest(
                        c4a0_cpp.GameMetadata(game_id, candidate_id, champion_id),
                        opening,
                    ),
                    c4a0_cpp.GameRequest(
                        c4a0_cpp.GameMetadata(game_id + 1, champion_id, candidate_id),
                        opening,
                    ),
                ]
            )
            game_id += 2

        def arena_telemetry(snapshot: dict[str, Any]) -> None:
            if progress is not None:
                progress(
                    {
                        **snapshot,
                        "completed_games": gate.games
                        + int(snapshot["completed_games"]),
                        "total_games": gate.maximum_games,
                        "wins": wins,
                        "draws": draws,
                        "losses": losses,
                        "llr": gate.llr,
                    }
                )

        results = c4a0_cpp.play_games_v2(
            requests,
            options,
            evaluator,
            None,
            cancelled.is_set,
            arena_telemetry,
        )
        for result in sorted(results.results, key=lambda game: game.metadata.game_id):
            red_score = float(result.player0_score())
            score = (
                red_score
                if int(result.metadata.player0_id) == candidate_id
                else 1.0 - red_score
            )
            if score == 1.0:
                wins += 1
            elif score == 0.5:
                draws += 1
            else:
                losses += 1
            gate.update(score)
        gate_decision = gate.decision()
        path.append(gate.llr)
        if progress is not None:
            progress(
                {
                    "completed_games": gate.games,
                    "total_games": gate.maximum_games,
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "llr": gate.llr,
                    "pair_batch_size": pair_count,
                }
            )
        if gate_decision is not None:
            decision = gate_decision
            break
    games = wins + draws + losses
    elapsed = max(time.monotonic() - started, 1e-9)
    if decision == "inconclusive":
        decision = "rejected"
    return {
        "decision": decision,
        "reason": "sprt" if gate.crossed_boundary else "max_games_inconclusive",
        "statistical_model": "paired-multinomial-gsprt-v1",
        "paired_outcomes": gate.counts.tolist(),
        "error_rates_calibrated_for_production_openings": False,
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": (wins + 0.5 * draws) / games,
        "games_per_second": games / elapsed,
        "neural_evaluations_per_second": (
            evaluator.neural_evaluations - starting_evaluations
        )
        / elapsed,
        "llr": gate.llr,
        "llr_path": path,
        "openings": games // 2,
        "opening_count": games // 2,
        "p0": gate.p0,
        "p1": gate.p1,
        "alpha": config.arena_alpha,
        "beta": config.arena_beta,
        "lower_bound": gate.lower,
        "upper_bound": gate.upper,
    }


class SprtGate:
    """Sequential probability-ratio gate using win-equivalent scores."""

    def __init__(
        self,
        p0: float,
        p1: float,
        alpha: float,
        beta: float,
        minimum_games: int,
        maximum_games: int,
    ) -> None:
        self.p0 = p0
        self.p1 = p1
        self.minimum_games = minimum_games
        self.maximum_games = maximum_games
        self.upper = math.log((1 - beta) / alpha)
        self.lower = math.log(beta / (1 - alpha))
        self.llr = 0.0
        self.games = 0
        self.crossed_boundary = False

    def update(self, score: float) -> None:
        if score not in {0.0, 0.5, 1.0}:
            raise ValueError("SPRT scores must be win, draw, or loss values")
        self.llr += score * math.log(self.p1 / self.p0) + (1 - score) * math.log(
            (1 - self.p1) / (1 - self.p0)
        )
        self.games += 1

    def decision(self) -> str | None:
        if self.games >= self.minimum_games:
            if self.llr >= self.upper:
                self.crossed_boundary = True
                return "accepted"
            if self.llr <= self.lower:
                self.crossed_boundary = True
                return "rejected"
        if self.games >= self.maximum_games:
            return "rejected"
        return None


def _owned_training_child(parent_pid: int, target, args) -> None:
    """Linux children cannot keep writing a run after their coordinator dies."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(
            ctypes.get_errno(), "cannot bind training child to coordinator lifetime"
        )
    if os.getppid() != parent_pid:
        return
    target(*args)


def _actor_process(
    config_data: dict[str, Any],
    command_queue: Any,
    event_queue: Any,
    cancelled: Any,
) -> None:
    try:
        config = TrainingV2Config.model_validate(config_data)
        device = torch.device(config.device)
        evaluator = _ActorEvaluator(
            device,
            config.run_seed,
            config.precision,
            config.inference_amp_min_batch_size,
            config.mcts_value_scale,
        )
        inference_batch = config.inference_batch_size
        while not cancelled.is_set():
            try:
                command = command_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if command["type"] == "stop":
                break
            if command["type"] == "reload":
                evaluator.configure([command["champion"], *command["archives"]])
                event_queue.put(
                    {
                        "type": "model_reloaded",
                        "champion_attempt": command["champion"]["attempt_n"],
                    }
                )
            elif command["type"] == "generate":
                models = [command["champion"], *command["archives"]]
                evaluator.configure(models)
                requests = _league_requests(
                    config,
                    int(command["champion"]["attempt_n"]),
                    [int(item["attempt_n"]) for item in command["archives"]],
                    int(command["first_game_id"]),
                )
                started = time.monotonic()
                starting_evaluations = evaluator.neural_evaluations
                while True:
                    try:
                        games = c4a0_cpp.play_games_v2(
                            requests,
                            _self_play_options(config, inference_batch),
                            evaluator,
                            None,
                            cancelled.is_set,
                            lambda snapshot: event_queue.put(
                                {
                                    "type": "self_play_progress",
                                    "champion_attempt": int(
                                        command["champion"]["attempt_n"]
                                    ),
                                    **snapshot,
                                }
                            ),
                        )
                        break
                    except RuntimeError as error:
                        if (
                            not _is_cuda_oom(error)
                            or device.type != "cuda"
                            or inference_batch <= 1
                        ):
                            raise
                        torch.cuda.empty_cache()
                        inference_batch = max(1, inference_batch // 2)
                        event_queue.put(
                            {
                                "type": "effective_batch",
                                "component": "actor",
                                "value": inference_batch,
                            }
                        )
                elapsed = time.monotonic() - started
                chunks = games.split_games(config.self_play_shard_games)
                prepared: list[dict[str, Any]] = []
                for chunk in chunks:
                    chunk_games = chunk.results
                    first_game_id = min(
                        int(game.metadata.game_id) for game in chunk_games
                    )
                    path = (
                        Path(config.base_dir) / "replay" / f"{first_game_id:020d}.cbor"
                    )
                    _atomic_bytes(path, chunk.to_cbor())
                    prepared.append(
                        {
                            "path": str(path),
                            "games": len(chunk_games),
                            "samples": sum(len(game.samples) for game in chunk_games),
                            "train_samples": sum(
                                len(game.samples)
                                for game in chunk_games
                                if not _validation_game(
                                    config.run_seed,
                                    int(game.metadata.game_id),
                                    config.validation_fraction,
                                )
                            ),
                            "first_game_id": first_game_id,
                        }
                    )
                batch_samples = sum(int(item["samples"]) for item in prepared)
                for index, shard in enumerate(prepared):
                    event_queue.put(
                        {
                            "type": "shard",
                            **shard,
                            "champion_attempt": int(command["champion"]["attempt_n"]),
                            "batch_final": index == len(prepared) - 1,
                            "batch_games": len(requests),
                            "batch_samples": batch_samples,
                            "batch_elapsed": elapsed,
                            "batch_neural_evaluations": (
                                evaluator.neural_evaluations - starting_evaluations
                            ),
                        }
                    )
            elif command["type"] == "arena":
                while True:
                    try:
                        arena_config = config.model_copy(
                            update={"inference_batch_size": inference_batch}
                        )
                        result = _run_arena(
                            arena_config,
                            evaluator,
                            command["champion"],
                            command["candidate"],
                            command["openings"],
                            cancelled,
                            lambda snapshot: event_queue.put(
                                {
                                    "type": "arena_progress",
                                    "attempt_n": command["candidate"]["attempt_n"],
                                    **snapshot,
                                }
                            ),
                        )
                        break
                    except RuntimeError as error:
                        if (
                            not _is_cuda_oom(error)
                            or device.type != "cuda"
                            or inference_batch <= 1
                        ):
                            raise
                        torch.cuda.empty_cache()
                        inference_batch = max(1, inference_batch // 2)
                        event_queue.put(
                            {
                                "type": "effective_batch",
                                "component": "actor",
                                "value": inference_batch,
                            }
                        )
                event_queue.put(
                    {
                        "type": "arena",
                        "attempt_n": command["candidate"]["attempt_n"],
                        "result": result,
                    }
                )
        event_queue.put({"type": "actor_stopped"})
    except BaseException as error:
        event_queue.put({"type": "error", "component": "actor", "message": repr(error)})


def _winning(board: list[list[int]], player: int) -> bool:
    for row in range(6):
        for col in range(7):
            for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
                if all(
                    0 <= row + dr * step < 6
                    and 0 <= col + dc * step < 7
                    and board[row + dr * step][col + dc * step] == player
                    for step in range(4)
                ):
                    return True
    return False


def _opening_is_nonterminal(moves: list[int]) -> bool:
    board = [[0] * 7 for _ in range(6)]
    heights = [0] * 7
    player = 1
    for move in moves:
        if move < 0 or move >= 7 or heights[move] >= 6:
            return False
        board[heights[move]][move] = player
        heights[move] += 1
        if _winning(board, player):
            return False
        player = 3 - player
    return True


def _opening_key(moves: Iterable[int]) -> tuple[int, ...]:
    board = [[0] * 7 for _ in range(6)]
    heights = [0] * 7
    player = 1
    for move in moves:
        board[heights[move]][move] = player
        heights[move] += 1
        player = 3 - player
    return tuple(cell for row in board for cell in row)


def _arena_openings(
    base_dir: Path, seed: int, total_games: int = 200
) -> list[list[int]]:
    path = base_dir / "arena_openings.json"
    opening_count = math.ceil(total_games / 2)
    if path.is_file():
        persisted = json.loads(path.read_text())
        if len(persisted) >= opening_count:
            return persisted[:opening_count]
    rng = random.Random(seed ^ 0xA2E4A)
    base: set[tuple[int, ...]] = set()
    positions: set[tuple[int, ...]] = set()
    while len(base) < math.ceil(opening_count / 2):
        depth = rng.randint(1, 6)
        moves = tuple(rng.randrange(7) for _ in range(depth))
        mirrored = tuple(6 - move for move in moves)
        position_key = _opening_key(moves)
        mirrored_key = _opening_key(mirrored)
        if (
            moves != mirrored
            and _opening_is_nonterminal(list(moves))
            and mirrored not in base
            and position_key != mirrored_key
            and position_key not in positions
            and mirrored_key not in positions
        ):
            base.add(moves)
            positions.add(position_key)
            positions.add(mirrored_key)
    openings: list[list[int]] = []
    for moves in sorted(base, key=lambda item: (len(item), item)):
        openings.append(list(moves))
        openings.append([6 - move for move in moves])
    _atomic_bytes(path, json.dumps(openings, indent=2).encode())
    return openings[:opening_count]


def _prune_rejected(
    manifest: RunManifest, retention: int, preserve_attempt: int | None = None
) -> None:
    rows = manifest.connection.execute(
        "SELECT attempt_n, checkpoint_path FROM attempts WHERE status = 'rejected' "
        "ORDER BY attempt_n DESC"
    ).fetchall()
    retained = {int(row["attempt_n"]) for row in rows[:retention]}
    if preserve_attempt is not None:
        retained.add(preserve_attempt)
    for row in rows:
        if int(row["attempt_n"]) in retained:
            continue
        try:
            manifest.resolve_artifact(row["checkpoint_path"]).unlink()
        except FileNotFoundError:
            pass


def _recover_orphaned_candidates(manifest: RunManifest) -> None:
    """Recover atomically saved candidates not registered before a crash."""
    champion_attempt = int(manifest.champion()["attempt_n"])

    def register(attempt_n: int, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        parent = int(payload.get("champion_attempt", champion_attempt))
        if manifest.pending_attempt() is None and parent == champion_attempt:
            manifest.add_pending_attempt(
                attempt_n,
                parent,
                checkpoint,
                float(payload.get("val_loss", float("inf"))),
                int(payload.get("training_steps", 0)),
                int(payload.get("training_samples", 0)),
                int(payload.get("fresh_games", 0)),
            )
        else:
            manifest.add_interrupted_attempt(attempt_n, parent, checkpoint)

    attempts_dir = manifest.base_dir / "attempts"
    if attempts_dir.is_dir():
        for checkpoint in sorted(attempts_dir.glob("*/model.pt")):
            try:
                attempt_n = int(checkpoint.parent.name)
            except ValueError:
                continue
            row = manifest.connection.execute(
                "SELECT 1 FROM attempts WHERE attempt_n = ?", (attempt_n,)
            ).fetchone()
            if row is None:
                register(attempt_n, checkpoint)
    orphaned_candidate = manifest.base_dir / ".candidate.pt"
    if orphaned_candidate.is_file():
        attempt_n = manifest.next_attempt()
        interrupted_path = attempts_dir / f"{attempt_n:06d}" / "model.pt"
        interrupted_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(orphaned_candidate, interrupted_path)
        _sync_directory(interrupted_path.parent)
        _sync_directory(manifest.base_dir)
        register(attempt_n, interrupted_path)


def run_async_training(
    config: TrainingV2Config,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    should_stop_after_candidate: Callable[[], bool] | None = None,
    arena_decision_override: Callable[[dict[str, Any]], str] | None = None,
) -> dict[str, Any]:
    config = config.model_copy(
        update={"base_dir": str(Path(config.base_dir).resolve())}
    )
    with training_run_owner(Path(config.base_dir)):
        with RunManifest.create(Path(config.base_dir), config) as manifest:
            return _run_async_training_owned(
                config,
                manifest,
                progress_callback,
                should_cancel,
                should_stop_after_candidate,
                arena_decision_override,
            )


def _run_async_training_owned(
    config: TrainingV2Config,
    manifest: RunManifest,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    should_stop_after_candidate: Callable[[], bool] | None = None,
    arena_decision_override: Callable[[dict[str, Any]], str] | None = None,
) -> dict[str, Any]:
    base_dir = Path(config.base_dir)
    persisted_config = manifest.get_state("config")
    if not isinstance(persisted_config, dict):
        raise RuntimeError("V2 run is missing its effective configuration")
    # Resume the exact data/search setup. Device and stopping limits are operational
    # controls and may safely change between invocations.
    persisted_config.update(
        {
            "base_dir": config.base_dir,
            "device": config.device,
            "max_gens": config.max_gens,
            "max_candidate_attempts": config.max_candidate_attempts,
        }
    )
    config = TrainingV2Config.model_validate(persisted_config)
    model_config = ModelConfig(
        n_residual_blocks=config.n_residual_blocks,
        conv_filter_size=config.conv_filter_size,
        n_policy_layers=config.n_policy_layers,
        n_value_layers=config.n_value_layers,
        lr_schedule=parse_lr_schedule(config.lr_schedule),
        l2_reg=config.l2_reg,
    )
    if manifest.get_state("champion_attempt") is None:
        random.seed(config.run_seed)
        np.random.seed(config.run_seed % (1 << 32))
        torch.manual_seed(config.run_seed)
        root_path = base_dir / "attempts" / "000000" / "model.pt"
        root = ConnectFourNet(model_config)
        root.mcts_value_scale = config.mcts_value_scale
        root.value_loss_weight = config.value_loss_weight
        _atomic_torch_save(root_path, _checkpoint_payload(root, status="accepted"))
        manifest.add_root(root_path)
    _recover_orphaned_candidates(manifest)
    if (
        manifest.pending_attempt() is None
        and config.max_gens is not None
        and int(manifest.get_state("accepted_count") or 0) >= config.max_gens
    ):
        return manifest.status()
    openings = _arena_openings(base_dir, config.run_seed, config.arena_max_games)
    champion = manifest.champion()
    pending = manifest.pending_attempt()
    active_shards = manifest.active_shards()
    context = mp.get_context("spawn")
    actor_commands = context.Queue()
    actor_events = context.Queue()
    trainer_commands = context.Queue()
    trainer_events = context.Queue()
    cancelled = context.Event()
    actor = context.Process(
        target=_owned_training_child,
        args=(
            os.getpid(),
            _actor_process,
            (config.model_dump(mode="json"), actor_commands, actor_events, cancelled),
        ),
        name="c4a0-self-play",
    )
    trainer = context.Process(
        target=_owned_training_child,
        args=(
            os.getpid(),
            _trainer_process,
            (
                config.model_dump(mode="json"),
                champion["checkpoint_path"],
                champion["attempt_n"],
                active_shards,
                pending is not None,
                trainer_commands,
                trainer_events,
                cancelled,
            ),
        ),
        name="c4a0-trainer",
    )
    writer: SummaryWriter | None = None
    try:
        actor.start()
        trainer.start()
        actor_busy = False
        arena_waiting = pending is not None
        pending_deletions: dict[int, str] = {}
        paused_for_debt = False
        trainer_budget = 0
        trainer_fresh_games = 0
        average_train_samples = 1.0
        stop_requested = False
        lifecycle_waiting: set[str] = set()
        candidate_decisions = 0
        last_logged_debt: int | None = None
        writer = SummaryWriter(log_dir=str(base_dir / "tensorboard"))

        def emit(phase: str, payload: dict[str, Any]) -> None:
            if progress_callback is not None:
                progress_callback(phase, payload)

        def actor_models():
            current = manifest.champion()
            archives = manifest.accepted_models(config.archive_depth)
            return current, archives

        def queue_generation() -> None:
            nonlocal actor_busy
            current, archives = actor_models()
            actor_commands.put(
                {
                    "type": "generate",
                    "champion": current,
                    "archives": archives,
                    "first_game_id": int(manifest.get_state("next_game_id") or 1),
                }
            )
            actor_busy = True

        if pending is not None:
            manifest.set_state("trainer_progress", {"state": "waiting_for_arena"})
            manifest.set_state(
                "arena_progress",
                {"state": "resuming", "attempt": int(pending["attempt_n"])},
            )
            trainer_commands.put({"type": "pending"})
            actor_commands.put(
                {
                    "type": "arena",
                    "champion": champion,
                    "candidate": pending,
                    "openings": openings,
                }
            )
            actor_busy = True
        else:
            queue_generation()
        while True:
            if should_cancel is not None and should_cancel():
                raise TrainingCancelled("asynchronous training cancelled")
            for source, events in (
                ("actor", actor_events),
                ("trainer", trainer_events),
            ):
                while True:
                    try:
                        event = events.get_nowait()
                    except queue.Empty:
                        break
                    kind = event["type"]
                    if kind == "error":
                        raise RuntimeError(
                            f"{event['component']} process failed: {event['message']}"
                        )
                    if kind == "shard":
                        actor_busy = not bool(event.get("batch_final", True))
                        trainer_budget += round(
                            int(event["train_samples"]) * config.replay_ratio
                        )
                        shard_id, retired = manifest.register_shard(event)
                        shard = {**event, "shard_id": shard_id, "active": 1}
                        trainer_commands.put({"type": "add_shard", "shard": shard})
                        for item in retired:
                            pending_deletions[int(item["shard_id"])] = str(item["path"])
                            trainer_commands.put(
                                {"type": "remove_shard", "shard_id": item["shard_id"]}
                            )
                        event_status = manifest.status()
                        step = int(manifest.get_state("next_game_id") or 1) - 1
                        writer.add_scalar(
                            "replay/games", event_status["replay_games"], step
                        )
                        if event.get("batch_final", True):
                            batch_games = int(event.get("batch_games", event["games"]))
                            batch_samples = int(
                                event.get("batch_samples", event["samples"])
                            )
                            batch_elapsed = float(
                                event.get("batch_elapsed", event.get("elapsed", 0.0))
                            )
                            batch_evaluations = int(
                                event.get(
                                    "batch_neural_evaluations",
                                    event.get("neural_evaluations", 0),
                                )
                            )
                            live = {
                                "state": "self_play",
                                "champion_attempt": int(event["champion_attempt"]),
                                "games_per_second": batch_games / batch_elapsed,
                                "positions_per_second": batch_samples / batch_elapsed,
                                "neural_evaluations_per_second": batch_evaluations
                                / batch_elapsed,
                            }
                            manifest.set_state("actor_progress", live)
                            emit(
                                "self_play",
                                {
                                    "component": source,
                                    "games": batch_games,
                                    "samples": batch_samples,
                                    **live,
                                    **event_status,
                                },
                            )
                            writer.add_scalar(
                                "actor/games_per_second",
                                batch_games / batch_elapsed,
                                step,
                            )
                            writer.add_scalar(
                                "actor/positions_per_second",
                                batch_samples / batch_elapsed,
                                step,
                            )
                            writer.add_scalar(
                                "actor/neural_evaluations_per_second",
                                batch_evaluations / batch_elapsed,
                                step,
                            )
                    elif kind == "self_play_progress":
                        live = {
                            "state": "self_play",
                            "champion_attempt": int(event["champion_attempt"]),
                            "completed_games": int(event["completed_games"]),
                            "total_games": int(event["total_games"]),
                            "mcts_iterations": int(event["mcts_iterations"]),
                            "neural_evaluations": int(event["neural_evaluations"]),
                            "neural_queue_depth": int(event["neural_queue_depth"]),
                            "mcts_queue_depth": int(event["mcts_queue_depth"]),
                            "elapsed_seconds": float(event["elapsed_seconds"]),
                        }
                        manifest.set_state("actor_progress", live)
                        emit("self_play_progress", {"component": source, **live})
                        step = int(event["mcts_iterations"])
                        writer.add_scalar(
                            "actor/neural_queue_depth",
                            int(event["neural_queue_depth"]),
                            step,
                        )
                        writer.add_scalar(
                            "actor/mcts_queue_depth",
                            int(event["mcts_queue_depth"]),
                            step,
                        )
                    elif kind == "candidate":
                        attempt_n = manifest.next_attempt()
                        attempt_path = (
                            base_dir / "attempts" / f"{attempt_n:06d}" / "model.pt"
                        )
                        attempt_path.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(event["path"], attempt_path)
                        _sync_directory(attempt_path.parent)
                        _sync_directory(base_dir)
                        champion = manifest.champion()
                        if stop_requested:
                            manifest.add_interrupted_attempt(
                                attempt_n,
                                int(champion["attempt_n"]),
                                attempt_path,
                            )
                            continue
                        manifest.add_pending_attempt(
                            attempt_n,
                            int(champion["attempt_n"]),
                            attempt_path,
                            float(event["val_loss"]),
                            int(event["training_steps"]),
                            int(event["training_samples"]),
                            int(event["submitted_fresh_games"]),
                        )
                        pending = manifest.pending_attempt()
                        assert pending is not None
                        arena_waiting = True
                        manifest.set_state(
                            "trainer_progress",
                            {
                                "state": "waiting_for_arena",
                                "steps": int(event["training_steps"]),
                            },
                        )
                        manifest.set_state(
                            "arena_progress",
                            {"state": "running", "attempt": attempt_n},
                        )
                        actor_commands.put(
                            {
                                "type": "arena",
                                "champion": champion,
                                "candidate": pending,
                                "openings": openings,
                            }
                        )
                        emit("arena", {"component": "arena", "attempt": attempt_n})
                        emit(
                            "candidate",
                            {
                                "component": source,
                                "attempt": attempt_n,
                                "candidate_latency_seconds": float(
                                    event["candidate_latency_seconds"]
                                ),
                            },
                        )
                        writer.add_scalar(
                            "trainer/candidate_latency_seconds",
                            float(event["candidate_latency_seconds"]),
                            attempt_n,
                        )
                    elif kind == "arena_progress":
                        live = {
                            "state": "running",
                            "attempt": int(event["attempt_n"]),
                            "completed_games": int(event["completed_games"]),
                            "total_games": int(event["total_games"]),
                            "wins": int(event.get("wins", 0)),
                            "draws": int(event.get("draws", 0)),
                            "losses": int(event.get("losses", 0)),
                            "llr": float(event.get("llr", 0.0)),
                            "mcts_iterations": int(event.get("mcts_iterations", 0)),
                            "neural_queue_depth": int(
                                event.get("neural_queue_depth", 0)
                            ),
                            "mcts_queue_depth": int(event.get("mcts_queue_depth", 0)),
                        }
                        manifest.set_state("arena_progress", live)
                        emit("arena_progress", {"component": "arena", **live})
                    elif kind == "arena":
                        actor_busy = False
                        arena_waiting = False
                        if arena_decision_override is not None:
                            overridden = arena_decision_override(event["result"])
                            if overridden not in {"accepted", "rejected"}:
                                raise ValueError(
                                    "arena decision override must accept or reject"
                                )
                            event["result"] = {
                                **event["result"],
                                "decision": overridden,
                                "reason": "injected_test_decision",
                            }
                        pending = manifest.pending_attempt()
                        if pending is None or int(pending["attempt_n"]) != int(
                            event["attempt_n"]
                        ):
                            raise RuntimeError(
                                "arena result does not match pending candidate"
                            )
                        accepted = manifest.decide_attempt(
                            int(event["attempt_n"]), event["result"]
                        )
                        candidate_decisions += 1
                        accepted_count = int(manifest.get_state("accepted_count") or 0)
                        stop_requested = (
                            bool(
                                should_stop_after_candidate is not None
                                and should_stop_after_candidate()
                            )
                            or bool(
                                config.max_gens is not None
                                and accepted_count >= config.max_gens
                            )
                            or bool(
                                config.max_candidate_attempts is not None
                                and candidate_decisions >= config.max_candidate_attempts
                            )
                        )
                        manifest.set_state(
                            "arena_progress",
                            {
                                "state": event["result"]["decision"],
                                "attempt": int(event["attempt_n"]),
                                "score": float(event["result"]["score"]),
                            },
                        )
                        champion = manifest.champion()
                        learner_incumbent_attempt: int | None
                        if accepted:
                            learner_path = str(champion["checkpoint_path"])
                            learner_incumbent_attempt = None
                            manifest.set_state("learner_incumbent_attempt", None)
                            manifest.set_state("learner_incumbent_score", None)
                        else:
                            score = float(event["result"]["score"])
                            incumbent_attempt = manifest.get_state(
                                "learner_incumbent_attempt"
                            )
                            incumbent_score = manifest.get_state(
                                "learner_incumbent_score"
                            )
                            incumbent_row = (
                                None
                                if incumbent_attempt is None
                                else manifest.connection.execute(
                                    "SELECT checkpoint_path FROM attempts "
                                    "WHERE attempt_n = ?",
                                    (int(incumbent_attempt),),
                                ).fetchone()
                            )
                            incumbent_path = (
                                None
                                if incumbent_row is None
                                else str(
                                    manifest.resolve_artifact(
                                        incumbent_row["checkpoint_path"]
                                    )
                                )
                            )
                            if score >= config.learner_incumbent_min_score and (
                                incumbent_score is None
                                or float(incumbent_score)
                                < config.learner_incumbent_min_score
                                or score > float(incumbent_score)
                                or incumbent_path is None
                                or not Path(incumbent_path).is_file()
                            ):
                                learner_incumbent_attempt = int(pending["attempt_n"])
                                learner_path = str(pending["checkpoint_path"])
                                manifest.set_state(
                                    "learner_incumbent_attempt",
                                    learner_incumbent_attempt,
                                )
                                manifest.set_state("learner_incumbent_score", score)
                            elif (
                                incumbent_score is not None
                                and float(incumbent_score)
                                >= config.learner_incumbent_min_score
                                and incumbent_path is not None
                                and Path(incumbent_path).is_file()
                            ):
                                assert incumbent_attempt is not None
                                learner_incumbent_attempt = int(incumbent_attempt)
                                learner_path = incumbent_path
                            else:
                                learner_incumbent_attempt = None
                                learner_path = str(champion["checkpoint_path"])
                                manifest.set_state("learner_incumbent_attempt", None)
                                manifest.set_state("learner_incumbent_score", None)
                        trainer_commands.put(
                            {
                                "type": "decision",
                                "accepted": accepted,
                                "champion_attempt": champion["attempt_n"],
                                "learner_path": learner_path,
                                "accepted_count": manifest.get_state("accepted_count"),
                                "submitted_fresh_games": int(
                                    pending.get("submitted_fresh_games", 0)
                                ),
                                "pause_after_decision": stop_requested,
                            }
                        )
                        current, archives = actor_models()
                        actor_commands.put(
                            {
                                "type": "reload",
                                "champion": current,
                                "archives": archives,
                            }
                        )
                        lifecycle_waiting.update({"actor", "trainer"})
                        _prune_rejected(
                            manifest,
                            config.rejected_weight_retention,
                            learner_incumbent_attempt,
                        )
                        emit(
                            "arena",
                            {
                                "component": "arena",
                                "attempt": event["attempt_n"],
                                **event["result"],
                                **manifest.status(),
                            },
                        )
                        writer.add_scalar(
                            "arena/score",
                            float(event["result"]["score"]),
                            int(event["attempt_n"]),
                        )
                        writer.add_scalar(
                            "arena/games_per_second",
                            float(event["result"]["games_per_second"]),
                            int(event["attempt_n"]),
                        )
                        writer.add_scalar(
                            "arena/neural_evaluations_per_second",
                            float(event["result"]["neural_evaluations_per_second"]),
                            int(event["attempt_n"]),
                        )
                        writer.add_scalar(
                            "champion/accepted_count",
                            int(manifest.get_state("accepted_count") or 0),
                            int(event["attempt_n"]),
                        )
                    elif kind == "shard_removed":
                        path = pending_deletions.pop(int(event["shard_id"]), None)
                        if path is not None:
                            try:
                                Path(path).unlink()
                            except FileNotFoundError:
                                pass
                    elif kind == "decision_applied":
                        lifecycle_waiting.discard("trainer")
                        trainer_fresh_games = int(event["fresh_games"])
                        manifest.set_state(
                            "trainer_progress",
                            {
                                "state": "ready",
                                "champion_attempt": int(event["champion_attempt"]),
                            },
                        )
                    elif kind == "model_reloaded":
                        lifecycle_waiting.discard("actor")
                        manifest.set_state(
                            "actor_progress",
                            {
                                "state": "ready",
                                "champion_attempt": int(event["champion_attempt"]),
                            },
                        )
                    elif kind == "trainer_progress":
                        trainer_budget = int(event["sample_budget"])
                        trainer_fresh_games = int(event["fresh_games"])
                        manifest.set_state(
                            "trainer_progress",
                            {
                                "state": "training",
                                "steps": int(event["steps"]),
                                "batches_per_second": float(
                                    event["batches_per_second"]
                                ),
                                "positions_per_second": float(
                                    event["positions_per_second"]
                                ),
                                "validation_loss": float(event["val_loss"]),
                                "champion_attempt": int(event["champion_attempt"]),
                                "training_queue_positions": int(event["sample_budget"]),
                                "fresh_game_queue": int(event["fresh_games"]),
                            },
                        )
                        emit("training", {"component": source, **event})
                        writer.add_scalar(
                            "trainer/batches_per_second",
                            float(event["batches_per_second"]),
                            int(event["steps"]),
                        )
                        writer.add_scalar(
                            "trainer/positions_per_second",
                            float(event["positions_per_second"]),
                            int(event["steps"]),
                        )
                        writer.add_scalar(
                            "trainer/validation_loss",
                            float(event["val_loss"]),
                            int(event["steps"]),
                        )
                        writer.add_scalar(
                            "trainer/training_queue_positions",
                            int(event["sample_budget"]),
                            int(event["steps"]),
                        )
                    elif kind == "effective_batch":
                        effective_config = manifest.get_state("config")
                        if not isinstance(effective_config, dict):
                            raise RuntimeError(
                                "V2 run is missing its effective configuration"
                            )
                        field = (
                            "inference_batch_size"
                            if event["component"] == "actor"
                            else "training_batch_size"
                        )
                        effective_config[field] = int(event["value"])
                        manifest.set_state("config", effective_config)
                        emit("training", {"component": source, **event})
            status = manifest.status()
            if status["replay_shards"]:
                average_train_samples = max(
                    1.0,
                    sum(int(item["train_samples"]) for item in manifest.active_shards())
                    / status["replay_shards"],
                )
            debt_shards = math.ceil(
                trainer_budget / (average_train_samples * config.replay_ratio)
            )
            was_paused_for_debt = paused_for_debt
            if debt_shards >= config.debt_pause_shards:
                paused_for_debt = True
            elif debt_shards <= config.debt_resume_shards:
                paused_for_debt = False
            if paused_for_debt != was_paused_for_debt:
                emit(
                    "backpressure",
                    {
                        "component": "actor",
                        "paused": paused_for_debt,
                        "training_debt_shards": debt_shards,
                        **status,
                    },
                )
            if debt_shards != last_logged_debt:
                writer.add_scalar(
                    "replay/training_debt_shards",
                    debt_shards,
                    int(status["next_game_id"]) - 1,
                )
                last_logged_debt = debt_shards
            if stop_requested and not arena_waiting and not lifecycle_waiting:
                return manifest.status()
            if (
                not actor_busy
                and not arena_waiting
                and not paused_for_debt
                and trainer_fresh_games < config.replay_warmup_games
                and not lifecycle_waiting
                and not stop_requested
            ):
                queue_generation()
            if not actor.is_alive() or not trainer.is_alive():
                if not cancelled.is_set():
                    raise RuntimeError(
                        "an asynchronous training process exited unexpectedly"
                    )
            time.sleep(0.02)
    finally:
        cancelled.set()
        actor_commands.put({"type": "stop"})
        trainer_commands.put({"type": "stop"})
        deadline = time.monotonic() + 3
        children = [child for child in (actor, trainer) if child.pid is not None]
        for child in children:
            child.join(timeout=max(0, deadline - time.monotonic()))
        for child in children:
            if child.is_alive():
                child.terminate()
        deadline = time.monotonic() + 1
        for child in children:
            child.join(timeout=max(0, deadline - time.monotonic()))
            if child.is_alive():
                child.kill()
        for child in children:
            child.join(timeout=1)
            if child.is_alive():
                raise RuntimeError(f"Training child did not exit: {child.name}")
        orphaned_candidate = base_dir / ".candidate.pt"
        if orphaned_candidate.is_file() and manifest.pending_attempt() is None:
            attempt_n = manifest.next_attempt()
            interrupted_path = base_dir / "attempts" / f"{attempt_n:06d}" / "model.pt"
            interrupted_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(orphaned_candidate, interrupted_path)
            _sync_directory(interrupted_path.parent)
            _sync_directory(base_dir)
            manifest.add_interrupted_attempt(
                attempt_n,
                int(manifest.champion()["attempt_n"]),
                interrupted_path,
            )
        if writer is not None:
            writer.flush()
            writer.close()
        for channel in (actor_commands, trainer_commands, actor_events, trainer_events):
            channel.cancel_join_thread()
            channel.close()
