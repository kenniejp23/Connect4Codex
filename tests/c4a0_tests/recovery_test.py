"""Crash recovery at real persistence boundaries, in isolated process groups."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from c4a0.training_v2 import RunManifest, load_champion_model
from c4a0_tests.training_v2_test import _tiny_config


RUNNER = r"""
import os, signal, sys
from pathlib import Path
import c4a0.training_v2 as training
from c4a0.config import TrainingV2Config
config = TrainingV2Config.model_validate_json(Path(sys.argv[1]).read_text())
point, marker = sys.argv[2], Path(sys.argv[3])
def crash(name):
    if point != name:
        return
    try:
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    os.kill(os.getpid(), signal.SIGKILL)
original_register = training.RunManifest.register_shard
def register(self, event):
    crash('shard_before')
    result = original_register(self, event)
    crash('shard_after')
    return result
training.RunManifest.register_shard = register
original_save = training._atomic_torch_save
def save(path, payload):
    if path.name == '.candidate.pt':
        crash('candidate_before')
    original_save(path, payload)
    if path.name == '.candidate.pt':
        crash('candidate_after')
training._atomic_torch_save = save
original_arena = training._run_arena
def arena(*args, **kwargs):
    crash('arena')
    return original_arena(*args, **kwargs)
training._run_arena = arena
original_decide = training.RunManifest.decide_attempt
def decide(self, attempt, result):
    crash('promotion_before')
    value = original_decide(self, attempt, result)
    crash('promotion_after')
    return value
training.RunManifest.decide_attempt = decide
if __name__ == '__main__':
    training.run_async_training(config, arena_decision_override=lambda _: 'accepted')
"""


@pytest.mark.parametrize(
    "point",
    [
        "shard_before",
        "shard_after",
        "candidate_before",
        "candidate_after",
        "arena",
        "promotion_before",
        "promotion_after",
    ],
)
def test_forced_termination_and_consistent_resume(tmp_path, point):
    config = _tiny_config(
        tmp_path / "run",
        max_candidate_attempts=1,
        validation_interval_steps=1,
        validation_batches=1,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json())
    runner = tmp_path / "runner.py"
    runner.write_text(RUNNER)
    marker = tmp_path / "crashed"
    with (tmp_path / "worker.log").open("w") as log:
        child = subprocess.Popen(
            [sys.executable, str(runner), str(config_path), point, str(marker)],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 35
            while (
                not marker.exists()
                and time.monotonic() < deadline
                and child.poll() is None
            ):
                time.sleep(0.05)
            assert marker.exists(), (tmp_path / "worker.log").read_text()
        finally:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)
    resumed = subprocess.run(
        [sys.executable, str(runner), str(config_path), "none", str(marker)],
        capture_output=True,
        text=True,
        timeout=40,
        start_new_session=True,
    )
    assert resumed.returncode == 0, resumed.stderr
    with RunManifest.open_existing(Path(config.base_dir)) as manifest:
        assert manifest.pending_attempt() is None
        assert (
            manifest.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        )
        assert Path(manifest.champion()["checkpoint_path"]).is_file()
        accepted = manifest.connection.execute(
            "SELECT count(*) FROM attempts WHERE status='accepted' AND attempt_n != 0"
        ).fetchone()[0]
        assert manifest.get_state("accepted_count") == accepted
        shards = manifest.active_shards()
        assert len({shard["first_game_id"] for shard in shards}) == len(shards)
        assert all(Path(shard["path"]).is_file() for shard in shards)
    load_champion_model(config.base_dir)


def test_solver_deadlines_reap_silent_or_closed_output_subprocess(
    tmp_path, monkeypatch
):
    from c4a0_tests.cpp_bridge_test import _small_games

    games = _small_games()
    book = tmp_path / "book"
    book.touch()
    monkeypatch.setenv("C4A0_SOLVER_TIMEOUT_SECONDS", ".15")
    for close_output in (False, True):
        solver = tmp_path / "solver"
        pid_file = tmp_path / "pid"
        solver.write_text(
            f'#!/bin/sh\necho $$ > "{pid_file}"\n'
            + ("exec >/dev/null 2>&1\n" if close_output else "")
            + "exec sleep 30\n"
        )
        solver.chmod(0o755)
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="deadline"):
            games.score_policies(
                str(solver), str(book), str(tmp_path / "cache.sqlite3")
            )
        assert time.monotonic() - started < 3
        pid = int(pid_file.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_training_child_exits_when_only_its_owner_is_killed(tmp_path):
    runner = tmp_path / "owner.py"
    marker = tmp_path / "child-pid"
    runner.write_text("""
import multiprocessing, os, sys, time
from pathlib import Path
from c4a0.training_v2 import _owned_training_child, training_run_owner

def wait(marker):
    Path(marker).write_text(str(os.getpid()))
    time.sleep(30)

if __name__ == '__main__':
    with training_run_owner(Path(sys.argv[2])):
        child = multiprocessing.get_context('spawn').Process(
            target=_owned_training_child, args=(os.getpid(), wait, (sys.argv[1],)))
        child.start()
        child.join()
""")
    owner = subprocess.Popen(
        [sys.executable, str(runner), str(marker), str(tmp_path / "run")]
    )
    pid = None
    try:
        deadline = time.monotonic() + 15
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists()
        pid = int(marker.read_text())
        owner.kill()
        owner.wait(timeout=3)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            stat = Path(f"/proc/{pid}/stat")
            if not stat.exists() or stat.read_text().rsplit(") ", 1)[1].startswith(
                "Z "
            ):
                break
            time.sleep(0.02)
        else:
            pytest.fail("Training child survived coordinator death")
        from c4a0.training_v2 import training_run_owner

        with training_run_owner(tmp_path / "run"):
            pass
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=3)
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
