import shutil
import subprocess
import sys
from typing import Any, cast

import pytest
import torch

from c4a0.config import TrainingV2Config
from c4a0.training_v2 import (
    RunManifest,
    training_run_owner,
    _atomic_torch_save,
    _checkpoint_payload,
    load_champion_model,
)
from c4a0_tests.release_correctness_test import _model


def test_single_coordinator_readers_and_owner_death(tmp_path):
    run = tmp_path / "run"
    with training_run_owner(run):
        with RunManifest.create(run, TrainingV2Config(base_dir=str(run))) as manifest:
            with RunManifest.open_existing(run) as reader:
                assert reader.get_state("format") == manifest.get_state("format")
                with pytest.raises(Exception, match="readonly"):
                    reader.set_state("test", 1)
        with pytest.raises(RuntimeError, match="Another trainer"):
            with training_run_owner(run):
                pass
    script = 'from pathlib import Path\nimport time\nfrom c4a0.training_v2 import training_run_owner\nwith training_run_owner(Path(__import__("sys").argv[1])):\n print("owned", flush=True)\n time.sleep(30)\n'
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(run)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "owned"
        with pytest.raises(RuntimeError, match="Another trainer"):
            with training_run_owner(run):
                pass
    finally:
        child.kill()
        child.wait(timeout=5)
        if child.stdout:
            child.stdout.close()
    with training_run_owner(run):
        pass


def test_run_move_migrates_old_absolute_artifacts_and_closes_readers(tmp_path):
    original = tmp_path / "original"
    config = TrainingV2Config(base_dir=str(original))
    path = original / "attempts/000000/model.pt"
    with RunManifest.create(original, config) as manifest:
        model = _model()
        model.mcts_value_scale = 0
        _atomic_torch_save(path, _checkpoint_payload(model))
        manifest.add_root(path)
        manifest.connection.execute(
            "UPDATE attempts SET checkpoint_path=?", (str(path),)
        )
        manifest.connection.commit()
    moved = tmp_path / "moved"
    shutil.move(original, moved)
    loaded = load_champion_model(str(moved))
    for expected, actual in zip(model.parameters(), loaded.parameters()):
        torch.testing.assert_close(expected, actual)
    with RunManifest.create(
        moved, config.model_copy(update={"base_dir": str(moved)})
    ) as manifest:
        stored = manifest.connection.execute(
            "SELECT checkpoint_path FROM attempts"
        ).fetchone()[0]
        assert stored == "attempts/000000/model.pt"
    with pytest.raises(Exception, match="closed"):
        manifest.get_state("config")


def test_failed_start_and_missing_terminal_event_finalize_queue(tmp_path, monkeypatch):
    from c4a0_tests.gui_test import _application, _wait_until
    from c4a0.gui.jobs import JobManager, QueuedJob

    application = _application(tmp_path)
    monkeypatch.setattr("c4a0.gui.jobs.sys.executable", str(tmp_path / "absent"))
    manager = cast(Any, JobManager())
    manager._queue.extend(
        [QueuedJob("validation", "first", {}), QueuedJob("validation", "second", {})]
    )
    manager._start_next()
    _wait_until(application, lambda: len(manager.recentJobs) == 2)
    assert not manager.active
    assert all(job["status"] == "Failed" for job in manager.recentJobs)
    executable = tmp_path / "silent-worker"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    monkeypatch.setattr("c4a0.gui.jobs.sys.executable", str(executable))
    manager._queue.append(QueuedJob("validation", "silent", {}))
    manager._start_next()
    _wait_until(application, lambda: not manager.active)
    assert manager.recentJobs[0]["status"] == "Failed"
    assert "before reporting" in manager.result


def test_cancel_timer_cannot_kill_next_job_and_shutdown_does_not_start_queue(
    tmp_path, monkeypatch
):
    from c4a0_tests.gui_test import _application, _wait_until
    from c4a0.gui.jobs import JobManager, QueuedJob

    application = _application(tmp_path)
    executable = tmp_path / "worker"
    executable.write_text("#!/bin/sh\nexec sleep 30\n")
    executable.chmod(0o755)
    monkeypatch.setattr("c4a0.gui.jobs.sys.executable", str(executable))
    manager = cast(Any, JobManager())
    manager._queue.extend(
        [QueuedJob("validation", "first", {}), QueuedJob("validation", "second", {})]
    )
    manager._start_next()
    _wait_until(
        application,
        lambda: (
            manager.phase != "Idle"
            and manager._process is not None
            and manager._process.processId() > 0
        ),
    )
    old = manager._process
    assert old is not None
    manager.cancel()
    _wait_until(
        application,
        lambda: (
            manager.currentTitle == "second"
            and manager._process is not None
            and manager._process.processId() > 0
        ),
    )
    manager._kill_if_running(old)
    assert manager.active and manager.currentTitle == "second"
    manager._queue.append(QueuedJob("validation", "never", {}))
    manager.shutdown()
    _wait_until(application, lambda: not manager.active)
    assert manager.queueDepth == 0
    assert all(job["title"] != "never" for job in manager.recentJobs)


def test_neural_sweep_completes_real_optimizer_trial(tmp_path, monkeypatch):
    import optuna
    import pytorch_lightning as pl
    import c4a0_cpp
    from c4a0.config import NNSweepConfig
    from c4a0.sweep import objective
    from c4a0_tests.cpp_bridge_test import _uniform_eval

    real_trainer = pl.Trainer
    monkeypatch.setattr(
        pl,
        "Trainer",
        lambda **kwargs: real_trainer(
            **{
                **kwargs,
                "accelerator": "cpu",
                "devices": 1,
                "logger": False,
                "enable_checkpointing": False,
            }
        ),
    )
    games = c4a0_cpp.play_games(
        [c4a0_cpp.GameMetadata(1, 0, 0)], 4, 1, 1.0, 0.01, _uniform_eval
    )
    config = NNSweepConfig(base_dir=str(tmp_path), max_epochs=1)
    trial = optuna.trial.FixedTrial(
        {
            "n_residual_blocks": config.residual_blocks_min,
            "conv_filter_size": config.filter_size_min,
            "n_policy_layers": config.policy_layers_min,
            "n_value_layers": config.value_layers_min,
            "learning_rate": config.learning_rate_min,
            "l2_reg": config.l2_reg_min,
            "batch_size": config.batch_sizes[0],
        }
    )
    value = objective(cast(Any, trial), list(games.results[0].samples), config)
    assert value >= 0 and __import__("math").isfinite(value)


def test_export_is_compact_and_safe_to_load(tmp_path):
    from c4a0.training_v2 import export_inference_model, load_model_checkpoint

    run = tmp_path / "run"
    with RunManifest.create(run, TrainingV2Config(base_dir=str(run))) as manifest:
        model = _model()
        model.mcts_value_scale = 0
        path = run / "attempts/000000/model.pt"
        _atomic_torch_save(path, _checkpoint_payload(model))
        manifest.add_root(path)
    exported = export_inference_model(str(run), tmp_path / "export.pt")
    payload = torch.load(exported, weights_only=True)
    assert "optimizer_state" not in payload and "numpy_rng_state" not in payload
    restored = load_model_checkpoint(exported)
    assert restored.mcts_value_scale == 0
    with pytest.raises(FileExistsError):
        export_inference_model(str(run), exported)


def test_malformed_worker_event_stays_failed_and_logs_remain_bounded(
    tmp_path, monkeypatch
):
    from c4a0_tests.gui_test import _application, _wait_until
    from c4a0.gui.jobs import JobManager, QueuedJob

    application = _application(tmp_path)
    executable = tmp_path / "malformed-worker"
    executable.write_text(
        '#!/bin/sh\nprintf \'{"version":1,"type":"progress","fraction":"invalid"}\\n\'\nexec sleep 30\n'
    )
    executable.chmod(0o755)
    monkeypatch.setattr("c4a0.gui.jobs.sys.executable", str(executable))
    manager = cast(Any, JobManager())
    manager._queue.append(QueuedJob("validation", "malformed", {}))
    manager._start_next()
    _wait_until(application, lambda: not manager.active)
    assert manager.recentJobs[0]["status"] == "Failed"
    assert "Invalid worker event" in manager.result
    for index in range(1200):
        manager._append_log(str(index))
    assert manager.logModel.rowCount() == 1000
    assert manager.logModel.data(manager.logModel.index(0, 0)) == "200"
    manager.clearLogs()
    assert manager.logModel.rowCount() == 0


def test_cancellation_kills_worker_descendants(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    from c4a0_tests.gui_test import _application, _wait_until
    from c4a0.gui.jobs import JobManager, QueuedJob

    application = _application(tmp_path)
    executable = tmp_path / "tree-worker"
    pid_file = tmp_path / "descendant"
    executable.write_text(
        f"#!{sys.executable}\nimport os,subprocess,json,time\nos.setsid()\n"
        f'child=subprocess.Popen(["sleep","30"])\nopen({str(pid_file)!r},"w").write(str(child.pid))\n'
        'print(json.dumps({"version":1,"type":"started","process_group":os.getpid()}),flush=True)\ntime.sleep(30)\n'
    )
    executable.chmod(0o755)
    monkeypatch.setattr("c4a0.gui.jobs.sys.executable", str(executable))
    manager = cast(Any, JobManager())
    manager._queue.append(QueuedJob("validation", "tree", {}))
    manager._start_next()
    _wait_until(application, lambda: manager._worker_group is not None)
    pid = int(pid_file.read_text())
    try:
        manager.cancel()
        _wait_until(application, lambda: not manager.active)

        def child_stopped():
            stat = Path(f"/proc/{pid}/stat")
            return not stat.exists() or stat.read_text().split(") ")[1].startswith("Z ")

        _wait_until(application, child_stopped)
    finally:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
        manager.shutdown()


def test_artifact_paths_reject_parent_traversal(tmp_path):
    with RunManifest.create(
        tmp_path, TrainingV2Config(base_dir=str(tmp_path))
    ) as manifest:
        with pytest.raises(ValueError, match="parent traversal"):
            manifest.resolve_artifact("attempts/../../outside.pt")
