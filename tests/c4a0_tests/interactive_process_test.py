import os
from pathlib import Path
import threading
import time

import numpy as np

from c4a0.gui.inference import ProcessEvaluator


class BlockingPlayer:
    def __init__(self, marker):
        self.marker = marker

    def forward_numpy(self, positions):
        Path(self.marker).write_text(str(os.getpid()))
        time.sleep(30)


def blocking_players(marker):
    return {0: BlockingPlayer(marker)}


def test_blocked_interactive_inference_closes_and_reaps_child(tmp_path):
    marker = tmp_path / "entered"
    evaluator = ProcessEvaluator(blocking_players, (str(marker),))
    errors = []

    def evaluate():
        try:
            evaluator.forward_numpy(0, np.zeros((1, 2, 6, 7), dtype=np.float32))
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=evaluate)
    thread.start()
    try:
        deadline = time.monotonic() + 15
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists()
        pid = int(marker.read_text())
        started = time.monotonic()
        evaluator.close()
        thread.join(timeout=3)
        assert not thread.is_alive() and errors
        assert time.monotonic() - started < 3
        assert not Path(f"/proc/{pid}").exists()
    finally:
        evaluator.close()
        thread.join(timeout=3)


def test_isolated_model_matches_saved_inference_settings(tmp_path):
    from c4a0.gui.inference import build_players
    from c4a0.training_v2 import RunManifest, _atomic_torch_save, _checkpoint_payload
    from c4a0.config import TrainingV2Config
    from c4a0_tests.release_correctness_test import _model

    model = _model().eval()
    model.mcts_value_scale = 0
    path = tmp_path / "attempts/000000/model.pt"
    with RunManifest.create(
        tmp_path, TrainingV2Config(base_dir=str(tmp_path))
    ) as manifest:
        _atomic_torch_save(path, _checkpoint_payload(model))
        manifest.add_root(path)
    positions = np.zeros((2, 2, 6, 7), dtype=np.float32)
    expected = model.forward_numpy(positions)
    evaluator = ProcessEvaluator(
        build_players, ("Human", "Latest Model", str(tmp_path), "cpu")
    )
    try:
        actual = evaluator.forward_numpy(1, positions)
        for first, second in zip(expected, actual):
            np.testing.assert_allclose(first, second, atol=1e-6)
    finally:
        evaluator.close()
