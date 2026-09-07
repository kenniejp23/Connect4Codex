"""Regression checks for release-review reproductions."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from c4a0.losses import policy_value_losses
from c4a0.minimax_harness import _has_four, _WINNING_MASKS
from c4a0.nn import ConnectFourNet, ModelConfig
from c4a0.training_v2 import (
    _checkpoint_payload,
    _atomic_torch_save,
    load_model_checkpoint,
    _ActorEvaluator,
)


def _model():
    return ConnectFourNet(
        ModelConfig(
            n_residual_blocks=0,
            conv_filter_size=2,
            n_policy_layers=1,
            n_value_layers=1,
            lr_schedule={0: 0.001},
            l2_reg=0,
        )
    )


def test_checkpoint_owns_parameters_buffers_and_optimizer(tmp_path):
    model = _model()
    optimizer = torch.optim.Adam(model.parameters())
    model(torch.randn(4, 2, 6, 7))[0].sum().backward()
    optimizer.step()
    payload = _checkpoint_payload(model, optimizer, training_steps=1)
    expected = {key: value.clone() for key, value in payload["state_dict"].items()}
    expected_optimizer = payload["optimizer_state"]["state"][0]["exp_avg"].clone()
    optimizer.zero_grad()
    model(torch.randn(4, 2, 6, 7))[0].sum().backward()
    optimizer.step()
    for key, value in payload["state_dict"].items():
        torch.testing.assert_close(value, expected[key])
    torch.testing.assert_close(
        payload["optimizer_state"]["state"][0]["exp_avg"], expected_optimizer
    )
    assert payload["training_steps"] == 1


@pytest.mark.parametrize("coefficient", [0.0, 0.1, 0.5, 1.0])
def test_value_coefficient_scales_loss_and_gradients(coefficient):
    prediction = torch.ones(32, requires_grad=True)
    policy = torch.full((32, 7), -np.log(7))
    target = torch.full((32, 7), 1 / 7)
    eligibility = torch.tensor([1.0] * 16 + [0.0] * 16)
    batch = (
        None,
        target,
        torch.zeros(32),
        torch.zeros(32),
        torch.ones(32),
        eligibility,
    )
    _, first, second = policy_value_losses(
        (policy, prediction, prediction), batch, coefficient
    )
    loss = first + second
    assert loss.item() == pytest.approx(2 * coefficient)
    loss.backward()
    assert prediction.grad is not None
    torch.testing.assert_close(prediction.grad[:16], torch.full((16,), coefficient / 4))
    assert prediction.grad[16:].count_nonzero() == 0


def test_minimax_edge_masks_and_native_differential():
    import c4a0_cpp

    assert len(_WINNING_MASKS) == 69
    assert not _has_four(sum(1 << bit for bit in [5, 6, 7, 8]))
    rng = np.random.default_rng(83)

    def evaluate(_, positions):
        return (
            np.zeros((len(positions), 7), np.float32),
            np.zeros(len(positions), np.float32),
            np.zeros(len(positions), np.float32),
        )

    game = c4a0_cpp.InteractivePlay(evaluate, 1, 1.0, 0.01, 0, 0)
    try:
        for _ in range(100):
            game.reset()
            heights = [0] * 7
            bits = [0, 0]
            for ply in range(42):
                column = int(rng.choice([c for c in range(7) if heights[c] < 6]))
                assert game.make_move(column)
                bits[ply % 2] |= 1 << (heights[column] * 7 + column)
                heights[column] += 1
                terminal = game.snapshot().terminal_state
                assert _has_four(bits[0]) == (terminal == "red_win")
                assert _has_four(bits[1]) == (terminal == "gold_win")
                if terminal != "ongoing":
                    break
    finally:
        game.close()


def test_saved_model_uses_arena_inference_settings(tmp_path):
    model = _model().eval()
    model.mcts_value_scale = 0.0
    with torch.no_grad():
        layer = model.fc_value[0]
        assert isinstance(layer, torch.nn.Linear)
        layer.weight.zero_()
        assert layer.bias is not None
        layer.bias.fill_(1)
    path = tmp_path / "model.pt"
    _atomic_torch_save(path, _checkpoint_payload(model))
    saved = load_model_checkpoint(path, trusted=True)
    actor = _ActorEvaluator(torch.device("cpu"), 1, value_scale=0.0)
    actor.configure([{"attempt_n": 0, "checkpoint_path": str(path)}])
    positions = np.zeros((2, 2, 6, 7), np.float32)

    def forbidden(input):
        raise AssertionError("disabled value head was evaluated")

    saved.fc_value.forward = forbidden
    for ordinary, arena in zip(saved.forward_numpy(positions), actor(0, positions)):
        np.testing.assert_array_equal(ordinary, arena)


def test_interactive_failure_retry_and_slow_evaluation_in_subprocess():
    script = """
import threading, time
import numpy as np
import c4a0_cpp
entered = threading.Event()
release = threading.Event()
fail = True
def evaluate(_, positions):
    entered.set()
    release.wait(3)
    if fail:
        raise ValueError("repeatable evaluator failure")
    return np.zeros((len(positions),7), np.float32), np.zeros(len(positions),np.float32), np.zeros(len(positions),np.float32)
game = c4a0_cpp.InteractivePlay(evaluate, 1, 1., .01, 0, 0)
assert entered.wait(3)
start = time.monotonic()
game.snapshot()
game.reset()
assert time.monotonic() - start < .2
release.set()
def wait_error():
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            game.snapshot()
        except RuntimeError as error:
            assert 'repeatable evaluator failure' in str(error)
            return
        time.sleep(.005)
    raise AssertionError('no evaluator error')
wait_error()
wait_error()
game.retry_evaluation()
wait_error()
fail = False
game.retry_evaluation()
assert game.make_move(3)
assert game.snapshot().move_history == [3]
game.reset()
assert game.snapshot().move_history == []
game.close()
game.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
