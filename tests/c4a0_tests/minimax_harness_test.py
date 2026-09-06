from types import SimpleNamespace

import numpy as np
import pytest
import torch

import c4a0_cpp
from c4a0.minimax_harness import (
    MODEL_ID,
    MinimaxHarnessConfig,
    _MinimaxEvaluator,
    _move_bit,
    evaluate_minimax_ladder,
)
from c4a0.nn import ConnectFourNet, ModelConfig


def _model() -> ConnectFourNet:
    return ConnectFourNet(
        ModelConfig(
            n_residual_blocks=0,
            conv_filter_size=4,
            n_policy_layers=1,
            n_value_layers=1,
            lr_schedule={0: 0.001},
            l2_reg=0,
        )
    )


def test_minimax_detects_an_immediate_win():
    # Current player has three pieces on the bottom row and wins in column 3.
    positions = np.zeros((1, 2, 6, 7), dtype=np.float32)
    positions[0, 0, 0, :3] = 1
    evaluator = _MinimaxEvaluator(depth=1)
    policy, q_penalty, q_no_penalty = evaluator(positions)
    assert policy[0, 3] == 0
    assert q_penalty.tolist() == [1.0]
    assert q_no_penalty.tolist() == [1.0]


def test_depth_two_minimax_blocks_an_immediate_opponent_win():
    positions = np.zeros((1, 2, 6, 7), dtype=np.float32)
    positions[0, 1, 0, :3] = 1
    policy, _, _ = _MinimaxEvaluator(depth=2)(positions)
    assert np.flatnonzero(policy[0] == 0).tolist() == [3]


def test_move_bit_uses_only_the_requested_column():
    # A piece in the adjacent column must not affect the next height in column 3.
    occupied = (1 << 0) | (1 << 1) | (1 << 7)
    assert _move_bit(occupied, 3) == 1 << 3
    assert _move_bit(occupied, 0) == 1 << 14


def test_ladder_auto_promotes_after_more_than_ten_points(monkeypatch):
    requests_seen = []

    def fake_play(requests, _options, _evaluator):
        requests_seen.append(requests)
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    metadata=request.metadata,
                    player0_score=lambda score=(
                        1.0 if request.metadata.player0_id == MODEL_ID else 0.0
                    ): score,
                )
                for request in requests
            ]
        )

    monkeypatch.setattr(c4a0_cpp, "play_games_v2", fake_play)
    result = evaluate_minimax_ladder(
        _model(),
        torch.device("cpu"),
        MinimaxHarnessConfig(max_depth=1, pair_batch_size=6, mcts_iterations=1),
    )

    random = result.levels[0]
    assert random.opponent == "random"
    assert random.games == 12
    assert random.points == 12
    assert random.promoted and random.auto_promoted
    assert all(
        sum(request.metadata.player0_id == MODEL_ID for request in requests) == len(requests) // 2
        for requests in requests_seen
    )


def test_ladder_stops_at_the_first_opponent_not_beaten(monkeypatch):
    def fake_play(requests, _options, _evaluator):
        return SimpleNamespace(
            results=[
                SimpleNamespace(metadata=request.metadata, player0_score=lambda: 0.0)
                for request in requests
            ]
        )

    monkeypatch.setattr(c4a0_cpp, "play_games_v2", fake_play)
    result = evaluate_minimax_ladder(
        _model(), torch.device("cpu"), MinimaxHarnessConfig(max_depth=2, mcts_iterations=1)
    )
    assert result.stopped_at == "random"
    assert result.levels[0].games == 20
    assert not result.levels[0].promoted


def test_forty_game_gate_requires_strictly_more_than_twenty_points(monkeypatch):
    def fake_play(requests, _options, _evaluator):
        return SimpleNamespace(
            results=[
                SimpleNamespace(metadata=request.metadata, player0_score=lambda: 0.5)
                for request in requests
            ]
        )

    monkeypatch.setattr(c4a0_cpp, "play_games_v2", fake_play)
    result = evaluate_minimax_ladder(
        _model(),
        torch.device("cpu"),
        MinimaxHarnessConfig(games_per_level=40, mcts_iterations=1),
    )
    assert result.promotion_points_threshold == 20
    assert result.levels[0].points == 20
    assert not result.levels[0].promoted


@pytest.mark.parametrize("games", [0, 3])
def test_ladder_requires_an_even_positive_game_count(games):
    with pytest.raises(ValueError, match="positive even"):
        MinimaxHarnessConfig(games_per_level=games)
