import pytest
from pydantic import ValidationError

from c4a0.config import (
    NNSweepConfig,
    SolverScoreConfig,
    TournamentConfig,
    TrainingConfig,
)


def test_training_presets_and_learning_rate_validation():
    cpu = TrainingConfig.preset("cpu")
    balanced = TrainingConfig.preset("balanced")
    gpu = TrainingConfig.preset("gpu")

    assert cpu.device == "cpu"
    assert cpu.n_self_play_games < balanced.n_self_play_games
    assert balanced.n_self_play_games < gpu.n_self_play_games
    assert gpu.lr_schedule == [0, 2e-3, 10, 8e-4]

    with pytest.raises(ValidationError, match="generation/rate pairs"):
        TrainingConfig(lr_schedule=[0, 0.001, 2])
    with pytest.raises(ValidationError, match="start at generation 0"):
        TrainingConfig(lr_schedule=[2, 0.001])
    with pytest.raises(ValidationError, match="provided together"):
        TrainingConfig(solver_path="solver")


def test_tournament_and_sweep_ranges_are_validated():
    with pytest.raises(ValidationError, match="must be even"):
        TournamentConfig(games_per_match=3)
    with pytest.raises(ValidationError, match="at least two"):
        TournamentConfig(players=["latest"])
    with pytest.raises(ValidationError, match="minimums cannot exceed"):
        NNSweepConfig(filter_size_min=128, filter_size_max=16)
    with pytest.raises(ValidationError, match="cannot contain duplicates"):
        SolverScoreConfig(solver_path="solver", book_path="book", generations=[2, 2])
