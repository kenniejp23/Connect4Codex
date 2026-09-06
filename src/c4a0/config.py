"""Shared, validated configuration models for the CLI and desktop application."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from c4a0.utils import get_torch_device


class StrictConfig(BaseModel):
    model_config = {"extra": "forbid", "validate_assignment": True}


class TrainingConfig(StrictConfig):
    base_dir: str = "training"
    device: str = Field(default_factory=lambda: str(get_torch_device()))
    n_self_play_games: int = Field(default=1700, ge=1)
    n_mcts_iterations: int = Field(default=1400, ge=1)
    c_exploration: float = Field(default=6.6, ge=0)
    c_ply_penalty: float = Field(default=0.01, ge=0)
    self_play_batch_size: int = Field(default=2000, ge=1)
    training_batch_size: int = Field(default=2000, ge=1)
    n_residual_blocks: int = Field(default=1, ge=0)
    conv_filter_size: int = Field(default=32, ge=1)
    n_policy_layers: int = Field(default=4, ge=1)
    n_value_layers: int = Field(default=2, ge=1)
    lr_schedule: list[float] = Field(default_factory=lambda: [0, 2e-3, 10, 8e-4])
    l2_reg: float = Field(default=4e-4, ge=0)
    max_gens: int | None = Field(default=None, ge=1)
    max_epochs: int = Field(default=100, ge=1)
    early_stopping_patience: int = Field(default=10, ge=0)
    solver_path: str | None = None
    book_path: str | None = None
    solutions_path: str = "./solutions.db"

    @model_validator(mode="after")
    def validate_training(self) -> "TrainingConfig":
        if len(self.lr_schedule) == 0 or len(self.lr_schedule) % 2 != 0:
            raise ValueError(
                "learning-rate schedule must contain generation/rate pairs"
            )
        generations = self.lr_schedule[::2]
        rates = self.lr_schedule[1::2]
        if any(value < 0 or not float(value).is_integer() for value in generations):
            raise ValueError("learning-rate generations must be non-negative integers")
        if generations != sorted(generations) or len(set(generations)) != len(
            generations
        ):
            raise ValueError("learning-rate generations must be unique and ascending")
        if generations[0] != 0:
            raise ValueError("learning-rate schedule must start at generation 0")
        if any(rate <= 0 for rate in rates):
            raise ValueError("learning rates must be positive")
        if bool(self.solver_path) != bool(self.book_path):
            raise ValueError(
                "solver executable and opening book must be provided together"
            )
        return self

    @classmethod
    def preset(cls, name: Literal["cpu", "balanced", "gpu"]) -> "TrainingConfig":
        if name == "cpu":
            return cls(
                device="cpu",
                n_self_play_games=100,
                n_mcts_iterations=100,
                self_play_batch_size=64,
                training_batch_size=128,
                conv_filter_size=16,
                n_policy_layers=2,
                n_value_layers=1,
                max_gens=3,
            )
        if name == "gpu":
            return cls()
        return cls(
            n_self_play_games=500,
            n_mcts_iterations=600,
            self_play_batch_size=512,
            training_batch_size=512,
            max_gens=5,
        )


class TrainingV2Config(StrictConfig):
    """Validated configuration for asynchronous neural-only training."""

    base_dir: str = "training-v2"
    device: str = Field(default_factory=lambda: str(get_torch_device()))
    run_seed: int = Field(default=1337, ge=0)
    n_mcts_iterations: int = Field(default=1400, ge=1)
    c_exploration: float = Field(default=6.6, ge=0)
    c_ply_penalty: float = Field(default=0.01, ge=0)
    mcts_value_scale: float = Field(default=0.0, ge=0, le=1)
    """Scale neural values used by MCTS; zero safely bootstraps from policy/terminals."""
    value_loss_weight: float = Field(default=0.0, ge=0, le=1)
    """Weight for neural value losses; zero avoids bootstrap interference."""
    self_play_shard_games: int = Field(default=256, ge=2)
    self_play_batch_games: int = Field(default=512, ge=2)
    replay_capacity_games: int = Field(default=20_000, ge=4)
    replay_warmup_games: int = Field(default=2_048, ge=2)
    replay_ratio: float = Field(default=4.0, gt=0)
    validation_fraction: float = Field(default=0.05, gt=0, lt=0.5)
    archive_depth: int = Field(default=8, ge=1)
    champion_self_play_weight: float = Field(default=0.65, ge=0, le=1)
    archive_weight: float = Field(default=0.30, ge=0, le=1)
    uniform_weight: float = Field(default=0.025, ge=0, le=1)
    random_weight: float = Field(default=0.025, ge=0, le=1)
    archive_recency: float = Field(default=0.7, gt=0, le=1)
    arena_min_games: int = Field(default=40, ge=2)
    arena_max_games: int = Field(default=800, ge=2, le=2_000)
    arena_p0: float = Field(default=0.50, gt=0, lt=1)
    arena_p1: float = Field(default=0.55, gt=0, lt=1)
    arena_alpha: float = Field(default=0.05, gt=0, lt=0.5)
    arena_beta: float = Field(default=0.05, gt=0, lt=0.5)
    arena_pair_batch_size: int = Field(default=100, ge=1, le=100)
    root_dirichlet_epsilon: float = Field(default=0.25, ge=0, le=1)
    root_dirichlet_alpha: float = Field(default=0.30, gt=0)
    temperature_cutoff_ply: int = Field(default=8, ge=0, le=42)
    early_temperature: float = Field(default=1.0, ge=0)
    late_temperature: float = Field(default=0.0, ge=0)
    training_batch_size: int = Field(default=512, ge=2)
    inference_batch_size: int = Field(default=128, ge=1)
    data_loader_workers: int = Field(default=2, ge=0)
    mcts_worker_threads: int = Field(default=0, ge=0)
    precision: Literal["auto", "16-mixed", "32-true"] = "auto"
    inference_amp_min_batch_size: int = Field(default=96, ge=1)
    n_residual_blocks: int = Field(default=1, ge=0)
    conv_filter_size: int = Field(default=32, ge=1)
    n_policy_layers: int = Field(default=4, ge=1)
    n_value_layers: int = Field(default=2, ge=1)
    lr_schedule: list[float] = Field(default_factory=lambda: [0, 2e-3, 10, 8e-4])
    l2_reg: float = Field(default=4e-4, ge=0)
    max_gens: int | None = Field(default=None, ge=1)
    max_candidate_attempts: int | None = Field(default=None, ge=1)
    validation_interval_steps: int = Field(default=100, ge=1)
    validation_batches: int = Field(default=8, ge=1)
    rejected_weight_retention: int = Field(default=3, ge=0)
    learner_incumbent_min_score: float = Field(default=0.5, ge=0, le=1)
    """Minimum arena score required to continue training a rejected learner."""
    debt_pause_shards: int = Field(default=2, ge=2)
    debt_resume_shards: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_v2(self) -> "TrainingV2Config":
        if self.self_play_batch_games < self.self_play_shard_games:
            raise ValueError("self-play batch cannot be smaller than its replay shard")
        if self.self_play_batch_games % self.self_play_shard_games:
            raise ValueError("self-play batch must contain whole replay shards")
        opponent_sum = (
            self.champion_self_play_weight
            + self.archive_weight
            + self.uniform_weight
            + self.random_weight
        )
        if abs(opponent_sum - 1.0) > 1e-9:
            raise ValueError("opponent weights must sum to 1")
        if self.replay_warmup_games >= self.replay_capacity_games:
            raise ValueError("replay warmup must be smaller than replay capacity")
        if self.arena_min_games % 2 or self.arena_max_games % 2:
            raise ValueError("arena game limits must be even")
        if self.arena_min_games > self.arena_max_games:
            raise ValueError("arena minimum games cannot exceed maximum games")
        if self.arena_p0 >= self.arena_p1:
            raise ValueError("arena p0 must be smaller than p1")
        if self.debt_resume_shards >= self.debt_pause_shards:
            raise ValueError("debt resume watermark must be below pause watermark")
        TrainingConfig(lr_schedule=self.lr_schedule)
        return self


class TournamentConfig(StrictConfig):
    players: list[str] = Field(default_factory=lambda: ["latest", "random"])
    base_dir: str = "training-v2"
    device: str = Field(default_factory=lambda: str(get_torch_device()))
    games_per_match: int = Field(default=2, ge=2)
    batch_size: int = Field(default=64, ge=1)
    mcts_iterations: int = Field(default=200, ge=1)
    exploration_constant: float = Field(default=6.6, ge=0)
    c_ply_penalty: float = Field(default=0.01, ge=0)

    @model_validator(mode="after")
    def validate_tournament(self) -> "TournamentConfig":
        if len(self.players) < 2:
            raise ValueError("select at least two tournament players")
        if self.games_per_match % 2:
            raise ValueError("games per match must be even")
        return self


class SolverScoreConfig(StrictConfig):
    solver_path: str
    book_path: str
    base_dir: str = "training"
    solutions_path: str = "./solutions.db"
    rescore: bool = False
    generations: list[int] | None = None

    @model_validator(mode="after")
    def validate_generations(self) -> "SolverScoreConfig":
        if self.generations is not None:
            if any(generation < 0 for generation in self.generations):
                raise ValueError("generation numbers must be non-negative")
            if len(set(self.generations)) != len(self.generations):
                raise ValueError("generation selection cannot contain duplicates")
        return self


class NNSweepConfig(StrictConfig):
    base_dir: str = "training"
    study_name: str = "sweep_hparam"
    n_gens: int = Field(default=5, ge=1)
    n_trials: int = Field(default=100, ge=1)
    max_epochs: int = Field(default=30, ge=1)
    residual_blocks_min: int = Field(default=0, ge=0)
    residual_blocks_max: int = Field(default=1, ge=0)
    filter_size_min: int = Field(default=16, ge=1)
    filter_size_max: int = Field(default=64, ge=1)
    policy_layers_min: int = Field(default=1, ge=1)
    policy_layers_max: int = Field(default=4, ge=1)
    value_layers_min: int = Field(default=1, ge=1)
    value_layers_max: int = Field(default=2, ge=1)
    learning_rate_min: float = Field(default=1e-4, gt=0)
    learning_rate_max: float = Field(default=1e-2, gt=0)
    l2_reg_min: float = Field(default=1e-5, ge=0)
    l2_reg_max: float = Field(default=1e-3, ge=0)
    batch_sizes: list[int] = Field(default_factory=lambda: [256, 512, 1024])

    @model_validator(mode="after")
    def validate_ranges(self) -> "NNSweepConfig":
        pairs = [
            (self.residual_blocks_min, self.residual_blocks_max),
            (self.filter_size_min, self.filter_size_max),
            (self.policy_layers_min, self.policy_layers_max),
            (self.value_layers_min, self.value_layers_max),
            (self.learning_rate_min, self.learning_rate_max),
            (self.l2_reg_min, self.l2_reg_max),
        ]
        if any(low > high for low, high in pairs):
            raise ValueError("sweep range minimums cannot exceed maximums")
        if not self.batch_sizes or any(size < 1 for size in self.batch_sizes):
            raise ValueError("provide at least one positive batch size")
        return self


class MCTSSweepConfig(StrictConfig):
    device: str = Field(default_factory=lambda: str(get_torch_device()))
    c_ply_penalty: float = Field(default=0.01, ge=0)
    self_play_batch_size: int = Field(default=2000, ge=1)
    training_batch_size: int = Field(default=2000, ge=1)
    n_residual_blocks: int = Field(default=1, ge=0)
    conv_filter_size: int = Field(default=32, ge=1)
    n_policy_layers: int = Field(default=4, ge=1)
    n_value_layers: int = Field(default=2, ge=1)
    lr_schedule: list[float] = Field(default_factory=lambda: [0, 2e-3])
    l2_reg: float = Field(default=4e-4, ge=0)
    base_training_dir: str = "training-sweeps"
    optuna_db_path: str = "optuna.db"
    n_trials: int = Field(default=100, ge=1)
    max_gens_per_trial: int = Field(default=10, ge=1)
    solver_path: str = "solver/c4solver"
    book_path: str = "solver/7x6.book"
    solutions_path: str = "./solutions.db"
    self_play_games_min: int = Field(default=1000, ge=1)
    self_play_games_max: int = Field(default=5000, ge=1)
    mcts_iterations_min: int = Field(default=100, ge=1)
    mcts_iterations_max: int = Field(default=1500, ge=1)
    exploration_min: float = Field(default=0.5, ge=0)
    exploration_max: float = Field(default=12.0, ge=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> "MCTSSweepConfig":
        if (
            self.self_play_games_min > self.self_play_games_max
            or self.mcts_iterations_min > self.mcts_iterations_max
            or self.exploration_min > self.exploration_max
        ):
            raise ValueError("sweep range minimums cannot exceed maximums")
        TrainingConfig(
            lr_schedule=self.lr_schedule,
            solver_path=self.solver_path,
            book_path=self.book_path,
        )
        return self


class ValidationConfig(StrictConfig):
    profile: Literal["lint", "typecheck", "test:cpp", "test:python", "check", "ci"] = (
        "check"
    )
    project_dir: str = Field(default_factory=lambda: str(Path.cwd()))
