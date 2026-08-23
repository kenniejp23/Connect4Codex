#!/usr/bin/env python

from enum import Enum
from pathlib import Path
import sys

from typing import List, Optional
import warnings

from loguru import logger
import optuna
import torch
import typer

# Ensure that the parent directory of this file exists on Python path
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from c4a0.nn import ModelConfig  # noqa: E402
from c4a0.config import (  # noqa: E402
    MCTSSweepConfig,
    NNSweepConfig,
    SolverScoreConfig,
    TrainingConfig,
)
from c4a0.sweep import perform_hparam_sweep_config  # noqa: E402
from c4a0.tournament import ModelID, RandomPlayer, UniformPlayer  # noqa: E402
from c4a0.training import (  # noqa: E402
    SolverConfig,
    TrainingGen,
    parse_lr_schedule,
    training_loop,
)
from c4a0.utils import get_torch_device  # noqa: E402

import c4a0_cpp  # noqa: E402

app = typer.Typer()


class GameMode(str, Enum):
    human_ai = "human-ai"
    human_human = "human-human"
    ai_ai = "ai-ai"


class HumanSide(str, Enum):
    red = "red"
    blue = "blue"


class PlayModel(str, Enum):
    best = "best"
    random = "random"
    uniform = "uniform"


@app.command()
def gui():
    """Launch the native desktop application."""
    from c4a0.gui import run_gui

    raise typer.Exit(run_gui())


@app.command()
def train(
    base_dir: str = "training",
    device: str = str(get_torch_device()),
    # These parameters were chosen based on the results of the nn_sweep and mcts_sweep
    n_self_play_games: int = 1700,
    n_mcts_iterations: int = 1400,
    c_exploration: float = 6.6,
    c_ply_penalty: float = 0.01,
    self_play_batch_size: int = 2000,
    training_batch_size: int = 2000,
    n_residual_blocks: int = 1,
    conv_filter_size: int = 32,
    n_policy_layers: int = 4,
    n_value_layers: int = 2,
    lr_schedule: List[float] = [0, 2e-3, 10, 8e-4],
    l2_reg: float = 4e-4,
    max_gens: Optional[int] = None,
    max_epochs: int = 100,
    early_stopping_patience: int = 10,
    solver_path: Optional[str] = None,
    book_path: Optional[str] = None,
    solutions_path: str = "./solutions.db",
):
    """Trains a model via self-play."""

    config = TrainingConfig(
        base_dir=base_dir,
        device=device,
        n_self_play_games=n_self_play_games,
        n_mcts_iterations=n_mcts_iterations,
        c_exploration=c_exploration,
        c_ply_penalty=c_ply_penalty,
        self_play_batch_size=self_play_batch_size,
        training_batch_size=training_batch_size,
        n_residual_blocks=n_residual_blocks,
        conv_filter_size=conv_filter_size,
        n_policy_layers=n_policy_layers,
        n_value_layers=n_value_layers,
        lr_schedule=lr_schedule,
        l2_reg=l2_reg,
        max_gens=max_gens,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        solver_path=solver_path,
        book_path=book_path,
        solutions_path=solutions_path,
    )

    model_config = ModelConfig(
        n_residual_blocks=config.n_residual_blocks,
        conv_filter_size=config.conv_filter_size,
        n_policy_layers=config.n_policy_layers,
        n_value_layers=config.n_value_layers,
        lr_schedule=parse_lr_schedule(config.lr_schedule),
        l2_reg=config.l2_reg,
    )

    if config.solver_path and config.book_path:
        logger.info("Using solver")
        solver_config = SolverConfig(
            solver_path=config.solver_path,
            book_path=config.book_path,
            solutions_path=config.solutions_path,
        )
    else:
        logger.info("Solver not provided, skipping solutions")
        solver_config = None

    training_loop(
        base_dir=config.base_dir,
        device=torch.device(config.device),
        n_self_play_games=config.n_self_play_games,
        n_mcts_iterations=config.n_mcts_iterations,
        c_exploration=config.c_exploration,
        c_ply_penalty=config.c_ply_penalty,
        self_play_batch_size=config.self_play_batch_size,
        training_batch_size=config.training_batch_size,
        model_config=model_config,
        max_gens=config.max_gens,
        solver_config=solver_config,
        max_epochs=config.max_epochs,
        early_stopping_patience=config.early_stopping_patience,
    )


@app.command()
def play(
    base_dir: str = "training",
    max_mcts_iters: int = 1400,
    c_exploration: float = 6.6,
    c_ply_penalty: float = 0.01,
    model: PlayModel = PlayModel.best,
    mode: GameMode = GameMode.human_ai,
    human_side: HumanSide = HumanSide.red,
):
    """Play against the AI, another human, or watch an AI-vs-AI game."""
    gen = TrainingGen.load_latest(base_dir)
    if model is PlayModel.best:
        nn = gen.get_model(base_dir)
        nn.eval()
    elif model is PlayModel.random:
        nn = RandomPlayer(ModelID(0))
    elif model is PlayModel.uniform:
        nn = UniformPlayer(ModelID(0))
    else:
        raise ValueError(f"unrecognized model: {model}")

    if mode is GameMode.human_ai:
        auto_red = human_side is HumanSide.blue
        auto_blue = human_side is HumanSide.red
    elif mode is GameMode.human_human:
        auto_red = False
        auto_blue = False
    elif mode is GameMode.ai_ai:
        auto_red = True
        auto_blue = True
    else:
        raise ValueError(f"unrecognized game mode: {mode}")

    c4a0_cpp.run_tui(  # type: ignore
        lambda model_id, x: nn.forward_numpy(x),
        max_mcts_iters,
        c_exploration,
        c_ply_penalty,
        auto_red,
        auto_blue,
    )


@app.command()
def nn_sweep(
    base_dir: str = "training",
    study_name: str = "sweep_hparam",
    n_gens: int = 5,
    n_trials: int = 100,
    max_epochs: int = 30,
    residual_blocks_min: int = 0,
    residual_blocks_max: int = 1,
    filter_size_min: int = 16,
    filter_size_max: int = 64,
    policy_layers_min: int = 1,
    policy_layers_max: int = 4,
    value_layers_min: int = 1,
    value_layers_max: int = 2,
    learning_rate_min: float = 1e-4,
    learning_rate_max: float = 1e-2,
    l2_reg_min: float = 1e-5,
    l2_reg_max: float = 1e-3,
    batch_sizes: List[int] = [256, 512, 1024],
):
    """
    Performs a hyperparameter sweep to determine best nn model params based on existing training
    data.
    """
    config = NNSweepConfig(
        base_dir=base_dir,
        study_name=study_name,
        n_gens=n_gens,
        n_trials=n_trials,
        max_epochs=max_epochs,
        residual_blocks_min=residual_blocks_min,
        residual_blocks_max=residual_blocks_max,
        filter_size_min=filter_size_min,
        filter_size_max=filter_size_max,
        policy_layers_min=policy_layers_min,
        policy_layers_max=policy_layers_max,
        value_layers_min=value_layers_min,
        value_layers_max=value_layers_max,
        learning_rate_min=learning_rate_min,
        learning_rate_max=learning_rate_max,
        l2_reg_min=l2_reg_min,
        l2_reg_max=l2_reg_max,
        batch_sizes=batch_sizes,
    )
    perform_hparam_sweep_config(config)


@app.command()
def mcts_sweep(
    device: str = str(get_torch_device()),
    c_ply_penalty: float = 0.01,
    self_play_batch_size: int = 2000,
    training_batch_size: int = 2000,
    # These NN parameters were chosen based on the results of the nn_sweep
    n_residual_blocks: int = 1,
    conv_filter_size: int = 32,
    n_policy_layers: int = 4,
    n_value_layers: int = 2,
    lr_schedule: List[float] = [0, 2e-3],
    l2_reg: float = 4e-4,
    # End NN parameters
    base_training_dir: str = "training-sweeps",
    optuna_db_path: str = "optuna.db",
    n_trials: int = 100,
    max_gens_per_trial: int = 10,
    solver_path: str = "solver/c4solver",
    book_path: str = "solver/7x6.book",
    solutions_path: str = "./solutions.db",
    self_play_games_min: int = 1000,
    self_play_games_max: int = 5000,
    mcts_iterations_min: int = 100,
    mcts_iterations_max: int = 1500,
    exploration_min: float = 0.5,
    exploration_max: float = 12.0,
):
    """
    Performs sweep of MCTS hyperparameters (e.g. n_self_play_games, n_mcts_iterations,
    c_exploration) to determine optimal values by performing `n_trials` independent training
    runs, each with `max_gens_per_trial` generations, seeking to maximize the solver score.
    """
    config = MCTSSweepConfig(
        device=device,
        c_ply_penalty=c_ply_penalty,
        self_play_batch_size=self_play_batch_size,
        training_batch_size=training_batch_size,
        n_residual_blocks=n_residual_blocks,
        conv_filter_size=conv_filter_size,
        n_policy_layers=n_policy_layers,
        n_value_layers=n_value_layers,
        lr_schedule=lr_schedule,
        l2_reg=l2_reg,
        base_training_dir=base_training_dir,
        optuna_db_path=optuna_db_path,
        n_trials=n_trials,
        max_gens_per_trial=max_gens_per_trial,
        solver_path=solver_path,
        book_path=book_path,
        solutions_path=solutions_path,
        self_play_games_min=self_play_games_min,
        self_play_games_max=self_play_games_max,
        mcts_iterations_min=mcts_iterations_min,
        mcts_iterations_max=mcts_iterations_max,
        exploration_min=exploration_min,
        exploration_max=exploration_max,
    )
    base_path = Path(config.base_training_dir)
    base_path.mkdir(exist_ok=True)

    model_config = ModelConfig(
        n_residual_blocks=config.n_residual_blocks,
        conv_filter_size=config.conv_filter_size,
        n_policy_layers=config.n_policy_layers,
        n_value_layers=config.n_value_layers,
        lr_schedule=parse_lr_schedule(config.lr_schedule),
        l2_reg=config.l2_reg,
    )

    def objective(trial: optuna.Trial):
        trial_path = base_path / f"trial_{trial.number}"
        trial_path.mkdir(exist_ok=False)
        gen = training_loop(
            base_dir=str(trial_path),
            device=torch.device(config.device),
            n_self_play_games=trial.suggest_int(
                "n_self_play_games",
                config.self_play_games_min,
                config.self_play_games_max,
            ),
            n_mcts_iterations=trial.suggest_int(
                "n_mcts_iterations",
                config.mcts_iterations_min,
                config.mcts_iterations_max,
            ),
            c_exploration=trial.suggest_float(
                "c_exploration", config.exploration_min, config.exploration_max
            ),
            c_ply_penalty=config.c_ply_penalty,
            self_play_batch_size=config.self_play_batch_size,
            training_batch_size=config.training_batch_size,
            model_config=model_config,
            max_gens=config.max_gens_per_trial,
            solver_config=SolverConfig(
                solver_path=config.solver_path,
                book_path=config.book_path,
                solutions_path=config.solutions_path,
            ),
        )
        logger.info(
            "Trial {} completed. Solver score: {}", trial.number, gen.solver_score
        )
        score = gen.solver_score
        assert score is not None
        return score

    storage_name = f"sqlite:///{config.optuna_db_path}"
    study = optuna.create_study(
        study_name="mcts_sweep",
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=config.n_trials)


@app.command()
def score(
    solver_path: str,
    book_path: str,
    base_dir: str = "training",
    solutions_path: str = "./solutions.db",
    generations: Optional[List[int]] = None,
    rescore: bool = False,
):
    """Scores the training generations using the given solver."""
    config = SolverScoreConfig(
        solver_path=solver_path,
        book_path=book_path,
        base_dir=base_dir,
        solutions_path=solutions_path,
        generations=generations,
        rescore=rescore,
    )
    gens = TrainingGen.load_all(config.base_dir)
    for gen in gens:
        if config.generations is not None and gen.gen_n not in config.generations:
            continue
        logger.info("Getting games for: {}", gen.gen_n)
        games = gen.get_games(config.base_dir)  # type: ignore
        if not games:
            continue
        if gen.solver_score is not None and not config.rescore:
            logger.info(f"Gen already has score: {gen.solver_score}")
            continue
        score = games.score_policies(  # type: ignore
            config.solver_path, config.book_path, config.solutions_path
        )
        gen.solver_score = score
        gen.save_metadata(config.base_dir)
        logger.info("Gen {} has score: {}", gen.gen_n, score)


if __name__ == "__main__":
    # Disable unnecessary pytorch warnings
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    app()
