#!/usr/bin/env python

from enum import Enum
import json
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
    TrainingV2Config,
)
from c4a0.sweep import perform_hparam_sweep_config  # noqa: E402
from c4a0.tournament import ModelID, RandomPlayer, UniformPlayer  # noqa: E402
from c4a0.training import (  # noqa: E402
    SolverConfig,
    TrainingGen,
    training_loop,
)
from c4a0.training_common import parse_lr_schedule  # noqa: E402
from c4a0.training_benchmark import (  # noqa: E402
    TrainingBenchmarkConfig,
    compare_training_benchmarks,
    load_benchmark_report,
    run_training_benchmark,
    write_benchmark_report,
)
from c4a0.minimax_harness import (  # noqa: E402
    MinimaxHarnessConfig,
    evaluate_minimax_ladder,
)
from c4a0.utils import get_torch_device  # noqa: E402
from c4a0.training_v2 import (  # noqa: E402
    is_v2_run,
    load_champion_model,
    load_model_checkpoint,
    run_async_training,
    training_status,
)

import c4a0_cpp  # noqa: E402

app = typer.Typer()


class GameMode(str, Enum):
    human_ai = "human-ai"
    human_human = "human-human"
    ai_ai = "ai-ai"


class TrainingPrecision(str, Enum):
    auto = "auto"
    mixed16 = "16-mixed"
    true32 = "32-true"


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


@app.command("train-legacy")
def train_legacy(
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
def train(
    base_dir: str = "training-v2",
    device: str = str(get_torch_device()),
    run_seed: int = 1337,
    n_mcts_iterations: int = 1400,
    c_exploration: float = 6.6,
    c_ply_penalty: float = 0.01,
    mcts_value_scale: float = 0.0,
    value_loss_weight: float = 0.0,
    self_play_shard_games: int = 256,
    self_play_batch_games: int = 512,
    replay_capacity_games: int = 20_000,
    replay_warmup_games: int = 2_048,
    replay_ratio: float = 4.0,
    validation_fraction: float = 0.05,
    archive_depth: int = 8,
    champion_self_play_weight: float = 0.65,
    archive_weight: float = 0.30,
    uniform_weight: float = 0.025,
    random_weight: float = 0.025,
    archive_recency: float = 0.7,
    training_batch_size: int = 512,
    inference_batch_size: int = 128,
    data_loader_workers: int = 2,
    mcts_worker_threads: int = 0,
    precision: TrainingPrecision = TrainingPrecision.auto,
    inference_amp_min_batch_size: int = 96,
    n_residual_blocks: int = 1,
    conv_filter_size: int = 32,
    n_policy_layers: int = 4,
    n_value_layers: int = 2,
    lr_schedule: List[float] = [0, 2e-3, 10, 8e-4],
    l2_reg: float = 4e-4,
    max_gens: Optional[int] = None,
    max_candidate_attempts: Optional[int] = None,
    arena_min_games: int = 40,
    arena_max_games: int = 800,
    arena_p0: float = 0.50,
    arena_p1: float = 0.55,
    arena_alpha: float = 0.05,
    arena_beta: float = 0.05,
    arena_pair_batch_size: int = 100,
    root_dirichlet_epsilon: float = 0.25,
    root_dirichlet_alpha: float = 0.30,
    temperature_cutoff_ply: int = 8,
    early_temperature: float = 1.0,
    late_temperature: float = 0.0,
    validation_interval_steps: int = 100,
    validation_batches: int = 8,
    rejected_weight_retention: int = 3,
    learner_incumbent_min_score: float = 0.5,
    debt_pause_shards: int = 2,
    debt_resume_shards: int = 1,
):
    """Train asynchronously with replay, a neural opponent league, and gating."""
    config = TrainingV2Config(
        base_dir=base_dir,
        device=device,
        run_seed=run_seed,
        n_mcts_iterations=n_mcts_iterations,
        c_exploration=c_exploration,
        c_ply_penalty=c_ply_penalty,
        mcts_value_scale=mcts_value_scale,
        value_loss_weight=value_loss_weight,
        self_play_shard_games=self_play_shard_games,
        self_play_batch_games=self_play_batch_games,
        replay_capacity_games=replay_capacity_games,
        replay_warmup_games=replay_warmup_games,
        replay_ratio=replay_ratio,
        validation_fraction=validation_fraction,
        archive_depth=archive_depth,
        champion_self_play_weight=champion_self_play_weight,
        archive_weight=archive_weight,
        uniform_weight=uniform_weight,
        random_weight=random_weight,
        archive_recency=archive_recency,
        training_batch_size=training_batch_size,
        inference_batch_size=inference_batch_size,
        data_loader_workers=data_loader_workers,
        mcts_worker_threads=mcts_worker_threads,
        precision=precision.value,
        inference_amp_min_batch_size=inference_amp_min_batch_size,
        n_residual_blocks=n_residual_blocks,
        conv_filter_size=conv_filter_size,
        n_policy_layers=n_policy_layers,
        n_value_layers=n_value_layers,
        lr_schedule=lr_schedule,
        l2_reg=l2_reg,
        max_gens=max_gens,
        max_candidate_attempts=max_candidate_attempts,
        arena_min_games=arena_min_games,
        arena_max_games=arena_max_games,
        arena_p0=arena_p0,
        arena_p1=arena_p1,
        arena_alpha=arena_alpha,
        arena_beta=arena_beta,
        arena_pair_batch_size=arena_pair_batch_size,
        root_dirichlet_epsilon=root_dirichlet_epsilon,
        root_dirichlet_alpha=root_dirichlet_alpha,
        temperature_cutoff_ply=temperature_cutoff_ply,
        early_temperature=early_temperature,
        late_temperature=late_temperature,
        validation_interval_steps=validation_interval_steps,
        validation_batches=validation_batches,
        rejected_weight_retention=rejected_weight_retention,
        learner_incumbent_min_score=learner_incumbent_min_score,
        debt_pause_shards=debt_pause_shards,
        debt_resume_shards=debt_resume_shards,
    )
    result = run_async_training(config)
    logger.info("Training stopped: {}", result)


@app.command("benchmark-production")
def benchmark_production_command(
    config_path: str, output: str, repeats: int = 3, timeout_seconds: float = 3600
):
    """Benchmark asynchronous training and independently evaluate saved candidates."""
    from pathlib import Path
    from c4a0.production_benchmark import run_production_benchmark

    config = TrainingV2Config.model_validate_json(Path(config_path).read_text())
    report = run_production_benchmark(config, repeats, timeout_seconds)
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    typer.echo(output)


@app.command("export-model")
def export_model_command(
    output: str, base_dir: str = "training-v2", attempt: Optional[int] = None
):
    """Export compact inference weights and settings from a trusted local run."""
    from c4a0.training_v2 import export_inference_model

    typer.echo(str(export_inference_model(base_dir, output, attempt)))


@app.command("training-config")
def training_config_command(base_dir: str = "training-v2", device: str = "cpu"):
    """Print the effective persisted experiment settings before resuming."""
    from c4a0.training_v2 import effective_training_config

    typer.echo(
        effective_training_config(
            TrainingV2Config(base_dir=base_dir, device=device)
        ).model_dump_json(indent=2)
    )


@app.command("training-status")
def training_status_command(base_dir: str = "training-v2"):
    """Show champion, pending candidate, and replay state for a V2 run."""
    typer.echo(json.dumps(training_status(base_dir), indent=2))


@app.command("minimax-test")
def minimax_test(
    base_dir: str = "training-v2",
    model_path: Optional[str] = None,
    device: str = str(get_torch_device()),
    games_per_level: int = 20,
    pair_batch_size: int = 6,
    max_depth: int = 6,
    mcts_value_scale: Optional[float] = None,
    trusted_checkpoint: bool = False,
    mcts_iterations: int = 64,
    inference_batch_size: int = 128,
    c_exploration: float = 1.4,
    c_ply_penalty: float = 0.01,
    worker_threads: int = 0,
    seed: int = 1337,
):
    """Test a model against random and successively deeper minimax opponents."""
    model = (
        load_model_checkpoint(
            model_path, mcts_value_scale=mcts_value_scale, trusted=trusted_checkpoint
        )
        if model_path is not None
        else load_champion_model(base_dir)
    )
    config = MinimaxHarnessConfig(
        games_per_level=games_per_level,
        pair_batch_size=pair_batch_size,
        max_depth=max_depth,
        mcts_iterations=mcts_iterations,
        inference_batch_size=inference_batch_size,
        c_exploration=c_exploration,
        c_ply_penalty=c_ply_penalty,
        worker_threads=worker_threads,
        seed=seed,
    )

    def report(snapshot: dict[str, object]) -> None:
        logger.info(
            "{}: {}/{} games, {} points",
            snapshot["opponent"],
            snapshot["completed_games"],
            snapshot["total_games"],
            snapshot["points"],
        )

    result = evaluate_minimax_ladder(model, torch.device(device), config, report)
    typer.echo(json.dumps(result.to_dict(), indent=2))


@app.command("benchmark-training")
def benchmark_training(
    output: str,
    workflow: str = "v2",
    device: str = "cpu",
    games: int = 64,
    arena_games: int = 32,
    mcts_iterations: int = 64,
    inference_batch_size: int = 64,
    training_batch_size: int = 128,
    training_steps: int = 25,
    repeats: int = 3,
    mcts_worker_threads: int = 0,
    precision: TrainingPrecision = TrainingPrecision.auto,
    inference_amp_min_batch_size: int = 96,
    solver_path: Optional[str] = None,
    book_path: Optional[str] = None,
    solver_cache_path: str = "benchmark-solutions.sqlite3",
):
    """Benchmark the fixed end-to-end V2 or frozen legacy Rust workflow."""
    config = TrainingBenchmarkConfig(
        workflow=workflow,  # type: ignore[arg-type]
        device=device,
        games=games,
        arena_games=arena_games,
        mcts_iterations=mcts_iterations,
        inference_batch_size=inference_batch_size,
        training_batch_size=training_batch_size,
        training_steps=training_steps,
        repeats=repeats,
        mcts_worker_threads=mcts_worker_threads,
        precision=precision.value,
        inference_amp_min_batch_size=inference_amp_min_batch_size,
        solver_path=solver_path,
        book_path=book_path,
        solver_cache_path=solver_cache_path,
    )
    report = run_training_benchmark(config)
    write_benchmark_report(report, output)
    typer.echo(json.dumps(report, indent=2))


@app.command("compare-training-benchmarks")
def compare_benchmarks(
    v2_report: str,
    legacy_rust_report: str,
    output: Optional[str] = None,
    throughput_ratio: float = 0.9,
    latency_ratio: float = 1.1,
):
    """Apply the fail-closed V2 replacement gate to two benchmark reports."""
    comparison = compare_training_benchmarks(
        load_benchmark_report(v2_report),
        load_benchmark_report(legacy_rust_report),
        throughput_ratio,
        latency_ratio,
    )
    if output is not None:
        write_benchmark_report(comparison, output)
    typer.echo(json.dumps(comparison, indent=2))
    if not comparison["replacement_approved"]:
        raise typer.Exit(2)


@app.command()
def play(
    base_dir: str = "training-v2",
    max_mcts_iters: int = 1400,
    c_exploration: float = 6.6,
    c_ply_penalty: float = 0.01,
    model: PlayModel = PlayModel.best,
    mode: GameMode = GameMode.human_ai,
    human_side: HumanSide = HumanSide.red,
):
    """Play against the AI, another human, or watch an AI-vs-AI game."""
    if model is PlayModel.best:
        nn = (
            load_champion_model(base_dir)
            if is_v2_run(base_dir)
            else TrainingGen.load_latest(base_dir).get_model(base_dir)
        )
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
