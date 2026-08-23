"""Isolated worker process used by the desktop application's job queue."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import traceback
from typing import Any

from loguru import logger
import optuna
import torch

from c4a0.config import (
    MCTSSweepConfig,
    NNSweepConfig,
    SolverScoreConfig,
    TournamentConfig,
    TrainingConfig,
    ValidationConfig,
)
from c4a0.nn import ModelConfig
from c4a0.sweep import perform_hparam_sweep_config
from c4a0.tournament import (
    ModelID,
    ModelPlayer,
    Player,
    PlayerName,
    RandomPlayer,
    UniformPlayer,
    play_tournament,
)
from c4a0.training import (
    SolverConfig,
    TrainingCancelled,
    TrainingGen,
    parse_lr_schedule,
    training_loop,
)


PROTOCOL_VERSION = 1
_cancelled = threading.Event()
_stop_after_generation = threading.Event()


def emit(event_type: str, **payload: Any) -> None:
    record = {
        "version": PROTOCOL_VERSION,
        "type": event_type,
        "timestamp": datetime.now().isoformat(),
        **payload,
    }
    print(json.dumps(record, default=str), flush=True)


def _handle_signal(_signum, _frame) -> None:
    _cancelled.set()


def _log_sink(message) -> None:
    emit("log", level=message.record["level"].name, message=str(message).rstrip())


def _listen_for_commands() -> None:
    """Receive non-blocking control messages after the initial job request."""
    for line in sys.stdin:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            emit("log", level="WARNING", message="Ignored malformed control message")
            continue
        command = message.get("command")
        if command == "stop_after_generation":
            _stop_after_generation.set()
            emit("log", level="INFO", message="Will stop after the current generation")


def run_training(data: dict[str, Any]) -> dict[str, Any]:
    config = TrainingConfig.model_validate(data)
    model_config = ModelConfig(
        n_residual_blocks=config.n_residual_blocks,
        conv_filter_size=config.conv_filter_size,
        n_policy_layers=config.n_policy_layers,
        n_value_layers=config.n_value_layers,
        lr_schedule=parse_lr_schedule(config.lr_schedule),
        l2_reg=config.l2_reg,
    )
    solver_config = None
    if config.solver_path and config.book_path:
        solver_config = SolverConfig(
            solver_path=config.solver_path,
            book_path=config.book_path,
            solutions_path=config.solutions_path,
        )

    def on_progress(phase: str, payload: dict[str, Any]) -> None:
        emit("phase", name=phase, **payload)
        current = payload.get("current")
        total = payload.get("total")
        if current is not None and total:
            emit("progress", current=current, total=total, fraction=current / total)
        for key, value in payload.get("metrics", {}).items():
            emit("metric", name=key, value=value)

    generation = training_loop(
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
        progress_callback=on_progress,
        should_cancel=_cancelled.is_set,
        should_stop_after_generation=_stop_after_generation.is_set,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    return {
        "generation": generation.gen_n,
        "val_loss": generation.val_loss,
        "solver_score": generation.solver_score,
    }


def run_score(data: dict[str, Any]) -> dict[str, Any]:
    config = SolverScoreConfig.model_validate(data)
    generations = TrainingGen.load_all(config.base_dir)
    selected = [
        generation
        for generation in generations
        if config.generations is None or generation.gen_n in config.generations
    ]
    selected = [
        generation
        for generation in selected
        if config.rescore or generation.solver_score is None
    ]
    scored = 0
    for index, generation in enumerate(selected, start=1):
        if _cancelled.is_set():
            raise TrainingCancelled("scoring cancelled")
        emit(
            "phase",
            name="solver",
            generation=generation.gen_n,
            current=index,
            total=len(selected),
        )
        games = generation.get_games(config.base_dir)
        if games is None:
            continue
        generation.solver_score = games.score_policies(
            config.solver_path, config.book_path, config.solutions_path
        )
        generation.save_metadata(config.base_dir)
        scored += 1
        emit(
            "metric",
            name=f"generation_{generation.gen_n}",
            value=generation.solver_score,
        )
        emit(
            "progress",
            current=index,
            total=len(selected),
            fraction=index / len(selected),
        )
    return {"scored_generations": scored}


def _load_tournament_player(
    spec: str, model_id: int, base_dir: str, device: torch.device
):
    identifier = ModelID(model_id)
    if spec == "random":
        player = RandomPlayer(identifier)
    elif spec == "uniform":
        player = UniformPlayer(identifier)
    else:
        generations = TrainingGen.load_all(base_dir)
        if not generations:
            raise FileNotFoundError(f"no trained generations in {base_dir}")
        if spec == "latest":
            generation = generations[0]
        elif spec.startswith("gen:"):
            requested = int(spec.removeprefix("gen:"))
            generation = next(
                (item for item in generations if item.gen_n == requested), None
            )
            if generation is None:
                raise ValueError(f"generation {requested} was not found")
        else:
            raise ValueError(f"unrecognized tournament player: {spec}")
        player = ModelPlayer(identifier, generation.get_model(base_dir), device=device)
    player.name = PlayerName(spec)
    return player


def run_tournament_job(data: dict[str, Any]) -> dict[str, Any]:
    config = TournamentConfig.model_validate(data)
    device = torch.device(config.device)
    players: list[Player] = [
        _load_tournament_player(spec, index, config.base_dir, device)
        for index, spec in enumerate(config.players)
    ]
    emit("phase", name="tournament", current=0, total=1)
    result = play_tournament(
        players=players,
        games_per_match=config.games_per_match,
        batch_size=config.batch_size,
        mcts_iterations=config.mcts_iterations,
        exploration_constant=config.exploration_constant,
        c_ply_penalty=config.c_ply_penalty,
        progress_callback=lambda current, total: emit(
            "progress", current=current, total=total, fraction=current / total
        ),
        cancelled_callback=_cancelled.is_set,
    )
    names = {player.model_id: str(player.name) for player in players}
    statistics = {
        name: {"player": name, "wins": 0, "draws": 0, "losses": 0, "points": 0.0}
        for name in names.values()
    }
    matchup: dict[str, dict[str, dict[str, Any]]] = {
        name: {} for name in names.values()
    }
    if result.games is not None:
        for game in result.games.results:
            red = names[ModelID(game.metadata.player0_id)]
            gold = names[ModelID(game.metadata.player1_id)]
            red_score = game.player0_score()
            for player, opponent, score in (
                (red, gold, red_score),
                (gold, red, 1.0 - red_score),
            ):
                cell = matchup[player].setdefault(
                    opponent,
                    {"opponent": opponent, "wins": 0, "draws": 0, "losses": 0},
                )
                statistics[player]["points"] += score
                if score == 1.0:
                    statistics[player]["wins"] += 1
                    cell["wins"] += 1
                elif score == 0.5:
                    statistics[player]["draws"] += 1
                    cell["draws"] += 1
                else:
                    statistics[player]["losses"] += 1
                    cell["losses"] += 1
    ranking = sorted(
        statistics.values(), key=lambda item: (-item["points"], item["player"])
    )
    return {
        "games": len(result.games.results if result.games else []),
        "ranking": ranking,
        "matchups": [
            {"player": player, "opponents": list(opponents.values())}
            for player, opponents in matchup.items()
        ],
    }


def run_nn_sweep(data: dict[str, Any]) -> dict[str, Any]:
    config = NNSweepConfig.model_validate(data)
    emit("phase", name="nn_sweep", current=0, total=config.n_trials)
    study = perform_hparam_sweep_config(
        config,
        progress_callback=lambda current, total: emit(
            "progress", current=current, total=total, fraction=current / total
        ),
        should_cancel=_cancelled.is_set,
    )
    if _cancelled.is_set():
        raise TrainingCancelled("neural sweep cancelled")
    return {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "trials": len(study.trials),
    }


def run_mcts_sweep(data: dict[str, Any]) -> dict[str, Any]:
    config = MCTSSweepConfig.model_validate(data)
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
        if _cancelled.is_set():
            raise optuna.TrialPruned("cancelled")
        emit(
            "phase",
            name="mcts_sweep",
            current=trial.number,
            total=config.n_trials,
        )
        trial_path = base_path / f"trial_{trial.number}"
        trial_path.mkdir(exist_ok=False)
        generation = training_loop(
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
            should_cancel=_cancelled.is_set,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        if generation.solver_score is None:
            raise RuntimeError("MCTS sweep requires a solver score")
        return generation.solver_score

    study = optuna.create_study(
        study_name="mcts_sweep",
        storage=f"sqlite:///{config.optuna_db_path}",
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )

    def on_trial_complete(
        current_study: optuna.Study, _trial: optuna.trial.FrozenTrial
    ) -> None:
        completed = len(current_study.trials)
        emit(
            "progress",
            current=completed,
            total=config.n_trials,
            fraction=completed / config.n_trials,
        )
        if _cancelled.is_set():
            current_study.stop()

    study.optimize(objective, n_trials=config.n_trials, callbacks=[on_trial_complete])
    if _cancelled.is_set():
        raise TrainingCancelled("MCTS sweep cancelled")
    return {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "trials": len(study.trials),
    }


def run_validation(data: dict[str, Any]) -> dict[str, Any]:
    config = ValidationConfig.model_validate(data)
    emit("phase", name=config.profile, current=0, total=1)
    process = subprocess.Popen(
        ["mise", "run", config.profile],
        cwd=config.project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    while process.poll() is None:
        line = process.stdout.readline()
        if line:
            emit("log", level="INFO", message=line.rstrip())
        if _cancelled.is_set():
            process.terminate()
            process.wait(timeout=5)
            raise TrainingCancelled("validation cancelled")
    for line in process.stdout:
        emit("log", level="INFO", message=line.rstrip())
    if process.returncode != 0:
        raise RuntimeError(f"validation exited with status {process.returncode}")
    return {"profile": config.profile, "exit_code": process.returncode}


RUNNERS = {
    "training": run_training,
    "solver_score": run_score,
    "tournament": run_tournament_job,
    "nn_sweep": run_nn_sweep,
    "mcts_sweep": run_mcts_sweep,
    "validation": run_validation,
}


def main() -> int:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    logger.remove()
    logger.add(_log_sink, colorize=False)
    try:
        request_line = sys.stdin.readline()
        if not request_line:
            raise ValueError("worker did not receive a job request")
        request = json.loads(request_line)
        if request.get("version") != PROTOCOL_VERSION:
            raise ValueError("unsupported worker protocol version")
        kind = request["kind"]
        runner = RUNNERS.get(kind)
        if runner is None:
            raise ValueError(f"unsupported job type: {kind}")
        emit("started", kind=kind)
        threading.Thread(target=_listen_for_commands, daemon=True).start()
        result = runner(request.get("config", {}))
        if _cancelled.is_set():
            emit("cancelled", message="job cancelled")
            return 2
        emit("completed", result=result)
        return 0
    except TrainingCancelled as error:
        emit("cancelled", message=str(error))
        return 2
    except Exception as error:
        emit(
            "failed",
            message=str(error),
            details="".join(traceback.format_exception(error)),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
