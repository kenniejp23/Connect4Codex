"""Reproducible end-to-end training throughput and replacement gates."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field
import torch
from torch.optim import Adam

from c4a0.nn import ConnectFourNet, ModelConfig
from c4a0.performance import UtilizationMonitor

try:
    import c4a0_cpp  # type: ignore
except ImportError:  # The frozen Rust benchmark environment intentionally lacks it.
    c4a0_cpp = None  # type: ignore


BENCHMARK_FORMAT = "c4a0-training-benchmark"
BENCHMARK_VERSION = 1


class TrainingBenchmarkConfig(BaseModel):
    workflow: Literal["v2", "legacy-rust"] = "v2"
    device: str = "cpu"
    seed: int = Field(default=1337, ge=0)
    games: int = Field(default=64, ge=2)
    arena_games: int = Field(default=32, ge=2)
    mcts_iterations: int = Field(default=64, ge=1)
    inference_batch_size: int = Field(default=64, ge=1)
    training_batch_size: int = Field(default=128, ge=2)
    training_steps: int = Field(default=25, ge=1)
    repeats: int = Field(default=3, ge=1)
    mcts_worker_threads: int = Field(default=0, ge=0)
    precision: Literal["auto", "16-mixed", "32-true"] = "auto"
    inference_amp_min_batch_size: int = Field(default=96, ge=1)
    solver_path: str | None = None
    book_path: str | None = None
    solver_cache_path: str = "benchmark-solutions.sqlite3"
    monitor_interval_seconds: float = Field(default=0.5, gt=0)

    def model_post_init(self, _context: Any) -> None:
        if self.games % 2 or self.arena_games % 2:
            raise ValueError("benchmark game counts must be even")
        if bool(self.solver_path) != bool(self.book_path):
            raise ValueError("solver executable and book must be supplied together")


class _Router:
    def __init__(
        self,
        models: dict[int, ConnectFourNet],
        device: torch.device,
        precision: Literal["auto", "16-mixed", "32-true"],
        amp_min_batch_size: int,
    ):
        self.models = models
        self.device = device
        self.use_amp = device.type == "cuda" and precision != "32-true"
        self.precision = precision
        self.amp_min_batch_size = amp_min_batch_size

    def __call__(self, model_id: int, positions: np.ndarray):
        model = self.models[int(model_id)]
        tensor = torch.from_numpy(positions).to(self.device)
        use_amp = self.use_amp and (
            self.precision == "16-mixed"
            or positions.shape[0] >= self.amp_min_batch_size
        )
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ),
        ):
            policy, q_penalty, q_no_penalty = model(tensor)
        return (
            np.ascontiguousarray(policy.float().cpu().numpy()),
            np.ascontiguousarray(q_penalty.float().cpu().numpy()),
            np.ascontiguousarray(q_no_penalty.float().cpu().numpy()),
        )


@dataclass(frozen=True)
class _Metadata:
    game_id: int
    player0_id: int
    player1_id: int


def _revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _model(seed: int) -> ConnectFourNet:
    torch.manual_seed(seed)
    return ConnectFourNet(
        ModelConfig(
            n_residual_blocks=1,
            conv_filter_size=32,
            n_policy_layers=4,
            n_value_layers=2,
            lr_schedule={0: 0.002},
            l2_reg=4e-4,
        )
    )


def _v2_play(
    metadata: list[Any],
    config: TrainingBenchmarkConfig,
    evaluator: _Router,
    *,
    arena: bool,
):
    if c4a0_cpp is None:
        raise RuntimeError("the V2 C++ extension is not installed")
    options = c4a0_cpp.SelfPlayOptions()
    options.max_nn_batch_size = config.inference_batch_size
    options.n_mcts_iterations = config.mcts_iterations
    options.c_exploration = 6.6
    options.c_ply_penalty = 0.01
    options.root_dirichlet_alpha = 0.0 if arena else 0.3
    options.root_dirichlet_epsilon = 0.0 if arena else 0.25
    options.temperature_midpoint_ply = 0 if arena else 8
    options.temperature_cutoff_ply = 0 if arena else 8
    options.early_temperature = 0.0 if arena else 1.0
    options.middle_temperature = 0.0 if arena else 1.0
    options.late_temperature = 0.0
    options.seed = config.seed
    options.worker_threads = config.mcts_worker_threads
    requests = [
        c4a0_cpp.GameRequest(
            c4a0_cpp.GameMetadata(
                item.game_id, item.player0_id, item.player1_id
            ),
            [],
        )
        for item in metadata
    ]
    return c4a0_cpp.play_games_v2(requests, options, evaluator)


def _legacy_rust_play(
    metadata: list[Any],
    config: TrainingBenchmarkConfig,
    evaluator: _Router,
    *,
    arena: bool,
):
    del arena
    try:
        import c4a0_rust  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "legacy-rust requires the frozen pre-C++ c4a0_rust extension; "
            "see docs/training-benchmark.md"
        ) from error
    requests = [
        c4a0_rust.GameMetadata(  # type: ignore[attr-defined]
            int(item.game_id), int(item.player0_id), int(item.player1_id)
        )
        for item in metadata
    ]
    return c4a0_rust.play_games(  # type: ignore[attr-defined]
        requests,
        config.inference_batch_size,
        config.mcts_iterations,
        6.6,
        0.01,
        evaluator,
    )


def _sample_batch(games: Any, batch_size: int, seed: int):
    samples = [sample for game in games.results for sample in game.samples]
    if not samples:
        raise RuntimeError("benchmark self-play returned no training positions")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(samples), size=batch_size)
    values = [samples[int(index)].to_numpy() for index in indices]
    return tuple(torch.stack([torch.from_numpy(item[column]) for item in values]) for column in range(4))


def _train_fixed_work(
    model: ConnectFourNet,
    games: Any,
    config: TrainingBenchmarkConfig,
    device: torch.device,
) -> tuple[float, int]:
    model.train().to(device)
    optimizer = Adam(
        model.parameters(),
        lr=0.002,
        weight_decay=4e-4,
        fused=device.type == "cuda",
    )
    use_amp = device.type == "cuda" and config.precision != "32-true"
    scaler = torch.GradScaler("cuda", enabled=use_amp)
    batch = tuple(
        value.to(device)
        for value in _sample_batch(games, config.training_batch_size, config.seed)
    )
    started = time.perf_counter()
    for _ in range(config.training_steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            policy_logprob, q_penalty, q_no_penalty = model(batch[0])
            policy_loss = -(batch[1] * policy_logprob).sum(dim=1).mean()
            value_loss = ((q_penalty - batch[2]) ** 2).mean()
            value_loss += ((q_no_penalty - batch[3]) ** 2).mean()
            loss = policy_loss + value_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - started, config.training_steps * config.training_batch_size


def _one_run(config: TrainingBenchmarkConfig) -> dict[str, Any]:
    device = torch.device(config.device)
    champion = _model(config.seed).eval().to(device)
    candidate = _model(config.seed).eval().to(device)
    router = _Router(
        {0: champion, 1: candidate},
        device,
        config.precision,
        config.inference_amp_min_batch_size,
    )
    play = _v2_play if config.workflow == "v2" else _legacy_rust_play
    total_started = time.perf_counter()
    monitor = UtilizationMonitor(config.monitor_interval_seconds)
    monitor.start()

    try:
        metadata = [
            _Metadata(game_id, 0, 0) for game_id in range(1, config.games + 1)
        ]
        started = time.perf_counter()
        with monitor.phase("self_play"):
            games = play(metadata, config, router, arena=False)
        self_play_seconds = time.perf_counter() - started

        candidate.load_state_dict(champion.state_dict())
        with monitor.phase("training"):
            training_seconds, trained_positions = _train_fixed_work(
                candidate, games, config, device
            )
        candidate.eval()
        candidate_latency = self_play_seconds + training_seconds

        arena_metadata = []
        for game_id in range(1, config.arena_games + 1, 2):
            arena_metadata.extend(
                [
                    _Metadata(game_id, 1, 0),
                    _Metadata(game_id + 1, 0, 1),
                ]
            )
        started = time.perf_counter()
        with monitor.phase("arena"):
            arena = play(arena_metadata, config, router, arena=True)
        arena_seconds = time.perf_counter() - started

        solver_score = None
        solver_seconds = 0.0
        if config.solver_path is not None and config.book_path is not None:
            started = time.perf_counter()
            with monitor.phase("solver"):
                solver_score = float(
                    games.score_policies(
                        config.solver_path,
                        config.book_path,
                        config.solver_cache_path,
                    )
                )
            solver_seconds = time.perf_counter() - started
    finally:
        monitor.stop()
    wall_clock_seconds = time.perf_counter() - total_started
    return {
        "games_per_second": config.games / self_play_seconds,
        "train_positions_per_second": trained_positions / training_seconds,
        "candidate_latency_seconds": candidate_latency,
        "arena_games_per_second": len(arena.results) / arena_seconds,
        "solver_score": solver_score,
        "solver_score_per_wall_clock_hour": (
            None
            if solver_score is None
            else solver_score * 3600.0 / wall_clock_seconds
        ),
        "wall_clock_seconds": wall_clock_seconds,
        "self_play_seconds": self_play_seconds,
        "training_seconds": training_seconds,
        "arena_seconds": arena_seconds,
        "solver_seconds": solver_seconds,
        "self_play_positions": sum(len(game.samples) for game in games.results),
        "utilization": monitor.summary(),
    }


def run_training_benchmark(config: TrainingBenchmarkConfig) -> dict[str, Any]:
    runs = [_one_run(config) for _ in range(config.repeats)]
    metric_names = [
        key
        for key, value in runs[0].items()
        if value is None or isinstance(value, (int, float))
    ]
    metrics = {
        key: (
            None
            if any(run[key] is None for run in runs)
            else statistics.median(float(run[key]) for run in runs)
        )
        for key in metric_names
    }
    return {
        "format": BENCHMARK_FORMAT,
        "version": BENCHMARK_VERSION,
        "workflow": config.workflow,
        "revision": _revision(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": config.device,
            "device_name": (
                torch.cuda.get_device_name(torch.device(config.device))
                if torch.device(config.device).type == "cuda"
                else platform.processor()
            ),
        },
        "config": config.model_dump(mode="json"),
        "metrics": metrics,
        "runs": runs,
    }


def write_benchmark_report(report: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")


def compare_training_benchmarks(
    v2: dict[str, Any],
    legacy: dict[str, Any],
    throughput_ratio: float = 0.9,
    latency_ratio: float = 1.1,
) -> dict[str, Any]:
    for name, report, workflow in (
        ("V2", v2, "v2"),
        ("legacy", legacy, "legacy-rust"),
    ):
        if report.get("format") != BENCHMARK_FORMAT:
            raise ValueError(f"{name} report has an unsupported format")
        if report.get("workflow") != workflow:
            raise ValueError(f"{name} report workflow must be {workflow}")
    comparable_fields = (
        "seed",
        "games",
        "arena_games",
        "mcts_iterations",
        "inference_batch_size",
        "training_batch_size",
        "training_steps",
    )
    mismatches = [
        field
        for field in comparable_fields
        if v2.get("config", {}).get(field) != legacy.get("config", {}).get(field)
    ]
    if mismatches:
        raise ValueError(
            "benchmark reports use different fixed workloads: " + ", ".join(mismatches)
        )
    v2_metrics = v2["metrics"]
    legacy_metrics = legacy["metrics"]
    gates: dict[str, dict[str, Any]] = {}
    for metric in (
        "games_per_second",
        "train_positions_per_second",
        "arena_games_per_second",
        "solver_score_per_wall_clock_hour",
    ):
        current = v2_metrics.get(metric)
        baseline = legacy_metrics.get(metric)
        ratio = None if current is None or baseline in {None, 0} else current / baseline
        gates[metric] = {
            "v2": current,
            "legacy_rust": baseline,
            "ratio": ratio,
            "required_ratio": throughput_ratio,
            "passed": ratio is not None and ratio >= throughput_ratio,
        }
    current_latency = v2_metrics.get("candidate_latency_seconds")
    baseline_latency = legacy_metrics.get("candidate_latency_seconds")
    latency = (
        None
        if current_latency is None or baseline_latency in {None, 0}
        else current_latency / baseline_latency
    )
    gates["candidate_latency_seconds"] = {
        "v2": current_latency,
        "legacy_rust": baseline_latency,
        "ratio": latency,
        "maximum_ratio": latency_ratio,
        "passed": latency is not None and latency <= latency_ratio,
    }
    approved = all(bool(gate["passed"]) for gate in gates.values())
    return {
        "format": "c4a0-training-benchmark-comparison",
        "version": 1,
        "replacement_approved": approved,
        "decision": "V2 may replace legacy Rust" if approved else "keep legacy Rust as baseline",
        "gates": gates,
        "v2_revision": v2.get("revision"),
        "legacy_revision": legacy.get("revision"),
    }


def load_benchmark_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _standalone() -> None:
    """Small entry point usable from the frozen legacy Rust worktree."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", choices=("v2", "legacy-rust"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--arena-games", type=int, default=32)
    parser.add_argument("--mcts-iterations", type=int, default=64)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--training-batch-size", type=int, default=128)
    parser.add_argument("--training-steps", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--solver-path")
    parser.add_argument("--book-path")
    parser.add_argument("--solver-cache-path", default="benchmark-solutions.sqlite3")
    args = parser.parse_args()
    config = TrainingBenchmarkConfig(**vars(args))
    report = run_training_benchmark(config)
    write_benchmark_report(report, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    _standalone()
