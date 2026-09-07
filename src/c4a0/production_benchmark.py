"""Measure the actual asynchronous lifecycle and saved-candidate quality separately."""

from pathlib import Path
import hashlib
import json
import math
import random
import statistics
import tempfile
import threading
import time
from typing import Any

import torch
import numpy as np
from c4a0.minimax_harness import _MinimaxEvaluator

from c4a0.config import TrainingV2Config
from c4a0.performance import UtilizationMonitor
from c4a0.training_v2 import (
    RunManifest,
    _ActorEvaluator,
    _arena_openings,
    _opening_is_nonterminal,
    _opening_key,
    _run_arena,
    run_async_training,
    export_inference_model,
)
from c4a0.training_benchmark import _revision, benchmark_provenance


def held_out_openings(count: int = 32) -> list[list[int]]:
    """Version-one fixed test bank: 8–12 plies, disjoint from arena's 1–6 plies."""
    rng = random.Random(0xC4A02026)
    positions = set()
    result = []
    while len(result) < count:
        moves = [rng.randrange(7) for _ in range(rng.randint(8, 12))]
        if _opening_is_nonterminal(moves) and _opening_key(moves) not in positions:
            positions.add(_opening_key(moves))
            result.append(moves)
    return result


def held_out_position_metrics(
    evaluator, champion_id: int, candidate_id: int, openings: list[list[int]]
) -> dict[str, float]:
    positions = np.zeros((len(openings), 2, 6, 7), dtype=np.float32)
    for index, moves in enumerate(openings):
        board = np.zeros((6, 7), dtype=np.int8)
        heights = [0] * 7
        for ply, column in enumerate(moves):
            board[heights[column], column] = 1 + ply % 2
            heights[column] += 1
        current = 1 + len(moves) % 2
        positions[index, 0] = board == current
        positions[index, 1] = board == 3 - current
    candidate_policy, _, _ = evaluator(candidate_id, positions)
    champion_policy, _, _ = evaluator(champion_id, positions)
    teacher_policy, _, _ = _MinimaxEvaluator(3, deadline=time.monotonic() + 30)(
        positions
    )
    legal = positions[:, :, -1, :].sum(axis=1) == 0

    def legal_logprob(policy):
        policy = np.where(legal, policy, -1e9)
        policy = policy - policy.max(axis=1, keepdims=True)
        return policy - np.log(np.exp(policy).sum(axis=1, keepdims=True))

    candidate_policy = legal_logprob(candidate_policy)
    champion_policy = legal_logprob(champion_policy)
    probabilities = np.exp(candidate_policy)
    chosen = candidate_policy.argmax(axis=1)
    return {
        "policy_entropy": float(-(probabilities * candidate_policy).sum(axis=1).mean()),
        "policy_kl_to_parent": float(
            (probabilities * (candidate_policy - champion_policy)).sum(axis=1).mean()
        ),
        "depth3_minimax_action_agreement": float(
            (teacher_policy[np.arange(len(chosen)), chosen] == 0).mean()
        ),
    }


def run_production_benchmark(
    config: TrainingV2Config, repeats: int = 3, timeout_seconds: float = 3600
) -> dict[str, Any]:
    if repeats < 1 or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Repeats and timeout must be positive")
    openings = held_out_openings()
    runs = []
    for repetition in range(repeats):
        with tempfile.TemporaryDirectory(prefix="c4a0-production-") as temporary:
            run_dir = Path(temporary) / "run"
            effective = config.model_copy(
                update={
                    "base_dir": str(run_dir),
                    "max_gens": None,
                    "max_candidate_attempts": 1,
                    "rejected_weight_retention": max(
                        1, config.rejected_weight_retention
                    ),
                }
            )
            events: list[dict[str, Any]] = []
            started = time.monotonic()
            deadline = started + timeout_seconds

            def progress(phase, payload):
                events.append(
                    {"seconds": time.monotonic() - started, "phase": phase, **payload}
                )

            monitor = UtilizationMonitor(include_children=True)
            monitor.start()
            try:
                with monitor.phase("production"):
                    status = run_async_training(
                        effective,
                        progress_callback=progress,
                        should_cancel=lambda: time.monotonic() >= deadline,
                    )
            finally:
                monitor.stop()
            elapsed = time.monotonic() - started
            with RunManifest.open_existing(run_dir) as manifest:
                rows = manifest.connection.execute(
                    "SELECT * FROM attempts WHERE status IN ('accepted','rejected') ORDER BY attempt_n"
                ).fetchall()
                if len(rows) != 2:
                    raise RuntimeError(
                        "Production benchmark did not persist exactly one candidate decision"
                    )
                champion, candidate = (manifest.artifact_row(row) for row in rows)
            selection = _arena_openings(
                run_dir, config.run_seed, config.arena_max_games
            )
            assert not {_opening_key(item) for item in selection} & {
                _opening_key(item) for item in openings
            }
            evaluation_config = effective.model_copy(
                update={
                    "arena_min_games": len(openings) * 2,
                    "arena_max_games": len(openings) * 2,
                }
            )
            evaluator = _ActorEvaluator(
                torch.device(config.device),
                config.run_seed,
                config.precision,
                config.inference_amp_min_batch_size,
                config.mcts_value_scale,
            )
            strength = _run_arena(
                evaluation_config,
                evaluator,
                champion,
                candidate,
                openings,
                threading.Event(),
            )
            # This fixed-budget score is descriptive evidence, not the selection decision.
            strength.pop("decision")
            position_metrics = held_out_position_metrics(
                evaluator,
                int(champion["attempt_n"]),
                int(candidate["attempt_n"]),
                openings,
            )
            export_path = export_inference_model(
                str(run_dir),
                Path(temporary) / "inference.pt",
                int(candidate["attempt_n"]),
            )
            runs.append(
                {
                    "repetition": repetition,
                    "wall_seconds": elapsed,
                    "replay_games": status["replay_games"],
                    "replay_samples": status["replay_samples"],
                    "training_steps": candidate["training_steps"],
                    "training_samples": candidate["training_samples"],
                    "candidate_status": candidate["status"],
                    "resumable_checkpoint_bytes": Path(candidate["checkpoint_path"])
                    .stat()
                    .st_size,
                    "inference_export_bytes": export_path.stat().st_size,
                    "held_out_strength": strength,
                    "held_out_positions": position_metrics,
                    "utilization": monitor.summary(),
                    "events": events,
                }
            )
    return {
        "format": "c4a0-production-benchmark",
        "version": 1,
        "revision": _revision(),
        "provenance": {
            **benchmark_provenance(config),
            "purpose": "production lifecycle and separate held-out candidate quality",
        },
        "measurement_scope": {
            "rss": "coordinator and descendants; shared pages can be counted more than once",
            "cpu": "coordinator only",
            "gpu": "device-wide, including other applications",
        },
        "config": config.model_dump(mode="json"),
        "repeats": repeats,
        "held_out_bank": {
            "version": 1,
            "sha256": hashlib.sha256(json.dumps(openings).encode()).hexdigest(),
            "openings": openings,
        },
        "throughput": {
            "median_candidate_seconds": statistics.median(
                run["wall_seconds"] for run in runs
            ),
            "median_replay_samples_per_second": statistics.median(
                run["replay_samples"] / run["wall_seconds"] for run in runs
            ),
        },
        "strength": {
            "mean_saved_candidate_score": statistics.mean(
                run["held_out_strength"]["score"] for run in runs
            )
        },
        "release_approved": False,
        "runs": runs,
    }
