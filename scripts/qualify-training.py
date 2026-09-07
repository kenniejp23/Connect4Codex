#!/usr/bin/env python3
"""Bounded CPU/CUDA acceptance, rejection, resume, cancellation, and soak qualification."""

import argparse
import json
from pathlib import Path
import tempfile
import time

import torch

from c4a0.config import TrainingV2Config
from c4a0.training_common import TrainingCancelled
import c4a0.training_v2 as training
from c4a0.performance import UtilizationMonitor

_ORIGINAL_TRAINER = training._trainer_process


def trainer_with_injected_oom(*args):
    original_loss = training._model_loss
    fired = False

    def loss(model, batch):
        nonlocal fired
        if not fired:
            fired = True
            raise torch.OutOfMemoryError("CUDA out of memory. Qualification injection")
        return original_loss(model, batch)

    training._model_loss = loss
    _ORIGINAL_TRAINER(*args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inject-oom", action="store_true")
    args = parser.parse_args()
    if args.seconds <= 0:
        raise ValueError("Soak duration must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA qualification requires a usable NVIDIA device")
    if args.inject_oom:
        if args.device != "cuda":
            raise ValueError("OOM recovery qualification requires CUDA")
        training._trainer_process = trainer_with_injected_oom
    with tempfile.TemporaryDirectory(
        prefix="c4a0-training-qualification-"
    ) as temporary:
        config = TrainingV2Config(
            base_dir=temporary,
            device=args.device,
            self_play_shard_games=16,
            self_play_batch_games=16,
            replay_warmup_games=16,
            replay_capacity_games=128,
            training_batch_size=32,
            inference_batch_size=32,
            n_mcts_iterations=16,
            arena_min_games=8,
            arena_max_games=16,
            arena_pair_batch_size=4,
            max_candidate_attempts=1,
            validation_interval_steps=5,
            validation_batches=2,
            mcts_worker_threads=2,
            precision="16-mixed" if args.device == "cuda" else "32-true",
        )
        results = {}
        for decision in ("accepted", "rejected"):
            results[decision] = training.run_async_training(
                config, arena_decision_override=lambda _, value=decision: value
            )
        assert results["accepted"]["accepted_champions"] == 1
        assert results["rejected"]["accepted_champions"] == 1
        if args.inject_oom:
            payload = torch.load(
                Path(temporary) / "learner.pt", map_location="cpu", weights_only=False
            )
            assert payload["effective_batch_size"] < config.training_batch_size
        monitor = UtilizationMonitor(include_children=True)
        monitor.start()
        started = time.monotonic()
        try:
            with monitor.phase("soak"):
                try:
                    training.run_async_training(
                        config.model_copy(update={"max_candidate_attempts": None}),
                        should_cancel=lambda: (
                            time.monotonic() - started >= args.seconds
                        ),
                    )
                except TrainingCancelled:
                    pass
        finally:
            monitor.stop()
        results["soak_seconds"] = time.monotonic() - started
        results["resume"] = training.run_async_training(config)
        results["utilization"] = monitor.summary()
        results["format"] = "c4a0-training-qualification"
        results["version"] = 1
        results["device"] = args.device
        results["torch"] = torch.__version__
        results["oom_injected"] = args.inject_oom
        results["forced_lifecycle_decisions"] = ["accepted", "rejected"]
        results["config"] = config.model_dump(mode="json")
        results["release_approved"] = False
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        print(args.output)


if __name__ == "__main__":
    main()
