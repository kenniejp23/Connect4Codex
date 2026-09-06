"""Small primitives shared by legacy and V2 training."""

from __future__ import annotations


class TrainingCancelled(RuntimeError):
    """Raised when a cooperative training cancellation is requested."""


def parse_lr_schedule(values: list[float]) -> dict[int, float]:
    """Parse generation/rate pairs into a learning-rate schedule."""
    if len(values) % 2:
        raise ValueError("learning-rate schedule must contain pairs")
    schedule: dict[int, float] = {}
    for index in range(0, len(values), 2):
        threshold = int(values[index])
        if threshold != values[index]:
            raise ValueError("learning-rate thresholds must be integers")
        schedule[threshold] = values[index + 1]
    return schedule
