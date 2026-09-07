"""Paired-outcome generalized SPRT with constrained multinomial likelihoods.

Each observation is the mean score of two games sharing one opening, with
colors exchanged. Error rates require empirical calibration for the chosen
opening distribution and batch schedule; truncation rejects inconclusive runs.
"""

import math
import numpy as np


def paired_log_likelihood(counts: np.ndarray, mean: float) -> float:
    support = np.arange(5) / 4
    delta = support - mean
    # Small regularization keeps the constrained MLE defined at sparse endpoints.
    frequency = (counts + 1e-6) / (counts.sum() + 5e-6)
    lower = -1 / delta.max() + 1e-12
    upper = -1 / delta.min() - 1e-12
    for _ in range(48):
        multiplier = (lower + upper) / 2
        derivative = np.sum(frequency * delta / (1 + multiplier * delta))
        if derivative > 0:
            lower = multiplier
        else:
            upper = multiplier
    probability = frequency / (1 + ((lower + upper) / 2) * delta)
    return float(np.dot(counts, np.log(probability)))


class PairedSprtGate:
    def __init__(self, p0, p1, alpha, beta, minimum_games, maximum_games):
        if not (0 < p0 < p1 < 1 and 0 < alpha < 0.5 and 0 < beta < 0.5):
            raise ValueError("Invalid paired SPRT hypotheses or error thresholds")
        if (
            minimum_games < 2
            or minimum_games % 2
            or maximum_games % 2
            or maximum_games < minimum_games
        ):
            raise ValueError("Paired SPRT requires even, ordered game limits")
        self.p0, self.p1 = p0, p1
        self.minimum_games, self.maximum_games = minimum_games, maximum_games
        self.upper = math.log((1 - beta) / alpha)
        self.lower = math.log(beta / (1 - alpha))
        self.counts = np.zeros(5, dtype=np.int64)
        self.games = 0
        self.llr = 0.0
        self.crossed_boundary = False
        self._first = None

    def update(self, score):
        if score not in (0.0, 0.5, 1.0):
            raise ValueError("Arena scores must be win, draw, or loss")
        self.games += 1
        if self._first is None:
            self._first = score
        else:
            self.counts[int(round(2 * (self._first + score)))] += 1
            self._first = None

    def decision(self):
        if self._first is not None:
            return None
        self.llr = paired_log_likelihood(self.counts, self.p1) - paired_log_likelihood(
            self.counts, self.p0
        )
        if self.games >= self.minimum_games:
            if self.llr >= self.upper:
                self.crossed_boundary = True
                return "accepted"
            if self.llr <= self.lower:
                self.crossed_boundary = True
                return "rejected"
        return "rejected" if self.games >= self.maximum_games else None
