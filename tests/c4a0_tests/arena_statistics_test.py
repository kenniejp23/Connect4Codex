import numpy as np
import pytest

from c4a0.arena_statistics import PairedSprtGate, paired_log_likelihood
from c4a0.production_benchmark import held_out_openings
from c4a0.training_v2 import _arena_openings, _opening_key


def test_pair_identity_changes_likelihood_despite_equal_marginal_score():
    low_variance = np.array([0, 0, 45, 10, 0])
    high_variance = np.array([20, 0, 10, 0, 25])

    def llr(counts):
        return paired_log_likelihood(counts, 0.55) - paired_log_likelihood(counts, 0.5)

    assert llr(low_variance) > llr(high_variance)
    gate = PairedSprtGate(0.5, 0.55, 0.05, 0.05, 2, 800)
    gate.update(1.0)
    assert gate.decision() is None
    gate.update(0.0)
    assert gate.counts.tolist() == [0, 0, 1, 0, 0]


@pytest.mark.parametrize(
    "distribution",
    [
        [0.5, 0, 0, 0, 0.5],
        [0.25, 0, 0.5, 0, 0.25],
        [0, 0.1, 0.8, 0.1, 0],
        [0, 0, 1, 0, 0],
    ],
)
def test_null_calibration_with_draws_pair_correlation_and_batch_boundaries(
    distribution,
):
    rng = np.random.default_rng(924)
    accepted = 0
    trials = 400
    for trial in range(trials):
        gate = PairedSprtGate(0.5, 0.55, 0.05, 0.05, 40, 800)
        outcomes = rng.choice(5, size=400, p=distribution)
        batch_pairs = (1, 10, 100)[trial % 3]
        for start in range(0, 400, batch_pairs):
            batch = outcomes[start : start + batch_pairs]
            gate.counts += np.bincount(batch, minlength=5)
            gate.games += 2 * len(batch)
            decision = gate.decision()
            if decision is not None:
                accepted += decision == "accepted"
                break
    # Finite Monte Carlo tolerance; this is not a claim about arbitrary openings.
    assert accepted / trials <= 0.08


def test_strength_bank_is_fixed_and_disjoint_from_selection(tmp_path):
    first = held_out_openings()
    assert first == held_out_openings()
    selection = _arena_openings(tmp_path, 1337, 800)
    assert not {_opening_key(item) for item in first} & {
        _opening_key(item) for item in selection
    }


@pytest.mark.parametrize(
    "distribution",
    [[0.45, 0, 0, 0, 0.55], [0.2025, 0, 0.495, 0, 0.3025], [0, 0.04, 0.72, 0.24, 0]],
)
def test_alternative_boundary_error_separates_budget_truncation(distribution):
    rng = np.random.default_rng(931)
    rejected_at_boundary = 0
    for trial in range(400):
        gate = PairedSprtGate(0.5, 0.55, 0.05, 0.05, 40, 800)
        outcomes = rng.choice(5, size=400, p=distribution)
        batch_pairs = (10, 100)[trial % 2]
        for start in range(0, 400, batch_pairs):
            batch = outcomes[start : start + batch_pairs]
            gate.counts += np.bincount(batch, minlength=5)
            gate.games += 2 * len(batch)
            decision = gate.decision()
            if decision is not None:
                rejected_at_boundary += decision == "rejected" and gate.crossed_boundary
                break
    assert rejected_at_boundary / 400 <= 0.08
