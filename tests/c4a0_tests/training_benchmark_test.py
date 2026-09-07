import pytest

from c4a0.training_benchmark import (
    BENCHMARK_FORMAT,
    compare_training_benchmarks,
)


def _report(workflow, scale=1.0, solver=True):
    return {
        "format": BENCHMARK_FORMAT,
        "version": 1,
        "workflow": workflow,
        "revision": workflow,
        "config": {
            "seed": 1337,
            "device": "cpu",
            "precision": "32-true",
            "games": 64,
            "arena_games": 32,
            "mcts_iterations": 64,
            "inference_batch_size": 64,
            "training_batch_size": 128,
            "training_steps": 25,
        },
        "metrics": {
            "games_per_second": 10 * scale,
            "train_positions_per_second": 100 * scale,
            "candidate_latency_seconds": 10 / scale,
            "arena_games_per_second": 8 * scale,
            "solver_score_per_wall_clock_hour": 2 * scale if solver else None,
        },
    }


def test_replacement_gate_passes_only_complete_comparable_reports():
    comparison = compare_training_benchmarks(
        _report("v2", 1.0), _report("legacy-rust", 1.0)
    )
    assert comparison["replacement_approved"]

    missing_solver = compare_training_benchmarks(
        _report("v2", 1.0, solver=False), _report("legacy-rust", 1.0)
    )
    assert not missing_solver["replacement_approved"]


def test_replacement_gate_rejects_mismatched_workloads():
    legacy = _report("legacy-rust")
    legacy["config"]["games"] = 32
    with pytest.raises(ValueError, match="different fixed workloads"):
        compare_training_benchmarks(_report("v2"), legacy)


@pytest.mark.parametrize(
    "defect", ["version", "workload", "infinite", "negative", "score"]
)
def test_invalid_reports_never_approve(defect):
    report = _report("v2")
    if defect == "version":
        report["version"] = 99
    elif defect == "workload":
        report["config"] = {}
    elif defect == "infinite":
        report["metrics"]["games_per_second"] = float("inf")
    elif defect == "negative":
        report["metrics"]["candidate_latency_seconds"] = -1
    else:
        report["metrics"]["solver_score"] = 1.1
    with pytest.raises(ValueError):
        compare_training_benchmarks(report, _report("legacy-rust"))
