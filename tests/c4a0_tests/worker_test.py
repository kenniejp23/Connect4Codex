import json
import subprocess
import sys


def test_worker_protocol_reports_structured_failure():
    request = {"version": 1, "kind": "does_not_exist", "config": {}}
    result = subprocess.run(
        [sys.executable, "-m", "c4a0.worker"],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert result.returncode == 1
    assert events[-1]["version"] == 1
    assert events[-1]["type"] == "failed"
    assert "unsupported job type" in events[-1]["message"]


def test_worker_tournament_reports_rankings_and_matchups():
    request = {
        "version": 1,
        "kind": "tournament",
        "config": {
            "players": ["random", "uniform"],
            "device": "cpu",
            "games_per_match": 2,
            "batch_size": 8,
            "mcts_iterations": 1,
            "exploration_constant": 1.0,
            "c_ply_penalty": 0.01,
        },
    }
    result = subprocess.run(
        [sys.executable, "-m", "c4a0.worker"],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    events = [json.loads(line) for line in result.stdout.splitlines()]
    completed = events[-1]
    assert result.returncode == 0
    assert completed["type"] == "completed"
    assert completed["result"]["games"] == 2
    assert len(completed["result"]["ranking"]) == 2
    assert len(completed["result"]["matchups"]) == 2
    assert all(
        row["wins"] + row["draws"] + row["losses"] == 2
        for row in completed["result"]["ranking"]
    )
