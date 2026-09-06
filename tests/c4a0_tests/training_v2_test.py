import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError
import torch

import c4a0_cpp
from c4a0.config import TrainingV2Config
from c4a0.nn import ConnectFourNet, ModelConfig
from c4a0.training_v2 import (
    RANDOM_MODEL_ID,
    ReplayPool,
    UNIFORM_MODEL_ID,
    _ActorEvaluator,
    RunManifest,
    SprtGate,
    _arena_openings,
    _atomic_torch_save,
    _baseline_kind_and_game,
    _baseline_model_id,
    _checkpoint_payload,
    _league_requests,
    _opening_key,
    _recover_orphaned_candidates,
    _run_arena,
    _validation_game,
    load_champion_model,
    run_async_training,
)


def _tiny_config(tmp_path: Path, **overrides):
    values = {
        "base_dir": str(tmp_path),
        "device": "cpu",
        "self_play_shard_games": 4,
        "self_play_batch_games": 4,
        "replay_capacity_games": 8,
        "replay_warmup_games": 4,
        "training_batch_size": 2,
        "inference_batch_size": 4,
        "n_mcts_iterations": 1,
        "arena_min_games": 2,
        "arena_max_games": 2,
        "n_residual_blocks": 0,
        "conv_filter_size": 4,
        "n_policy_layers": 1,
        "n_value_layers": 1,
        "lr_schedule": [0, 0.001],
        "l2_reg": 0,
        "mcts_worker_threads": 1,
    }
    values.update(overrides)
    return TrainingV2Config(**values)


def test_v2_config_validates_cross_field_invariants(tmp_path):
    assert _tiny_config(tmp_path).base_dir == str(tmp_path)
    with pytest.raises(ValidationError, match="opponent weights"):
        _tiny_config(tmp_path, champion_self_play_weight=0.5)
    with pytest.raises(ValidationError, match="warmup"):
        _tiny_config(tmp_path, replay_warmup_games=8)
    with pytest.raises(ValidationError, match="must be even"):
        _tiny_config(tmp_path, arena_min_games=3)
    with pytest.raises(ValidationError, match="less than or equal to 2000"):
        _tiny_config(tmp_path, arena_max_games=2002)
    with pytest.raises(ValidationError, match="resume watermark"):
        _tiny_config(tmp_path, debt_pause_shards=4, debt_resume_shards=4)
    with pytest.raises(ValidationError, match="whole replay shards"):
        _tiny_config(tmp_path, self_play_batch_games=6)


def test_sprt_gate_waits_for_minimum_and_rejects_inconclusive():
    accepting = SprtGate(0.5, 0.55, 0.05, 0.05, 40, 200)
    for _ in range(39):
        accepting.update(1.0)
    assert accepting.decision() is None
    accepting.update(1.0)
    assert accepting.decision() == "accepted"

    rejecting = SprtGate(0.5, 0.55, 0.05, 0.05, 40, 200)
    for _ in range(40):
        rejecting.update(0.0)
    assert rejecting.decision() == "rejected"

    draw = SprtGate(0.5, 0.55, 0.05, 0.05, 2, 200)
    draw.update(0.5)
    assert draw.games == 1
    assert draw.llr == pytest.approx(
        0.5 * np.log(0.55 / 0.5) + 0.5 * np.log(0.45 / 0.5)
    )

    inconclusive = SprtGate(0.5, 0.55, 0.05, 0.05, 2, 4)
    for score in (1.0, 0.0, 0.5, 0.5):
        inconclusive.update(score)
    assert inconclusive.decision() == "rejected"
    assert not inconclusive.crossed_boundary


def test_arena_openings_are_unique_mirrored_and_nonempty(tmp_path):
    openings = _arena_openings(tmp_path, 1337)
    assert len(openings) == 100
    assert len({tuple(opening) for opening in openings}) == 100
    assert len({_opening_key(opening) for opening in openings}) == 100
    assert all(1 <= len(opening) <= 6 for opening in openings)
    values = {tuple(opening) for opening in openings}
    assert all(tuple(6 - move for move in opening) in values for opening in openings)
    extended = _arena_openings(tmp_path, 1337, 800)
    assert len(extended) == 400
    assert len({_opening_key(opening) for opening in extended}) == 400


def test_league_allocation_is_balanced_and_bootstraps_without_archive(tmp_path):
    config = _tiny_config(
        tmp_path, self_play_shard_games=40, self_play_batch_games=40
    )
    requests = _league_requests(config, 7, [], 1)
    assert len(requests) == 40
    assert all(
        _baseline_kind_and_game(int(request.metadata.player0_id))
        != (RANDOM_MODEL_ID, int(request.metadata.game_id))
        or request.metadata.player1_id == 7
        for request in requests
    )
    baselines = [
        request
        for request in requests
        if _baseline_kind_and_game(int(request.metadata.player0_id)) is not None
        or _baseline_kind_and_game(int(request.metadata.player1_id)) is not None
    ]
    assert len(baselines) == 2
    baseline_kinds = set()
    for request in baselines:
        baseline = _baseline_kind_and_game(
            int(
                request.metadata.player0_id
                if request.metadata.player0_id != 7
                else request.metadata.player1_id
            )
        )
        assert baseline is not None
        baseline_kinds.add(baseline[0])
    assert baseline_kinds == {RANDOM_MODEL_ID, UNIFORM_MODEL_ID}


def test_league_balances_champion_colour_within_each_category(tmp_path):
    config = _tiny_config(
        tmp_path, self_play_shard_games=40, self_play_batch_games=40
    )
    requests = _league_requests(config, 7, [5, 4], 100)
    colours: dict[str, list[bool]] = {}
    for request in requests:
        if (
            int(request.metadata.player0_id) == 7
            and int(request.metadata.player1_id) == 7
        ):
            continue
        opponent = (
            int(request.metadata.player1_id)
            if int(request.metadata.player0_id) == 7
            else int(request.metadata.player0_id)
        )
        baseline = _baseline_kind_and_game(opponent)
        category = "archive" if baseline is None and opponent != 7 else str(baseline)
        colours.setdefault(category, []).append(int(request.metadata.player0_id) == 7)
    assert all(
        abs(sum(values) - (len(values) - sum(values))) <= 1
        for values in colours.values()
    )


def test_random_baseline_is_stable_for_run_game_and_position():
    evaluator = _ActorEvaluator(torch.device("cpu"), 1337)
    positions = np.zeros((2, 2, 6, 7), dtype=np.float32)
    positions[1, 0, 0, 0] = 1.0
    model_id = _baseline_model_id(RANDOM_MODEL_ID, 19)
    first = evaluator(model_id, positions)
    repeated = evaluator(model_id, positions[::-1].copy())
    assert np.array_equal(first[0][0], repeated[0][1])
    assert np.array_equal(first[0][1], repeated[0][0])
    changed_game = evaluator(_baseline_model_id(RANDOM_MODEL_ID, 20), positions)
    assert not np.array_equal(first[0], changed_game[0])


def test_actor_value_scale_can_disable_uncalibrated_search_values():
    evaluator = _ActorEvaluator(torch.device("cpu"), 1337, value_scale=0.0)
    positions = np.zeros((2, 2, 6, 7), dtype=np.float32)
    policy, q_penalty, q_no_penalty = evaluator(RANDOM_MODEL_ID, positions)
    assert np.any(policy != 0.0)
    assert np.all(q_penalty == 0.0)
    assert np.all(q_no_penalty == 0.0)


def test_random_league_uses_one_vectorizable_model_id(tmp_path):
    config = _tiny_config(
        tmp_path, self_play_shard_games=80, self_play_batch_games=80
    )
    requests = _league_requests(config, 7, [6], 100)
    random_ids = []
    for request in requests:
        for model_id in (
            int(request.metadata.player0_id),
            int(request.metadata.player1_id),
        ):
            baseline = _baseline_kind_and_game(model_id)
            if baseline is not None and baseline[0] == RANDOM_MODEL_ID:
                random_ids.append(model_id)
    assert len(random_ids) == 2
    assert set(random_ids) == {RANDOM_MODEL_ID}


def test_arena_evaluates_multiple_pairs_in_one_native_batch(tmp_path, monkeypatch):
    config = _tiny_config(
        tmp_path,
        arena_min_games=8,
        arena_max_games=8,
        arena_pair_batch_size=4,
    )
    evaluator = _ActorEvaluator(torch.device("cpu"), config.run_seed)
    evaluator.configure = lambda _entries: None  # type: ignore[method-assign]
    batch_sizes = []

    class Result:
        def __init__(self, metadata):
            self.metadata = metadata

        def player0_score(self):
            return 0.5

    def fake_play(requests, *_args):
        batch_sizes.append(len(requests))
        return SimpleNamespace(results=[Result(item.metadata) for item in requests])

    monkeypatch.setattr(c4a0_cpp, "play_games_v2", fake_play)
    result = _run_arena(
        config,
        evaluator,
        {"attempt_n": 0, "checkpoint_path": "champion"},
        {"attempt_n": 1, "checkpoint_path": "candidate"},
        [[0], [6], [1], [5]],
        SimpleNamespace(is_set=lambda: False),
    )
    assert batch_sizes == [8]
    assert result["games"] == 8


def test_manifest_promotes_only_accepted_candidates(tmp_path):
    config = _tiny_config(tmp_path)
    manifest = RunManifest.create(tmp_path, config)
    model = ConnectFourNet(
        ModelConfig(
            n_residual_blocks=0,
            conv_filter_size=4,
            n_policy_layers=1,
            n_value_layers=1,
            lr_schedule={0: 0.001},
            l2_reg=0,
        )
    )
    root = tmp_path / "attempts/000000/model.pt"
    _atomic_torch_save(root, _checkpoint_payload(model))
    manifest.add_root(root)
    candidate = tmp_path / "attempts/000001/model.pt"
    _atomic_torch_save(candidate, _checkpoint_payload(model))
    manifest.add_pending_attempt(1, 0, candidate, 1.0, 1, 2, 4)
    assert not manifest.decide_attempt(1, {"decision": "rejected"})
    assert manifest.champion()["attempt_n"] == 0

    second = tmp_path / "attempts/000002/model.pt"
    _atomic_torch_save(second, _checkpoint_payload(model))
    manifest.add_pending_attempt(2, 0, second, 0.5, 2, 4, 4)
    assert manifest.decide_attempt(2, {"decision": "accepted"})
    assert manifest.champion()["attempt_n"] == 2
    assert isinstance(load_champion_model(str(tmp_path)), ConnectFourNet)


def test_v2_refuses_and_preserves_an_unmarked_legacy_directory(tmp_path):
    legacy = tmp_path / "model.pkl"
    legacy.write_bytes(b"legacy-model")
    with pytest.raises(FileExistsError, match="empty or V2"):
        RunManifest.create(tmp_path, _tiny_config(tmp_path))
    assert legacy.read_bytes() == b"legacy-model"
    assert not (tmp_path / "run.sqlite3").exists()


def test_manifest_retires_oldest_whole_shards_at_capacity(tmp_path):
    config = _tiny_config(tmp_path)
    manifest = RunManifest.create(tmp_path, config)
    model = ConnectFourNet(
        ModelConfig(
            n_residual_blocks=0,
            conv_filter_size=4,
            n_policy_layers=1,
            n_value_layers=1,
            lr_schedule={0: 0.001},
            l2_reg=0,
        )
    )
    root = tmp_path / "attempts/000000/model.pt"
    _atomic_torch_save(root, _checkpoint_payload(model))
    manifest.add_root(root)
    retired = []
    for index in range(3):
        _, retired = manifest.register_shard(
            {
                "path": str(tmp_path / f"{index}.cbor"),
                "games": 4,
                "samples": 20,
                "train_samples": 16,
                "champion_attempt": 0,
                "first_game_id": 1 + index * 4,
            }
        )
    assert [item["shard_id"] for item in manifest.active_shards()] == [2, 3]
    assert [item["shard_id"] for item in retired] == [1]
    assert manifest.status()["replay_games"] == 8


def test_manifest_recovers_fully_written_crash_candidate_as_pending(tmp_path):
    config = _tiny_config(tmp_path)
    manifest = RunManifest.create(tmp_path, config)
    model = ConnectFourNet(
        ModelConfig(
            n_residual_blocks=0,
            conv_filter_size=4,
            n_policy_layers=1,
            n_value_layers=1,
            lr_schedule={0: 0.001},
            l2_reg=0,
        )
    )
    root = tmp_path / "attempts/000000/model.pt"
    _atomic_torch_save(root, _checkpoint_payload(model))
    manifest.add_root(root)
    _atomic_torch_save(
        tmp_path / ".candidate.pt",
        _checkpoint_payload(
            model,
            champion_attempt=0,
            val_loss=0.5,
            training_steps=2,
            training_samples=4,
            fresh_games=4,
        ),
    )
    _recover_orphaned_candidates(manifest)
    row = manifest.connection.execute(
        "SELECT status FROM attempts WHERE attempt_n = 1"
    ).fetchone()
    assert row["status"] == "pending"
    assert not (tmp_path / ".candidate.pt").exists()


def _native_games(requests):
    def uniform(_model_id, positions):
        batch = positions.shape[0]
        return (
            np.zeros((batch, 7), dtype=np.float32),
            np.zeros(batch, dtype=np.float32),
            np.zeros(batch, dtype=np.float32),
        )

    options = c4a0_cpp.SelfPlayOptions()
    options.max_nn_batch_size = 8
    options.n_mcts_iterations = 1
    options.c_exploration = 1.0
    options.c_ply_penalty = 0.01
    options.temperature_midpoint_ply = 8
    options.temperature_cutoff_ply = 8
    options.early_temperature = 1.0
    options.middle_temperature = 1.0
    options.late_temperature = 0.0
    options.seed = 17
    options.worker_threads = 1
    return c4a0_cpp.play_games_v2(requests, options, uniform)


def test_replay_partition_sampling_recency_and_head_weights(tmp_path):
    config = _tiny_config(tmp_path, validation_fraction=0.1)
    train_game_ids = [
        game_id
        for game_id in range(1, 100)
        if not _validation_game(config.run_seed, game_id, config.validation_fraction)
    ][:2]
    games = _native_games(
        [
            c4a0_cpp.GameRequest(c4a0_cpp.GameMetadata(train_game_ids[0], 7, 7), []),
            c4a0_cpp.GameRequest(c4a0_cpp.GameMetadata(train_game_ids[1], 7, 5), []),
        ]
    )
    shard_path = tmp_path / "replay.cbor"
    shard_path.write_bytes(games.to_cbor())
    shard = {
        "shard_id": 1,
        "path": str(shard_path),
        "games": 2,
        "samples": sum(len(game.samples) for game in games.results),
        "train_samples": sum(len(game.samples) for game in games.results),
        "champion_attempt": 7,
    }
    pool = ReplayPool(config, [shard])
    batch = pool.batch(512, torch.device("cpu"), False)
    policy_weights = set(batch[4].tolist())
    assert policy_weights == {0.0, 1.0}
    assert torch.all(batch[5] == config.value_loss_weight)
    assert len(pool.cache) == 1

    fake_shards = [
        {"shard_id": index, "games": 1, "samples": 1, "path": str(index)}
        for index in range(1, 9)
    ]
    recency_pool = ReplayPool(config, fake_shards)
    newest = sum(
        int(recency_pool._choose_shard()["shard_id"] >= 7) for _ in range(4000)
    )
    assert 0.59 < newest / 4000 < 0.66


def test_validation_assignment_is_stable_by_seed_and_game_id():
    first = [_validation_game(1337, game_id, 0.05) for game_id in range(1, 500)]
    assert first == [_validation_game(1337, game_id, 0.05) for game_id in range(1, 500)]
    assert first != [_validation_game(1338, game_id, 0.05) for game_id in range(1, 500)]


def test_v2_native_openings_and_temperature_options():
    def uniform(_model_id, positions):
        batch = positions.shape[0]
        return (
            np.zeros((batch, 7), dtype=np.float32),
            np.zeros(batch, dtype=np.float32),
            np.zeros(batch, dtype=np.float32),
        )

    options = c4a0_cpp.SelfPlayOptions()
    options.max_nn_batch_size = 4
    options.n_mcts_iterations = 1
    options.c_exploration = 1.0
    options.c_ply_penalty = 0.01
    options.temperature_midpoint_ply = 8
    options.temperature_cutoff_ply = 8
    options.early_temperature = 1.0
    options.middle_temperature = 1.0
    options.late_temperature = 0.0
    options.seed = 9
    result = c4a0_cpp.play_games_v2(
        [c4a0_cpp.GameRequest(c4a0_cpp.GameMetadata(1, 3, 7), [3, 2])],
        options,
        uniform,
    )
    assert len(result.results) == 1
    assert result.results[0].metadata.player0_id == 3
    with pytest.raises(ValueError, match="non-terminal"):
        c4a0_cpp.play_games_v2(
            [
                c4a0_cpp.GameRequest(
                    c4a0_cpp.GameMetadata(2, 0, 0), [0, 1, 0, 1, 0, 1, 0]
                )
            ],
            options,
            uniform,
        )


def test_async_worker_completes_after_one_candidate_decision(tmp_path):
    config = _tiny_config(
        tmp_path,
        self_play_batch_games=8,
        replay_ratio=0.03,
        validation_fraction=0.25,
        validation_interval_steps=1,
        validation_batches=1,
        debt_pause_shards=2,
        debt_resume_shards=1,
    )
    request = {
        "version": 1,
        "kind": "training",
        "config": config.model_dump(mode="json"),
    }
    controls = (
        json.dumps(request)
        + "\n"
        + json.dumps({"command": "stop_after_generation"})
        + "\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "c4a0.worker"],
        input=controls,
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert result.returncode == 0, result.stdout + result.stderr
    assert events[-1]["type"] == "completed"
    assert any(
        event["type"] == "phase" and event.get("name") == "self_play"
        for event in events
    )
    assert any(
        event["type"] == "phase" and event.get("name") == "arena" for event in events
    )
    assert events[-1]["result"]["replay_games"] <= config.replay_capacity_games
    assert events[-1]["result"]["components"]["actor"]["champion_attempt"] == 0
    assert events[-1]["result"]["components"]["trainer"]["champion_attempt"] == 0
    assert (tmp_path / "run.sqlite3").is_file()
    assert not any("solver" in line.lower() for line in result.stdout.splitlines())


def test_async_pipeline_forced_acceptance_reloads_champion(tmp_path):
    config = _tiny_config(
        tmp_path,
        self_play_batch_games=8,
        replay_ratio=0.03,
        validation_fraction=0.25,
        validation_interval_steps=1,
        validation_batches=1,
        debt_pause_shards=2,
        debt_resume_shards=1,
        max_candidate_attempts=1,
    )
    status = run_async_training(
        config, arena_decision_override=lambda _result: "accepted"
    )
    assert status["champion"] == 1
    assert status["accepted_champions"] == 1
    assert status["components"]["actor"]["champion_attempt"] == 1
    assert status["components"]["trainer"]["champion_attempt"] == 1
    manifest = RunManifest.open_existing(tmp_path)
    assert manifest.champion()["status"] == "accepted"
    assert manifest.accepted_models(8)[0]["attempt_n"] == 0
    shards = manifest.active_shards()
    assert [int(shard["games"]) for shard in shards] == [4, 4]
    first_ids = [int(shard["first_game_id"]) for shard in shards]
    assert first_ids[1] == first_ids[0] + 4


def test_async_pipeline_emits_candidate_at_replay_resume_watermark(tmp_path):
    """Normal replay weighting must not prevent a candidate from being emitted."""
    config = _tiny_config(
        tmp_path,
        replay_ratio=4.0,
        replay_capacity_games=20,
        replay_warmup_games=12,
        validation_interval_steps=1000,
        debt_pause_shards=4,
        debt_resume_shards=2,
        max_candidate_attempts=1,
    )
    status = run_async_training(
        config, arena_decision_override=lambda _result: "rejected"
    )
    manifest = RunManifest.open_existing(tmp_path)
    candidate = manifest.connection.execute(
        "SELECT * FROM attempts WHERE attempt_n = 1"
    ).fetchone()

    assert candidate is not None
    assert candidate["status"] == "rejected"
    assert int(candidate["submitted_fresh_games"]) >= config.replay_warmup_games
    assert status["pending_candidate"] is None
