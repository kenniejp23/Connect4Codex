import gc
import pickle
import threading
import time

import numpy as np
import pytest

import c4a0_cpp


def _uniform_eval(_model_id, positions):
    assert positions.dtype == np.float32
    assert positions.flags.c_contiguous
    assert positions.shape[1:] == (2, 6, 7)
    batch_size = positions.shape[0]
    return (
        np.zeros((batch_size, c4a0_cpp.N_COLS), dtype=np.float32),
        np.zeros((batch_size,), dtype=np.float32),
        np.zeros((batch_size,), dtype=np.float32),
    )


def _small_games(count=1):
    return c4a0_cpp.play_games(
        [c4a0_cpp.GameMetadata(game_id, 0, 0) for game_id in range(count)],
        16,
        2,
        1.4,
        0.01,
        _uniform_eval,
    )


def test_public_api_and_numpy_contract():
    expected_exports = {
        "BUF_N_CHANNELS",
        "N_COLS",
        "N_ROWS",
        "GameMetadata",
        "GameResult",
        "PlayGamesResult",
        "Sample",
        "play_games",
        "run_tui",
    }
    assert set(c4a0_cpp.__all__) == expected_exports
    assert expected_exports <= set(dir(c4a0_cpp))
    assert (c4a0_cpp.N_ROWS, c4a0_cpp.N_COLS, c4a0_cpp.BUF_N_CHANNELS) == (
        6,
        7,
        2,
    )
    assert c4a0_cpp.GameMetadata.__module__ == "c4a0_cpp"
    assert c4a0_cpp.Sample.__module__ == "c4a0_cpp"
    assert c4a0_cpp.GameResult.__module__ == "c4a0_cpp"
    assert c4a0_cpp.PlayGamesResult.__module__ == "c4a0_cpp"

    games = _small_games()
    sample = games.results[0].samples[0]
    position, policy, q_penalty, q_no_penalty = sample.to_numpy()
    assert position.shape == (2, 6, 7)
    assert policy.shape == (7,)
    assert q_penalty.shape == ()
    assert q_no_penalty.shape == ()
    assert all(
        value.dtype == np.float32
        for value in (position, policy, q_penalty, q_no_penalty)
    )
    assert position.flags.c_contiguous
    assert policy.flags.c_contiguous

    expected_position = position.copy()
    expected_policy = policy.copy()
    del sample
    del games
    gc.collect()
    np.testing.assert_array_equal(position, expected_position)
    np.testing.assert_array_equal(policy, expected_policy)

    metadata = c4a0_cpp.GameMetadata(1, 2, 3)
    for field in ("game_id", "player0_id", "player1_id"):
        with pytest.raises(AttributeError):
            setattr(metadata, field, 99)


def test_empty_requests_return_without_calling_callback():
    def unexpected_callback(*_args):
        raise AssertionError("empty self-play must not evaluate positions")

    result = c4a0_cpp.play_games(
        [], 0, 0, float("nan"), float("nan"), unexpected_callback
    )
    assert result.results == []


def test_one_iteration_self_play_never_samples_a_full_column():
    result = c4a0_cpp.play_games(
        [c4a0_cpp.GameMetadata(game_id, 0, 0) for game_id in range(32)],
        16,
        1,
        1.4,
        0.01,
        _uniform_eval,
    )
    assert len(result.results) == 32


@pytest.mark.parametrize(
    ("batch_size", "iterations", "exploration", "ply_penalty"),
    [
        (0, 2, 1.4, 0.01),
        (8, 0, 1.4, 0.01),
        (8, 2, float("nan"), 0.01),
        (8, 2, 1.4, -0.01),
    ],
)
def test_invalid_engine_arguments_are_rejected(
    batch_size, iterations, exploration, ply_penalty
):
    with pytest.raises(ValueError):
        c4a0_cpp.play_games(
            [c4a0_cpp.GameMetadata(0, 0, 0)],
            batch_size,
            iterations,
            exploration,
            ply_penalty,
            _uniform_eval,
        )


def test_callback_exception_is_propagated_without_hanging():
    class CallbackFailure(RuntimeError):
        pass

    def fail(_model_id, _positions):
        raise CallbackFailure("model failed")

    with pytest.raises(CallbackFailure, match="model failed"):
        c4a0_cpp.play_games([c4a0_cpp.GameMetadata(0, 0, 0)], 8, 2, 1.4, 0.01, fail)


@pytest.mark.parametrize(
    "failure",
    [
        "tuple",
        "dtype",
        "shape",
        "contiguous",
        "q_dtype",
        "q_shape",
        "q_contiguous",
        "nan",
        "positive_inf",
        "all_invalid",
    ],
)
def test_invalid_callback_outputs_are_rejected(failure):
    def invalid(_model_id, positions):
        batch_size = positions.shape[0]
        policy = np.zeros((batch_size, 7), dtype=np.float32)
        q_value = np.zeros((batch_size,), dtype=np.float32)
        if failure == "tuple":
            return policy, q_value
        if failure == "dtype":
            policy = policy.astype(np.float64)
        elif failure == "shape":
            policy = np.zeros((batch_size, 6), dtype=np.float32)
        elif failure == "contiguous":
            policy = np.zeros((batch_size, 14), dtype=np.float32)[:, ::2]
        elif failure == "q_dtype":
            q_value = q_value.astype(np.float64)
        elif failure == "q_shape":
            q_value = np.zeros((batch_size, 1), dtype=np.float32)
        elif failure == "q_contiguous" and batch_size > 1:
            q_value = np.zeros((batch_size * 2,), dtype=np.float32)[::2]
        elif failure == "nan":
            policy[0, 0] = np.nan
        elif failure == "positive_inf":
            policy[0, 0] = np.inf
        elif failure == "all_invalid":
            policy.fill(-np.inf)
        return policy, q_value, q_value

    with pytest.raises((TypeError, ValueError)):
        game_count = 8 if failure == "q_contiguous" else 1
        c4a0_cpp.play_games(
            [c4a0_cpp.GameMetadata(i, 0, 0) for i in range(game_count)],
            8,
            2,
            1.4,
            0.01,
            invalid,  # pyright: ignore[reportArgumentType]
        )


def test_another_python_thread_progresses_during_native_self_play():
    active = threading.Event()
    stop = threading.Event()
    progress = 0

    def worker():
        nonlocal progress
        active.wait()
        while not stop.is_set():
            progress += 1

    def slow_eval(model_id, positions):
        time.sleep(0.001)
        return _uniform_eval(model_id, positions)

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        before = progress
        active.set()
        result = c4a0_cpp.play_games(
            [c4a0_cpp.GameMetadata(0, 0, 0)],
            8,
            2,
            1.4,
            0.01,
            slow_eval,
        )
        after = progress
    finally:
        stop.set()
        active.set()
        thread.join(timeout=2)

    assert len(result.results) == 1
    assert after > before
    assert not thread.is_alive()


def test_result_cbor_pickle_addition_and_split_round_trips():
    games = _small_games(2)
    cbor_round_trip = c4a0_cpp.PlayGamesResult.from_cbor(games.to_cbor())
    pickle_round_trip = pickle.loads(pickle.dumps(games))
    assert [game.metadata.game_id for game in cbor_round_trip.results] == [
        game.metadata.game_id for game in games.results
    ]
    assert [sample.pos_str() for sample in pickle_round_trip.results[0].samples] == [
        sample.pos_str() for sample in games.results[0].samples
    ]

    combined = games + cbor_round_trip
    assert len(combined.results) == 4
    train, test = games.split_train_test(0.5, 1337)
    assert train
    assert test
    assert len(train) + len(test) == sum(
        len(game.samples) for game in games.results
    )
    with pytest.raises(ValueError):
        games.split_train_test(1.1, 1337)
    with pytest.raises(ValueError):
        c4a0_cpp.PlayGamesResult.from_cbor(b"not CBOR")
