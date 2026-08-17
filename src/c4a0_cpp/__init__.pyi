from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

N_COLS: int
N_ROWS: int
BUF_N_CHANNELS: int
__all__: list[str]

class GameMetadata:
    def __init__(self, game_id: int, player0_id: int, player1_id: int) -> None: ...
    @property
    def game_id(self) -> int: ...
    @property
    def player0_id(self) -> int: ...
    @property
    def player1_id(self) -> int: ...

class Sample:
    def flip_h(self) -> Sample: ...
    def to_numpy(
        self,
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
    ]: ...
    def pos_str(self) -> str: ...

class GameResult:
    @property
    def metadata(self) -> GameMetadata: ...
    @property
    def samples(self) -> list[Sample]: ...
    def player0_score(self) -> float: ...

class PlayGamesResult:
    def __init__(self) -> None: ...
    @property
    def results(self) -> list[GameResult]: ...
    def __add__(self, other: PlayGamesResult) -> PlayGamesResult: ...
    def split_train_test(
        self, train_frac: float, seed: int
    ) -> tuple[list[Sample], list[Sample]]: ...
    def score_policies(
        self,
        solver_path: str,
        solver_book_path: str,
        solution_cache_path: str,
    ) -> float: ...
    def unique_positions(self) -> int: ...
    def to_cbor(self) -> bytes: ...
    @staticmethod
    def from_cbor(data: bytes) -> PlayGamesResult: ...

EvalCallback = Callable[
    [int, NDArray[np.float32]],
    tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]],
]

def play_games(
    reqs: Sequence[GameMetadata],
    max_nn_batch_size: int,
    n_mcts_iterations: int,
    c_exploration: float,
    c_ply_penalty: float,
    py_eval_pos_cb: EvalCallback,
) -> PlayGamesResult: ...
def run_tui(
    py_eval_pos_cb: EvalCallback,
    max_mcts_iters: int,
    c_exploration: float,
    c_ply_penalty: float,
    auto_red: bool = False,
    auto_blue: bool = False,
) -> None: ...
