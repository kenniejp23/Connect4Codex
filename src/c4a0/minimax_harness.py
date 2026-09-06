"""Batched strength testing against random play and depth-limited minimax."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Callable

import numpy as np
import torch

import c4a0_cpp
from c4a0.nn import ConnectFourNet


MODEL_ID = 0
OPPONENT_ID = 1
_CENTER_FIRST = (3, 2, 4, 1, 5, 0, 6)
_COLUMN_MASKS = tuple(sum(1 << (row * 7 + col) for row in range(6)) for col in range(7))


@dataclass(frozen=True)
class MinimaxHarnessConfig:
    """Parameters for a model-versus-baseline strength ladder."""

    games_per_level: int = 20
    pair_batch_size: int = 6
    max_depth: int = 42
    mcts_iterations: int = 64
    inference_batch_size: int = 128
    c_exploration: float = 1.4
    c_ply_penalty: float = 0.01
    worker_threads: int = 0
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.games_per_level <= 0 or self.games_per_level % 2:
            raise ValueError("games_per_level must be a positive even number")
        if self.pair_batch_size <= 0:
            raise ValueError("pair_batch_size must be positive")
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least one")
        if self.mcts_iterations < 1 or self.inference_batch_size < 1:
            raise ValueError("MCTS and inference batch sizes must be positive")

    @property
    def promotion_points_threshold(self) -> float:
        """Strict promotion boundary; a draw counts as half a point."""
        return self.games_per_level / 2


@dataclass(frozen=True)
class LadderLevelResult:
    opponent: str
    depth: int | None
    games: int
    wins: int
    draws: int
    losses: int
    points: float
    promoted: bool
    auto_promoted: bool
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MinimaxHarnessResult:
    levels: list[LadderLevelResult]
    stopped_at: str | None
    promotion_points_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "levels": [level.to_dict() for level in self.levels],
            "stopped_at": self.stopped_at,
            "promotion_points_threshold": self.promotion_points_threshold,
            "highest_defeated": self.levels[-2].opponent
            if self.stopped_at is not None and len(self.levels) > 1
            else (self.levels[-1].opponent if self.stopped_at is None and self.levels else None),
        }


def _has_four(bits: int) -> bool:
    for shift in (1, 6, 7, 8):
        paired = bits & (bits >> shift)
        if paired & (paired >> (2 * shift)):
            return True
    return False


def _legal_moves(occupied: int) -> list[int]:
    return [col for col in _CENTER_FIRST if not occupied & (1 << (35 + col))]


def _move_bit(occupied: int, col: int) -> int:
    filled = occupied & _COLUMN_MASKS[col]
    return 1 << ((filled.bit_count() * 7) + col)


class _MinimaxEvaluator:
    """Negamax evaluator operating on relative player-to-move bitboards."""

    def __init__(self, depth: int):
        self.depth = depth
        self.cache: dict[tuple[int, int, int], int] = {}

    def _score(self, current: int, opponent: int, depth: int) -> int:
        key = (current, opponent, depth)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        occupied = current | opponent
        legal = _legal_moves(occupied)
        if not legal:
            return 0
        best = -1
        for col in legal:
            after_move = current | _move_bit(occupied, col)
            if _has_four(after_move):
                best = 1
                break
            if depth > 1:
                best = max(best, -self._score(opponent, after_move, depth - 1))
            else:
                best = max(best, 0)
            if best == 1:
                break
        self.cache[key] = best
        return best

    @staticmethod
    def _bitboards(positions: np.ndarray, row: int) -> tuple[int, int]:
        current = opponent = 0
        for board_row in range(6):
            for col in range(7):
                bit = 1 << (board_row * 7 + col)
                if positions[row, 0, board_row, col] != 0:
                    current |= bit
                elif positions[row, 1, board_row, col] != 0:
                    opponent |= bit
        return current, opponent

    def __call__(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        policies = np.full((len(positions), 7), -1.0e9, dtype=np.float32)
        values = np.zeros(len(positions), dtype=np.float32)
        for row in range(len(positions)):
            current, opponent = self._bitboards(positions, row)
            occupied = current | opponent
            best = -2
            best_moves: list[int] = []
            for col in _legal_moves(occupied):
                after_move = current | _move_bit(occupied, col)
                score = 1 if _has_four(after_move) else (
                    -self._score(opponent, after_move, self.depth - 1)
                    if self.depth > 1
                    else 0
                )
                if score > best:
                    best, best_moves = score, [col]
                elif score == best:
                    best_moves.append(col)
            policies[row, best_moves] = 0.0
            values[row] = best if best != -2 else 0.0
        return policies, values, values.copy()


def _random_evaluation(
    positions: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose one legal move uniformly, keeping the random baseline truly random."""
    policies = np.full((len(positions), 7), -1.0e9, dtype=np.float32)
    for row in range(len(positions)):
        occupied = np.logical_or(positions[row, 0], positions[row, 1])
        legal = [col for col in range(7) if not occupied[5, col]]
        policies[row, int(rng.choice(legal))] = 0.0
    values = np.zeros(len(positions), dtype=np.float32)
    return policies, values, values.copy()


def _options(config: MinimaxHarnessConfig) -> c4a0_cpp.SelfPlayOptions:
    options = c4a0_cpp.SelfPlayOptions()
    options.max_nn_batch_size = config.inference_batch_size
    options.n_mcts_iterations = config.mcts_iterations
    options.c_exploration = config.c_exploration
    options.c_ply_penalty = config.c_ply_penalty
    options.root_dirichlet_alpha = 0.0
    options.root_dirichlet_epsilon = 0.0
    options.temperature_midpoint_ply = 0
    options.temperature_cutoff_ply = 0
    options.early_temperature = 0.0
    options.middle_temperature = 0.0
    options.late_temperature = 0.0
    options.seed = config.seed
    options.worker_threads = config.worker_threads
    return options


def evaluate_minimax_ladder(
    model: ConnectFourNet,
    device: torch.device,
    config: MinimaxHarnessConfig = MinimaxHarnessConfig(),
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> MinimaxHarnessResult:
    """Test ``model`` from random through increasing minimax depths.

    Each level uses paired colors. Native ``play_games_v2`` executes a batch of
    pairs concurrently; the score is checked after each batch so a model that
    has already exceeded ten points promotes without playing needless games.
    """
    model.eval().to(device)
    rng = np.random.default_rng(config.seed)
    options = _options(config)
    levels: list[LadderLevelResult] = []
    next_game_id = 0
    promotion_threshold = config.promotion_points_threshold

    for depth in [None, *range(1, config.max_depth + 1)]:
        opponent_name = "random" if depth is None else f"minimax{depth}"
        opponent = None if depth is None else _MinimaxEvaluator(depth)
        wins = draws = losses = games = 0
        started = time.monotonic()

        while games < config.games_per_level:
            pair_count = min(config.pair_batch_size, (config.games_per_level - games) // 2)
            requests = []
            for _ in range(pair_count):
                requests.extend(
                    [
                        c4a0_cpp.GameRequest(
                            c4a0_cpp.GameMetadata(next_game_id, MODEL_ID, OPPONENT_ID)
                        ),
                        c4a0_cpp.GameRequest(
                            c4a0_cpp.GameMetadata(next_game_id + 1, OPPONENT_ID, MODEL_ID)
                        ),
                    ]
                )
                next_game_id += 2

            def evaluator(model_id: int, positions: np.ndarray):
                if model_id == MODEL_ID:
                    return model.forward_numpy(positions)
                if opponent is None:
                    return _random_evaluation(positions, rng)
                return opponent(positions)

            result = c4a0_cpp.play_games_v2(requests, options, evaluator)
            for game in result.results:
                red_score = float(game.player0_score())
                model_score = red_score if game.metadata.player0_id == MODEL_ID else 1.0 - red_score
                if model_score == 1.0:
                    wins += 1
                elif model_score == 0.5:
                    draws += 1
                else:
                    losses += 1
            games += len(result.results)
            points = wins + 0.5 * draws
            if progress is not None:
                progress(
                    {
                        "opponent": opponent_name,
                        "completed_games": games,
                        "total_games": config.games_per_level,
                        "wins": wins,
                        "draws": draws,
                        "losses": losses,
                        "points": points,
                    }
                )
            if points > promotion_threshold:
                break

        points = wins + 0.5 * draws
        auto_promoted = games < config.games_per_level and points > promotion_threshold
        promoted = points > promotion_threshold
        level = LadderLevelResult(
            opponent=opponent_name,
            depth=depth,
            games=games,
            wins=wins,
            draws=draws,
            losses=losses,
            points=points,
            promoted=promoted,
            auto_promoted=auto_promoted,
            elapsed_seconds=time.monotonic() - started,
        )
        levels.append(level)
        if not promoted:
            return MinimaxHarnessResult(levels, opponent_name, promotion_threshold)
    return MinimaxHarnessResult(levels, None, promotion_threshold)
