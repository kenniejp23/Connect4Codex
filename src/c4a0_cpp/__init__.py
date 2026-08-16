"""C++ backend for the c4a0 Connect Four engine."""

from ._native import (  # type: ignore[reportMissingImports]
    BUF_N_CHANNELS,
    N_COLS,
    N_ROWS,
    GameMetadata,
    GameResult,
    PlayGamesResult,
    Sample,
    play_games,
    run_tui,
)

__all__ = [
    "BUF_N_CHANNELS",
    "N_COLS",
    "N_ROWS",
    "GameMetadata",
    "GameResult",
    "PlayGamesResult",
    "Sample",
    "play_games",
    "run_tui",
]
