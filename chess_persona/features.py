"""Per-move style features, used to label game sides aggressive / balanced / steady.

A move counts toward aggression if it gives check, captures, or lands within two
squares (king distance) of the enemy king. The per-side score is the share of the
side's moves that do at least one of these. Labels are terciles of that score within
an Elo bucket, so "aggressive" means aggressive relative to players of that level.
"""
from __future__ import annotations

import chess


def move_flags(board: chess.Board, move: chess.Move) -> tuple[bool, bool, bool]:
    gives_check = board.gives_check(move)
    capture = board.is_capture(move)
    king = board.king(not board.turn)
    near_king = king is not None and chess.square_distance(move.to_square, king) <= 2
    return gives_check, capture, near_king


def aggression_score(n_moves: int, n_aggressive: int) -> float:
    return n_aggressive / max(n_moves, 1)
