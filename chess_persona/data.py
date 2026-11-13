"""Read PGN games (plain or .zst) and turn them into move-token sequences."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Iterator

import chess
import chess.pgn
import zstandard

from .vocab import VOCAB, BOS


@dataclass
class Game:
    tokens: list[int]                 # BOS + one token per move
    headers: dict = field(default_factory=dict)
    result: float | None = None       # 1 white win, 0.5 draw, 0 black win


def open_pgn(path: str) -> io.TextIOBase:
    if path.endswith(".zst"):
        fh = open(path, "rb")
        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


RESULT = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def tokenize_game(game: chess.pgn.Game) -> Game | None:
    board = game.board()
    if board.fen() != chess.STARTING_FEN:  # skip Chess960 / set-up positions
        return None
    toks = [BOS]
    for mv in game.mainline_moves():
        toks.append(VOCAB.encode(board, mv))
        board.push(mv)
    return Game(toks, dict(game.headers), RESULT.get(game.headers.get("Result", "*")))


def iter_games(path: str, limit: int | None = None, min_plies: int = 2) -> Iterator[Game]:
    fh = open_pgn(path)
    n = 0
    while limit is None or n < limit:
        g = chess.pgn.read_game(fh)
        if g is None:
            break
        if g.errors:
            continue
        tg = tokenize_game(g)
        if tg is None or len(tg.tokens) - 1 < min_plies:
            continue
        n += 1
        yield tg
