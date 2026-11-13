"""PGN text -> arrays of move tokens + per-game metadata (shared by Modal and local runs)."""
from __future__ import annotations

import io

import chess
import chess.pgn
import numpy as np

from .features import move_flags
from .vocab import VOCAB

RESULT = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
MAX_MOVES = 255


def tokenize_pgn_text(text: str, min_moves: int = 10) -> dict[str, np.ndarray]:
    fh = io.StringIO(text)
    toks, offs = [], [0]
    meta = {k: [] for k in ("white_elo", "black_elo", "result", "white_agg", "black_agg", "white_n", "black_n")}
    while True:
        g = chess.pgn.read_game(fh)
        if g is None:
            break
        h = g.headers
        if g.errors or h.get("Result") not in RESULT or "WhiteElo" not in h or "BlackElo" not in h:
            continue
        try:
            we, be = int(h["WhiteElo"]), int(h["BlackElo"])
        except ValueError:
            continue
        board = g.board()
        if board.fen() != chess.STARTING_FEN:
            continue
        moves, agg, cnt = [], [0, 0], [0, 0]
        for mv in g.mainline_moves():
            side = 0 if board.turn == chess.WHITE else 1
            c, x, k = move_flags(board, mv)
            agg[side] += int(c or x or k)
            cnt[side] += 1
            moves.append(VOCAB.encode(board, mv))
            board.push(mv)
            if len(moves) == MAX_MOVES:
                break
        if len(moves) < min_moves:
            continue
        toks.extend(moves)
        offs.append(len(toks))
        for k, v in zip(meta, (we, be, RESULT[h["Result"]], agg[0], agg[1], cnt[0], cnt[1])):
            meta[k].append(v)
    out = {"tokens": np.array(toks, np.uint16), "offsets": np.array(offs, np.int64)}
    out["result"] = np.array(meta.pop("result"), np.float32)
    out.update({k: np.array(v, np.int16) for k, v in meta.items()})
    return out
