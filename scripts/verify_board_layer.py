"""Exactness test: the compiled DeltaNet board must equal python-chess after every move.

usage: python scripts/verify_board_layer.py PGN[.zst] [--games N] [--batch 256] [--device cpu]
"""
import argparse
import time

import chess
import torch

from chess_persona.board_layer import CompiledBoardLayer, decode_board
from chess_persona.data import iter_games
from chess_persona.vocab import PAD, VOCAB

ap = argparse.ArgumentParser()
ap.add_argument("pgn"); ap.add_argument("--games", type=int, default=20000)
ap.add_argument("--batch", type=int, default=256); ap.add_argument("--device", default="cpu")
a = ap.parse_args()

layer = CompiledBoardLayer().to(a.device)
games = list(iter_games(a.pgn, limit=a.games))
plies = sum(len(g.tokens) - 1 for g in games)
print(f"{len(games)} games, {plies} plies; max length {max(len(g.tokens) for g in games)}")

import collections
kinds = collections.Counter()
checked = wrong = 0
worst = 0.0
first_error = None
t0 = time.time()
for i in range(0, len(games), a.batch):
    chunk = games[i:i + a.batch]
    T = max(len(g.tokens) for g in chunk)
    toks = torch.full((len(chunk), T), PAD, dtype=torch.long)
    for j, g in enumerate(chunk):
        toks[j, :len(g.tokens)] = torch.tensor(g.tokens)
    states = layer(toks.to(a.device)).cpu()
    for j, g in enumerate(chunk):
        board = chess.Board()
        for t in range(1, len(g.tokens)):
            mv = VOCAB.decode(board, g.tokens[t])
            assert mv is not None, "tokenizer/decoder disagree"
            d = VOCAB.tokens[g.tokens[t]]
            kinds[d[0] if d[0] != 'promo' else ('promo_' + chess.piece_symbol(d[3]))] += 1
            board.push(mv)
            S = states[j, t]
            pieces, white = decode_board(S)
            truth = {sq: (p.piece_type, p.color) for sq, p in board.piece_map().items()}
            # how far from exact 0/1 any entry drifted
            planes = S[:12, :64]
            worst = max(worst, float((planes - planes.round()).abs().max()))
            checked += 1
            if pieces != truth or white != (board.turn == chess.WHITE):
                wrong += 1
                if first_error is None:
                    first_error = (g.headers.get("Site"), t, board.fen())
print("move kinds checked:", dict(kinds))
print(f"checked {checked} positions in {time.time()-t0:.1f}s: {checked-wrong}/{checked} exact; "
      f"max float drift {worst:.2e}")
if first_error:
    print("first mismatch:", first_error)
