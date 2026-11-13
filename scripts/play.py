"""Play against the model in the terminal, or watch it play Stockfish.

  python scripts/play.py --persona tal --style aggressive --guard value
  python scripts/play.py --persona kasparov --vs-stockfish 1700 --guard stockfish --games 2

Moves are typed in UCI (e2e4) or SAN (Nf3). --guard: none | value | stockfish.
"""
import argparse
import os

import chess
import chess.engine
import torch

from chess_persona.play import Player

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="models/big_board/ckpt.pt")
ap.add_argument("--persona", default="", help="name of an adapter in models/personas/<name>.pt")
ap.add_argument("--style", default="unknown", choices=["unknown", "steady", "balanced", "aggressive"])
ap.add_argument("--guard", default="value", choices=["none", "value", "stockfish"])
ap.add_argument("--margin-cp", type=int, default=50)
ap.add_argument("--color", default="black", choices=["white", "black"], help="the model's colour")
ap.add_argument("--vs-stockfish", type=int, default=0, help="watch the model play Stockfish at this UCI_Elo")
ap.add_argument("--games", type=int, default=1)
ap.add_argument("--stockfish", default="stockfish")
a = ap.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"
adapter = f"models/personas/big_board/{a.persona}.pt" if a.persona else None
need_sf = a.guard == "stockfish" or a.vs_stockfish
me = Player(a.ckpt, adapter=adapter, device=device, stockfish=a.stockfish if need_sf else None)
guard = None if a.guard == "none" else a.guard
label = f"{a.persona or 'base'} / {a.style} / guard={a.guard}"

if a.vs_stockfish:
    sf = chess.engine.SimpleEngine.popen_uci(a.stockfish)
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": a.vs_stockfish})
    for gi in range(a.games):
        board, model_white = chess.Board(), gi % 2 == 0
        while not board.is_game_over(claim_draw=True):
            if (board.turn == chess.WHITE) == model_white:
                mv = me.choose(board, a.style, guard, margin_cp=a.margin_cp, opp_elo=a.vs_stockfish)
            else:
                mv = sf.play(board, chess.engine.Limit(time=0.05)).move
            print(board.san(mv), end=" ", flush=True)
            board.push(mv)
        print(f"\n[{label}] as {'white' if model_white else 'black'}: {board.result(claim_draw=True)}")
    sf.quit()
else:
    board = chess.Board()
    model_white = a.color == "white"
    print(f"You are {'black' if model_white else 'white'}. Model: {label}")
    while not board.is_game_over(claim_draw=True):
        print(board, "\n")
        if (board.turn == chess.WHITE) == model_white:
            mv = me.choose(board, a.style, guard, margin_cp=a.margin_cp)
            print("model plays", board.san(mv))
        else:
            txt = input("your move: ").strip()
            try:
                mv = board.parse_uci(txt) if len(txt) >= 4 and txt[1].isdigit() else board.parse_san(txt)
            except ValueError:
                print("not a legal move"); continue
        board.push(mv)
    print(board, "\nresult:", board.result(claim_draw=True))
me.close()
