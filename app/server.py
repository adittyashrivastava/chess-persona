"""Demo app: play the model (as Tal or Kasparov) and watch its DeltaNet board memory update.

  uv run python app/server.py            # then open http://127.0.0.1:8000
  uv run python app/server.py --run main_board

Every memory step shown in the UI is computed with the compiled layer's own key / beta /
value tables and the same delta-rule update the model runs, so the panel is the model's
real state, not a re-drawing of the board.
"""
from __future__ import annotations

import argparse
import os
import uuid

import chess
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from chess_persona.board_layer import (D_V, N_COLS, N_STEPS, TURN_A, TURN_B, CompiledBoardLayer,
                                       decode_board, initial_state)
from chess_persona.play import Player
from chess_persona.vocab import VOCAB

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIECE_NAMES = ["white pawn", "white knight", "white bishop", "white rook", "white queen", "white king",
               "black pawn", "black knight", "black bishop", "black rook", "black queen", "black king"]
PIECE_SYMBOLS = "PNBRQKpnbrqk"
PERSONAS = {"base": "Base model (no persona)", "tal": "Mikhail Tal", "kasparov": "Garry Kasparov"}

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="big_board")
ap.add_argument("--port", type=int, default=8000)
args = ap.parse_args()
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CKPT = os.path.join(ROOT, "models", args.run, "ckpt.pt")

LAYER = CompiledBoardLayer()
_players: dict[str, Player] = {}


def player(persona: str) -> Player:
    if persona not in _players:
        adapter = None
        if persona != "base":
            adapter = os.path.join(ROOT, "models", "personas", args.run, f"{persona}.pt")
            if not os.path.exists(adapter):
                raise HTTPException(400, f"no {persona} adapter for {args.run}")
        _players[persona] = Player(CKPT, adapter=adapter, device=DEVICE)
    return _players[persona]


def col_name(c: int) -> str:
    return chess.square_name(c) if c < 64 else ("turn A" if c == TURN_A else "turn B")


def piece_at(S: torch.Tensor, col: int) -> str | None:
    c = S[:12, col]
    return PIECE_NAMES[int(c.argmax())] if c.max() > 0.5 else None


def describe_step(k: torch.Tensor, beta: float, v: torch.Tensor, S_before: torch.Tensor) -> dict | None:
    """Human description of one delta-rule step, read off the compiled key/beta/value."""
    if beta == 0:
        return None
    cols = [int(i) for i in torch.nonzero(k.abs() > 1e-6).flatten()]
    names = [col_name(c) for c in cols]
    piece, frm, to = None, None, None
    if beta == 2.0 and len(cols) == 2:
        op = "swap"
        if set(cols) == {TURN_A, TURN_B}:
            op = "turn"
            text = "flip whose turn it is (swap the two turn columns)"
        else:
            a, b = piece_at(S_before, cols[0]), piece_at(S_before, cols[1])
            if a and not b:
                text = f"swap {names[0]} ↔ {names[1]}: the {a} moves {names[0]} → {names[1]}"
                piece, frm, to = a, names[0], names[1]
            elif b and not a:
                text = f"swap {names[0]} ↔ {names[1]}: the {b} moves {names[1]} → {names[0]}"
                piece, frm, to = b, names[1], names[0]
            else:
                text = f"swap {names[0]} ↔ {names[1]}"
                frm, to = names[0], names[1]
    elif v.abs().sum() == 0:
        op = "clear"
        held = piece_at(S_before, cols[0])
        text = f"clear {names[0]}" + (f": captures the {held}" if held else " (already empty: nothing captured)")
        piece, to = held, names[0]
        if not held:
            op = "noop"
    else:
        op = "set"
        piece, to = PIECE_NAMES[int(v[:12].argmax())], names[0]
        text = f"set {names[0]} = {piece}"
    sym = PIECE_SYMBOLS[PIECE_NAMES.index(piece)] if piece else None
    return {"op": op, "text": text, "cols": cols, "beta": beta, "piece": sym, "from": frm, "to": to,
            "key": {col_name(c): round(float(k[c]), 4) for c in cols},
            "value": [round(float(x), 3) for x in v]}


def apply_token(S: torch.Tensor, token: int) -> tuple[torch.Tensor, list[dict]]:
    """Run one token through the compiled layer step by step, recording every state."""
    steps = []
    for j in range(N_STEPS):
        k = LAYER.key_table[token, j]
        b = float(LAYER.beta_table[token, j])
        v = LAYER.value_table[token, j]
        d = describe_step(k, b, v, S)
        S = S - b * torch.outer(S @ k, k) + b * torch.outer(v, k)   # S (I - b k k^T) + b v k^T
        if d:
            d["state"] = S.round(decimals=4).tolist()
            steps.append(d)
    return S, steps


def memory_board(S: torch.Tensor) -> dict:
    pieces, white = decode_board(S)
    return {"squares": {chess.square_name(sq): (PIECE_SYMBOLS[["P", "N", "B", "R", "Q", "K"].index(
        chess.piece_symbol(pt).upper()) + (0 if col else 6)]) for sq, (pt, col) in pieces.items()},
        "white_to_move": white}


def token_label(token: int) -> str:
    d = VOCAB.tokens[token]
    n = chess.square_name
    if d[0] == "move":
        return f"move {n(d[1])}→{n(d[2])}"
    if d[0] == "promo":
        return f"promotion {n(d[1])}→{n(d[2])}={chess.piece_symbol(d[3]).upper()}"
    if d[0] == "castle":
        return f"castle {'white' if d[1] else 'black'} {'O-O' if d[2] == 'K' else 'O-O-O'}"
    if d[0] == "ep":
        return f"en passant {n(d[1])}→{n(d[2])}"
    return d[0]


class Game:
    def __init__(self, persona: str, style: str, guard: str, model_color: str):
        self.board = chess.Board()
        self.S = initial_state()
        self.persona, self.style, self.guard = persona, style, guard
        self.model_white = model_color == "white"
        self.plies: list[dict] = []

    def push(self, move: chess.Move, who: str, analysis: dict | None = None) -> dict:
        token = VOCAB.encode(self.board, move)
        san = self.board.san(move)
        self.board.push(move)
        self.S, steps = apply_token(self.S, token)
        ply = {"who": who, "san": san, "uci": move.uci(), "token": token,
               "token_desc": token_label(token), "steps": steps, "fen": self.board.fen(),
               "analysis": analysis}
        self.plies.append(ply)
        return ply

    def model_move(self) -> dict:
        p = player(self.persona)
        probs, value = p.policy(self.board, self.style)
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:6]
        guard = None if self.guard == "none" else self.guard
        mv = p.choose(self.board, style=self.style, guard=guard)
        analysis = {"top": [{"san": self.board.san(m), "uci": m.uci(), "p": round(pr, 4)} for m, pr in top],
                    "expected_score": round(value, 4), "chosen": self.board.san(mv),
                    "guard": self.guard, "persona": PERSONAS[self.persona], "style": self.style}
        return self.push(mv, "model", analysis)

    def snapshot(self, new_plies: list[dict]) -> dict:
        mem = memory_board(self.S)
        real = {chess.square_name(sq): pc.symbol() for sq, pc in self.board.piece_map().items()}
        return {"fen": self.board.fen(), "moves": [p["san"] for p in self.plies],
                "legal": [m.uci() for m in self.board.legal_moves],
                "model_white": self.model_white, "turn": "white" if self.board.turn else "black",
                "game_over": self.board.is_game_over(claim_draw=True),
                "result": self.board.result(claim_draw=True) if self.board.is_game_over(claim_draw=True) else None,
                "memory": self.S.round(decimals=4).tolist(), "memory_board": mem,
                "memory_matches": mem["squares"] == real and mem["white_to_move"] == self.board.turn,
                "new_plies": new_plies}


GAMES: dict[str, Game] = {}
app = FastAPI()


class NewGame(BaseModel):
    persona: str = "tal"
    style: str = "unknown"
    guard: str = "value"
    model_color: str = "black"


class Move(BaseModel):
    sid: str
    uci: str


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "app", "static", "index.html"))


@app.get("/api/meta")
def meta():
    rows = [PIECE_SYMBOLS[i] for i in range(12)] + ["turn"]
    cols = [col_name(c) for c in range(N_COLS)]
    return {"personas": PERSONAS, "styles": ["unknown", "aggressive", "balanced", "steady"],
            "guards": ["none", "value"], "rows": rows, "cols": cols, "run": args.run,
            "device": DEVICE, "d_v": D_V, "n_cols": N_COLS}


@app.post("/api/new")
def new_game(req: NewGame):
    if req.persona not in PERSONAS:
        raise HTTPException(400, "unknown persona")
    g = Game(req.persona, req.style, req.guard, req.model_color)
    sid = uuid.uuid4().hex
    GAMES[sid] = g
    new = [g.model_move()] if g.model_white else []
    return {"sid": sid, **g.snapshot(new)}


@app.post("/api/move")
def move(req: Move):
    g = GAMES.get(req.sid)
    if g is None:
        raise HTTPException(404, "no such game")
    try:
        mv = chess.Move.from_uci(req.uci)
    except ValueError:
        raise HTTPException(400, "bad move")
    if mv not in g.board.legal_moves:
        raise HTTPException(400, "illegal move")
    new = [g.push(mv, "you")]
    if not g.board.is_game_over(claim_draw=True):
        new.append(g.model_move())
    return {"sid": req.sid, **g.snapshot(new)}


if __name__ == "__main__":
    print(f"model {CKPT} on {DEVICE}; open http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
