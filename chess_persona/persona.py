"""Persona adapters: LoRA trained only on one player's own moves.

The base model stays frozen; an adapter is a small file (~1-2% of the weights)
that shifts the policy toward how that player moves. `clone` = fetch your games
(scripts/fetch_player.py) + train_adapter.
"""
from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass

import chess
import chess.pgn
import torch
import torch.nn.functional as F

from .model import N_PREFIX, ChessPersona, Config, build_prefix
from .vocab import PAD, VOCAB

MAX_MOVES = 255


@dataclass
class PlayerGame:
    moves: list[int]
    player_white: bool
    white_elo: int
    black_elo: int


def _elo(h, key, default):
    try:
        return int(h.get(key, "")) or default
    except ValueError:
        return default


def load_player_games(pgn_path: str, ids: list[str], player_default_elo: int = 2600,
                      opp_default_elo: int = 2450) -> list[PlayerGame]:
    games = []
    with open(pgn_path, encoding="utf-8", errors="replace") as fh:
        while True:
            g = chess.pgn.read_game(fh)
            if g is None:
                break
            h = g.headers
            w, b = h.get("White", "").lower(), h.get("Black", "").lower()
            is_w = any(i.lower() in w for i in ids)
            is_b = any(i.lower() in b for i in ids)
            if is_w == is_b:
                continue
            board, moves = g.board(), []
            for mv in g.mainline_moves():
                moves.append(VOCAB.encode(board, mv)); board.push(mv)
                if len(moves) == MAX_MOVES:
                    break
            if len(moves) < 6:
                continue
            we = _elo(h, "WhiteElo", player_default_elo if is_w else opp_default_elo)
            be = _elo(h, "BlackElo", player_default_elo if is_b else opp_default_elo)
            games.append(PlayerGame(moves, is_w, we, be))
    return games


def batchify(games: list[PlayerGame], device: str):
    T = N_PREFIX + MAX_MOVES
    tok = torch.full((len(games), T), PAD, dtype=torch.long)
    tgt = torch.full((len(games), T), -100, dtype=torch.long)
    for i, g in enumerate(games):
        seq = build_prefix(g.white_elo, g.black_elo) + g.moves
        tok[i, :len(seq)] = torch.tensor(seq)
        for t, m in enumerate(g.moves):            # move t is played from position N_PREFIX-1+t
            white_moves = t % 2 == 0
            if white_moves == g.player_white:
                tgt[i, N_PREFIX - 1 + t] = m
    return tok.to(device), tgt.to(device)


def load_base(ckpt_path: str, device: str) -> ChessPersona:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ChessPersona(Config(**ck["cfg"]))
    model.load_state_dict(ck["model"])
    return model.to(device)


@torch.no_grad()
def move_match(model: ChessPersona, games: list[PlayerGame], device: str, batch: int = 32) -> float:
    model.eval()
    ok = n = 0
    for i in range(0, len(games), batch):
        tok, tgt = batchify(games[i:i + batch], device)
        logits, _ = model(tok)
        m = tgt != -100
        ok += int(((logits.argmax(-1) == tgt) & m).sum()); n += int(m.sum())
    return ok / max(n, 1)


def train_adapter(model: ChessPersona, games: list[PlayerGame], device: str, steps: int = 3000,
                  batch: int = 32, lr: float = 2e-4, r: int = 8, eval_every: int = 25,
                  patience: int = 4, max_epochs: float = 2.0) -> list[dict]:
    """LoRA on the player's own moves, with early stopping on a validation slice.

    10% of `games` is held back for validation; the adapter with the best validation
    move-match is kept. Training also stops after `max_epochs` passes over the games,
    because a few thousand games overfit quickly.
    """
    tr, va = split_games(games, held_frac=0.1, seed=1)
    steps = min(steps, max(eval_every, int(max_epochs * len(tr) / batch)))
    model.add_persona_adapter(r=r)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    best = move_match(model, va, device)
    best_state = {k: v.clone() for k, v in model.adapter_state().items()}
    log, bad = [{"step": 0, "val_match": best}], 0
    model.train()
    for step in range(1, steps + 1):
        tok, tgt = batchify(random.sample(tr, min(batch, len(tr))), device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits, _ = model(tok)
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), tgt.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            v = move_match(model, va, device)
            model.train()
            log.append({"step": step, "loss": loss.item(), "val_match": v})
            print(json.dumps(log[-1]), flush=True)
            if v > best:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in model.adapter_state().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    model.load_state_dict({k: t.to(device) for k, t in best_state.items()}, strict=False)
    log.append({"best_val_match": best})
    return log


def split_games(games: list[PlayerGame], held_frac: float = 0.1, seed: int = 0):
    g = games[:]
    random.Random(seed).shuffle(g)
    k = max(50, int(len(g) * held_frac))
    return g[k:], g[:k]
