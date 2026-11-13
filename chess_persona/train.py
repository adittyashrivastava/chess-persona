"""Train the chess core on tokenized Lichess shards (runs on a Modal GPU or locally)."""
from __future__ import annotations

import glob
import json
import math
import os
import time

import chess
import numpy as np
import torch
import torch.nn.functional as F

from .model import N_PREFIX, ELO_BUCKETS, STYLES, ChessPersona, Config, elo_token, style_token
from .vocab import BOS, PAD, VOCAB

MAX_MOVES = 255


class Shards:
    """All games of a split held in memory as flat token array + offsets + metadata."""

    def __init__(self, pattern: str) -> None:
        files = sorted(glob.glob(pattern))
        assert files, pattern
        parts = []
        for f in files:
            try:
                z = np.load(f)
                if len(z["offsets"]) - 1 == len(z["result"]):
                    parts.append({k: z[k] for k in z.files})
            except Exception as e:  # a shard caught mid-write when its job was stopped
                print(f"skipping unreadable shard {f}: {e!r}")
        toks, starts, ends, base = [], [], [], 0
        for p in parts:
            o = p["offsets"]
            toks.append(p["tokens"]); starts.append(o[:-1] + base); ends.append(o[1:] + base)
            base += len(p["tokens"])
        self.tokens = np.concatenate(toks)
        self.starts = np.concatenate(starts)
        self.ends = np.concatenate(ends)
        cat = lambda k: np.concatenate([p[k] for p in parts])
        self.white_elo, self.black_elo, self.result = cat("white_elo"), cat("black_elo"), cat("result")
        self.white_agg = cat("white_agg") / np.maximum(cat("white_n"), 1)
        self.black_agg = cat("black_agg") / np.maximum(cat("black_n"), 1)
        self.n = len(self.starts)

    def style_thresholds(self) -> dict[int, tuple[float, float]]:
        """Tercile cut points of the aggression score per Elo bucket."""
        th = {}
        elo = np.concatenate([self.white_elo, self.black_elo])
        agg = np.concatenate([self.white_agg, self.black_agg])
        bucket = np.searchsorted(ELO_BUCKETS, elo, side="right") - 1
        for b in range(len(ELO_BUCKETS)):
            a = agg[bucket == b]
            th[b] = (float(np.quantile(a, 1 / 3)), float(np.quantile(a, 2 / 3))) if len(a) > 100 else (0.3, 0.45)
        return th


def style_label(score: float, elo: int, th: dict) -> str:
    b = int(np.searchsorted(ELO_BUCKETS, elo, side="right") - 1)
    lo, hi = th[b]
    return "steady" if score < lo else ("aggressive" if score > hi else "balanced")


def make_batch(data: Shards, idx: np.ndarray, th: dict, style_dropout: float, rng: np.random.Generator):
    """Returns tokens (B, T), policy targets (B, T), value targets (B, T); -100 = ignore."""
    B = len(idx)
    T = N_PREFIX + MAX_MOVES
    tokens = np.full((B, T), PAD, np.int64)
    pol = np.full((B, T), -100, np.int64)
    val = np.full((B, T), -100, np.int64)
    for i, g in enumerate(idx):
        mv = data.tokens[data.starts[g]:data.ends[g]][:MAX_MOVES].astype(np.int64)
        we, be = int(data.white_elo[g]), int(data.black_elo[g])
        ws = style_label(data.white_agg[g], we, th) if rng.random() > style_dropout else "unknown"
        bs = style_label(data.black_agg[g], be, th) if rng.random() > style_dropout else "unknown"
        prefix = [BOS, elo_token(we), elo_token(be), style_token(ws), style_token(bs)]
        seq = prefix + mv.tolist()
        L = len(seq)
        tokens[i, :L] = seq
        pol[i, N_PREFIX - 1:L - 1] = mv                      # position t predicts token t+1
        r = data.result[g]
        for t in range(N_PREFIX - 1, L):
            white_to_move = (t - (N_PREFIX - 1)) % 2 == 0     # after the prefix, white moves first
            score = r if white_to_move else 1 - r
            val[i, t] = 0 if score == 0 else (1 if score == 0.5 else 2)   # loss / draw / win
    return torch.from_numpy(tokens), torch.from_numpy(pol), torch.from_numpy(val)


@torch.no_grad()
def evaluate(model, data: Shards, th: dict, device: str, n_games: int = 4000, batch: int = 128,
             legality_games: int = 1000) -> dict:
    model.eval()
    rng = np.random.default_rng(0)
    idx = np.arange(min(n_games, data.n))
    tot = correct = 0
    vloss = vn = 0.0
    by_ply = {}
    legal_stats = {}
    for i in range(0, len(idx), batch):
        tok, pol, val = make_batch(data, idx[i:i + batch], th, 0.0, rng)
        tok, pol, val = tok.to(device), pol.to(device), val.to(device)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            logits, vlog = model(tok)
        m = pol != -100
        pred = logits.argmax(-1)
        ok = (pred == pol) & m
        tot += int(m.sum()); correct += int(ok.sum())
        ply = torch.arange(tok.shape[1], device=device) - (N_PREFIX - 1)
        for name, lo, hi in [("1-20", 0, 20), ("21-40", 20, 40), ("41-80", 40, 80), ("81+", 80, 10_000)]:
            sel = m & (ply >= lo) & (ply < hi)
            a = by_ply.setdefault(name, [0, 0]); a[0] += int((ok & sel).sum()); a[1] += int(sel.sum())
        vm = val != -100
        vloss += float(F.cross_entropy(vlog[vm].float(), val[vm], reduction="sum")); vn += int(vm.sum())
        if i < legality_games:
            probs = logits.float().softmax(-1).cpu()
            pred_c = pred.cpu()
            for j in range(tok.shape[0]):
                g = idx[i + j]
                mv = data.tokens[data.starts[g]:data.ends[g]][:MAX_MOVES]
                board = chess.Board()
                for t, m_tok in enumerate(mv):
                    pos = N_PREFIX - 1 + t
                    legal = VOCAB.legal_mask(board)
                    lp = float(probs[j, pos, legal].sum())
                    ply_name = "1-20" if t < 20 else "21-40" if t < 40 else "41-80" if t < 80 else "81+"
                    a = legal_stats.setdefault(ply_name, [0, 0, 0.0])
                    a[0] += int(int(pred_c[j, pos]) not in legal); a[1] += 1; a[2] += lp
                    board.push(VOCAB.decode(board, int(m_tok)))
    model.train()
    ill = {k: v[0] / v[1] for k, v in legal_stats.items()}
    mass = {k: v[2] / v[1] for k, v in legal_stats.items()}
    n_all = sum(v[1] for v in legal_stats.values())
    return {"top1": correct / max(tot, 1), "value_ce": vloss / max(vn, 1),
            "top1_by_ply": {k: v[0] / max(v[1], 1) for k, v in by_ply.items()},
            "illegal_top1": sum(v[0] for v in legal_stats.values()) / max(n_all, 1),
            "illegal_top1_by_ply": ill,
            "legal_mass": sum(v[2] for v in legal_stats.values()) / max(n_all, 1),
            "legal_mass_by_ply": mass}


def train(train_dir: str, eval_dir: str, out_dir: str, use_board: bool = True, steps: int = 12000,
          batch: int = 256, d: int = 512, layers: int = 8, lr: float = 3e-4, warmup: int = 500,
          style_dropout: float = 0.2, eval_every: int = 2000, commit=None, device: str | None = None,
          square_policy: bool = False) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    data = Shards(f"{train_dir}/*.npz")
    held = Shards(f"{eval_dir}/*.npz")
    th = data.style_thresholds()
    print(f"train games {data.n}, eval games {held.n}, device {device}, use_board {use_board}")
    cfg = Config(d=d, layers=layers, heads=max(4, d // 64), use_board=use_board, square_policy=square_policy)
    model = ChessPersona(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    rng = np.random.default_rng(1)
    log = []
    t0 = time.time()
    for step in range(1, steps + 1):
        cur_lr = lr * min(1, step / warmup) * 0.5 * (1 + math.cos(math.pi * min(1, step / steps)))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        idx = rng.integers(0, data.n, batch)
        tok, pol, val = (x.to(device, non_blocking=True) for x in make_batch(data, idx, th, style_dropout, rng))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits, vlog = model(tok)
            loss_p = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), pol.view(-1), ignore_index=-100)
            loss_v = F.cross_entropy(vlog.float().view(-1, 3), val.view(-1), ignore_index=-100)
            loss = loss_p + 0.5 * loss_v
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"step {step} loss_p {loss_p.item():.3f} loss_v {loss_v.item():.3f} lr {cur_lr:.2e} "
                  f"{(time.time()-t0)/step:.2f}s/step", flush=True)
        if step % eval_every == 0 or step == steps:
            ev = evaluate(model, held, th, device)
            ev.update(step=step, loss_p=loss_p.item(), elapsed=time.time() - t0)
            log.append(ev)
            print("EVAL", json.dumps(ev), flush=True)
            torch.save({"cfg": cfg.__dict__, "model": model.state_dict(), "style_thresholds": th},
                       f"{out_dir}/ckpt.pt")
            json.dump({"params": n_params, "use_board": use_board, "square_policy": square_policy, "log": log}, open(f"{out_dir}/log.json", "w"))
            if commit:
                commit()
    return {"params": n_params, "final": log[-1]}
