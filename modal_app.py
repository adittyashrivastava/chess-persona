"""Modal jobs: preprocess Lichess months, train the core (with / without board), evaluate.

  modal run modal_app.py::preprocess --month 2017-01 --max-games 6000000 --split train
  modal run modal_app.py::preprocess --month 2017-02 --max-games 60000 --split eval
  modal run modal_app.py::train --run board  --use-board true
  modal run modal_app.py::train --run noboard --use-board false
"""
from __future__ import annotations

import io
import os

import modal

app = modal.App("chess-persona")
vol = modal.Volume.from_name("chess-persona", create_if_missing=True)
VOL = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("stockfish")
    .pip_install("torch", "python-chess==1.999", "numpy", "zstandard", "requests", "tqdm")
    .add_local_python_source("chess_persona")
)

GAMES_PER_CHUNK = 10_000


def _base_seconds(tc: str) -> int:
    try:
        return int(tc.split("+")[0])
    except Exception:
        return 0


@app.function(image=image, timeout=6 * 3600, cpu=2, memory=4096)
def split_month(month: str, max_games: int, split: str, min_base: int = 180) -> int:
    """Stream a monthly dump, keep games with base time >= min_base, and hand each
    10k-game text chunk to a tokenize_chunk worker as soon as it is cut."""
    import requests
    import zstandard

    url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(r.raw), encoding="utf-8", errors="replace")
    handles, cur, kept, game, tc = [], [], 0, [], ""

    def dispatch(text_chunk):
        handles.append(tokenize_chunk.spawn((text_chunk, month, len(handles), split)))

    for line in text:
        if line.startswith("[Event ") and game:
            if _base_seconds(tc) >= min_base:
                cur.append("".join(game)); kept += 1
                if len(cur) == GAMES_PER_CHUNK:
                    dispatch("".join(cur)); cur = []
                if kept >= max_games:
                    break
            game, tc = [], ""
        if line.startswith("[TimeControl "):
            tc = line.split('"')[1]
        game.append(line)
    if cur:
        dispatch("".join(cur))
    print(f"{month}: kept {kept} games, {len(handles)} chunks dispatched")
    return sum(h.get() for h in handles)


@app.function(image=image, volumes={VOL: vol}, timeout=3600, cpu=1, memory=2048, max_containers=64)
def tokenize_chunk(args: tuple[str, str, int, str]) -> int:
    import numpy as np

    from chess_persona.prep import tokenize_pgn_text

    text, month, idx, split = args
    arrays = tokenize_pgn_text(text)
    out = f"{VOL}/tok/{split}/{month}_{idx:05d}.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, **arrays)
    vol.commit()
    return len(arrays["result"])


@app.local_entrypoint()
def preprocess(month: str = "2017-01", max_games: int = 6_000_000, split: str = "train"):
    n = split_month.remote(month, max_games, split)
    print(f"tokenized {n} games from {month} into split '{split}'")


# --------------------------------------------------------------------------- training

@app.function(image=image, volumes={VOL: vol}, gpu="H100", timeout=12 * 3600, memory=65536)
def train_remote(run: str, use_board: bool, steps: int, batch: int, d: int, layers: int, lr: float,
                 square_policy: bool = False) -> dict:
    from chess_persona.train import train
    return train(f"{VOL}/tok/train", f"{VOL}/tok/eval", f"{VOL}/runs/{run}", use_board=use_board,
                 steps=steps, batch=batch, d=d, layers=layers, lr=lr, commit=vol.commit,
                 square_policy=square_policy)


@app.local_entrypoint()
def train(run: str = "board", use_board: bool = True, steps: int = 12000, batch: int = 256,
          d: int = 512, layers: int = 8, lr: float = 3e-4, square_policy: bool = False):
    print(train_remote.remote(run, use_board, steps, batch, d, layers, lr, square_policy))


# --------------------------------------------------------------------------- evaluation: play vs Stockfish

@app.function(image=image, volumes={VOL: vol}, cpu=4, memory=8192, timeout=4 * 3600, max_containers=40)
def play_match(cfg: dict) -> dict:
    """cfg: run, adapter(optional), opp_elo, games, guard, style, margin_cp, seed."""
    import random

    import chess
    import chess.engine
    import torch

    from chess_persona.features import move_flags
    from chess_persona.play import Player

    torch.set_num_threads(4)
    adapter = f"{VOL}/personas/{cfg['run']}/{cfg['adapter']}.pt" if cfg.get("adapter") else None
    me = Player(f"{VOL}/runs/{cfg['run']}/ckpt.pt", adapter=adapter, stockfish="/usr/games/stockfish")
    opp = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
    opp.configure({"UCI_LimitStrength": True, "UCI_Elo": cfg["opp_elo"]})
    rng = random.Random(cfg.get("seed", 0))
    score, agg, moves_n, out = 0.0, 0, 0, []
    for gi in range(cfg["games"]):
        board, we_white = chess.Board(), gi % 2 == 0
        while not board.is_game_over(claim_draw=True) and board.ply() < 300:
            if (board.turn == chess.WHITE) == we_white:
                mv = me.choose(board, style=cfg.get("style", "unknown"), guard=cfg.get("guard"),
                               margin_cp=cfg.get("margin_cp", 50), our_elo=cfg.get("our_elo", 2200),
                               opp_elo=cfg["opp_elo"], temperature=cfg.get("temperature", 0.0))
                c, x, k = move_flags(board, mv)
                agg += int(c or x or k); moves_n += 1
            else:
                mv = opp.play(board, chess.engine.Limit(time=0.05)).move
            board.push(mv)
        res = board.result(claim_draw=True)
        s = 0.5 if res in ("1/2-1/2", "*") else (1.0 if (res == "1-0") == we_white else 0.0)
        score += s
        out.append({"white": we_white, "result": res, "plies": board.ply()})
    me.close(); opp.quit()
    return {**{k: v for k, v in cfg.items()}, "score": score / cfg["games"],
            "aggression": agg / max(moves_n, 1), "games_detail": out}


def perf_elo(score: float, opp: int) -> float:
    import math
    s = min(max(score, 0.01), 0.99)
    return opp + 400 * math.log10(s / (1 - s))


@app.local_entrypoint()
def strength(runs: str = "board,noboard", levels: str = "1400,1700,2000", games: int = 40,
             guard: str = "none", style: str = "unknown", adapter: str = "", tag: str = "strength",
             margin_cp: int = 50):
    import json
    cfgs = []
    for run in runs.split(","):
        for lv in map(int, levels.split(",")):
            for shard in range(4):  # 4 containers per level, games split evenly
                cfgs.append({"run": run, "opp_elo": lv, "games": games // 4, "seed": shard,
                             "guard": None if guard == "none" else guard, "style": style,
                             "adapter": adapter or None, "margin_cp": margin_cp})
    res = list(play_match.map(cfgs))
    summary = {}
    for r in res:
        key = (r["run"], r["opp_elo"])
        a = summary.setdefault(key, [0.0, 0, 0.0])
        a[0] += r["score"] * r["games"]; a[1] += r["games"]; a[2] += r["aggression"] * r["games"]
    rows = []
    for (run, lv), (s, n, ag) in sorted(summary.items()):
        rows.append({"run": run, "opp_elo": lv, "games": n, "score": s / n, "perf_elo": round(perf_elo(s / n, lv)),
                     "aggression": round(ag / n, 3), "guard": guard, "style": style, "adapter": adapter,
                     "margin_cp": margin_cp})
        print(json.dumps(rows[-1]))
    os.makedirs("results", exist_ok=True)
    with open(f"results/{tag}.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# --------------------------------------------------------------------------- personas

@app.function(image=image, volumes={VOL: vol}, gpu="A10G", timeout=3 * 3600, memory=32768)
def personas_remote(run: str, players: dict, steps: int = 3000) -> dict:
    """players: {name: {"pgn": text, "ids": [...]}}. Trains one adapter each, then a cross-match table."""
    import os
    import tempfile

    import torch

    from chess_persona.persona import load_base, load_player_games, move_match, split_games, train_adapter

    held, results = {}, {}
    for name, p in players.items():
        with tempfile.NamedTemporaryFile("w", suffix=".pgn", delete=False) as f:
            f.write(p["pgn"])
        games = load_player_games(f.name, p["ids"])
        tr, he = split_games(games)
        held[name] = he
        model = load_base(f"{VOL}/runs/{run}/ckpt.pt", "cuda")
        base = move_match(model, he, "cuda")
        log = train_adapter(model, tr, "cuda", steps=steps)
        os.makedirs(f"{VOL}/personas/{run}", exist_ok=True)
        torch.save(model.adapter_state(), f"{VOL}/personas/{run}/{name}.pt")
        results[name] = {"games": len(games), "train": len(tr), "held": len(he),
                         "base_match": base, "adapter_match": move_match(model, he, "cuda"), "log": log}
        print(name, results[name]["base_match"], results[name]["adapter_match"], flush=True)
    vol.commit()
    # cross table: adapter (row) on player's held-out games (column)
    cross = {}
    for a in players:
        model = load_base(f"{VOL}/runs/{run}/ckpt.pt", "cuda")
        model.load_adapter(torch.load(f"{VOL}/personas/{run}/{a}.pt", weights_only=False), r=8)
        model.to("cuda")
        cross[a] = {b: move_match(model, held[b], "cuda") for b in players}
    base_model = load_base(f"{VOL}/runs/{run}/ckpt.pt", "cuda")
    cross["base"] = {b: move_match(base_model, held[b], "cuda") for b in players}
    return {"per_player": results, "cross": cross}


@app.local_entrypoint()
def personas(run: str = "board", names: str = "tal,kasparov,vidit", steps: int = 3000):
    import json
    players = {}
    for n in names.split(","):
        meta = json.load(open(f"data/players/{n}.json"))
        players[n] = {"pgn": open(f"data/players/{n}.pgn").read(), "ids": meta["ids"]}
    res = personas_remote.remote(run, players, steps)
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/personas_{run}.json", "w"), indent=1)
    print(json.dumps(res["cross"], indent=1))
    for n, r in res["per_player"].items():
        print(n, {k: r[k] for k in ("games", "base_match", "adapter_match")})


# --------------------------------------------------------------------------- head-to-head between two runs

@app.function(image=image, volumes={VOL: vol}, cpu=4, memory=8192, timeout=4 * 3600, max_containers=40)
def play_h2h(cfg: dict) -> dict:
    """cfg: run_a, run_b, games, seed, opening_plies. Returns run_a's score."""
    import random

    import chess
    import torch

    from chess_persona.play import Player

    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"])
    A = Player(f"{VOL}/runs/{cfg['run_a']}/ckpt.pt")
    B = Player(f"{VOL}/runs/{cfg['run_b']}/ckpt.pt")
    score, details = 0.0, []
    for gi in range(cfg["games"]):
        board, a_white = chess.Board(), gi % 2 == 0
        while not board.is_game_over(claim_draw=True) and board.ply() < 300:
            side = A if (board.turn == chess.WHITE) == a_white else B
            temp = 1.0 if board.ply() < cfg.get("opening_plies", 6) else 0.0
            board.push(side.choose(board, temperature=temp, our_elo=2000, opp_elo=2000))
        res = board.result(claim_draw=True)
        s = 0.5 if res in ("1/2-1/2", "*") else (1.0 if (res == "1-0") == a_white else 0.0)
        score += s
        details.append({"a_white": a_white, "result": res, "plies": board.ply()})
    return {"score_a": score, "games": cfg["games"], "details": details}


@app.local_entrypoint()
def h2h(run_a: str = "main_board", run_b: str = "main_noboard", games: int = 200, shards: int = 10):
    import json
    import math
    res = list(play_h2h.map([{"run_a": run_a, "run_b": run_b, "games": games // shards, "seed": s}
                             for s in range(shards)]))
    n = sum(r["games"] for r in res)
    s = sum(r["score_a"] for r in res) / n
    wins = sum(1 for r in res for d in r["details"] if (d["result"] == "1-0") == d["a_white"] and d["result"] in ("1-0", "0-1"))
    draws = sum(1 for r in res for d in r["details"] if d["result"] not in ("1-0", "0-1"))
    se = math.sqrt(max(s * (1 - s), 1e-9) / n)
    elo = lambda p: 400 * math.log10(min(max(p, 1e-3), 1 - 1e-3) / (1 - min(max(p, 1e-3), 1 - 1e-3)))
    out = {"run_a": run_a, "run_b": run_b, "games": n, "wins_a": wins, "draws": draws, "losses_a": n - wins - draws,
           "score_a": round(s, 3), "elo_diff": round(elo(s)), "elo_95ci": [round(elo(s - 1.96 * se)), round(elo(s + 1.96 * se))]}
    print(json.dumps(out))
    os.makedirs("results", exist_ok=True)
    with open("results/h2h.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")
