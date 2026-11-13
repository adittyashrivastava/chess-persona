"""Clone a player's style into a persona adapter, locally (Apple GPU / CPU) in minutes.

  python scripts/clone.py --name me --lichess <your-username>
  python scripts/clone.py --name me --chesscom <your-username> --min-base 180

Fetches the player's games, trains a LoRA adapter on that player's own moves only,
and reports how often the model predicts the player's real move on held-out games,
before and after. Writes models/personas/<name>.pt.
"""
import argparse
import json
import os
import subprocess
import sys

import torch

from chess_persona.persona import load_base, load_player_games, move_match, split_games, train_adapter

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--lichess"); ap.add_argument("--chesscom"); ap.add_argument("--pgnmentor")
ap.add_argument("--min-base", type=int, default=0)
ap.add_argument("--ckpt", default="models/big_board/ckpt.pt")
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--skip-fetch", action="store_true")
a = ap.parse_args()

root = os.path.join(os.path.dirname(__file__), "..")
if not a.skip_fetch:
    cmd = [sys.executable, os.path.join(root, "scripts", "fetch_player.py"), a.name, "--min-base", str(a.min_base)]
    for flag in ("lichess", "chesscom", "pgnmentor"):
        if getattr(a, flag):
            cmd += [f"--{flag}", getattr(a, flag)]
    subprocess.run(cmd, check=True)

meta = json.load(open(os.path.join(root, "data", "players", f"{a.name}.json")))
games = load_player_games(os.path.join(root, "data", "players", f"{a.name}.pgn"), meta["ids"])
train, held = split_games(games)
print(f"{len(games)} games: {len(train)} to train, {len(held)} held out")
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model = load_base(a.ckpt, device)
before = move_match(model, held, device)
train_adapter(model, train, device, steps=a.steps)
after = move_match(model, held, device)
os.makedirs(os.path.join(root, "models", "personas", "big_board"), exist_ok=True)
torch.save(model.adapter_state(), os.path.join(root, "models", "personas", "big_board", f"{a.name}.pt"))
print(f"move match on held-out games: base {before:.3f} -> {a.name} adapter {after:.3f}")
