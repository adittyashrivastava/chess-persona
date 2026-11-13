"""Collect one player's games into data/players/<name>.pgn.

  python scripts/fetch_player.py tal       --pgnmentor Tal
  python scripts/fetch_player.py kasparov  --pgnmentor Kasparov
  python scripts/fetch_player.py vidit     --chesscom viditchess --min-base 180
  python scripts/fetch_player.py me        --lichess <your-username>

Also writes data/players/<name>.json with the player's identity string(s) as they
appear in the PGN headers, so training can tell which side the player was.
"""
import argparse
import io
import json
import os
import time
import zipfile

import chess.pgn
import requests

UA = {"User-Agent": "chess-persona (personal research project)"}
ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--pgnmentor")
ap.add_argument("--lichess")
ap.add_argument("--chesscom")
ap.add_argument("--min-base", type=int, default=0, help="drop games with base time below this (seconds)")
ap.add_argument("--max-games", type=int, default=20000)
a = ap.parse_args()

out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "players")
os.makedirs(out_dir, exist_ok=True)
texts, ids = [], []

if a.pgnmentor:
    r = requests.get(f"https://www.pgnmentor.com/players/{a.pgnmentor}.zip", headers=UA, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for n in z.namelist():
            if n.lower().endswith(".pgn"):
                texts.append(z.read(n).decode("latin-1"))
    ids.append(a.pgnmentor)  # PGN Mentor uses "Surname, Firstname"; matched by surname
if a.lichess:
    url = f"https://lichess.org/api/games/user/{a.lichess}"
    params = {"rated": "true", "max": a.max_games, "perfType": "blitz,rapid,classical,correspondence"}
    r = requests.get(url, params=params, headers={**UA, "Accept": "application/x-chess-pgn"}, timeout=600)
    r.raise_for_status()
    texts.append(r.text)
    ids.append(a.lichess)
if a.chesscom:
    arch = requests.get(f"https://api.chess.com/pub/player/{a.chesscom}/games/archives", headers=UA, timeout=60).json()
    for month_url in reversed(arch.get("archives", [])):
        r = requests.get(month_url + "/pgn", headers=UA, timeout=120)
        if r.ok:
            texts.append(r.text)
        time.sleep(0.3)
    ids.append(a.chesscom)

kept = 0
path = os.path.join(out_dir, f"{a.name}.pgn")
with open(path, "w") as out:
    for t in texts:
        fh = io.StringIO(t)
        while kept < a.max_games:
            g = chess.pgn.read_game(fh)
            if g is None:
                break
            h = g.headers
            if h.get("Variant", "Standard") not in ("Standard", "") or g.board().fen() != chess.STARTING_FEN:
                continue
            players = (h.get("White", "") + " " + h.get("Black", "")).lower()
            if not any(i.lower() in players for i in ids):
                continue
            if a.min_base:
                try:
                    if int(h.get("TimeControl", "0").split("+")[0].split("/")[-1]) < a.min_base:
                        continue
                except ValueError:
                    pass
            print(g, file=out, end="\n\n")
            kept += 1
json.dump({"name": a.name, "ids": ids}, open(os.path.join(out_dir, f"{a.name}.json"), "w"))
print(f"{a.name}: {kept} games -> {path}")
