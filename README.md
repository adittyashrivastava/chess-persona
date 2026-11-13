# chess-persona

A small chess transformer where **every move is one token**, with three additions:

1. **An exact board inside the model.** A hand-compiled Gated-DeltaNet layer (no training)
   keeps the full chess position in its recurrent state and updates it exactly on every
   move token. The learned layers read that board instead of having to infer it from the
   move list.
2. **Personas.** Small LoRA adapters trained on one player's own moves: Mikhail Tal,
   Garry Kasparov, or yourself from your Lichess / Chess.com username.
3. **Style instructions with a safety guard.** Ask for `aggressive`, `balanced` or
   `steady` play; a guard drops moves that give away the position before the persona and
   style pick among the rest.

## Results

All models are trained from scratch on 5.3M Lichess games (January 2017, blitz and slower) and
evaluated on 58,887 held-out games from February 2017. Single training seed each.

**1. The compiled board is exact.** Replayed through the layer, all 8,155,187 positions of
121,332 games (Lichess, January 2013) match python-chess, including 179,695 castles, 3,496 en
passant captures and 21,695 promotions (1,035 underpromotions). Every value stays within 1.2e-7
of an exact 0 or 1. `tests/` covers each special move.

**2. Giving the model the exact board helps, most in long games.** Same 27M-parameter model,
same data, same 20k steps; the only difference is the board input.

| | top-1 move match | illegal top choice | value loss |
|---|---|---|---|
| with exact board | **43.4%** | **1.9%** | **0.675** |
| moves only | 41.3% | 3.1% | 0.689 |

| ply range | top-1 with / without board | illegal with / without |
|---|---|---|
| 1-20 | 49.7 / 49.4 | 0.09 / 0.10% |
| 41-80 | 38.0 / 35.0 | 3.2 / 4.8% |
| 81+ | 42.0 / 37.1 | 4.9 / 9.2% |

Head to head, 200 games: with-board vs moves-only = 76 wins, 98 draws, 26 losses, about
**+89 Elo** (95% interval +40 to +141).

**3. Scaling.** An 89M version (768 wide, 12 layers, 40k steps) reaches 48.1% top-1 and 0.6%
illegal top choices, and beats the 27M model by about +137 Elo (104-67-29, 95% interval +87 to +192).

**4. Personas.** Rank-8 LoRA adapters, trained only on that player's moves with early stopping.
Held-out move match on the 89M model (the adapter predicts its own player best, and is worse
than the base model on the other player):

| adapter \ held-out games of | Tal | Kasparov |
|---|---|---|
| base | 43.8% | 42.9% |
| Tal | **44.3%** | 42.6% |
| Kasparov | 43.1% | **44.4%** |

**5. Style and the guard.** Against Stockfish at fixed strength (UCI_Elo, 0.05 s/move, 40 games):

| setup | vs 1400 | vs 1700 | vs 2000 | vs 2400 |
|---|---|---|---|---|
| 27M model alone | 0.14 | 0.06 | 0.04 | |
| 89M model alone | 0.34 | 0.11 | 0.05 | |
| 89M + Stockfish guard (within 50 cp, depth 10) | | 0.95 | 0.79 | 0.40 |

The guard keeps the game on track, but most of that strength is Stockfish's: it only lets
near-best moves through, and the model picks among them. The style token does change play:
without the guard, the share of moves that check, capture or land next to the enemy king is
30.6% with `aggressive` vs 24.5% with `steady` (27M model, vs Stockfish 1700). Inside the
guard the gap shrinks to 45.9% vs 42.8%.

**What didn't work.** A square-pair policy head (Lc0-style) learned faster at first but plateaued
worse. The model's own value head as a guard did not help. Persona adapters trained for many
passes overfit (Tal: 39.2% -> 30.7%) until early stopping was added.

## Demo app

Play Tal or Kasparov and watch the DeltaNet memory change on every move:

```bash
uv run python app/server.py          # http://127.0.0.1:8000  (needs models/ downloaded)
```

The page shows the board, the model's top candidate moves and win estimate, and the memory
state S as a 13 x 66 heatmap. Each move is replayed step by step (clear / swap / set, with the
key, beta and changed cells highlighted) using the compiled layer's own tables, and a second
board is decoded from the memory alone to show it matches the real one.

## How the compiled board works, with one move

The model's state is a 13 x 66 matrix `S`. Columns 0-63 are the squares a1..h8; a column
holds the piece on that square as a one-hot over 12 piece kinds (white P N B R Q K, black
P N B R Q K), or all zeros if the square is empty. Columns 64/65 hold whose turn it is.

Each move token applies up to four standard delta-rule steps (the DeltaProduct form of
Gated DeltaNet):

```
S  <-  S (I - beta k k^T) + beta v k^T
```

with the key `k`, `beta` and value `v` read straight from a table indexed by the token.
Three kinds of step are enough for all of chess:

| step | k | beta | v | effect |
|---|---|---|---|---|
| clear square s | e_s | 1 | 0 | column s becomes empty |
| set square s to piece p | e_s | 1 | onehot(p) | column s holds p |
| swap squares a, b | (e_a - e_b)/sqrt(2) | 2 | 0 | columns a and b trade places (a Householder reflection) |

`e2e4` is: clear e4 (in case of a capture), swap e2 <-> e4, swap the two turn columns.
Moving a piece by **swapping** columns is the trick: the layer never has to look up which
piece is on e2, so the update depends on the token alone. Castling is two swaps, en
passant a swap plus a clear, promotion a clear + swap + set. Castling and en passant get
their own tokens because `e1g1` can also be an ordinary queen move.

Read-out: 66 read heads with one-hot queries return every column, i.e. the whole board
after each move, which the learned model consumes as 12 x 64 piece planes + a turn bit.

**Verified exact:** see `scripts/verify_board_layer.py` and `tests/`.

For training speed the four steps of each token are fused into one affine map
`S -> S A_tok + C_tok` (the same math), so the whole board for a game costs one batched
matmul per move.

## Where the idea comes from, and what is new

- **qwenFS** ([Vishal Singh Baraiya](https://github.com/vixhal-baraiya/qwenFS)) hand-writes
  transformer layers that act as a filesystem inside Qwen3.5, using the Gated DeltaNet
  state as exact key-value tables. This repo uses the same "compile a program into
  DeltaNet state" idea for a chess board.
- **Qwenputer / qwenchess** ([blog](https://divagr.com/blog/qwenputer),
  [live](https://github.com/siddhantdubey/qwenchess-live)) compile a whole RISC-V CPU into
  Qwen3.5, with RAM in DeltaNet state, and run a classical chess engine inside the
  weights. There the board lives in an emulated CPU's memory and the engine plays.
- **Move-token chess transformers** (e.g. The Chess Transformer 2020, Chess-GPT 2024)
  learn the board only implicitly and approximately.
- **Style / persona conditioning**: Maia's individual-player models (2022) fine-tune per
  player; CHSM8 and UniMaia (2026) steer chess policies with natural language.

New here, as far as I can find: a board compiled **directly** (one token per move, exact,
no CPU emulation) feeding a **learned** move model, an ablation of whether that exact
board helps, and per-player LoRA adapters plus a guarded style control on top.

## Layout

```
chess_persona/
  vocab.py         one token per move (2064 tokens)
  board_layer.py   the hand-compiled DeltaNet board
  model.py         transformer core + board input + LoRA adapters
  train.py         training / evaluation (move match, illegal-move rate, value)
  persona.py       per-player adapters
  play.py          move choice: persona + style + guard
  prep.py, data.py, features.py
modal_app.py       preprocessing, training and evaluation jobs on Modal
scripts/           verify_board_layer.py, fetch_player.py, clone.py, play.py
app/               demo web app (server.py + static/index.html)
tests/             exactness tests for every special move
```

## Quick start

```bash
uv sync
uv run pytest -q tests                                     # board layer is exact
uv run python app/server.py                                # demo app
uv run python scripts/play.py --persona tal --style aggressive --guard value
uv run python scripts/clone.py --name me --lichess <your-username>
```
