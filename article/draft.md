# I taught a tiny chess transformer to remember the board and play like Tal

*A weekend project: one token per move, an exact board in memory, and a personality on top.*

[IMAGE 01_app.png — the demo: play against "Tal" and watch the model's memory update on every move]

---

Training a transformer to play chess is a solved weekend problem. Give it millions of games, make every move a token, and ask it to predict the next one. It learns to play surprisingly well.

I wanted two things it doesn't have out of the box:

**1. Memory.** A move-predicting model never actually *sees* the board. It only sees the list of moves so far, and has to rebuild the position in its head. Deep into a game, it starts to lose track, and suggests moves for pieces that aren't there anymore.

**2. Personality.** A model trained on millions of online games plays like the average online player. I wanted it to play like a specific person: Mikhail Tal, the attacking genius, or Garry Kasparov.

So the plan was three steps.

[IMAGE 02_pipeline.png]

---

## Step 1: Pretrain on 5.3M games, one token per move

I took 5.3 million Lichess games from January 2017 (blitz and slower) and turned each move into a single token: `e2e4` is one token, `g1f3` is another. 2,064 tokens cover every possible chess move, including castling, en passant and promotions.

The model is a plain causal transformer: read the moves so far, predict the next one. It also predicts who's winning. I trained a 27M version first, then a bigger 89M one, on Modal GPUs.

Nothing new here. This is the baseline everything else is measured against.

---

## Step 2: Give it a memory that holds the exact board

This is the fun part.

There's a layer type called **Gated DeltaNet**, used in newer models like Qwen3.5. Instead of looking back over every previous token, it keeps a small grid of numbers (its "state") and edits that grid with each new token.

I realised that grid can hold a chess board exactly, if you set it up right. So I didn't train this layer at all. I **wrote its weights by hand**.

The memory is a 13 × 66 grid:
- one column per square (a1 … h8), plus two for whose turn it is
- one row per piece type (white pawn, white knight, … black king)
- a lit cell means "this square has this piece"

[IMAGE 06_heatmap.png — the actual memory, mid-game. You can see the pawns on rank 2, the pieces on rank 1, and black's pieces at the end]

Each move token triggers a few fixed edits to that grid. It turns out you only need three kinds of edit for all of chess:

[IMAGE 04_legend.png]

- **Swap** two columns: a piece moves from one square to another
- **Clear** a column: a piece gets captured
- **Set** a column: a pawn promotes

The trick that makes it work is the swap. To move a knight from f3 to d4, the layer doesn't need to know *what* is on f3. It just swaps the two columns. So the edit depends only on the move token, and can be baked into a fixed table.

Here it is running live. Each move is a capture-check, a swap, and a turn flip:

[GIF 03_memory.gif — four moves of a Sicilian, each memory edit highlighted as it happens]

And to prove it's exact, the app decodes the board back from memory alone, next to the real one:

[IMAGE 05_memboard.png — "identical to the real board"]

I didn't just eyeball it. I replayed **8,155,187 positions from 121,332 real games** through the layer, and every single one matched, including 179,695 castles, 3,496 en passant captures and 21,695 promotions.

### Does an exact board actually help?

I trained the same 27M model twice, same data, same steps. One gets the board from memory, the other only sees the moves.

[IMAGE 08_memory_results.png]

Early in the game there's no difference: the position is easy to track. Late in the game, the model with memory is clearly better and plays half as many illegal moves. In a 200-game match, the memory version won **76–26 with 98 draws, about +89 Elo**.

[IMAGE 10_elo.png]

---

## Step 3: Give it a personality with LoRA

Now the model knows the board. Next: make it play like someone.

I collected Tal's games (2,419) and Kasparov's (2,127) and trained a small **LoRA adapter** for each: a tiny add-on that nudges the frozen model's choices, trained only on that player's moves. The big model stays the same; you swap a ~5 MB file in and out to switch personality.

I played the same White moves against both. They chose different Sicilians from move 2: Tal went 2…d6 and then 5…a6, the sharp Najdorf. Kasparov went 2…e6, and at move 5 his adapter's top choice is …d6, the Scheveningen setup he played for years:

[IMAGE 07_tal.png] [IMAGE 07_kasparov.png]

Each adapter predicts its own player's real moves better than the base model, and the other player's slightly worse:

[IMAGE 09_personas.png]

The effect is small (+0.5 and +1.5 points), which makes sense: there are only about 2,000 games per player, against 5.3 million for pretraining. But it's consistently in the right direction.

---

## What didn't work

- **Training the personas for longer made them worse.** Tal's adapter dropped from 39% to 31% move-match after many passes over his games. It memorised instead of learning style. Early stopping fixed it.
- **A fancier move head** (one score per from-square/to-square pair, like Leela Chess Zero) learned faster at first but ended up worse.
- **Personality doesn't make it strong.** On its own, the 89M model scores 34% against Stockfish at 1400 Elo. To keep games watchable, the app has an optional "guard" that blocks moves which throw the game away. The persona then picks among the safe moves. That's honest about what's doing the heavy lifting.

---

## Try it

The demo app lets you play Tal or Kasparov, pick a style (aggressive / steady), and watch the memory grid update square by square on every move.

[link to repo]

Credit where it's due: the "write a program into a DeltaNet's memory by hand" idea comes from Vishal Singh Baraiya's **qwenFS** (a filesystem inside Qwen3.5) and **Qwenputer** (divagr.com), which compiled a whole CPU into Qwen and ran a chess engine on it. This project points the same trick at a chess board, and puts a learned player on top of it.
