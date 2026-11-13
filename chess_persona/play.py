"""Move choice at play time: persona + style + a guard that keeps the game on track.

1. The model gives a probability for every legal move, conditioned on ratings,
   the style token for our side ("aggressive" / "balanced" / "steady") and, if
   loaded, a persona adapter.
2. Guard: drop moves that lose too much. With Stockfish: keep moves within
   `margin_cp` centipawns of Stockfish's best. Without: use the model's own value
   head on each candidate and keep moves within `margin_score` expected score of
   the best one.
3. Play the most likely remaining move (or sample with `temperature`).
"""
from __future__ import annotations

import math

import chess
import chess.engine
import torch

from .model import ChessPersona, build_prefix
from .persona import load_base
from .vocab import VOCAB


class Player:
    def __init__(self, ckpt: str, adapter: str | None = None, device: str = "cpu",
                 stockfish: str | None = None) -> None:
        self.device = device
        self.model: ChessPersona = load_base(ckpt, device).eval()
        if adapter:
            self.model.load_adapter(torch.load(adapter, map_location="cpu", weights_only=False), r=8)
            self.model.to(device)
        self.sf = chess.engine.SimpleEngine.popen_uci(stockfish) if stockfish else None

    def close(self):
        if self.sf:
            self.sf.quit()

    def _tokens(self, board: chess.Board) -> list[int]:
        replay, toks = chess.Board(), []
        for mv in board.move_stack:
            toks.append(VOCAB.encode(replay, mv)); replay.push(mv)
        return toks

    @torch.no_grad()
    def policy(self, board: chess.Board, style: str = "unknown", our_elo: int = 2400, opp_elo: int = 2400):
        """{move: prob} over legal moves, and expected score for the side to move."""
        white = board.turn == chess.WHITE
        prefix = build_prefix(our_elo if white else opp_elo, opp_elo if white else our_elo,
                              style if white else "unknown", "unknown" if white else style)
        seq = (prefix + self._tokens(board))[-260:]
        logits, vlog = self.model(torch.tensor([seq], device=self.device))
        legal = list(board.legal_moves)
        idx = [VOCAB.encode(board, m) for m in legal]
        p = logits[0, -1, idx].float().softmax(-1).cpu().tolist()
        v = vlog[0, -1].float().softmax(-1).cpu().tolist()   # loss, draw, win for side to move
        return dict(zip(legal, p)), v[2] + 0.5 * v[1]

    @torch.no_grad()
    def value_after(self, board: chess.Board, moves: list[chess.Move], our_elo: int, opp_elo: int) -> list[float]:
        """Expected score for the mover after each candidate move (model's value head, one batch)."""
        white = board.turn == chess.WHITE
        prefix = build_prefix(our_elo if white else opp_elo, opp_elo if white else our_elo)
        base = prefix + self._tokens(board)
        seqs = [(base + [VOCAB.encode(board, m)])[-260:] for m in moves]
        L = max(map(len, seqs))
        x = torch.zeros(len(seqs), L, dtype=torch.long)
        last = []
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s); last.append(len(s) - 1)
        _, vlog = self.model(x.to(self.device))
        v = vlog[torch.arange(len(seqs)), torch.tensor(last)].float().softmax(-1).cpu()
        return (v[:, 0] + 0.5 * v[:, 1]).tolist()   # opponent's loss + half draw = our score

    def choose(self, board: chess.Board, style: str = "unknown", guard: str | None = None,
               margin_cp: int = 50, margin_score: float = 0.05, sf_depth: int = 10,
               temperature: float = 0.0, our_elo: int = 2400, opp_elo: int = 2400) -> chess.Move:
        probs, _ = self.policy(board, style, our_elo, opp_elo)
        moves = list(probs)
        if guard == "stockfish" and self.sf and len(moves) > 1:
            infos = self.sf.analyse(board, chess.engine.Limit(depth=sf_depth), multipv=len(moves))
            cp = {i["pv"][0]: i["score"].pov(board.turn).score(mate_score=100000) for i in infos}
            best = max(cp.values())
            moves = [m for m in moves if cp.get(m, -10**9) >= best - margin_cp] or moves
        elif guard == "value" and len(moves) > 1:
            vals = self.value_after(board, moves, our_elo, opp_elo)
            best = max(vals)
            moves = [m for m, v in zip(moves, vals) if v >= best - margin_score] or moves
        if temperature <= 0:
            return max(moves, key=lambda m: probs[m])
        w = torch.tensor([math.log(max(probs[m], 1e-12)) / temperature for m in moves]).softmax(0)
        return moves[int(torch.multinomial(w, 1))]
