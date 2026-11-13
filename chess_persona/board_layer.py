"""Hand-compiled board layer: an exact chess board kept in a DeltaNet state.

Nothing here is trained. Following qwenFS (github.com/vixhal-baraiya/qwenFS), the
weights are written by hand so the layer runs an exact program.

State S has shape (13, 66):
  columns 0..63  = squares a1..h8; the column holds the piece on that square as a
                   one-hot over 12 piece kinds (dims 0..11), or all zeros if empty.
  columns 64, 65 = turn register: dim 12 of column 64 is 1 when white is to move;
                   a swap of columns 64/65 after every move flips it.

Each token applies N_STEPS delta-rule steps (the DeltaProduct form of Gated
DeltaNet, Siems et al. 2025):
        S <- S (I - beta k k^T) + beta v k^T          (gate alpha = 1)
with k (unit), beta in [0, 2] and v taken straight from the token's compiled
embedding, so the update never needs to read the board:
  clear square s:   k = e_s,                 beta = 1, v = 0
  set square s=p:   k = e_s,                 beta = 1, v = onehot(p)
  swap squares a,b: k = (e_a - e_b)/sqrt(2), beta = 2, v = 0   (a Householder reflection)
A normal move e2e4 is: clear e4 (capture), swap e2<->e4, swap turn columns.
Moving the piece by swapping columns is what removes the need for a lookup.

Read-out: 66 read heads with one-hot queries return every column, i.e. the full
board after each move. The learned model reads those 12*64 + 1 numbers.
"""
from __future__ import annotations

import math

import chess
import torch
from torch import nn

from .vocab import VOCAB, MoveVocab, piece_index

N_COLS = 66          # 64 squares + 2 turn columns
D_V = 13             # 12 piece kinds + turn marker
N_STEPS = 4          # max delta steps per token (promotion with capture needs all 4)
TURN_A, TURN_B = 64, 65
BOARD_FEATURES = 12 * 64 + 1


def _clear(s):
    k = torch.zeros(N_COLS); k[s] = 1.0
    return k, 1.0, torch.zeros(D_V)


def _set(s, p):
    k = torch.zeros(N_COLS); k[s] = 1.0
    v = torch.zeros(D_V); v[p] = 1.0
    return k, 1.0, v


def _swap(a, b):
    k = torch.zeros(N_COLS); k[a] = 1 / math.sqrt(2); k[b] = -1 / math.sqrt(2)
    return k, 2.0, torch.zeros(D_V)


def _noop():
    k = torch.zeros(N_COLS); k[0] = 1.0  # any unit key; beta = 0 makes it identity
    return k, 0.0, torch.zeros(D_V)


def compile_token_program(vocab: MoveVocab, token: int) -> list[tuple]:
    """The delta steps one token performs, as (k, beta, v)."""
    d = vocab.tokens[token]
    steps = []
    if d[0] == "move":
        _, a, b = d
        steps = [_clear(b), _swap(a, b)]
    elif d[0] == "promo":
        _, a, b, piece = d
        color = chess.WHITE if chess.square_rank(b) == 7 else chess.BLACK
        steps = [_clear(b), _swap(a, b), _set(b, piece_index(piece, color))]
    elif d[0] == "castle":
        _, color, side = d
        r = 0 if color == chess.WHITE else 7
        k_from, k_to = chess.square(4, r), chess.square(6 if side == "K" else 2, r)
        r_from, r_to = chess.square(7 if side == "K" else 0, r), chess.square(5 if side == "K" else 3, r)
        steps = [_swap(k_from, k_to), _swap(r_from, r_to)]
    elif d[0] == "ep":
        _, a, b = d
        captured = chess.square(chess.square_file(b), chess.square_rank(a))
        steps = [_swap(a, b), _clear(captured)]
    else:  # pad / bos / eos / control tokens: no board change, no turn change
        return [_noop()] * N_STEPS
    steps.append(_swap(TURN_A, TURN_B))
    steps += [_noop()] * (N_STEPS - len(steps))
    assert len(steps) == N_STEPS
    return steps


def initial_state(board: chess.Board | None = None) -> torch.Tensor:
    board = board or chess.Board()
    S = torch.zeros(D_V, N_COLS)
    for sq, pc in board.piece_map().items():
        S[piece_index(pc.piece_type, pc.color), sq] = 1.0
    if board.turn == chess.WHITE:
        S[12, TURN_A] = 1.0
    else:
        S[12, TURN_B] = 1.0
    return S


class CompiledBoardLayer(nn.Module):
    """Frozen DeltaProduct layer whose per-token keys/betas/values are compiled tables."""

    def __init__(self, vocab: MoveVocab = VOCAB) -> None:
        super().__init__()
        V = len(vocab)
        K = torch.zeros(V, N_STEPS, N_COLS)
        B = torch.zeros(V, N_STEPS)
        Vv = torch.zeros(V, N_STEPS, D_V)
        for t in range(V):
            for j, (k, beta, v) in enumerate(compile_token_program(vocab, t)):
                K[t, j], B[t, j], Vv[t, j] = k, beta, v
        # These are the layer's "projections": token -> (k, beta, v) per step.
        self.register_buffer("key_table", K)
        self.register_buffer("beta_table", B)
        self.register_buffer("value_table", Vv)
        self.register_buffer("s0", initial_state())
        self.register_buffer("read_queries", torch.eye(N_COLS))  # 66 one-hot read heads

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, s0: torch.Tensor | None = None) -> torch.Tensor:
        """tokens: (B, T) -> board after each token: (B, T, 13, 66)."""
        Bsz, T = tokens.shape
        S = (self.s0 if s0 is None else s0).expand(Bsz, D_V, N_COLS).clone()
        K = self.key_table[tokens]      # (B, T, steps, 66)
        beta = self.beta_table[tokens]  # (B, T, steps)
        V = self.value_table[tokens]    # (B, T, steps, 13)
        out = []
        for t in range(T):
            for j in range(N_STEPS):
                k = K[:, t, j]                      # (B, 66)
                b = beta[:, t, j].view(Bsz, 1, 1)
                Sk = torch.einsum("bdc,bc->bd", S, k)                 # S k
                S = S - b * torch.einsum("bd,bc->bdc", Sk, k)        # S (I - b k k^T)
                S = S + b * torch.einsum("bd,bc->bdc", V[:, t, j], k)  # + b v k^T
            out.append(torch.einsum("bdc,cq->bdq", S, self.read_queries))
        return torch.stack(out, 1)

    def fused_tables(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Collapse each token's delta steps into one affine map S -> S @ A + C.

        Every step is S -> S (I - b k k^T) + b v k^T = S M_j + c_j, so the composition
        is S -> S (M1 M2 M3 M4) + (c1 M2 M3 M4 + c2 M3 M4 + c3 M4 + c4). Same math,
        one matmul per token instead of four rank-1 updates.
        """
        Vn = self.key_table.shape[0]
        A = torch.eye(N_COLS).expand(Vn, N_COLS, N_COLS).clone()
        C = torch.zeros(Vn, D_V, N_COLS)
        for j in range(N_STEPS):
            k = self.key_table[:, j].cpu()                       # (V, 66)
            b = self.beta_table[:, j].cpu().view(Vn, 1, 1)
            M = torch.eye(N_COLS) - b * torch.einsum("vi,vj->vij", k, k)
            c = b * torch.einsum("vd,vc->vdc", self.value_table[:, j].cpu(), k)
            A = A @ M
            C = C @ M + c
        return A, C

    @torch.no_grad()
    def fast_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Same output as forward(), using the fused per-token affine maps."""
        if not hasattr(self, "_A"):
            A, C = self.fused_tables()
            self._A, self._C = A.to(tokens.device), C.to(tokens.device)
        Bsz, T = tokens.shape
        S = self.s0.expand(Bsz, D_V, N_COLS).clone()
        out = torch.empty(Bsz, T, D_V, N_COLS, device=tokens.device)
        for t in range(T):
            S = torch.bmm(S, self._A[tokens[:, t]]) + self._C[tokens[:, t]]
            out[:, t] = S
        return out

    @staticmethod
    def features(states: torch.Tensor) -> torch.Tensor:
        """(B, T, 13, 66) -> (B, T, 769): 12x64 piece planes + white-to-move bit."""
        planes = states[:, :, :12, :64].flatten(2)
        turn = states[:, :, 12, TURN_A].unsqueeze(-1)
        return torch.cat([planes, turn], -1)


def decode_board(state: torch.Tensor) -> tuple[dict[int, tuple[int, bool]], bool]:
    """State (13, 66) -> ({square: (piece_type, color)}, white_to_move)."""
    pieces = {}
    for sq in range(64):
        col = state[:12, sq]
        if col.max() > 0.5:
            p = int(col.argmax())
            pieces[sq] = (PIECE_TYPES_[p % 6], p < 6)
    return pieces, bool(state[12, TURN_A] > 0.5)


PIECE_TYPES_ = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
