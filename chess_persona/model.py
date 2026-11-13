"""Chess core: a causal transformer over move tokens that also reads the exact board.

Input at position t = embedding(move token t) + W_board @ board_after_move_t
where board_after_move_t comes from the frozen CompiledBoardLayer (12x64 piece
planes + side-to-move bit). With use_board=False the model sees moves only, like
earlier move-token chess transformers; that is the ablation.

Sequence layout:  BOS, <white-elo>, <black-elo>, <white-style>, <black-style>, m1, m2, ...
Control tokens use the reserved vocab range and leave the board unchanged.

Heads:
  policy: next move token
  value:  game result from the side to move after token t (loss / draw / win)

Persona adapters: LoRA on every attention and MLP projection (see LoRALinear).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .board_layer import BOARD_FEATURES, CompiledBoardLayer
from .vocab import VOCAB, STYLE_BASE

# ----- control tokens (reserved vocab slots) ---------------------------------
ELO_BUCKETS = [0, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600]   # 10 buckets
STYLES = ["unknown", "steady", "balanced", "aggressive"]
ELO_TOKEN0 = STYLE_BASE                       # 3 .. 12
STYLE_TOKEN0 = ELO_TOKEN0 + len(ELO_BUCKETS)  # 13 .. 16
N_PREFIX = 5                                  # BOS + 2 elo + 2 style


def elo_token(elo: int | None) -> int:
    if not elo:
        elo = 1500
    b = max(i for i, lo in enumerate(ELO_BUCKETS) if elo >= lo)
    return ELO_TOKEN0 + b


def style_token(style: str) -> int:
    return STYLE_TOKEN0 + STYLES.index(style)


@dataclass
class Config:
    vocab: int = len(VOCAB)
    d: int = 512
    layers: int = 8
    heads: int = 8
    max_len: int = 260
    dropout: float = 0.0
    use_board: bool = True
    square_policy: bool = False
    square_dim: int = 64


class LoRALinear(nn.Module):
    """y = W x + (alpha/r) B A x. W frozen when an adapter is active; A, B are the persona."""

    def __init__(self, d_in: int, d_out: int, bias: bool = False) -> None:
        super().__init__()
        self.base = nn.Linear(d_in, d_out, bias=bias)
        self.r = 0
        self.A = self.B = None
        self.scale = 1.0

    def add_lora(self, r: int = 16, alpha: float = 32.0) -> None:
        self.r = r
        self.A = nn.Parameter(torch.randn(r, self.base.in_features) / math.sqrt(self.base.in_features))
        self.B = nn.Parameter(torch.zeros(self.base.out_features, r))
        self.scale = alpha / r

    def remove_lora(self) -> None:
        self.r, self.A, self.B = 0, None, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r:
            y = y + (x @ self.A.t() @ self.B.t()) * self.scale
        return y


class Block(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        self.h = c.heads
        self.ln1 = nn.LayerNorm(c.d)
        self.qkv = LoRALinear(c.d, 3 * c.d)
        self.proj = LoRALinear(c.d, c.d)
        self.ln2 = nn.LayerNorm(c.d)
        self.fc = LoRALinear(c.d, 4 * c.d)
        self.out = LoRALinear(4 * c.d, c.d)
        self.drop = c.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, -1)
        q, k, v = (t.view(B, T, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.drop if self.training else 0.0)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        x = x + self.out(F.gelu(self.fc(self.ln2(x))))
        return x


class ChessPersona(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d)
        self.pos = nn.Embedding(c.max_len, c.d)
        self.board = CompiledBoardLayer()          # frozen, no parameters
        self.board_in = nn.Linear(BOARD_FEATURES, c.d) if c.use_board else None
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.ln = nn.LayerNorm(c.d)
        self.policy = nn.Linear(c.d, c.vocab, bias=False)
        self.value = nn.Linear(c.d, 3)
        if c.square_policy:
            # Square-pair policy (as in Lc0's attention policy): logit(a->b) += q(a).k(b).
            # Square a's representation = its square embedding + the piece on it (from the
            # exact board, if used) + the position's context vector.
            dq = c.square_dim
            self.sq_emb = nn.Parameter(torch.randn(64, dq) * 0.02)
            self.sq_piece = nn.Linear(12, dq, bias=False) if c.use_board else None
            self.sq_ctx = nn.Linear(c.d, dq)
            self.sq_q = nn.Linear(dq, dq)
            self.sq_k = nn.Linear(dq, dq)
            from .vocab import VOCAB as _V
            frm, to, kind = [], [], []
            KIND = {"move": 0, "promo": 1, "castle": 2, "ep": 3}
            for t in _V.tokens:
                if t[0] in ("move", "ep"):
                    frm.append(t[1]); to.append(t[2])
                elif t[0] == "promo":
                    frm.append(t[1]); to.append(t[2])
                elif t[0] == "castle":
                    r = 0 if t[1] else 7
                    frm.append(4 + 8 * r); to.append((6 if t[2] == "K" else 2) + 8 * r)
                else:
                    frm.append(0); to.append(0)
                kind.append(KIND.get(t[0], -1))
            self.register_buffer("tok_from", torch.tensor(frm))
            self.register_buffer("tok_to", torch.tensor(to))
            self.register_buffer("tok_is_move", torch.tensor([k >= 0 for k in kind]))

    def forward(self, tokens: torch.Tensor, board_feats: torch.Tensor | None = None):
        B, T = tokens.shape
        x = self.tok(tokens) + self.pos(torch.arange(T, device=tokens.device))
        if self.c.use_board:
            if board_feats is None:
                board_feats = CompiledBoardLayer.features(self.board.fast_forward(tokens))
            x = x + self.board_in(board_feats)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        logits = self.policy(x)
        if self.c.square_policy:
            sq = self.sq_emb + self.sq_ctx(x).unsqueeze(2)                     # (B, T, 64, dq)
            if self.c.use_board:
                planes = board_feats[..., :768].view(B, T, 12, 64).transpose(2, 3)  # (B, T, 64, 12)
                sq = sq + self.sq_piece(planes)
            sq = F.gelu(sq)
            pair = torch.einsum("ntad,ntcd->ntac", self.sq_q(sq), self.sq_k(sq)) / math.sqrt(sq.shape[-1])
            pair_logits = pair.flatten(2)[..., self.tok_from * 64 + self.tok_to]   # (B, T, vocab)
            logits = logits + pair_logits * self.tok_is_move
        return logits, self.value(x)

    # ---- persona adapters ----
    def lora_modules(self):
        return [m for m in self.modules() if isinstance(m, LoRALinear)]

    def add_persona_adapter(self, r: int = 8, alpha: float = 32.0) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        for m in self.lora_modules():
            m.add_lora(r, alpha)

    def adapter_state(self) -> dict:
        return {n: p.detach().cpu() for n, p in self.named_parameters() if n.endswith(".A") or n.endswith(".B")}

    def load_adapter(self, state: dict, r: int = 8, alpha: float = 32.0) -> None:
        if not any(m.r for m in self.lora_modules()):
            for m in self.lora_modules():
                m.add_lora(r, alpha)
        missing = self.load_state_dict(state, strict=False)
        assert not [k for k in missing.unexpected_keys], missing.unexpected_keys

    def clear_adapter(self) -> None:
        for m in self.lora_modules():
            m.remove_lora()


def build_prefix(white_elo=None, black_elo=None, white_style="unknown", black_style="unknown") -> list[int]:
    from .vocab import BOS
    return [BOS, elo_token(white_elo), elo_token(black_elo), style_token(white_style), style_token(black_style)]
