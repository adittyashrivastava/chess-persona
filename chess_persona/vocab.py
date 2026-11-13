"""Move vocabulary: every move is exactly one token.

Token families
- specials: PAD, BOS, EOS, plus reserved style/control slots
- normal:    (from, to) for every queen-line or knight-jump pair (1792 tokens).
             Covers every non-special move of every piece, including king steps.
- promotion: pawn reaching the last rank, per target piece (176 tokens)
- castling:  O-O / O-O-O per colour (4 tokens). Separate from the king's (from,to)
             token because e1g1 can also be an ordinary queen/rook move.
- en passant: separate tokens (28), because e5d6 can also be an ordinary move.

Splitting castling and en passant out means each token alone says exactly which
squares change, so the compiled board layer never has to look at the board to
apply a move.
"""
from __future__ import annotations

import chess

SQUARES = 64
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
PROMO_PIECES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

PAD, BOS, EOS = 0, 1, 2
N_RESERVED = 64  # control / style tokens live in [3, 64)
STYLE_BASE = 3


def piece_index(piece_type: int, color: bool) -> int:
    """0..11: white P N B R Q K, then black P N B R Q K."""
    return PIECE_TYPES.index(piece_type) + (0 if color == chess.WHITE else 6)


def _queen_or_knight(a: int, b: int) -> bool:
    fa, ra, fb, rb = chess.square_file(a), chess.square_rank(a), chess.square_file(b), chess.square_rank(b)
    df, dr = abs(fa - fb), abs(ra - rb)
    if a == b:
        return False
    return df == 0 or dr == 0 or df == dr or {df, dr} == {1, 2}


class MoveVocab:
    def __init__(self) -> None:
        self.tokens: list[tuple] = []  # descriptor per token id
        self.index: dict[tuple, int] = {}

        def add(desc: tuple) -> None:
            self.index[desc] = len(self.tokens)
            self.tokens.append(desc)

        add(("pad",)); add(("bos",)); add(("eos",))
        for i in range(STYLE_BASE, N_RESERVED):
            add(("reserved", i))
        for a in range(SQUARES):
            for b in range(SQUARES):
                if _queen_or_knight(a, b):
                    add(("move", a, b))
        for color in (chess.WHITE, chess.BLACK):
            from_rank, to_rank = (6, 7) if color == chess.WHITE else (1, 0)
            for f in range(8):
                for df in (-1, 0, 1):
                    if 0 <= f + df < 8:
                        a, b = chess.square(f, from_rank), chess.square(f + df, to_rank)
                        for p in PROMO_PIECES:
                            add(("promo", a, b, p))
        for color in (chess.WHITE, chess.BLACK):
            add(("castle", color, "K"))
            add(("castle", color, "Q"))
        for color in (chess.WHITE, chess.BLACK):
            from_rank, to_rank = (4, 5) if color == chess.WHITE else (3, 2)
            for f in range(8):
                for df in (-1, 1):
                    if 0 <= f + df < 8:
                        add(("ep", chess.square(f, from_rank), chess.square(f + df, to_rank)))

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, board: chess.Board, move: chess.Move) -> int:
        """Token for `move` played from `board` (board needed only to tell castling/ep apart)."""
        if board.is_castling(move):
            side = "K" if board.is_kingside_castling(move) else "Q"
            return self.index[("castle", board.turn, side)]
        if board.is_en_passant(move):
            return self.index[("ep", move.from_square, move.to_square)]
        if move.promotion:
            return self.index[("promo", move.from_square, move.to_square, move.promotion)]
        return self.index[("move", move.from_square, move.to_square)]

    def decode(self, board: chess.Board, token: int) -> chess.Move | None:
        """Move for `token` on `board`, or None if the token is not a legal move there."""
        d = self.tokens[token]
        if d[0] == "move":
            m = chess.Move(d[1], d[2])
        elif d[0] == "promo":
            m = chess.Move(d[1], d[2], promotion=d[3])
        elif d[0] == "castle":
            if d[1] != board.turn:
                return None
            rank = 0 if d[1] == chess.WHITE else 7
            m = chess.Move(chess.square(4, rank), chess.square(6 if d[2] == "K" else 2, rank))
        elif d[0] == "ep":
            m = chess.Move(d[1], d[2])
        else:
            return None
        if m not in board.legal_moves or self.encode(board, m) != token:
            return None
        return m

    def legal_mask(self, board: chess.Board) -> list[int]:
        return [self.encode(board, m) for m in board.legal_moves]


VOCAB = MoveVocab()
