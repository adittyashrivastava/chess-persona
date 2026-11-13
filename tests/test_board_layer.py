"""The compiled board must match python-chess exactly, including every special move."""
import chess
import pytest
import torch

from chess_persona.board_layer import CompiledBoardLayer, decode_board, initial_state
from chess_persona.vocab import BOS, VOCAB


def run_moves(start_fen: str, ucis: list[str]):
    board = chess.Board(start_fen)
    layer = CompiledBoardLayer()
    toks = [BOS]
    for u in ucis:
        mv = chess.Move.from_uci(u)
        assert mv in board.legal_moves, (u, board.fen())
        toks.append(VOCAB.encode(board, mv))
        board.push(mv)
    s0 = initial_state(chess.Board(start_fen)).unsqueeze(0)
    states = layer(torch.tensor([toks]), s0=s0)
    fast = layer.fast_forward(torch.tensor([toks])) if start_fen == chess.STARTING_FEN else None
    return board, states[0, -1], (fast[0, -1] if fast is not None else None)


def assert_same(board, state):
    pieces, white = decode_board(state)
    assert pieces == {sq: (p.piece_type, p.color) for sq, p in board.piece_map().items()}
    assert white == (board.turn == chess.WHITE)


CASES = {
    "white castles kingside": ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ["e1g1"]),
    "white castles queenside": ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ["e1c1"]),
    "black castles both ways": ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", ["e8c8", "e1g1"]),
    "queen e1g1 is not castling": ("4k3/8/8/8/8/8/8/4Q1K1 w - - 0 1", ["e1f1"]),
    "white en passant": ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", ["e5d6"]),
    "black en passant": ("4k3/8/8/8/4Pp2/8/8/4K3 b - e3 0 1", ["f4e3"]),
    "promotion with capture to queen": ("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1", ["a7b8q"]),
    "underpromotion to knight": ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", ["a7a8n"]),
    "black underpromotion to rook with capture": ("4k3/8/8/8/8/8/p7/1N2K3 b - - 0 1", ["a2b1r"]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_special_moves(name):
    fen, moves = CASES[name]
    board, state, _ = run_moves(fen, moves)
    assert_same(board, state)


def test_full_game_and_fast_path_agree():
    moves = "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 e1g1 f6e4 d2d4 e5d4 f1e1 d7d5 c4d5 d8d5 b1c3".split()
    board, state, fast = run_moves(chess.STARTING_FEN, moves)
    assert_same(board, state)
    assert_same(board, fast)


def test_vocab_roundtrip_every_legal_move_in_many_positions():
    import random
    rng = random.Random(0)
    for _ in range(200):
        board = chess.Board()
        for _ in range(rng.randint(0, 80)):
            moves = list(board.legal_moves)
            if not moves:
                break
            mv = rng.choice(moves)
            tok = VOCAB.encode(board, mv)
            assert VOCAB.decode(board, tok) == mv
            board.push(mv)
