import random

import chess
import numpy as np
import pytest

from chessnet.encode import (
    MOVES,
    NUM_MOVES,
    NUM_PLANES,
    decode_move,
    encode_board,
    encode_move,
    legal_move_mask,
)


def random_positions(count: int, seed: int = 0) -> list[chess.Board]:
    """Positions from random games, including promotions, castling and en passant."""
    rng = random.Random(seed)
    positions = []
    while len(positions) < count:
        board = chess.Board()
        while not board.is_game_over() and board.ply() < 300:
            positions.append(board.copy())
            board.push(rng.choice(list(board.legal_moves)))
    return positions[:count]


POSITIONS = random_positions(3000)


# --- move table --------------------------------------------------------------

def test_move_table_has_1858_unique_moves():
    assert len(MOVES) == NUM_MOVES == 1858
    assert len(set(MOVES)) == NUM_MOVES


def test_move_table_contents():
    assert "e2e4" in MOVES
    assert "g1f3" in MOVES  # knight
    assert "a1h8" in MOVES  # long diagonal
    assert "e1g1" in MOVES  # castling is a king move
    assert "a7b8n" in MOVES  # capturing under-promotion
    assert "a7a8q" not in MOVES  # queen promotion uses plain a7a8
    assert "a1b3" in MOVES and "a1c4" not in MOVES


# --- moves -----------------------------------------------------------------

def test_every_legal_move_round_trips():
    for board in POSITIONS:
        indices = set()
        for move in board.legal_moves:
            index = encode_move(move, board.turn)
            assert decode_move(index, board) == move, (board.fen(), move)
            indices.add(index)
        assert len(indices) == board.legal_moves.count(), board.fen()


def test_black_moves_are_mirrored():
    board = chess.Board()
    board.push_uci("e2e4")
    assert encode_move(chess.Move.from_uci("e7e5"), chess.BLACK) == MOVES.index("e2e4")


def test_promotions():
    board = chess.Board("8/P6k/8/8/8/8/6Kp/8 w - - 0 1")
    assert MOVES[encode_move(chess.Move.from_uci("a7a8q"), chess.WHITE)] == "a7a8"
    assert MOVES[encode_move(chess.Move.from_uci("a7a8n"), chess.WHITE)] == "a7a8n"
    assert decode_move(MOVES.index("a7a8"), board) == chess.Move.from_uci("a7a8q")

    board.turn = chess.BLACK
    # Black's h2-h1 is "h7h8" in Black's frame.
    assert MOVES[encode_move(chess.Move.from_uci("h2h1r"), chess.BLACK)] == "h7h8r"
    assert decode_move(MOVES.index("h7h8"), board) == chess.Move.from_uci("h2h1q")


def test_legal_move_mask_matches_legal_moves():
    for board in POSITIONS[:500]:
        mask = legal_move_mask(board)
        assert mask.sum() == board.legal_moves.count()


# --- board planes -------------------------------------------------------------

def test_start_position_planes():
    planes = encode_board(chess.Board())
    assert planes.shape == (NUM_PLANES, 8, 8) and planes.dtype == np.uint8
    assert planes[0, 1].tolist() == [1] * 8  # our pawns on rank 2
    assert planes[6, 6].tolist() == [1] * 8  # their pawns on rank 7
    assert planes[5, 0, 4] == 1  # our king on e1
    assert planes[11, 7, 4] == 1  # their king on e8
    assert planes[12].sum() == 0  # White to move
    assert all(planes[p].all() for p in (13, 14, 15, 16))  # all castling rights
    assert planes[17].sum() == 0
    assert planes[:12].sum() == 32


def test_colour_mirrored_positions_encode_identically():
    # board.mirror() flips the board, swaps colours and the side to move,
    # so apart from the "Black to move" plane the encoding must be identical.
    for board in POSITIONS[:500]:
        a, b = encode_board(board), encode_board(board.mirror())
        np.testing.assert_array_equal(np.delete(a, 12, axis=0), np.delete(b, 12, axis=0))
        assert a[12].all() != b[12].all()


def test_castling_planes_follow_side_to_move():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1")  # White: O-O only, Black: O-O-O only
    planes = encode_board(board)
    assert not planes[13].any() and planes[14].all()  # us = Black: queenside only
    assert planes[15].all() and not planes[16].any()  # them = White: kingside only


@pytest.mark.parametrize(
    ("fen", "expected"),
    [
        # d5 pawn can take e6 en passant: marked (Black's e6 is real e6, White to move, so no mirror).
        ("rnbqkbnr/pppp1ppp/8/3Pp3/8/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 3", (5, 4)),
        # Double push but no pawn can capture: not marked even though FEN lists e3.
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", None),
    ],
)
def test_en_passant_plane_only_when_capture_is_legal(fen, expected):
    plane = encode_board(chess.Board(fen))[17]
    if expected is None:
        assert plane.sum() == 0
    else:
        assert plane.sum() == 1 and plane[expected] == 1
