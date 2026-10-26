"""Board and move encoding shared by training and the browser engine.

THIS IS THE SOURCE OF TRUTH. web/lib/engine/encode.ts must produce exactly the
same planes and move indices; shared test vectors keep the two in sync.

Everything is encoded from the point of view of the side to move: when Black
is to move, the board is mirrored vertically (rank 1 <-> rank 8) and colours
are swapped, so "our" pawns always advance up the board. The network then only
ever has to learn to play one side.

Planes (18 x 8 x 8, uint8), indexed [plane, row, col] with row 0 = rank 1 and
col 0 = file a, *after* the mirror:
     0-5   our   pawns, knights, bishops, rooks, queens, king
     6-11  their pawns, knights, bishops, rooks, queens, king
     12    all ones if the real side to move is Black
     13-14 our kingside / queenside castling right (whole plane)
     15-16 their kingside / queenside castling right (whole plane)
     17    en-passant target square, only if an en-passant capture is legal

Moves (1858 policy slots), also in the side-to-move frame:
    - every from->to pair a queen or knight could make on an empty board
      (1792). Normal moves, castling (king e1->g1 / e1->c1) and queen
      promotions all use these.
    - under-promotions to knight, bishop, rook from rank 7 to rank 8,
      straight or capturing left/right (66).
"""

from __future__ import annotations

import chess
import numpy as np

NUM_PLANES = 18
NUM_MOVES = 1858

_PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
_UNDERPROMOTIONS = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _build_move_table() -> list[str]:
    moves: list[str] = []
    for from_sq in chess.SQUARES:
        fr, ff = chess.square_rank(from_sq), chess.square_file(from_sq)
        for to_sq in chess.SQUARES:
            if to_sq == from_sq:
                continue
            dr, df = chess.square_rank(to_sq) - fr, chess.square_file(to_sq) - ff
            queen_like = dr == 0 or df == 0 or abs(dr) == abs(df)
            knight_like = sorted((abs(dr), abs(df))) == [1, 2]
            if queen_like or knight_like:
                moves.append(chess.square_name(from_sq) + chess.square_name(to_sq))

    for file in range(8):
        for df in (-1, 0, 1):  # capture left, straight, capture right
            to_file = file + df
            if not 0 <= to_file < 8:
                continue
            for piece in _UNDERPROMOTIONS:
                uci = chess.FILE_NAMES[file] + "7" + chess.FILE_NAMES[to_file] + "8"
                moves.append(uci + chess.piece_symbol(piece))
    return moves


#: Policy index -> move in UCI notation, in the side-to-move frame.
MOVES: list[str] = _build_move_table()
#: Move in UCI notation (side-to-move frame) -> policy index.
MOVE_TO_INDEX: dict[str, int] = {uci: i for i, uci in enumerate(MOVES)}

assert len(MOVES) == NUM_MOVES, len(MOVES)


def _orient(square: chess.Square, turn: chess.Color) -> chess.Square:
    """Map a real square into the side-to-move frame (and back: it's an involution)."""
    return square if turn == chess.WHITE else chess.square_mirror(square)


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a position as an (18, 8, 8) uint8 array in the side-to-move frame."""
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.uint8)
    us = board.turn
    them = not us

    for offset, color in ((0, us), (6, them)):
        for i, piece_type in enumerate(_PIECE_ORDER):
            for square in board.pieces(piece_type, color):
                sq = _orient(square, us)
                planes[offset + i, chess.square_rank(sq), chess.square_file(sq)] = 1

    if us == chess.BLACK:
        planes[12] = 1
    planes[13] = board.has_kingside_castling_rights(us)
    planes[14] = board.has_queenside_castling_rights(us)
    planes[15] = board.has_kingside_castling_rights(them)
    planes[16] = board.has_queenside_castling_rights(them)

    # python-chess keeps ep_square after any double push; only mark it when
    # the capture is actually possible, so the encoding depends on the
    # position alone (chess.js and FEN writers differ on this).
    if board.ep_square is not None and board.has_legal_en_passant():
        sq = _orient(board.ep_square, us)
        planes[17, chess.square_rank(sq), chess.square_file(sq)] = 1

    return planes


def encode_move(move: chess.Move, turn: chess.Color) -> int:
    """Policy index of a move played by `turn`."""
    from_sq, to_sq = _orient(move.from_square, turn), _orient(move.to_square, turn)
    uci = chess.square_name(from_sq) + chess.square_name(to_sq)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        uci += chess.piece_symbol(move.promotion)
    return MOVE_TO_INDEX[uci]


def decode_move(index: int, board: chess.Board) -> chess.Move:
    """Real-board move for a policy index in `board`. Not checked for legality."""
    uci = MOVES[index]
    turn = board.turn
    from_sq = _orient(chess.parse_square(uci[0:2]), turn)
    to_sq = _orient(chess.parse_square(uci[2:4]), turn)
    if len(uci) == 5:
        promotion = chess.Piece.from_symbol(uci[4]).piece_type
    elif board.piece_type_at(from_sq) == chess.PAWN and chess.square_rank(to_sq) in (0, 7):
        promotion = chess.QUEEN  # plain pawn move to the last rank = queen promotion
    else:
        promotion = None
    return chess.Move(from_sq, to_sq, promotion)


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Boolean (1858,) mask of policy slots that are legal moves in `board`."""
    mask = np.zeros(NUM_MOVES, dtype=bool)
    for move in board.legal_moves:
        mask[encode_move(move, board.turn)] = True
    return mask
