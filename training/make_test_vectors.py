"""Write shared/encoding_vectors.json: reference encodings the browser must reproduce.

    python make_test_vectors.py

web/lib/engine/encode.test.ts loads the file and checks that the TypeScript
encoder gives identical planes and move indices. Re-run after any change to
chessnet/encode.py (tests/test_vectors.py fails if the file is stale).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import chess
import numpy as np

from chessnet.encode import ENCODING_VERSION, MOVES, encode_board, encode_move

OUT = Path(__file__).resolve().parent.parent / "shared" / "encoding_vectors.json"

# Positions random games rarely reach.
SPECIAL_FENS = [
    chess.STARTING_FEN,
    "rnbqkbnr/pppp1ppp/8/3Pp3/8/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 3",  # legal en passant (White)
    "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3",  # legal en passant (Black)
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",  # all castling
    "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1",  # partial castling, Black to move
    "8/P6k/8/8/8/8/6Kp/8 w - - 0 1",  # White promotes
    "8/P6k/8/8/8/8/6Kp/8 b - - 0 1",  # Black promotes
    "1r5k/P7/8/8/8/8/8/K7 w - - 0 1",  # promotion by capture
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",  # stalemate: no legal moves
]


def random_fens(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    fens: list[str] = []
    while len(fens) < count:
        board = chess.Board()
        while not board.is_game_over() and len(fens) < count:
            if rng.random() < 0.15:  # sample positions from all game phases
                fens.append(board.fen())
            board.push(rng.choice(list(board.legal_moves)))
    return fens


def vector(fen: str) -> dict:
    board = chess.Board(fen)
    planes = encode_board(board)
    return {
        "fen": fen,
        # 18x8x8 bits, packed in plane / rank / file order, as hex.
        "planes": np.packbits(planes.reshape(-1)).tobytes().hex(),
        "legal": {move.uci(): encode_move(move, board.turn) for move in board.legal_moves},
    }


def build() -> dict:
    return {
        "encoding_version": ENCODING_VERSION,
        "moves": MOVES,
        "positions": [vector(fen) for fen in SPECIAL_FENS + random_fens(300, seed=1)],
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = build()
    OUT.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print(f"wrote {len(data['positions'])} positions to {OUT}")


if __name__ == "__main__":
    main()
