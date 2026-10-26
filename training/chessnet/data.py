"""Training examples: extraction from PGN games and on-disk shard format.

A shard is a compressed .npz file with three aligned arrays:
    planes  (N, 144) uint8  the 18x8x8 binary planes, bit-packed (8x smaller)
    move    (N,)     int16  policy index of the move actually played
    result  (N,)     int8   game result for the side to move: WIN, DRAW or LOSS
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from chessnet.encode import NUM_PLANES, encode_board, encode_move

PACKED_BYTES = NUM_PLANES * 64 // 8  # 144

WIN, DRAW, LOSS = 0, 1, 2
_WHITE_RESULT = {"1-0": WIN, "1/2-1/2": DRAW, "0-1": LOSS}


def pack_planes(planes: np.ndarray) -> np.ndarray:
    """(N, 18, 8, 8) 0/1 uint8 -> (N, 144) uint8."""
    return np.packbits(planes.reshape(len(planes), -1), axis=1)


def unpack_planes(packed: np.ndarray) -> np.ndarray:
    """(N, 144) uint8 -> (N, 18, 8, 8) 0/1 uint8."""
    return np.unpackbits(packed, axis=1).reshape(-1, NUM_PLANES, 8, 8)


def game_examples(game: chess.pgn.Game, skip_plies: int = 0) -> Iterator[tuple[np.ndarray, int, int]]:
    """(planes, move_index, result) for every position in a finished game.

    The first `skip_plies` positions are skipped: openings repeat across
    thousands of games and would otherwise dominate the data.
    """
    white_result = _WHITE_RESULT.get(game.headers.get("Result", "*"))
    if white_result is None:
        return
    board = game.board()
    for ply, move in enumerate(game.mainline_moves()):
        if ply >= skip_plies:
            result = white_result if board.turn == chess.WHITE else 2 - white_result
            yield encode_board(board), encode_move(move, board.turn), result
        board.push(move)


class ShardWriter:
    """Buffers examples and writes them out as numbered shards of fixed size."""

    def __init__(self, out_dir: Path, prefix: str, shard_size: int):
        self.out_dir = Path(out_dir)
        self.prefix = prefix
        self.shard_size = shard_size
        self.shards_written = 0
        self.total = 0
        self._planes: list[np.ndarray] = []
        self._moves: list[int] = []
        self._results: list[int] = []
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, planes: np.ndarray, move: int, result: int) -> None:
        self._planes.append(planes)
        self._moves.append(move)
        self._results.append(result)
        if len(self._moves) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._moves:
            return
        path = self.out_dir / f"{self.prefix}-{self.shards_written:04d}.npz"
        np.savez_compressed(
            path,
            planes=pack_planes(np.stack(self._planes)),
            move=np.array(self._moves, dtype=np.int16),
            result=np.array(self._results, dtype=np.int8),
        )
        self.total += len(self._moves)
        self.shards_written += 1
        self._planes, self._moves, self._results = [], [], []


def load_shard(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in ("planes", "move", "result")}
