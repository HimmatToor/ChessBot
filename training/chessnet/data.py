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


def game_arrays(game: chess.pgn.Game, skip_plies: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All of a game's examples as (packed planes, moves, results) arrays."""
    examples = list(game_examples(game, skip_plies))
    if not examples:
        return np.zeros((0, PACKED_BYTES), np.uint8), np.zeros(0, np.int16), np.zeros(0, np.int8)
    planes, moves, results = zip(*examples)
    return pack_planes(np.stack(planes)), np.array(moves, np.int16), np.array(results, np.int8)


class ShardWriter:
    """Buffers examples and writes them out as numbered shards of exactly `shard_size` (the last may be smaller)."""

    def __init__(self, out_dir: Path, prefix: str, shard_size: int):
        self.out_dir = Path(out_dir)
        self.prefix = prefix
        self.shard_size = shard_size
        self.shards_written = 0
        self.total = 0
        self._chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._buffered = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def positions(self) -> int:
        """Examples added so far, written or still buffered."""
        return self.total + self._buffered

    def add(self, planes: np.ndarray, move: int, result: int) -> None:
        """One example with unpacked (18, 8, 8) planes."""
        self.add_packed(pack_planes(planes[None]), np.array([move], np.int16), np.array([result], np.int8))

    def add_packed(self, planes: np.ndarray, moves: np.ndarray, results: np.ndarray) -> None:
        """Many examples at once, planes already bit-packed (N, 144)."""
        if len(moves) == 0:
            return
        self._chunks.append((planes, moves, results))
        self._buffered += len(moves)
        while self._buffered >= self.shard_size:
            self._write(self.shard_size)

    def flush(self) -> None:
        if self._buffered:
            self._write(self._buffered)

    def _write(self, count: int) -> None:
        planes, moves, results = (np.concatenate(parts) for parts in zip(*self._chunks))
        path = self.out_dir / f"{self.prefix}-{self.shards_written:04d}.npz"
        np.savez_compressed(path, planes=planes[:count], move=moves[:count], result=results[:count])
        rest = (planes[count:], moves[count:], results[count:])
        self._chunks = [rest] if len(rest[1]) else []
        self._buffered -= count
        self.total += count
        self.shards_written += 1


def load_shard(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in ("planes", "move", "result")}
