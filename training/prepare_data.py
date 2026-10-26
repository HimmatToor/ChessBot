"""Turn PGN game files into training shards, using all CPU cores.

Example (one month of the Lichess Elite Database, ~85 MB download):

    mkdir -p data/raw
    curl -L -o data/raw/lichess_elite_2025-06.zip \\
        https://database.nikonoel.fr/lichess_elite_2025-06.zip
    python prepare_data.py data/raw/lichess_elite_2025-06.zip --max-games 20000

Writes data/processed/train-0000.npz, ... and val-0000.npz, ...
Roughly 1 in 100 games goes to validation, split by game (not by position)
so the same game never appears on both sides.

The main process only splits the file into game texts; worker processes
parse and encode them. Results are consumed in order, so the output is
identical for any --workers value.
"""

from __future__ import annotations

import argparse
import io
import itertools
import os
import time
import zipfile
from collections.abc import Iterable, Iterator
from multiprocessing import Pool
from pathlib import Path

import chess.pgn
import numpy as np

from chessnet.data import ShardWriter, game_arrays

GAMES_PER_TASK = 200

GameArrays = tuple[np.ndarray, np.ndarray, np.ndarray]


def open_pgn_streams(path: Path) -> Iterator[io.TextIOBase]:
    """Text streams for a .pgn file or every .pgn inside a .zip."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.endswith(".pgn"):
                    with archive.open(name) as raw:
                        yield io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            yield f


def split_games(stream: Iterable[str]) -> Iterator[str]:
    """Raw PGN text of each game: a header line after movetext starts a new game."""
    lines: list[str] = []
    in_movetext = False
    for line in stream:
        if line.startswith("[") and in_movetext:
            yield "".join(lines)
            lines, in_movetext = [], False
        elif line.strip() and not line.startswith("["):
            in_movetext = True
        lines.append(line)
    if any(line.strip() for line in lines):
        yield "".join(lines)


def time_control_seconds(header: str) -> float | None:
    """Estimated game length in seconds: base + 40 * increment. None if unknown."""
    try:
        base, increment = header.split("+")
        return int(base) + 40 * int(increment)
    except ValueError:
        return None


def accept_game(headers: chess.pgn.Headers, min_elo: int, min_seconds: int) -> bool:
    if headers.get("Variant", "Standard") != "Standard" or "FEN" in headers:
        return False  # chess960, custom start positions
    if headers.get("Result") not in ("1-0", "0-1", "1/2-1/2"):
        return False
    if headers.get("Termination") == "Abandoned":
        return False
    try:
        if min(int(headers["WhiteElo"]), int(headers["BlackElo"])) < min_elo:
            return False
    except (KeyError, ValueError):
        return False
    seconds = time_control_seconds(headers.get("TimeControl", ""))
    return seconds is not None and seconds >= min_seconds


def process_games(task: tuple[list[str], int, int, int]) -> list[GameArrays | None]:
    """Worker: encode each game text, or None if it's rejected. Runs in a subprocess."""
    texts, min_elo, min_seconds, skip_plies = task
    out: list[GameArrays | None] = []
    for text in texts:
        game = chess.pgn.read_game(io.StringIO(text))
        if game is None or game.errors or not accept_game(game.headers, min_elo, min_seconds):
            out.append(None)
        else:
            out.append(game_arrays(game, skip_plies))
    return out


def encoded_games(tasks: Iterator[tuple], workers: int) -> Iterator[GameArrays | None]:
    """Results of process_games for each task, in order."""
    if workers <= 1:
        for task in tasks:
            yield from process_games(task)
        return
    with Pool(workers) as pool:
        # Submit a bounded window at a time: Pool.imap would otherwise read
        # the whole input into memory as fast as it can.
        while window := list(itertools.islice(tasks, workers * 4)):
            for result in pool.imap(process_games, window):
                yield from result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help=".pgn or .zip files")
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--min-elo", type=int, default=2000, help="both players at least this rating")
    parser.add_argument("--min-seconds", type=int, default=180, help="skip faster games (base + 40*inc)")
    parser.add_argument("--skip-plies", type=int, default=6, help="skip the first N positions of each game")
    parser.add_argument("--max-games", type=int, default=None, help="stop after this many accepted games")
    parser.add_argument("--shard-size", type=int, default=500_000, help="positions per shard")
    parser.add_argument("--val-every", type=int, default=100, help="every Nth accepted game goes to validation")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="parallel processes")
    args = parser.parse_args(argv)

    def game_texts() -> Iterator[str]:
        for path in args.inputs:
            for stream in open_pgn_streams(path):
                yield from split_games(stream)

    texts = game_texts()
    tasks = iter(lambda: list(itertools.islice(texts, GAMES_PER_TASK)), [])
    tasks = ((batch, args.min_elo, args.min_seconds, args.skip_plies) for batch in tasks)

    train = ShardWriter(args.out, "train", args.shard_size)
    val = ShardWriter(args.out, "val", args.shard_size)
    seen = accepted = 0
    start = time.time()

    def report() -> None:
        elapsed = max(time.time() - start, 1e-9)
        print(f"games seen {seen:,}  accepted {accepted:,}  ({accepted / elapsed:.0f}/s)  "
              f"positions {train.positions + val.positions:,}", flush=True)

    for arrays in encoded_games(tasks, args.workers):
        seen += 1
        if arrays is None:
            continue
        (val if accepted % args.val_every == 0 else train).add_packed(*arrays)
        accepted += 1
        if accepted % 10_000 == 0:
            report()
        if args.max_games is not None and accepted >= args.max_games:
            break

    train.flush()
    val.flush()
    report()
    print(f"wrote {train.total:,} train positions ({train.shards_written} shards) and "
          f"{val.total:,} val positions ({val.shards_written} shards) to {args.out}")


if __name__ == "__main__":
    main()
