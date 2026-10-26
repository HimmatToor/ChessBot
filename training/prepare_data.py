"""Turn PGN game files into training shards.

Example (one month of the Lichess Elite Database, ~85 MB download):

    mkdir -p data/raw
    curl -L -o data/raw/lichess_elite_2025-06.zip \\
        https://database.nikonoel.fr/lichess_elite_2025-06.zip
    python prepare_data.py data/raw/lichess_elite_2025-06.zip --max-games 20000

Writes data/processed/train-0000.npz, ... and val-0000.npz, ...
Roughly 1 in 100 games goes to validation, split by game (not by position)
so the same game never appears on both sides.
"""

from __future__ import annotations

import argparse
import io
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path

import chess.pgn

from chessnet.data import ShardWriter, game_examples


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
    args = parser.parse_args(argv)

    train = ShardWriter(args.out, "train", args.shard_size)
    val = ShardWriter(args.out, "val", args.shard_size)
    seen = accepted = 0
    start = time.time()

    def report() -> None:
        rate = accepted / max(time.time() - start, 1e-9)
        print(f"games seen {seen:,}  accepted {accepted:,}  ({rate:.0f}/s)", flush=True)

    for path in args.inputs:
        for stream in open_pgn_streams(path):
            while args.max_games is None or accepted < args.max_games:
                game = chess.pgn.read_game(stream)
                if game is None:
                    break
                seen += 1
                if game.errors or not accept_game(game.headers, args.min_elo, args.min_seconds):
                    continue
                writer = val if accepted % args.val_every == 0 else train
                for example in game_examples(game, args.skip_plies):
                    writer.add(*example)
                accepted += 1
                if accepted % 5000 == 0:
                    report()

    train.flush()
    val.flush()
    report()
    print(f"wrote {train.total:,} train positions ({train.shards_written} shards) and "
          f"{val.total:,} val positions ({val.shards_written} shards) to {args.out}")


if __name__ == "__main__":
    main()
