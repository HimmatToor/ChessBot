"""Estimate the bot's rating at each difficulty level by playing rated Stockfish.

    python evaluate.py runs/full-v1/full-v1/best.pt --levels hard --opponents 1400 1700 2000 --games 20

Each level plays every opponent strength from a set of standard openings with
both colours. Stockfish's UCI_Elo (limited-strength mode, calibrated to human
CCRL-style ratings) gives the opponents' ratings; the bot's rating is the
maximum-likelihood fit to all its scores. Writes summary.json and games.pgn to --out.

Requires Stockfish: on PATH, or the official build unpacked in training/tools/
(see "Measuring strength" in README.md).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

from chessnet.checkpoint import load_checkpoint
from chessnet.mcts import pick_move, search

# Mirrors web/lib/engine/levels.ts. The browser's Max level is "5 seconds",
# which is ~360-470 simulations depending on the machine; 400 here.
LEVELS = {
    "easy": {"simulations": 1, "temperature": 1.5, "random_move_chance": 0.15},
    "medium": {"simulations": 50, "temperature": 0.5, "random_move_chance": 0.0},
    "hard": {"simulations": 200, "temperature": 0.0, "random_move_chance": 0.0},
    "max": {"simulations": 400, "temperature": 0.0, "random_move_chance": 0.0},
}

# Standard openings (SAN) so deterministic levels don't replay the same game.
OPENINGS = [
    "e4 e5 Nf3 Nc6", "e4 c5 Nf3 d6", "e4 e6 d4 d5", "e4 c6 d4 d5", "e4 c5 Nc3 Nc6",
    "d4 d5 c4 e6", "d4 Nf6 c4 g6", "d4 Nf6 c4 e6", "d4 d5 c4 c6", "c4 e5 Nc3 Nf6",
    "Nf3 d5 g3 Nf6", "e4 e5 Nf3 Nf6", "e4 d5 exd5 Qxd5", "d4 f5 g3 Nf6", "e4 g6 d4 Bg7",
]

MAX_PLIES = 400  # adjudicate as a draw after this many half-moves


def find_stockfish() -> str | None:
    """Stockfish on PATH, else the official build unpacked under training/tools/ (see README)."""
    on_path = shutil.which("stockfish")
    if on_path:
        return on_path
    local = sorted(Path(__file__).resolve().parent.glob("tools/stockfish*/stockfish/stockfish-*"))
    return str(local[-1]) if local else None


class NetPlayer:
    def __init__(self, model: torch.nn.Module, level: str, seed: int = 0):
        self.model, self.settings, self.name = model, LEVELS[level], f"ChessBot {level}"
        self.rng = random.Random(seed)

    def evaluate(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            policy, wdl = self.model(torch.from_numpy(planes))
        return policy.numpy(), wdl.numpy()

    def choose(self, board: chess.Board) -> chess.Move:
        result = search(board, self.evaluate, self.settings["simulations"])
        if self.rng.random() < self.settings["random_move_chance"]:
            return self.rng.choice(result.candidates).move
        return pick_move(result.candidates, self.settings["temperature"], self.rng).move


class StockfishPlayer:
    def __init__(self, engine: chess.engine.SimpleEngine, elo: int, movetime: float):
        self.engine, self.elo, self.movetime, self.name = engine, elo, movetime, f"Stockfish {elo}"

    def choose(self, board: chess.Board) -> chess.Move:
        self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": self.elo})
        return self.engine.play(board, chess.engine.Limit(time=self.movetime)).move


def play_game(white, black, opening: str, max_plies: int = MAX_PLIES) -> chess.pgn.Game:
    """Plays from the opening position; the PGN's Result header holds the outcome."""
    board = chess.Board()
    for san in opening.split():
        board.push_san(san)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        board.push((white if board.turn == chess.WHITE else black).choose(board))

    game = chess.pgn.Game.from_board(board)
    game.headers.update({"White": white.name, "Black": black.name, "Opening": opening})
    game.headers["Result"] = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
    if board.ply() >= max_plies and not board.is_game_over(claim_draw=True):
        game.headers["Termination"] = f"adjudicated draw after {max_plies} plies"
    return game


def expected_score(rating: float, opponent: float) -> float:
    return 1 / (1 + 10 ** ((opponent - rating) / 400))


def estimate_elo(games: list[tuple[float, float]]) -> tuple[float, float]:
    """(opponent rating, score 0/0.5/1) pairs -> (maximum-likelihood rating, standard error).

    Solves sum(score - expected) = 0 by bisection; clamps to +-800 of the
    opponents when the bot won or lost every game (the MLE is then infinite).
    """
    lo = min(r for r, _ in games) - 800
    hi = max(r for r, _ in games) + 800
    for _ in range(100):
        mid = (lo + hi) / 2
        if sum(s - expected_score(mid, r) for r, s in games) > 0:
            lo = mid
        else:
            hi = mid
    rating = (lo + hi) / 2
    information = sum(expected_score(rating, r) * (1 - expected_score(rating, r)) for r, _ in games)
    stderr = (400 / math.log(10)) / math.sqrt(information) if information > 0 else math.inf
    return rating, stderr


@dataclass
class Record:
    level: str
    opponent: int
    bot_color: str
    opening: str
    result: str
    score: float  # for the bot
    plies: int


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--levels", nargs="+", choices=LEVELS, default=["hard"])
    parser.add_argument("--opponents", nargs="+", type=int, default=[1400, 1700, 2000],
                        help="Stockfish UCI_Elo values (Stockfish accepts 1320-3190)")
    parser.add_argument("--games", type=int, default=20, help="games per level per opponent (even: both colours)")
    parser.add_argument("--movetime", type=float, default=0.1, help="Stockfish seconds per move")
    parser.add_argument("--stockfish", default=find_stockfish())
    parser.add_argument("--out", type=Path, default=Path("runs/eval"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not args.stockfish:
        parser.error("Stockfish not found: see 'Measuring strength' in README.md, or pass --stockfish PATH")

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    model, checkpoint = load_checkpoint(args.checkpoint)
    args.out.mkdir(parents=True, exist_ok=True)
    pgn_file = open(args.out / "games.pgn", "w")
    records: list[Record] = []
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    engine.configure({"Threads": 1, "Hash": 16})
    print(f"model step {checkpoint.get('step')}  stockfish {engine.id.get('name')}  movetime {args.movetime}s")

    try:
        for level in args.levels:
            bot = NetPlayer(model, level, seed=args.seed)
            for opponent in args.opponents:
                sf = StockfishPlayer(engine, opponent, args.movetime)
                for g in range(args.games):
                    opening = OPENINGS[(g // 2) % len(OPENINGS)]
                    bot_white = g % 2 == 0
                    start = time.time()
                    game = play_game(bot, sf, opening) if bot_white else play_game(sf, bot, opening)
                    result = game.headers["Result"]
                    score = {"1-0": 1.0, "0-1": 0.0}.get(result, 0.5)
                    score = score if bot_white else 1 - score
                    plies = len(list(game.mainline_moves()))
                    records.append(Record(level, opponent, "white" if bot_white else "black", opening, result, score, plies))
                    print(game, file=pgn_file, end="\n\n", flush=True)
                    print(f"  {level:<6} vs {opponent}  game {g + 1:>2}/{args.games}  bot {'W' if bot_white else 'B'}  "
                          f"{result:<7} ({plies} plies, {time.time() - start:.0f}s)", flush=True)
    finally:
        engine.quit()
        pgn_file.close()

    summary = {"checkpoint": str(args.checkpoint), "step": checkpoint.get("step"), "movetime": args.movetime, "levels": {}}
    print()
    for level in args.levels:
        mine = [r for r in records if r.level == level]
        rating, stderr = estimate_elo([(r.opponent, r.score) for r in mine])
        per_opponent = {
            o: sum(r.score for r in mine if r.opponent == o) / max(1, sum(1 for r in mine if r.opponent == o))
            for o in args.opponents
        }
        summary["levels"][level] = {"rating": round(rating), "stderr": round(stderr), "games": len(mine),
                                    "score_vs": {str(o): round(s, 3) for o, s in per_opponent.items()}}
        scores = "  ".join(f"vs {o}: {s:.0%}" for o, s in per_opponent.items())
        print(f"{level:<6} rating ≈ {rating:.0f} ± {stderr:.0f}   ({len(mine)} games; {scores})")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "games.json").write_text(json.dumps([r.__dict__ for r in records], indent=1) + "\n")
    print(f"wrote {args.out}/summary.json, games.pgn")


if __name__ == "__main__":
    main()
