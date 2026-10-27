import random
import shutil

import chess
import pytest
import torch

import evaluate
from chessnet.model import ChessNet, ModelConfig


class RandomPlayer:
    def __init__(self, seed):
        self.rng, self.name = random.Random(seed), f"random-{seed}"

    def choose(self, board):
        return self.rng.choice(list(board.legal_moves))


def test_estimate_elo_known_cases():
    rating, _ = evaluate.estimate_elo([(1500, 1), (1500, 0)] * 10)  # 50% vs 1500
    assert rating == pytest.approx(1500, abs=1)
    rating, _ = evaluate.estimate_elo([(1500, 1), (1500, 1), (1500, 1), (1500, 0)] * 10)  # 75%
    assert rating == pytest.approx(1500 + 400 * 0.4771, abs=2)  # 400 * log10(3)
    draws, _ = evaluate.estimate_elo([(1800, 0.5)] * 10)
    assert draws == pytest.approx(1800, abs=1)


def test_estimate_elo_combines_opponents_and_shrinks_error_with_more_games():
    rating, se_small = evaluate.estimate_elo([(1400, 1), (1700, 0.5), (2000, 0)] * 4)
    assert 1650 < rating < 1750
    _, se_big = evaluate.estimate_elo([(1400, 1), (1700, 0.5), (2000, 0)] * 40)
    assert se_big < se_small / 3


def test_estimate_elo_is_finite_for_a_clean_sweep():
    rating, _ = evaluate.estimate_elo([(2000, 1.0)] * 10)
    assert 2000 < rating <= 2800


def test_play_game_reaches_a_legal_result_from_an_opening():
    game = evaluate.play_game(RandomPlayer(1), RandomPlayer(2), "e4 e5 Nf3 Nc6")
    moves = list(game.mainline_moves())
    assert [m.uci() for m in moves[:4]] == ["e2e4", "e7e5", "g1f3", "b8c6"]
    assert game.headers["Result"] in ("1-0", "0-1", "1/2-1/2")
    assert len(moves) <= evaluate.MAX_PLIES


def test_max_plies_adjudicates_a_draw():
    game = evaluate.play_game(RandomPlayer(3), RandomPlayer(4), "e4 e5", max_plies=10)
    assert game.headers["Result"] == "1/2-1/2"
    assert "adjudicated" in game.headers["Termination"]


def test_openings_are_legal():
    for opening in evaluate.OPENINGS:
        board = chess.Board()
        for san in opening.split():
            board.push_san(san)


@pytest.mark.parametrize("level", list(evaluate.LEVELS))
def test_net_player_plays_legal_moves_at_every_level(level):
    model = ChessNet(ModelConfig.tiny()).eval()
    bot = evaluate.NetPlayer(model, level)
    bot.settings = {**bot.settings, "simulations": min(bot.settings["simulations"], 20)}  # keep the test fast
    board = chess.Board()
    for _ in range(6):
        move = bot.choose(board)
        assert move in board.legal_moves
        board.push(move)


@pytest.mark.skipif(shutil.which("stockfish") is None, reason="Stockfish not installed")
def test_stockfish_player_makes_legal_moves():
    engine = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish"))
    try:
        sf = evaluate.StockfishPlayer(engine, 1500, 0.01)
        board = chess.Board()
        assert sf.choose(board) in board.legal_moves
    finally:
        engine.quit()
