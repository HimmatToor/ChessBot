"""Same scenarios as web/lib/engine/mcts.test.ts, so the two searches behave alike."""

import random

import chess
import numpy as np

from chessnet.encode import NUM_MOVES, encode_move
from chessnet.mcts import Candidate, pick_move, search


def uniform(planes):
    """Knows nothing: equal priors, every position a coin flip."""
    return np.zeros((len(planes), NUM_MOVES), np.float32), np.zeros((len(planes), 3), np.float32)


PIECE_VALUES = np.array([1, 3, 3, 5, 9], np.float32)  # pawn..queen; kings ignored


def material(planes):
    """Equal priors; value = material balance for the side to move."""
    ours = planes[:, 0:5].sum(axis=(2, 3)) @ PIECE_VALUES
    theirs = planes[:, 6:11].sum(axis=(2, 3)) @ PIECE_VALUES
    balance = ours - theirs
    wdl = np.stack([balance / 2, np.zeros_like(balance), -balance / 2], axis=1)
    return np.zeros((len(planes), NUM_MOVES), np.float32), wdl.astype(np.float32)


def test_finds_mate_in_one_by_search_alone():
    result = search(chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"), uniform, simulations=300)
    assert result.move.uci() == "a1a8"
    assert result.value > 0.5


def test_wins_a_hanging_queen():
    result = search(chess.Board("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1"), material, simulations=300)
    assert result.move.uci() == "d2d5"


def test_refuses_a_pawn_that_loses_the_queen():
    result = search(chess.Board("1r2k3/1p6/8/8/8/8/8/1Q2K3 w - - 0 1"), material, simulations=400)
    assert result.move.uci() != "b1b7"
    best = result.candidates[0]
    grab = next(c for c in result.candidates if c.move.uci() == "b1b7")
    assert grab.visits > 1
    assert grab.q < best.q - 0.15
    assert grab.visits < best.visits / 2


def test_one_simulation_plays_the_policy_favourite():
    def prefers_nf3(planes):
        policy, wdl = uniform(planes)
        policy[:, encode_move(chess.Move.from_uci("g1f3"), chess.WHITE)] = 5
        return policy, wdl

    result = search(chess.Board(), prefers_nf3, simulations=1)
    assert result.move.uci() == "g1f3" and result.simulations == 1


def test_exact_simulations_in_batches_and_board_restored():
    board = chess.Board()
    board.push_san("e4")
    fen, calls = board.fen(), []

    def counting(planes):
        calls.append(len(planes))
        return uniform(planes)

    result = search(board, counting, simulations=200, batch_size=8)
    assert result.simulations == 200
    assert sum(c.visits for c in result.candidates) == 199
    assert max(calls) <= 8 and len(calls) < 100
    assert board.fen() == fen and len(board.move_stack) == 1


def test_time_limit():
    assert search(chess.Board(), uniform, simulations=1_000_000, time_limit=0).simulations == 1


def test_pick_move_matches_the_browser_rules():
    cands = [Candidate(chess.Move.null(), v, p, 0.0) for v, p in ((60, 0.1), (30, 0.2), (10, 0.7))]
    assert pick_move(cands, 0) is cands[0]
    counts = [0, 0, 0]
    rng = random.Random(0)
    for _ in range(20_000):
        counts[cands.index(pick_move(cands, 1, rng))] += 1
    np.testing.assert_allclose(np.array(counts) / 20_000, [0.6, 0.3, 0.1], atol=0.02)
    unsearched = [Candidate(c.move, 0, c.prior, 0.0) for c in cands]
    counts = [0, 0, 0]
    for _ in range(20_000):
        counts[unsearched.index(pick_move(unsearched, 1, rng))] += 1
    np.testing.assert_allclose(np.array(counts) / 20_000, [0.1, 0.2, 0.7], atol=0.02)
