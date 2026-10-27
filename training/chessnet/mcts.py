"""PUCT Monte-Carlo tree search, a port of web/lib/engine/mcts.ts.

Same algorithm and constants as the browser engine, so strength measured here
(evaluate.py) matches what players face, and self-play can reuse it.

Value convention: values are in [-1, 1]. A node's value_sum is from the point
of view of the player who made the move leading to it, so a parent prefers
children with a high average value.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import chess
import numpy as np

from chessnet.encode import encode_board, encode_move

#: planes (B, 18, 8, 8) float32 -> (policy logits (B, 1858), wdl logits (B, 3))
Evaluator = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]

DEFAULT_BATCH = 16
DEFAULT_C_PUCT = 1.5
FPU_REDUCTION = 0.25  # unvisited moves are assumed a bit worse than their parent


class Node:
    __slots__ = ("move", "prior", "visits", "value_sum", "children", "terminal_value", "net_value", "pending")

    def __init__(self, move: chess.Move | None, prior: float):
        self.move = move
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children: list[Node] | None = None  # None = not expanded; [] = terminal
        self.terminal_value: float | None = None
        self.net_value = 0.0  # network value for the side to move here
        self.pending = False  # queued for evaluation in the current batch

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class Candidate:
    move: chess.Move
    visits: int
    prior: float
    q: float


@dataclass
class SearchResult:
    move: chess.Move
    value: float  # expected score for the side to move, -1..1
    candidates: list[Candidate]  # most visited first
    pv: list[chess.Move]
    simulations: int


def _select(parent: Node, c_puct: float) -> Node:
    sqrt_visits = math.sqrt(parent.visits)
    fpu = parent.net_value - FPU_REDUCTION
    best, best_score = None, -math.inf
    for child in parent.children:
        q = child.q if child.visits else fpu
        score = q + c_puct * child.prior * sqrt_visits / (1 + child.visits)
        if score > best_score:
            best, best_score = child, score
    return best


def _terminal_value(board: chess.Board, has_legal_moves: bool) -> float | None:
    if not has_legal_moves:
        return -1.0 if board.is_check() else 0.0  # checkmated, or stalemate
    if board.is_insufficient_material() or board.is_repetition(3) or board.is_fifty_moves():
        return 0.0
    return None


def _backpropagate(path: list[Node], value: float) -> None:
    """`value` is for the side to move at the leaf; the leaf's own sum is from its mover's view."""
    v = -value
    for node in reversed(path):
        node.visits += 1
        node.value_sum += v
        v = -v


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _expand(node: Node, legal: list[chess.Move], turn: chess.Color, policy: np.ndarray, wdl: np.ndarray) -> float:
    priors = _softmax(policy[[encode_move(m, turn) for m in legal]])
    order = np.argsort(-priors, kind="stable")
    node.children = [Node(legal[i], float(priors[i])) for i in order]
    w, _, l = _softmax(wdl)
    node.net_value = float(w - l)
    return node.net_value


def search(
    board: chess.Board,
    evaluate: Evaluator,
    simulations: int,
    batch_size: int = DEFAULT_BATCH,
    c_puct: float = DEFAULT_C_PUCT,
    time_limit: float | None = None,
) -> SearchResult:
    """Searches the current position. `board` is modified during the search and restored."""
    deadline = math.inf if time_limit is None else time.perf_counter() + time_limit
    root_legal = list(board.legal_moves)
    if not root_legal:
        raise ValueError("no legal moves to search")

    root = Node(None, 1.0)
    policy, wdl = evaluate(encode_board(board)[None].astype(np.float32))
    _backpropagate([root], _expand(root, root_legal, board.turn, policy[0], wdl[0]))

    while root.visits < simulations and time.perf_counter() < deadline:
        leaves: list[tuple[list[Node], list[chess.Move], chess.Color]] = []
        planes: list[np.ndarray] = []

        while len(leaves) < batch_size and root.visits + len(leaves) < simulations:
            path, node = [root], root
            while node.children:
                node = _select(node, c_puct)
                board.push(node.move)
                path.append(node)

            if node.pending:  # another leaf in this batch took this path: evaluate what we have
                for _ in path[1:]:
                    board.pop()
                break

            legal = [] if node.children is not None else list(board.legal_moves)
            terminal = node.terminal_value if node.terminal_value is not None else _terminal_value(board, bool(legal))
            if terminal is not None:
                node.terminal_value = terminal
                node.children = []
                _backpropagate(path, terminal)
            else:
                planes.append(encode_board(board))
                node.pending = True
                for n in path[1:]:  # virtual loss: the rest of the batch explores elsewhere
                    n.visits += 1
                    n.value_sum -= 1
                leaves.append((path, legal, board.turn))
            for _ in path[1:]:
                board.pop()

        if not leaves:
            continue
        policy, wdl = evaluate(np.stack(planes).astype(np.float32))
        for row, (path, legal, turn) in enumerate(leaves):
            leaf = path[-1]
            leaf.pending = False
            for n in path[1:]:
                n.visits -= 1
                n.value_sum += 1
            _backpropagate(path, _expand(leaf, legal, turn, policy[row], wdl[row]))

    children = sorted(root.children, key=lambda c: (-c.visits, -c.prior))
    pv, node = [], children[0]
    while node is not None:
        pv.append(node.move)
        node = max(node.children, key=lambda c: c.visits) if node.children else None
        if node is not None and node.visits == 0:
            break

    return SearchResult(
        move=children[0].move,
        value=-root.q,
        candidates=[Candidate(c.move, c.visits, c.prior, c.q) for c in children],
        pv=pv,
        simulations=root.visits,
    )


def pick_move(candidates: list[Candidate], temperature: float, rng: random.Random | None = None) -> Candidate:
    """Port of web/lib/engine/selection.ts: temperature 0 = best; else sample visits^(1/T) (priors if unsearched)."""
    if temperature <= 0:
        return candidates[0]
    rng = rng or random
    searched = any(c.visits > 0 for c in candidates)
    weights = [(c.visits if searched else c.prior) ** (1 / temperature) for c in candidates]
    total = sum(weights)
    if not total > 0:
        return candidates[0]
    r = rng.random() * total
    for c, w in zip(candidates, weights):
        r -= w
        if r < 0:
            return c
    return candidates[-1]
