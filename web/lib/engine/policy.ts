// Turning raw network output into move probabilities and a position value.

import type { Chess } from "chess.js";
import { legalMoves, NUM_MOVES, type LegalMove } from "./encode";

export type RankedMove = LegalMove & { prior: number };

export type Evaluation = {
  /** Legal moves with their probability under the policy, best first. */
  moves: RankedMove[];
  /** [win, draw, loss] probabilities for the side to move. */
  wdl: [number, number, number];
  /** Expected score for the side to move, from -1 (lost) to +1 (won). */
  value: number;
};

/**
 * Softmax over the legal moves only: illegal slots are ignored, so the
 * probabilities of legal moves sum to 1. `row` picks the position in a batch.
 */
export function interpret(game: Chess, policy: Float32Array, wdl: Float32Array, row = 0): Evaluation {
  return interpretLegal(legalMoves(game), policy, wdl, row);
}

/** Same as interpret, for a legal-move list computed earlier (the search does this). */
export function interpretLegal(legal: LegalMove[], policy: Float32Array, wdl: Float32Array, row = 0): Evaluation {
  const logits = policy.subarray(row * NUM_MOVES, (row + 1) * NUM_MOVES);

  let max = -Infinity;
  for (const { index } of legal) max = Math.max(max, logits[index]);
  let sum = 0;
  const moves = legal.map((entry) => {
    const weight = Math.exp(logits[entry.index] - max); // subtract max for numerical stability
    sum += weight;
    return { ...entry, prior: weight };
  });
  for (const m of moves) m.prior /= sum;
  moves.sort((a, b) => b.prior - a.prior);

  const [w, d, l] = softmax(wdl.subarray(row * 3, row * 3 + 3));
  return { moves, wdl: [w, d, l], value: w - l };
}

function softmax(logits: ArrayLike<number>): number[] {
  const max = Math.max(...Array.from(logits));
  const exps = Array.from(logits, (x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}
