// Fast move generation for the search, via chess.js internals.
//
// chess.js's public moves({ verbose: true }) / move() / undo() build full FEN
// strings for every move object (~2 ms per call), which would dominate the
// search. These internal methods do the same work without that. They are
// private API, so:
//   - chess.js is pinned to an exact version in package.json;
//   - this is the only file that touches them, and it checks they exist;
//   - fastMoves.test.ts checks results against the public API.

import type { Chess } from "chess.js";

/** chess.js's internal move: squares are 0x88 indices. */
export type InternalMove = { from: number; to: number; promotion?: string; flags: number };

export type FastMove = { from: string; to: string; promotion?: string; internal: InternalMove };

interface ChessInternals {
  _moves(options?: { legal?: boolean }): InternalMove[];
  _makeMove(move: InternalMove): void;
  _undoMove(): InternalMove | null;
  _incPositionCount(): void;
  _decPositionCount(hash: bigint): void;
  _hash: bigint;
}

function internals(game: Chess): ChessInternals {
  const g = game as unknown as ChessInternals;
  if (
    typeof g._moves !== "function" ||
    typeof g._makeMove !== "function" ||
    typeof g._undoMove !== "function" ||
    typeof g._incPositionCount !== "function" ||
    typeof g._decPositionCount !== "function" ||
    typeof g._hash !== "bigint"
  ) {
    throw new Error("chess.js internals changed: update lib/engine/fastMoves.ts for this chess.js version");
  }
  return g;
}

const FILES = "abcdefgh";
/** 0x88 index (a8 = 0, h1 = 119) -> algebraic. */
const algebraic = (sq: number) => FILES[sq & 0xf] + String(8 - (sq >> 4));

export function fastLegalMoves(game: Chess): FastMove[] {
  return internals(game)
    ._moves({ legal: true })
    .map((m) => ({ from: algebraic(m.from), to: algebraic(m.to), promotion: m.promotion, internal: m }));
}

/** Plays a move from fastLegalMoves, keeping repetition counts correct. */
export function pushMove(game: Chess, move: InternalMove): void {
  const g = internals(game);
  g._makeMove(move);
  g._incPositionCount();
}

/** Takes back the last move made with pushMove (or move()). */
export function popMove(game: Chess): void {
  const g = internals(game);
  const hash = g._hash;
  if (!g._undoMove()) throw new Error("no move to undo");
  g._decPositionCount(hash);
}
