// Board and move encoding for the neural network.
//
// A line-by-line port of training/chessnet/encode.py, which is the source of
// truth (see the docstring there for the full layout). encode.test.ts checks
// this file against shared/encoding_vectors.json, generated from the Python.
//
// Squares are numbered like python-chess: a1 = 0, b1 = 1, ..., h8 = 63, so
// rank = sq >> 3 and file = sq & 7. Everything is in the side-to-move frame:
// for Black the board is mirrored vertically (sq ^ 56) and colours swapped.

import type { Chess, Color, PieceSymbol, Square } from "chess.js";

export const NUM_PLANES = 18;
export const NUM_MOVES = 1858;
export const ENCODING_VERSION = 1;
/** Floats per encoded position (18 planes x 64 squares). */
export const PLANES_SIZE = NUM_PLANES * 64;

export type MoveInput = { from: string; to: string; promotion?: string };

const FILES = "abcdefgh";
const PIECE_ORDER: PieceSymbol[] = ["p", "n", "b", "r", "q", "k"];
const UNDERPROMOTIONS = ["n", "b", "r"];

const squareName = (sq: number) => FILES[sq & 7] + String((sq >> 3) + 1);
const squareIndex = (name: string) => FILES.indexOf(name[0]) + (Number(name[1]) - 1) * 8;
/** Real square <-> side-to-move frame (its own inverse). */
const orient = (sq: number, turn: Color) => (turn === "w" ? sq : sq ^ 56);

function buildMoveTable(): string[] {
  const moves: string[] = [];
  for (let from = 0; from < 64; from++) {
    for (let to = 0; to < 64; to++) {
      if (to === from) continue;
      const dr = (to >> 3) - (from >> 3);
      const df = (to & 7) - (from & 7);
      const queenLike = dr === 0 || df === 0 || Math.abs(dr) === Math.abs(df);
      const knightLike =
        (Math.abs(dr) === 1 && Math.abs(df) === 2) || (Math.abs(dr) === 2 && Math.abs(df) === 1);
      if (queenLike || knightLike) moves.push(squareName(from) + squareName(to));
    }
  }
  for (let file = 0; file < 8; file++) {
    for (const df of [-1, 0, 1]) {
      const toFile = file + df;
      if (toFile < 0 || toFile >= 8) continue;
      for (const piece of UNDERPROMOTIONS) {
        moves.push(FILES[file] + "7" + FILES[toFile] + "8" + piece);
      }
    }
  }
  return moves;
}

/** Policy index -> move in UCI notation, in the side-to-move frame. */
export const MOVES: readonly string[] = buildMoveTable();
const MOVE_TO_INDEX = new Map(MOVES.map((uci, i) => [uci, i]));

if (MOVES.length !== NUM_MOVES) throw new Error(`move table has ${MOVES.length} entries`);

/**
 * Writes the position's 18x8x8 planes into `out` at `offset` (plane, rank,
 * file order) and returns `out`. Pass a shared buffer to batch positions.
 */
export function encodeBoard(
  game: Chess,
  out: Float32Array = new Float32Array(PLANES_SIZE),
  offset = 0,
): Float32Array {
  out.fill(0, offset, offset + PLANES_SIZE);
  const us = game.turn();
  const them: Color = us === "w" ? "b" : "w";
  const set = (plane: number, sq: number) => (out[offset + plane * 64 + sq] = 1);
  const fillPlane = (plane: number) => out.fill(1, offset + plane * 64, offset + (plane + 1) * 64);

  for (const row of game.board()) {
    for (const piece of row) {
      if (!piece) continue;
      const plane = PIECE_ORDER.indexOf(piece.type) + (piece.color === us ? 0 : 6);
      set(plane, orient(squareIndex(piece.square), us));
    }
  }

  if (us === "b") fillPlane(12);
  const ours = game.getCastlingRights(us);
  const theirs = game.getCastlingRights(them);
  if (ours.k) fillPlane(13);
  if (ours.q) fillPlane(14);
  if (theirs.k) fillPlane(15);
  if (theirs.q) fillPlane(16);

  // Only when an en-passant capture is actually legal (matches the Python).
  const enPassant = game.moves({ verbose: true }).find((m) => m.isEnPassant());
  if (enPassant) set(17, orient(squareIndex(enPassant.to), us));

  return out;
}

/** Policy index of a move played by `turn`. Queen promotions use the plain pawn move. */
export function encodeMove(move: MoveInput, turn: Color): number {
  let uci = squareName(orient(squareIndex(move.from), turn)) + squareName(orient(squareIndex(move.to), turn));
  if (move.promotion && move.promotion !== "q") uci += move.promotion;
  const index = MOVE_TO_INDEX.get(uci);
  if (index === undefined) throw new Error(`move ${move.from}${move.to} has no policy slot`);
  return index;
}

/** Real-board move for a policy index in `game`. Not checked for legality. */
export function decodeMove(index: number, game: Chess): MoveInput {
  const uci = MOVES[index];
  const turn = game.turn();
  const from = squareName(orient(squareIndex(uci.slice(0, 2)), turn));
  const to = squareName(orient(squareIndex(uci.slice(2, 4)), turn));
  if (uci.length === 5) return { from, to, promotion: uci[4] };
  const piece = game.get(from as Square);
  const lastRank = to[1] === "1" || to[1] === "8";
  return piece?.type === "p" && lastRank ? { from, to, promotion: "q" } : { from, to };
}

/** Every legal move in `game` with its policy index. */
export function legalMoves(game: Chess): { index: number; move: MoveInput }[] {
  const turn = game.turn();
  return game.moves({ verbose: true }).map((m) => {
    const move: MoveInput = { from: m.from, to: m.to, promotion: m.promotion };
    return { index: encodeMove(move, turn), move };
  });
}
