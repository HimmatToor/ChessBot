import type { Chess, Color, PieceSymbol } from "chess.js";

type CountedPiece = Exclude<PieceSymbol, "k">;
type Board = ReturnType<Chess["board"]>;

export const PIECE_VALUES: Record<CountedPiece, number> = { q: 9, r: 5, b: 3, n: 3, p: 1 };
const DISPLAY_ORDER: CountedPiece[] = ["q", "r", "b", "n", "p"];

export type MaterialSide = {
  /** Opponent pieces this side is "up" on, most valuable first. */
  captured: CountedPiece[];
  /** Material lead in pawns; 0 for the side that is level or behind. */
  advantage: number;
};

/**
 * Lichess-style material diff: compare piece counts on the board rather than
 * replaying captures, so promotions and trades come out right automatically.
 */
export function materialBalance(board: Board): Record<Color, MaterialSide> {
  const counts: Record<Color, Record<CountedPiece, number>> = {
    w: { q: 0, r: 0, b: 0, n: 0, p: 0 },
    b: { q: 0, r: 0, b: 0, n: 0, p: 0 },
  };
  for (const row of board) {
    for (const square of row) {
      if (square && square.type !== "k") counts[square.color][square.type]++;
    }
  }

  const result: Record<Color, MaterialSide> = {
    w: { captured: [], advantage: 0 },
    b: { captured: [], advantage: 0 },
  };
  let score = 0; // positive = white ahead
  for (const type of DISPLAY_ORDER) {
    const diff = counts.w[type] - counts.b[type];
    const leader = diff > 0 ? result.w : result.b;
    for (let i = 0; i < Math.abs(diff); i++) leader.captured.push(type);
    score += diff * PIECE_VALUES[type];
  }
  if (score > 0) result.w.advantage = score;
  if (score < 0) result.b.advantage = -score;
  return result;
}
