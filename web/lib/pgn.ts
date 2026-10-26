import { Chess } from "chess.js";
import { gameResult } from "@/lib/gameResult";

type PgnOptions = {
  white?: string;
  black?: string;
  date?: Date;
};

/** PGN for the game so far, with standard Seven Tag Roster headers. */
export function exportPgn(game: Chess, { white = "White", black = "Black", date = new Date() }: PgnOptions = {}): string {
  // Replay onto a fresh board so we don't touch the live game's headers.
  const copy = new Chess();
  for (const san of game.history()) copy.move(san);

  copy.setHeader("Event", "ChessBot game");
  copy.setHeader("Site", "ChessBot");
  copy.setHeader("Date", formatPgnDate(date));
  copy.setHeader("Round", "-");
  copy.setHeader("White", white);
  copy.setHeader("Black", black);
  copy.setHeader("Result", pgnResult(game));
  return copy.pgn();
}

export function pgnResult(game: Chess): string {
  const result = gameResult(game);
  if (!result) return "*";
  if (result.winner === "w") return "1-0";
  if (result.winner === "b") return "0-1";
  return "1/2-1/2";
}

function formatPgnDate(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}.${pad(date.getMonth() + 1)}.${pad(date.getDate())}`;
}
