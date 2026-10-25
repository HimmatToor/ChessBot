import type { Chess, Color } from "chess.js";

export type GameResult = {
  title: string;
  detail: string;
  /** null for a draw */
  winner: Color | null;
};

/** How the game ended, or null while it's still in progress. */
export function gameResult(game: Chess): GameResult | null {
  if (game.isCheckmate()) {
    const winner = game.turn() === "w" ? "b" : "w";
    return { title: "Checkmate", detail: `${winner === "w" ? "White" : "Black"} wins`, winner };
  }
  if (game.isStalemate()) return draw("by stalemate");
  if (game.isThreefoldRepetition()) return draw("by threefold repetition");
  if (game.isInsufficientMaterial()) return draw("by insufficient material");
  if (game.isDrawByFiftyMoves()) return draw("by the 50-move rule");
  return null;
}

function draw(detail: string): GameResult {
  return { title: "Draw", detail, winner: null };
}
