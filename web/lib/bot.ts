import type { Chess } from "chess.js";

export type BotMove = { from: string; to: string; promotion?: string };

/**
 * Anything that can pick a move. It gets the whole game (not just the
 * position) so engines can account for repetitions.
 */
export interface Bot {
  name: string;
  /** Optional warm-up, e.g. downloading a model before the first move. */
  prepare?(): Promise<unknown>;
  chooseMove(game: Chess): Promise<BotMove>;
}

export const randomBot: Bot = {
  name: "Random bot",
  async chooseMove(game) {
    const moves = game.moves({ verbose: true });
    if (moves.length === 0) throw new Error("No legal moves in this position");
    const move = moves[Math.floor(Math.random() * moves.length)];
    return { from: move.from, to: move.to, promotion: move.promotion };
  },
};
