import { Chess } from "chess.js";

export type BotMove = { from: string; to: string; promotion?: string };

/**
 * Anything that can pick a move. The random bot below is a placeholder;
 * the neural-network bot (ONNX in a Web Worker) will implement the same
 * interface, which is why chooseMove is async.
 */
export interface Bot {
  name: string;
  chooseMove(fen: string): Promise<BotMove>;
}

export const randomBot: Bot = {
  name: "Random bot",
  async chooseMove(fen) {
    const moves = new Chess(fen).moves({ verbose: true });
    if (moves.length === 0) throw new Error("No legal moves in this position");
    const move = moves[Math.floor(Math.random() * moves.length)];
    return { from: move.from, to: move.to, promotion: move.promotion };
  },
};
