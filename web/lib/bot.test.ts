import { describe, expect, it } from "vitest";
import { Chess } from "chess.js";
import { randomBot } from "./bot";

describe("randomBot", () => {
  it("always returns a legal move", async () => {
    const game = new Chess();
    // Play a whole random game; game.move throws if the bot ever picks an illegal move.
    while (!game.isGameOver() && game.history().length < 200) {
      game.move(await randomBot.chooseMove(game));
    }
    expect(game.history().length).toBeGreaterThan(0);
  });

  it("finds the only legal move", async () => {
    // Black king on h8 in check from the rook on h1; Kg8 is the only move (g7 is covered by the queen).
    const move = await randomBot.chooseMove(new Chess("7k/8/5Q2/8/8/8/8/K6R b - - 0 1"));
    expect(move).toEqual({ from: "h8", to: "g8", promotion: undefined });
  });

  it("includes the promotion piece when promoting", async () => {
    // The black queen on c2 boxes the a1 king in, so White's only legal
    // moves are the four promotions of the a7 pawn.
    const move = await randomBot.chooseMove(new Chess("7k/P7/8/8/8/8/2q5/K7 w - - 0 1"));
    expect(move.from).toBe("a7");
    expect(move.to).toBe("a8");
    expect(["q", "r", "b", "n"]).toContain(move.promotion);
  });

  it("refuses positions with no legal moves", async () => {
    await expect(randomBot.chooseMove(new Chess("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"))).rejects.toThrow();
  });
});
