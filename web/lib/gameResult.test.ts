import { describe, expect, it } from "vitest";
import { Chess } from "chess.js";
import { gameResult } from "./gameResult";

function play(...moves: string[]) {
  const game = new Chess();
  moves.forEach((m) => game.move(m));
  return game;
}

describe("gameResult", () => {
  it("is null while the game is in progress", () => {
    expect(gameResult(play("e4", "e5"))).toBeNull();
  });

  it("names the winner on checkmate", () => {
    expect(gameResult(play("f3", "e5", "g4", "Qh4#"))).toEqual({
      title: "Checkmate",
      detail: "Black wins",
      winner: "b",
    });
  });

  it("reports stalemate as a draw", () => {
    expect(gameResult(new Chess("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"))).toMatchObject({
      title: "Draw",
      detail: "by stalemate",
      winner: null,
    });
  });

  it("reports threefold repetition", () => {
    const game = play("Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8");
    expect(gameResult(game)?.detail).toBe("by threefold repetition");
  });

  it("reports insufficient material", () => {
    expect(gameResult(new Chess("8/8/4k3/8/8/3K4/8/8 w - - 0 1"))?.detail).toBe(
      "by insufficient material",
    );
  });

  it("reports the 50-move rule", () => {
    expect(gameResult(new Chess("8/8/4k3/8/8/3K4/4R3/8 w - - 100 80"))?.detail).toBe(
      "by the 50-move rule",
    );
  });
});
