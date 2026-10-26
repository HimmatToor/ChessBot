import { describe, expect, it } from "vitest";
import { Chess } from "chess.js";
import { exportPgn } from "./pgn";

function play(...moves: string[]) {
  const game = new Chess();
  moves.forEach((m) => game.move(m));
  return game;
}

const DATE = new Date(2026, 9, 25);

describe("exportPgn", () => {
  it("writes headers and moves for a finished game", () => {
    const pgn = exportPgn(play("f3", "e5", "g4", "Qh4#"), { date: DATE });
    expect(pgn).toContain('[Date "2026.10.25"]');
    expect(pgn).toContain('[White "White"]');
    expect(pgn).toContain('[Result "0-1"]');
    expect(pgn).toContain("1. f3 e5 2. g4 Qh4# 0-1");
  });

  it("marks an unfinished game with *", () => {
    const pgn = exportPgn(play("e4"), { date: DATE, white: "You", black: "ChessBot" });
    expect(pgn).toContain('[Result "*"]');
    expect(pgn).toContain('[Black "ChessBot"]');
    expect(pgn).toContain("1. e4 *");
  });

  it("round-trips: the exported PGN loads back to the same position", () => {
    const game = play("e4", "e5", "Nf3", "Nc6", "Bb5", "a6");
    const loaded = new Chess();
    loaded.loadPgn(exportPgn(game, { date: DATE }));
    expect(loaded.fen()).toBe(game.fen());
  });

  it("does not modify the live game", () => {
    const game = play("e4");
    exportPgn(game, { date: DATE });
    expect(game.getHeaders().Event).not.toBe("ChessBot game");
  });
});
