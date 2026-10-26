import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { Chess } from "chess.js";
import { fastLegalMoves, popMove, pushMove } from "./fastMoves";

const fens: string[] = JSON.parse(
  readFileSync(new URL("../../../shared/encoding_vectors.json", import.meta.url), "utf8"),
).positions.map((p: { fen: string }) => p.fen);

const uci = (m: { from: string; to: string; promotion?: string }) => m.from + m.to + (m.promotion ?? "");

describe("fastMoves (chess.js internals)", () => {
  it(`generates exactly the public legal moves in ${fens.length} positions`, () => {
    for (const fen of fens) {
      const game = new Chess(fen);
      const fast = fastLegalMoves(game).map(uci).sort();
      const pub = game.moves({ verbose: true }).map(uci).sort();
      expect(fast, fen).toEqual(pub);
    }
  });

  it("push/pop matches public move/undo and restores the position exactly", () => {
    for (const fen of fens.slice(0, 100)) {
      const game = new Chess(fen);
      for (const move of fastLegalMoves(game)) {
        const reference = new Chess(fen);
        reference.move({ from: move.from, to: move.to, promotion: move.promotion });

        pushMove(game, move.internal);
        expect(game.fen(), `${fen} ${uci(move)}`).toBe(reference.fen());
        popMove(game);
        expect(game.fen()).toBe(fen);
      }
    }
  });

  it("keeps repetition counts correct", () => {
    const game = new Chess();
    const shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"];
    const play = (u: string) =>
      pushMove(game, fastLegalMoves(game).find((m) => uci(m) === u)!.internal);

    [...shuffle, ...shuffle].forEach(play); // start position now seen 3 times
    expect(game.isThreefoldRepetition()).toBe(true);
    popMove(game);
    expect(game.isThreefoldRepetition()).toBe(false);
  });

  it("works alongside the public API (moves made with move() can be popped)", () => {
    const game = new Chess();
    game.move("e4");
    popMove(game);
    expect(game.fen()).toBe(new Chess().fen());
  });
});
