// Runs the real exported model (public/models/chessnet.onnx) through
// onnxruntime-web, exactly as the worker does in the browser.

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { Chess } from "chess.js";
import { encodeBoard, PLANES_SIZE } from "./encode";
import { Network } from "./network";
import { interpret } from "./policy";

const modelBytes = new Uint8Array(readFileSync(new URL("../../public/models/chessnet.onnx", import.meta.url)));

describe("Network with the shipped model", () => {
  it("evaluates a batch and every chosen move is legal", async () => {
    const network = await Network.load(modelBytes);

    const games: Chess[] = [];
    const game = new Chess();
    for (const san of ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"]) {
      games.push(new Chess(game.fen()));
      game.move(san);
    }

    const planes = new Float32Array(games.length * PLANES_SIZE);
    games.forEach((g, i) => encodeBoard(g, planes, i * PLANES_SIZE));
    const { policy, wdl } = await network.evaluate(planes, games.length);

    games.forEach((g, i) => {
      const evaluation = interpret(g, policy, wdl, i);
      const best = evaluation.moves[0].move;
      expect(() => new Chess(g.fen()).move(best), g.fen()).not.toThrow();
      expect(evaluation.value).toBeGreaterThanOrEqual(-1);
      expect(evaluation.value).toBeLessThanOrEqual(1);
    });
  });

  it("has learned something: likes a sensible first move", async () => {
    const network = await Network.load(modelBytes);
    const start = new Chess();
    const { policy, wdl } = await network.evaluate(encodeBoard(start), 1);
    const top5 = interpret(start, policy, wdl).moves.slice(0, 5).map((m) => m.move.from + m.move.to);
    // A trained net puts mainstream openings first, not e.g. a2a3/h2h4.
    expect(top5.some((m) => ["e2e4", "d2d4", "g1f3", "c2c4"].includes(m)), top5.join(" ")).toBe(true);
  });
});
