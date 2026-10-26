import { describe, expect, it } from "vitest";
import { Chess } from "chess.js";
import { encodeMove, NUM_MOVES } from "./encode";
import { interpret } from "./policy";

const wdlLogits = (w: number, d: number, l: number) => new Float32Array([w, d, l]);

describe("interpret", () => {
  it("ranks legal moves by logit and ignores illegal slots", () => {
    const game = new Chess();
    const policy = new Float32Array(NUM_MOVES);
    policy[encodeMove({ from: "e2", to: "e4" }, "w")] = 5;
    policy[encodeMove({ from: "d2", to: "d4" }, "w")] = 4;
    policy[encodeMove({ from: "e2", to: "e5" }, "w")] = 100; // illegal: must be ignored

    const { moves } = interpret(game, policy, wdlLogits(0, 0, 0));
    expect(moves).toHaveLength(20);
    expect(moves[0].move).toMatchObject({ from: "e2", to: "e4" });
    expect(moves[1].move).toMatchObject({ from: "d2", to: "d4" });
    expect(moves.reduce((sum, m) => sum + m.prior, 0)).toBeCloseTo(1, 5);
  });

  it("reads the right row of a batch", () => {
    const game = new Chess();
    const policy = new Float32Array(NUM_MOVES * 2);
    policy[NUM_MOVES + encodeMove({ from: "g1", to: "f3" }, "w")] = 10;
    const wdl = new Float32Array([0, 0, 0, 10, 0, 0]);
    const evaluation = interpret(game, policy, wdl, 1);
    expect(evaluation.moves[0].move).toMatchObject({ from: "g1", to: "f3" });
    expect(evaluation.value).toBeGreaterThan(0.99);
  });

  it("turns WDL logits into probabilities and an expected score", () => {
    const { wdl, value } = interpret(new Chess(), new Float32Array(NUM_MOVES), wdlLogits(0, 0, 0));
    wdl.forEach((p) => expect(p).toBeCloseTo(1 / 3, 5));
    expect(value).toBeCloseTo(0, 5);
  });

  it("handles extreme logits without overflowing", () => {
    const policy = new Float32Array(NUM_MOVES).fill(1000);
    const { moves, value } = interpret(new Chess(), policy, wdlLogits(-1000, 0, 1000));
    expect(moves.every((m) => Number.isFinite(m.prior))).toBe(true);
    expect(value).toBeCloseTo(-1, 5);
  });
});
