import { describe, expect, it } from "vitest";
import { Chess } from "chess.js";
import { encodeMove, NUM_MOVES, PLANES_SIZE } from "./encode";
import { search, type Evaluator } from "./mcts";

/** Knows nothing: equal priors, every position a coin flip. */
const uniform: Evaluator = async (_planes, batch) => ({
  policy: new Float32Array(batch * NUM_MOVES),
  wdl: new Float32Array(batch * 3),
});

/** Counts material from the planes: equal priors, value = material balance for the side to move. */
const PIECE_VALUES = [1, 3, 3, 5, 9]; // pawn..queen; kings ignored
const material: Evaluator = async (planes, batch) => {
  const wdl = new Float32Array(batch * 3);
  for (let b = 0; b < batch; b++) {
    let balance = 0;
    PIECE_VALUES.forEach((value, piece) => {
      for (let sq = 0; sq < 64; sq++) {
        balance += value * planes[b * PLANES_SIZE + piece * 64 + sq]; // ours
        balance -= value * planes[b * PLANES_SIZE + (piece + 6) * 64 + sq]; // theirs
      }
    });
    wdl[b * 3] = balance / 2; // win logit up, loss logit down
    wdl[b * 3 + 2] = -balance / 2;
  }
  return { policy: new Float32Array(batch * NUM_MOVES), wdl };
};

function counting(evaluator: Evaluator) {
  const calls: number[] = [];
  const wrapped: Evaluator = (planes, batch) => {
    calls.push(batch);
    return evaluator(planes, batch);
  };
  return { wrapped, calls };
}

const uci = (m: { from: string; to: string; promotion?: string }) => m.from + m.to + (m.promotion ?? "");

describe("search", () => {
  it("finds mate in one by search alone (the evaluator knows nothing)", async () => {
    const game = new Chess("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"); // Ra8#
    const result = await search(game, uniform, { simulations: 300 });
    expect(uci(result.move)).toBe("a1a8");
    expect(result.value).toBeGreaterThan(0.5);
  });

  it("wins a hanging queen", async () => {
    const game = new Chess("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1"); // Rxd5
    const result = await search(game, material, { simulations: 300 });
    expect(uci(result.move)).toBe("d2d5");
  });

  it("looks ahead: refuses a pawn that loses the queen", async () => {
    // Qxb7?? is answered by ...Rxb7. Plain material at depth 1 would grab it.
    const game = new Chess("1r2k3/1p6/8/8/8/8/8/1Q2K3 w - - 0 1");
    const result = await search(game, material, { simulations: 400 });
    expect(uci(result.move)).not.toBe("b1b7");
    // Not by luck: the search tried Qxb7, found ...Rxb7, and abandoned it. (Its
    // average q still includes the optimistic visits from before the recapture
    // was found, so compare it with the best move rather than an absolute value.)
    const best = result.candidates[0];
    const grab = result.candidates.find((c) => uci(c.move) === "b1b7")!;
    expect(grab.visits).toBeGreaterThan(1);
    expect(grab.q).toBeLessThan(best.q - 0.15);
    expect(grab.visits).toBeLessThan(best.visits / 2);
  });

  it("with one simulation, plays the policy's favourite move", async () => {
    const game = new Chess();
    const prefersNf3: Evaluator = async (_p, batch) => {
      const policy = new Float32Array(batch * NUM_MOVES);
      policy[encodeMove({ from: "g1", to: "f3" }, "w")] = 5;
      return { policy, wdl: new Float32Array(batch * 3) };
    };
    const result = await search(game, prefersNf3, { simulations: 1 });
    expect(uci(result.move)).toBe("g1f3");
    expect(result.simulations).toBe(1);
  });

  it("runs exactly the requested simulations, in batches, and restores the game", async () => {
    const game = new Chess();
    game.move("e4");
    const fenBefore = game.fen();
    const { wrapped, calls } = counting(uniform);

    const result = await search(game, wrapped, { simulations: 200, batchSize: 8 });

    expect(result.simulations).toBe(200);
    // Root's own evaluation is the first simulation; the rest are spread over its children.
    expect(result.candidates.reduce((sum, c) => sum + c.visits, 0)).toBe(199);
    expect(Math.max(...calls)).toBeLessThanOrEqual(8);
    expect(calls.length).toBeLessThan(200 / 2); // batching actually happened
    expect(game.fen()).toBe(fenBefore);
    expect(game.history()).toEqual(["e4"]);
  });

  it("stops at the time limit", async () => {
    const result = await search(new Chess(), uniform, { simulations: 1_000_000, timeLimitMs: 0 });
    expect(result.simulations).toBe(1);
  });

  it("returns a principal variation starting with the chosen move", async () => {
    const result = await search(new Chess("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"), uniform, { simulations: 100 });
    expect(result.pv[0]).toBe(uci(result.move));
  });
});
