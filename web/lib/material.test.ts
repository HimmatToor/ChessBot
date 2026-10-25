import { describe, expect, it } from "vitest";
import { Chess } from "chess.js";
import { materialBalance } from "./material";

const balance = (fen?: string) => materialBalance(new Chess(fen).board());

describe("materialBalance", () => {
  it("is level at the start", () => {
    expect(balance()).toEqual({
      w: { captured: [], advantage: 0 },
      b: { captured: [], advantage: 0 },
    });
  });

  it("credits a captured knight to the capturing side", () => {
    // Black's b8 knight is missing.
    const result = balance("r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    expect(result.w).toEqual({ captured: ["n"], advantage: 3 });
    expect(result.b).toEqual({ captured: [], advantage: 0 });
  });

  it("cancels out an even trade", () => {
    // Both sides have lost their queen.
    const result = balance("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1");
    expect(result.w.captured).toEqual([]);
    expect(result.b.captured).toEqual([]);
  });

  it("handles promotion: the extra queen counts, the lost pawn counts against", () => {
    // White's a-pawn promoted to a second queen on a8, taking the rook.
    const result = balance("Qnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR b KQk - 0 1");
    expect(result.w.captured).toEqual(["q", "r"]);
    expect(result.b.captured).toEqual(["p"]);
    expect(result.w.advantage).toBe(9 + 5 - 1);
  });

  it("puts the advantage on the side that is behind in captures but ahead in value", () => {
    // White has an extra rook; black has two extra pawns.
    const result = balance("1nbqkbnr/pppppppp/8/8/8/8/2PPPPPP/RNBQKBNR w KQk - 0 1");
    expect(result.w).toEqual({ captured: ["r"], advantage: 3 });
    expect(result.b).toEqual({ captured: ["p", "p"], advantage: 0 });
  });
});
