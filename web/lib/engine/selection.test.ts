import { describe, expect, it } from "vitest";
import { pickMove } from "./selection";

const candidates = [
  { name: "best", visits: 60, prior: 0.1 },
  { name: "second", visits: 30, prior: 0.2 },
  { name: "third", visits: 10, prior: 0.7 },
];

/** Picks with `random` returning each of n evenly spaced values, and counts the results. */
function distribution(temperature: number, list = candidates, n = 1000) {
  const counts: Record<string, number> = {};
  for (let i = 0; i < n; i++) {
    const { name } = pickMove(list, temperature, () => (i + 0.5) / n);
    counts[name] = (counts[name] ?? 0) + 1;
  }
  return counts;
}

describe("pickMove", () => {
  it("plays the most-visited move at temperature 0", () => {
    expect(pickMove(candidates, 0, () => 0.99).name).toBe("best");
  });

  it("samples in proportion to visits at temperature 1", () => {
    expect(distribution(1)).toEqual({ best: 600, second: 300, third: 100 });
  });

  it("sharpens towards the best move at low temperature", () => {
    // T = 0.5 squares the visits: 3600 : 900 : 100.
    const counts = distribution(0.5);
    expect(counts.best).toBeGreaterThan(750);
    expect(counts.third).toBeLessThan(30);
  });

  it("uses policy priors when nothing was searched", () => {
    const unsearched = candidates.map((c) => ({ ...c, visits: 0 }));
    expect(distribution(1, unsearched)).toEqual({ best: 100, second: 200, third: 700 });
  });

  it("never picks a move the search gave zero visits", () => {
    const list = [...candidates, { name: "unvisited", visits: 0, prior: 0.9 }];
    expect(distribution(2, list).unvisited).toBeUndefined();
  });
});
