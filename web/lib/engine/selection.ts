// Choosing which move to actually play from the search results.

export type Candidate = { visits: number; prior: number };

/**
 * Temperature 0 plays the top candidate (the list is most-visited first).
 * Higher temperatures sample in proportion to visits^(1/T), so weaker
 * levels sometimes play their second or third choice. When nothing was
 * searched (1 simulation), the policy priors are used instead of visits.
 */
export function pickMove<T extends Candidate>(candidates: T[], temperature: number, random = Math.random): T {
  if (candidates.length === 0) throw new Error("no candidates");
  if (temperature <= 0) return candidates[0];

  const searched = candidates.some((c) => c.visits > 0);
  const weights = candidates.map((c) => (searched ? c.visits : c.prior) ** (1 / temperature));
  const total = weights.reduce((a, b) => a + b, 0);
  if (!(total > 0)) return candidates[0];

  let r = random() * total;
  for (let i = 0; i < candidates.length; i++) {
    r -= weights[i];
    if (r < 0) return candidates[i];
  }
  return candidates[candidates.length - 1];
}
