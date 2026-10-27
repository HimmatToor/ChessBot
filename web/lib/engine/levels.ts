// Difficulty levels: all use the same network, with different search budgets
// and amounts of randomness.

import type { SearchSettings } from "./protocol";

export type LevelId = "easy" | "medium" | "hard" | "max";

export type Level = {
  id: LevelId;
  label: string;
  description: string;
  settings: SearchSettings;
};

export const LEVELS: Level[] = [
  {
    id: "easy",
    label: "Easy",
    description: "Plays on instinct and makes mistakes",
    settings: { simulations: 1, temperature: 1.5, randomMoveChance: 0.15 },
  },
  {
    id: "medium",
    label: "Medium",
    description: "Thinks briefly, sometimes picks a second-best move",
    settings: { simulations: 50, temperature: 0.5 },
  },
  {
    id: "hard",
    label: "Hard",
    description: "Searches 150 positions and plays its best move",
    // Sized for the 10x128 model on single-threaded WASM (~19 ms per simulation).
    settings: { simulations: 150, timeLimitMs: 4000, temperature: 0 },
  },
  {
    id: "max",
    label: "Max",
    description: "Thinks for 6 seconds every move",
    settings: { simulations: 1_000_000, timeLimitMs: 6000, temperature: 0 },
  },
];

export const DEFAULT_LEVEL: LevelId = "medium";

export const levelById = (id: LevelId): Level => LEVELS.find((l) => l.id === id)!;
