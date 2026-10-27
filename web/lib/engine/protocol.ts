// Messages between the page and engine.worker.ts.

import type { MoveInput } from "./encode";

export type SearchSettings = {
  /** 1 = play the network's favourite move without searching. */
  simulations: number;
  timeLimitMs?: number;
  /** 0 = always the best move; higher = more varied (see selection.ts). */
  temperature?: number;
  /** Chance (0..1) of ignoring the search and playing a random legal move. */
  randomMoveChance?: number;
};

export type WorkerRequest =
  /** `numThreads` overrides the automatic choice (see defaultThreadCount). */
  | { id: number; type: "init"; modelUrl: string; wasmPaths: string; numThreads?: number }
  /** `moves` is the game so far in UCI (e.g. "e2e4", "e7e8q") from the standard start. */
  | { id: number; type: "search"; moves: string[]; settings: SearchSettings };

export type BestMove = {
  move: MoveInput;
  /** Expected score for the side to move, -1..1. */
  value: number;
  /** Expected line of play, in UCI. */
  pv: string[];
  simulations: number;
  timeMs: number;
  /** CPU threads the network ran on. */
  threads: number;
};

export type WorkerResponse =
  | { id: number; ok: true; result: BestMove | null }
  | { id: number; ok: false; error: string };
