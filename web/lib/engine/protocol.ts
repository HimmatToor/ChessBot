// Messages between the page and engine.worker.ts.

import type { MoveInput } from "./encode";

export type WorkerRequest =
  | { id: number; type: "init"; modelUrl: string; wasmPaths: string }
  /** `moves` is the game so far in UCI (e.g. "e2e4", "e7e8q") from the standard start. */
  | { id: number; type: "bestMove"; moves: string[] };

export type BestMove = { move: MoveInput; prior: number; value: number };

export type WorkerResponse =
  | { id: number; ok: true; result: BestMove | null }
  | { id: number; ok: false; error: string };
