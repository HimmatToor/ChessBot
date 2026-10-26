// The neural-network bot: a Bot whose moves come from engine.worker.ts.

import type { Chess } from "chess.js";
import type { Bot, BotMove } from "@/lib/bot";
import type { BestMove, WorkerRequest, WorkerResponse } from "./protocol";

const MODEL_URL = "/models/chessnet.onnx";
const WASM_PATHS = "/ort/";

type DistributiveOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;

export function createNeuralBot(name = "ChessBot"): Bot {
  let worker: Worker | null = null;
  let ready: Promise<unknown> | null = null;
  let nextId = 0;
  const pending = new Map<number, { resolve: (r: BestMove | null) => void; reject: (e: Error) => void }>();

  function request(message: DistributiveOmit<WorkerRequest, "id">): Promise<BestMove | null> {
    if (!worker) {
      // Created lazily: this module is also evaluated during the static build, where Worker doesn't exist.
      worker = new Worker(new URL("./engine.worker.ts", import.meta.url), { type: "module" });
      worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
        const handlers = pending.get(event.data.id);
        pending.delete(event.data.id);
        if (!handlers) return;
        if (event.data.ok) handlers.resolve(event.data.result);
        else handlers.reject(new Error(event.data.error));
      };
      worker.onerror = (event) => {
        const error = new Error(event.message || "engine worker crashed");
        pending.forEach(({ reject }) => reject(error));
        pending.clear();
      };
    }
    const id = nextId++;
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
      worker!.postMessage({ ...message, id } as WorkerRequest);
    });
  }

  function prepare(): Promise<unknown> {
    ready ??= request({ type: "init", modelUrl: MODEL_URL, wasmPaths: WASM_PATHS }).catch((error) => {
      ready = null; // allow a retry on the next move
      throw error;
    });
    return ready;
  }

  return {
    name,
    prepare,
    async chooseMove(game: Chess): Promise<BotMove> {
      // Snapshot the moves before awaiting: the caller's game object may change meanwhile.
      const moves = game.history({ verbose: true }).map((m) => m.from + m.to + (m.promotion ?? ""));
      await prepare();
      const best = await request({ type: "bestMove", moves });
      if (!best) throw new Error("engine returned no move");
      return best.move;
    },
  };
}
