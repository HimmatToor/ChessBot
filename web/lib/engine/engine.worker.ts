// Web Worker that owns the neural network and the search, so thinking never blocks the page.

/// <reference lib="webworker" />

import { Chess } from "chess.js";
import { search } from "./mcts";
import { defaultThreadCount, Network } from "./network";
import type { BestMove, WorkerRequest, WorkerResponse } from "./protocol";
import { pickMove } from "./selection";

let network: Promise<Network> | null = null;
let threads = 1;

async function handle(request: WorkerRequest): Promise<BestMove | null> {
  switch (request.type) {
    case "init":
      if (!network) {
        // An explicit count (e.g. ?threads=N for benchmarking) still needs isolation to work.
        threads = self.crossOriginIsolated ? (request.numThreads ?? defaultThreadCount()) : 1;
        network = Network.load(request.modelUrl, { wasmPaths: request.wasmPaths, numThreads: threads });
      }
      await network;
      return null;

    case "search": {
      if (!network) throw new Error("engine not initialised");
      const net = await network;
      // Replay the whole game so repetition draws are detected during the search.
      const game = new Chess();
      for (const uci of request.moves) {
        game.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] });
      }
      const { settings } = request;
      const start = performance.now();
      const result = await search(game, (planes, batch) => net.evaluate(planes, batch), settings);
      const chosen =
        Math.random() < (settings.randomMoveChance ?? 0)
          ? result.candidates[Math.floor(Math.random() * result.candidates.length)]
          : pickMove(result.candidates, settings.temperature ?? 0);
      return {
        move: chosen.move,
        value: result.value,
        pv: result.pv,
        simulations: result.simulations,
        timeMs: Math.round(performance.now() - start),
        threads,
      };
    }
  }
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const { id } = event.data;
  let response: WorkerResponse;
  try {
    response = { id, ok: true, result: await handle(event.data) };
  } catch (error) {
    response = { id, ok: false, error: error instanceof Error ? error.message : String(error) };
  }
  self.postMessage(response);
};
