// Web Worker that owns the neural network, so inference never blocks the page.

/// <reference lib="webworker" />

import { Chess } from "chess.js";
import { encodeBoard } from "./encode";
import { Network } from "./network";
import { interpret } from "./policy";
import type { BestMove, WorkerRequest, WorkerResponse } from "./protocol";

let network: Promise<Network> | null = null;

async function handle(request: WorkerRequest): Promise<BestMove | null> {
  switch (request.type) {
    case "init":
      network ??= Network.load(request.modelUrl, { wasmPaths: request.wasmPaths });
      await network;
      return null;

    case "bestMove": {
      if (!network) throw new Error("engine not initialised");
      const game = new Chess();
      for (const uci of request.moves) {
        game.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] });
      }
      const { policy, wdl } = await (await network).evaluate(encodeBoard(game), 1);
      const evaluation = interpret(game, policy, wdl);
      const best = evaluation.moves[0];
      if (!best) throw new Error("no legal moves");
      return { move: best.move, prior: best.prior, value: evaluation.value };
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
