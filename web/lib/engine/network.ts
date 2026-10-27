// Thin wrapper around an onnxruntime-web session for chessnet.onnx.
// Interface documented in training/export_onnx.py.

import * as ort from "onnxruntime-web/wasm";
import { NUM_MOVES, NUM_PLANES, PLANES_SIZE } from "./encode";

export type NetworkOutput = {
  /** batch x 1858 move logits, row-major. */
  policy: Float32Array;
  /** batch x 3 win/draw/loss logits, row-major. */
  wdl: Float32Array;
};

export type LoadOptions = {
  /** Where ort-wasm-simd-threaded.{mjs,wasm} are served from, e.g. "/ort/". */
  wasmPaths?: string;
  /** CPU threads for inference. >1 only works when the page is cross-origin isolated. */
  numThreads?: number;
};

/**
 * Threads to use in this context: about one per physical core (half the logical
 * cores), capped at 4, when the page is cross-origin isolated (see
 * public/_headers); otherwise 1, since onnxruntime can't share memory between
 * threads without isolation. Measured on a 4-core/8-thread laptop: 1 thread
 * 25 ms/simulation, 2-4 threads 12-15 ms, but 8 threads 59 ms (the inference
 * threads fight the search and the page for the same cores).
 */
export function defaultThreadCount(): number {
  if (typeof globalThis.crossOriginIsolated === "undefined" || !globalThis.crossOriginIsolated) return 1;
  const logical = globalThis.navigator?.hardwareConcurrency ?? 1;
  return Math.max(1, Math.min(4, Math.floor(logical / 2)));
}

export class Network {
  private constructor(private readonly session: ort.InferenceSession) {}

  /** `model` is a URL or the .onnx file's bytes. */
  static async load(model: string | Uint8Array, options: LoadOptions = {}): Promise<Network> {
    if (options.wasmPaths) ort.env.wasm.wasmPaths = options.wasmPaths;
    ort.env.wasm.numThreads = options.numThreads ?? 1;
    // Two calls because create() has separate URL and bytes overloads.
    const session =
      typeof model === "string"
        ? await ort.InferenceSession.create(model, { executionProviders: ["wasm"] })
        : await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
    const io = [...session.inputNames, ...session.outputNames].join(",");
    if (io !== "planes,policy,wdl") throw new Error(`unexpected model interface: ${io}`);
    return new Network(session);
  }

  /** Evaluates the first `batch` positions in `planes` (each PLANES_SIZE floats). */
  async evaluate(planes: Float32Array, batch: number): Promise<NetworkOutput> {
    const input = new ort.Tensor("float32", planes.subarray(0, batch * PLANES_SIZE), [batch, NUM_PLANES, 8, 8]);
    const output = await this.session.run({ planes: input });
    const policy = output.policy.data as Float32Array;
    const wdl = output.wdl.data as Float32Array;
    if (policy.length !== batch * NUM_MOVES || wdl.length !== batch * 3) {
      throw new Error("model output has the wrong shape");
    }
    return { policy, wdl };
  }
}
