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
};

export class Network {
  private constructor(private readonly session: ort.InferenceSession) {}

  /** `model` is a URL or the .onnx file's bytes. */
  static async load(model: string | Uint8Array, options: LoadOptions = {}): Promise<Network> {
    if (options.wasmPaths) ort.env.wasm.wasmPaths = options.wasmPaths;
    // Multi-threading needs cross-origin isolation headers; stay single-threaded for now.
    ort.env.wasm.numThreads = 1;
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
