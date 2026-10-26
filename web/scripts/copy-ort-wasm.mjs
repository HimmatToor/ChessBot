// Copies onnxruntime-web's WebAssembly runtime into public/ort/ so the site
// serves it itself (no CDN). Runs automatically before `dev` and `build`.
import { copyFileSync, mkdirSync } from "node:fs";

const src = new URL("../node_modules/onnxruntime-web/dist/", import.meta.url);
const dest = new URL("../public/ort/", import.meta.url);
mkdirSync(dest, { recursive: true });
for (const file of ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"]) {
  copyFileSync(new URL(file, src), new URL(file, dest));
}
console.log("copied onnxruntime-web wasm to public/ort/");
