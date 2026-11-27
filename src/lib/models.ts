// Served from R2. Keys are versioned and cached as immutable: bump the version when a file changes.
import { env } from "onnxruntime-web/webgpu";

export const MODELS = {
  seg: "/models/segformer_b2_clothes-fp32-v1.onnx",
  hmr: "/models/instanthmr-medium-fit2-v1.1.onnx",
  body: "/models/mhr-body-v1.1.onnx",
  pose: "/models/rtmpose-m-humanart-v1.onnx",   // illustrations' 2D keypoints (made by tools/convert_rtmpose.py)
} as const;

export function configureOrt() {
  env.wasm.wasmPaths = `/models/ort-${env.versions.web}/`;
  env.logLevel = "error";
}
