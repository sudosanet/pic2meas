import * as Comlink from "comlink";
import type { Backend, InferApi, Progress } from "./types";

const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
const api = Comlink.wrap<InferApi>(worker);

// One call at a time: overlapping session.run calls are not safe.
let queue: Promise<unknown> = Promise.resolve();
const serial = <T,>(f: () => Promise<T>): Promise<T> => { const p = queue.then(f, f); queue = p.catch(() => {}); return p; };

// ?ep=wasm or ?ep=webgpu forces a backend.
const requestedBackend = (): Backend | "auto" => {
  const ep = new URLSearchParams(location.search).get("ep");
  return ep === "wasm" || ep === "webgpu" ? ep : "auto";
};

export const initInference = (onProgress: (p: Progress) => void) => serial(() => api.init(requestedBackend(), Comlink.proxy(onProgress)));
export const analyseImage = (image: Blob, onProgress: (p: Progress) => void) => serial(() => api.analyse(image, Comlink.proxy(onProgress)));
