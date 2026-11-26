import type { Level } from "@/lib/measure";

export type Backend = "webgpu" | "wasm";
export type Progress = { stage: "download" | "session" | "segment" | "body" | "sections"; loaded?: number; total?: number; label: string };

export type Pt = { x: number; y: number };

/** What the worker derives from one image, in working-image pixels (w × h). */
export type Analysis = {
  w: number;
  h: number;
  backend: Backend;
  ms: { segment: number; body: number; sections: number };
  /** Transparent background: the alpha channel is the person mask. */
  transparent: boolean;
  person: Uint8Array;
  garment: Uint8Array;
  garmentKind: "skirt" | "dress" | "none";
  garmentLines: { waistY: number; hemY: number } | null;
  silhouette: { x0: number; x1: number; y0: number; y1: number; touchTop: boolean; touchBottom: boolean };
  /** Bottom of the face segment (≈ chin). */
  chinY: number | null;
  joints2d: Float32Array; // 70 × 2
  meshPoints: Float32Array; // projected, subsampled
  meshMask: Uint8Array; // projected silhouette
  meshSpan: { headY: number; soleY: number };
  meshStatureM: number;
  /** Every 1 cm along the body axis; y, cx = image position of the axis there. */
  sections: { t: number; y: number; cx: number; hull: Float32Array; armContact: number }[];
  autoLevels: Record<Level, number>;
};

export interface InferApi {
  init(backend: Backend | "auto", onProgress: (p: Progress) => void): Promise<Backend>;
  analyse(image: Blob, onProgress: (p: Progress) => void): Promise<Analysis>;
}
