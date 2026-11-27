import type { Ratios } from "@/lib/fit";
import type { Level, Pt } from "@/lib/measure";

export type { Pt };
export type Backend = "webgpu" | "wasm";
export type Progress = { stage: "download" | "session" | "segment" | "body" | "sections"; loaded?: number; total?: number; label: string };

/** What the worker derives from one image. 2D values are in working-image pixels (w × h). */
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
  /** The person touches the top or bottom edge of the image. */
  cut: boolean;
  joints2d: Float32Array; // 70 × 2
  /** Landmarks found by walking the masks along the skeleton. */
  found: { headTop: Pt; chin: Pt | null; sole: { l: Pt; r: Pt }; waist: Pt | null; hem: { l: Pt | null; r: Pt | null } };
  /** The same, with RTMPose's keypoints (trained on drawings) where it is confident: used for illustrations. */
  art: { joints2d: Float32Array; found: Analysis["found"] };
  meshPoints: Float32Array; // posed mesh, projected, subsampled
  meshMask: Uint8Array; // posed mesh silhouette
  /** The same body unposed into a standing pose (arms raised). Heights in metres above mid-hip. */
  rest: {
    statureM: number;
    floorM: number; // sole height (negative)
    torsoM: number; // mid-hip to neck
    jointUp: Float32Array; // 70 joint heights
    joints: Float32Array; // 70 × 3
    crotchF: number; // crotch between hip (0) and knee (1)
    girthM: { neck: number; thigh: number; calf: number };
  };
  /** Rest-pose sections every 1 cm; t = height above mid-hip (m). */
  sections: { t: number; hull: Float32Array; armContact: number }[];
  autoLevels: Record<Level, number>;
};

export type Refit = Pick<Analysis, "rest" | "sections" | "autoLevels"> & { residual: number };

export interface InferApi {
  init(backend: Backend | "auto", onProgress: (p: Progress) => void): Promise<Backend>;
  analyse(image: Blob, onProgress: (p: Progress) => void): Promise<Analysis>;
  /** Re-measure the last analysis with its skeleton fitted to these proportions. */
  refit(target: Ratios): Promise<Refit>;
}
