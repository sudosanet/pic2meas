import type { Caps } from "@/lib/breast";
import type { Ratios } from "@/lib/fit";
import type { Level, Line2d, Pt } from "@/lib/measure";

export type { Pt };
export type Backend = "webgpu" | "wasm";
export type Progress = { stage: "download" | "session" | "segment" | "body" | "breast" | "sections"; loaded?: number; total?: number; label: string };

/** What the worker derives from one image. 2D values are in working-image pixels (w × h). */
export type Analysis = {
  w: number;
  h: number;
  backend: Backend;
  ms: { segment: number; body: number; breast: number; sections: number };
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
    front: 1 | -1; // which way the body faces in the sections' y
    jointUp: Float32Array; // 70 joint heights
    joints: Float32Array; // 70 × 3
    crotchF: number; // crotch between hip (0) and knee (1)
    girthM: { neck: number; thigh: number; calf: number };
    verts: Float32Array; // the unposed (arms-down) mesh, 3 per vertex, same frame as joints
  };
  /** Rest-pose sections every 1 cm; t = height above mid-hip (m); line2d = where the section lies in the image. */
  sections: { t: number; hull: Float32Array; armContact: number; line2d: Line2d | null }[];
  /**
   * The chest surface integrated from surface normals: d (grid × grid, height in m towards the camera) over
   * the image square at (x0, y0) of `side` px (NaN outside the chest); mPerCell = metres per grid cell at the
   * chest. caps: the two spherical caps fitted to it, in image px.
   */
  /** relief: the chest's normals directly (on its cells): spread of their vertical part, and how far they turn down. */
  breast: { d: Float32Array; x0: number; y0: number; side: number; grid: number; mPerCell: number; caps: Caps; relief: { sy: number; down: number } };
  /** The body mesh's triangles (for the 3D view of rest.verts). */
  mesh: { faces: Int32Array };
  /** The posed mesh's projected neck and mid-hip, which line2d is relative to. */
  meshAnchor: { neck: Pt; hip: Pt };
  autoLevels: Record<Level, number>;
};

export type Refit = Pick<Analysis, "rest" | "sections" | "autoLevels"> & { residual: number };

export interface InferApi {
  init(backend: Backend | "auto", onProgress: (p: Progress) => void): Promise<Backend>;
  analyse(image: Blob, onProgress: (p: Progress) => void): Promise<Analysis>;
  /** Re-measure the last analysis with its skeleton fitted to these proportions. */
  refit(target: Ratios): Promise<Refit>;
}
