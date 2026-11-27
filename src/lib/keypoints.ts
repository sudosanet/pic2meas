// RTMPose (COCO 17 keypoints, SimCC outputs) → MHR70-layout 2D joints for illustrations.

export const POSE_W = 192, POSE_H = 256, SIMCC_SPLIT = 2;

/** Argmax of each keypoint's x and y SimCC bins → crop-pixel position and score (the weaker of the two maxima). */
export function decodeSimcc(sx: Float32Array, sy: Float32Array, k: number) {
  const nx = sx.length / k, ny = sy.length / k, out: { x: number; y: number; score: number }[] = [];
  const argmax = (a: Float32Array, o: number, n: number) => { let b = 0; for (let i = 1; i < n; i++) if (a[o + i] > a[o + b]) b = i; return b; };
  for (let i = 0; i < k; i++) {
    const bx = argmax(sx, i * nx, nx), by = argmax(sy, i * ny, ny);
    out.push({ x: bx / SIMCC_SPLIT, y: by / SIMCC_SPLIT, score: Math.min(sx[i * nx + bx], sy[i * ny + by]) });
  }
  return out;
}

// COCO index → MHR70 index (eyes, ears, shoulders, elbows, wrists, hips, knees, ankles, nose).
const COCO_TO_MHR = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14];

/**
 * Replace the MHR joints that RTMPose also detects (score ≥ minScore) with its positions; the neck (base of the
 * neck, which COCO lacks) sits above the shoulders' midpoint by `neckRise` × the shoulder-to-hip distance.
 */
export function mergeCoco(mhr: Float32Array, coco: { x: number; y: number; score: number }[], neckRise: number, minScore = 0.3) {
  const out = mhr.slice();
  coco.forEach((p, i) => { if (p.score >= minScore) out.set([p.x, p.y], COCO_TO_MHR[i] * 2); });
  const [ls, rs, lh, rh] = [5, 6, 11, 12].map((i) => coco[i]);
  if ([ls, rs, lh, rh].every((p) => p.score >= minScore)) {
    const sx = (ls.x + rs.x) / 2, sy = (ls.y + rs.y) / 2, hx = (lh.x + rh.x) / 2, hy = (lh.y + rh.y) / 2;
    out.set([sx + (sx - hx) * neckRise, sy + (sy - hy) * neckRise], 69 * 2);
  }
  return out;
}
