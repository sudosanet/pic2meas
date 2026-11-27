// Fit the 3D body's proportions to a drawing: MHR skeleton scales (parameters 136–203) chosen so the unposed
// body's segment ratios match the ones measured along the drawing's skeleton.

import { J, mid2, type Pt } from "./measure";

/** MHR scale parameters that set head, torso, shoulder width, upper arm, hip width, thigh and shin. */
export const FIT_PARAMS = [140, 139, 141, 142, 146, 149, 150];
export const RATIO_KEYS = ["head", "torso", "shoulders", "upperArm", "hips", "thigh", "shin"] as const;
export type Ratios = Record<(typeof RATIO_KEYS)[number], number>;

const d = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y);

/** Segment lengths over the head-to-sole chain (head + torso + thigh + shin + foot), from 2D points. */
export function ratios2d(p: { headTop: Pt; neck: Pt; hip: { l: Pt; r: Pt }; knee: { l: Pt; r: Pt }; ankle: { l: Pt; r: Pt }; sole: { l: Pt; r: Pt }; shoulder: { l: Pt; r: Pt }; elbow: { l: Pt; r: Pt } }): Ratios {
  const avg = (f: (s: "l" | "r") => number) => (f("l") + f("r")) / 2;
  const thigh = avg((s) => d(p.hip[s], p.knee[s])), shin = avg((s) => d(p.knee[s], p.ankle[s]));
  const head = d(p.headTop, p.neck), torso = d(p.neck, mid2(p.hip.l, p.hip.r));
  const chain = head + torso + thigh + shin + avg((s) => d(p.ankle[s], p.sole[s]));
  return {
    head: head / chain, torso: torso / chain, thigh: thigh / chain, shin: shin / chain,
    shoulders: d(p.shoulder.l, p.shoulder.r) / chain, hips: d(p.hip.l, p.hip.r) / chain, upperArm: avg((s) => d(p.shoulder[s], p.elbow[s])) / chain,
  };
}

/** The same ratios from an unposed (standing) body: joints 70×3 and the heights of its top and sole along `up`. */
export function ratios3d(joints: Float32Array, up: [number, number, number], top: number, sole: number): Ratios {
  const j = (i: number) => [joints[i * 3], joints[i * 3 + 1], joints[i * 3 + 2]];
  const len = (a: number, b: number) => Math.hypot(...j(a).map((v, k) => v - j(b)[k]));
  const h = (i: number) => j(i).reduce((s, v, k) => s + v * up[k], 0);
  const avg = (a: number, b: number) => (a + b) / 2, hip = avg(h(J.lHip), h(J.rHip));
  const head = top - h(J.neck), torso = h(J.neck) - hip, foot = avg(h(J.lAnkle), h(J.rAnkle)) - sole;
  const thigh = avg(len(J.lHip, J.lKnee), len(J.rHip, J.rKnee)), shin = avg(len(J.lKnee, J.lAnkle), len(J.rKnee, J.rAnkle));
  const chain = head + torso + thigh + shin + foot;
  return {
    head: head / chain, torso: torso / chain, thigh: thigh / chain, shin: shin / chain,
    shoulders: len(J.lShoulder, J.rShoulder) / chain, hips: len(J.lHip, J.rHip) / chain,
    upperArm: avg(len(J.lShoulder, J.lElbow), len(J.rShoulder, J.rElbow)) / chain,
  };
}

/** Solve a small dense system A x = b (Gaussian elimination with partial pivoting). */
export function solve(A: number[][], b: number[]) {
  const n = b.length, M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
  return x;
}

/**
 * Levenberg–Marquardt on FIT_PARAMS (forward-difference Jacobian), each kept within ±limit of its start and pulled
 * gently towards it. `evaluate` runs the body model for a full 204-vector and returns its ratios.
 */
export async function fitScales(evaluate: (params: Float32Array) => Promise<Ratios>, target: Ratios, start: Float32Array, { iterations = 6, limit = 3, prior = 0.02, step = 0.05 } = {}) {
  const res = (r: Ratios) => RATIO_KEYS.map((k) => r[k] - target[k]);
  const cost = (r: number[], p: Float32Array) => r.reduce((s, v) => s + v * v, 0) + prior ** 2 * FIT_PARAMS.reduce((s, i) => s + (p[i] - start[i]) ** 2, 0);
  let p = start.slice(), r = res(await evaluate(p)), c = cost(r, p), lambda = 1e-3;
  for (let it = 0; it < iterations; it++) {
    const Jm: number[][] = RATIO_KEYS.map(() => []);
    for (const [k, i] of FIT_PARAMS.entries()) {
      const q = p.slice(); q[i] += step;
      const rq = res(await evaluate(q));
      rq.forEach((v, row) => (Jm[row][k] = (v - r[row]) / step));
    }
    // (JᵀJ + λ diag + prior²) Δ = −Jᵀ r − prior² (p − start)
    const n = FIT_PARAMS.length, A = Array.from({ length: n }, (_, a) => Array.from({ length: n }, (_, b) => Jm.reduce((s, row) => s + row[a] * row[b], 0)));
    const g = Array.from({ length: n }, (_, a) => Jm.reduce((s, row, k) => s + row[a] * r[k], 0) + prior ** 2 * (p[FIT_PARAMS[a]] - start[FIT_PARAMS[a]]));
    for (let tries = 0; tries < 4; tries++) {
      const delta = solve(A.map((row, a) => row.map((v, b) => v + (a === b ? lambda * (v || 1) + prior ** 2 : 0))), g.map((v) => -v));
      const q = p.slice();
      FIT_PARAMS.forEach((i, k) => (q[i] = Math.min(start[i] + limit, Math.max(start[i] - limit, q[i] + delta[k]))));
      const rq = res(await evaluate(q)), cq = cost(rq, q);
      if (cq < c) { p = q; r = rq; c = cq; lambda /= 3; break; }
      lambda *= 10;
    }
  }
  return { params: p, residual: Math.sqrt(r.reduce((s, v) => s + v * v, 0) / r.length) };
}
