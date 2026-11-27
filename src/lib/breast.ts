// Breasts as two spherical caps on the chest wall: fitted to the normal-derived surface, placed on the bust section.
import { solve } from "./fit";
import { convexHull, perimeter, type Pt } from "./measure";

/** Left/right cap centres and base radius in image px, base radius in metres, height (projection) in metres. */
export type Caps = { l: Pt; r: Pt; a: number; aM: number; h: number };

/** Height of a cap (base radius a, height h) at distance rho from its centre; 0 outside. */
export function capHeight(rho: number, a: number, h: number) {
  if (rho >= a || h <= 0) return 0;
  const R = (a * a + h * h) / (2 * h);
  return Math.sqrt(R * R - rho * rho) - (R - h);
}

/** Volume of one cap (m³): π h (3a² + h²) / 6. */
export const capVolume = (aM: number, h: number) => (Math.PI * h * (3 * aM * aM + h * h)) / 6;

type Grid = { d: Float32Array; x0: number; y0: number; side: number; grid: number; mPerCell: number };

/**
 * Fit d (NaN = not chest) ≈ plane + two caps sharing base radius and height, by Levenberg–Marquardt with a
 * finite-difference Jacobian. The plane absorbs a tilt between the drawing and the body model. `init` gives the
 * starting centres (image px) and base radius (px).
 */
export function fitCaps(g: Grid, init: { l: Pt; r: Pt; a: number }, iterations = 30): Caps | null {
  const G = g.grid, cell = g.side / G, toG = (p: Pt) => ({ x: (p.x - g.x0) / cell - 0.5, y: (p.y - g.y0) / cell - 0.5 });
  const cells: number[] = [];
  for (let i = 0; i < G * G; i++) if (Number.isFinite(g.d[i])) cells.push(i);
  if (cells.length < 50) return null;
  const L = toG(init.l), R = toG(init.r), a0 = init.a / cell;
  // p = [plane c, cx, cy, lx, ly, rx, ry, a (cells), h (m)]
  let p = [0, 0, 0, L.x, L.y, R.x, R.y, a0, 0.02];
  const model = (q: number[], i: number) => {
    const x = i % G, y = (i / G) | 0, a = Math.max(1, q[7]) * g.mPerCell;
    return q[0] + q[1] * (x - G / 2) * 0.01 + q[2] * (y - G / 2) * 0.01
      + capHeight(Math.hypot(x - q[3], y - q[4]) * g.mPerCell, a, q[8]) + capHeight(Math.hypot(x - q[5], y - q[6]) * g.mPerCell, a, q[8]);
  };
  const res = (q: number[]) => cells.map((i) => model(q, i) - g.d[i]);
  const cost = (r: number[]) => r.reduce((s, v) => s + v * v, 0);
  // Centres stay within half a base radius of where they started, so the caps can't wander onto the sleeves.
  const near = (v: number, v0: number) => Math.min(v0 + 0.5 * a0, Math.max(v0 - 0.5 * a0, v));
  const clamp = (q: number[]) => {
    q[3] = near(q[3], L.x); q[4] = near(q[4], L.y); q[5] = near(q[5], R.x); q[6] = near(q[6], R.y);
    q[7] = Math.min(1.6 * a0, Math.max(0.5 * a0, q[7])); q[8] = Math.min(0.12, Math.max(0, q[8]));
    return q;
  };
  let r = res(p), c = cost(r), lambda = 1e-2;
  const steps = [1e-4, 1e-3, 1e-3, 0.5, 0.5, 0.5, 0.5, 0.5, 1e-3];
  for (let it = 0; it < iterations; it++) {
    const J = p.map((_, k) => { const q = p.slice(); q[k] += steps[k]; return res(q).map((v, i) => (v - r[i]) / steps[k]); });
    const n = p.length, A = Array.from({ length: n }, (_, u) => Array.from({ length: n }, (_, v) => J[u].reduce((s, x, i) => s + x * J[v][i], 0)));
    const b = Array.from({ length: n }, (_, u) => -J[u].reduce((s, x, i) => s + x * r[i], 0));
    let improved = false;
    for (let tries = 0; tries < 5 && !improved; tries++) {
      const delta = solve(A.map((row, u) => row.map((v, w) => v + (u === w ? lambda * (v || 1) : 0))), b);
      const q = clamp(p.map((v, k) => v + delta[k])), rq = res(q), cq = cost(rq);
      if (cq < c) { p = q; r = rq; c = cq; lambda /= 3; improved = true; } else lambda *= 10;
    }
    if (!improved) break;
  }
  const px = (x: number, y: number): Pt => ({ x: g.x0 + (x + 0.5) * cell, y: g.y0 + (y + 0.5) * cell });
  return { l: px(p[3], p[4]), r: px(p[5], p[6]), a: p[7] * cell, aM: p[7] * g.mPerCell, h: p[8] };
}

/**
 * A rest-pose section (hull: x across, y = front·forward) with the caps added where the level line crosses them.
 * `place` gives each cap's position across the section (x, m) and its vertical offset from the section plane (m).
 */
export function withCaps(hull: Float32Array, front: 1 | -1, caps: { x: number; dz: number; aM: number; h: number }[]) {
  const pts = Array.from(hull);
  for (const { x, dz, aM, h } of caps) {
    if (h <= 0) continue;
    const R = (aM * aM + h * h) / (2 * h), rr = R * R - dz * dz;
    if (rr <= 0) continue;
    // Front of the chest wall at x: where the hull's edges cross x, the furthest forward.
    let wall = -Infinity;
    for (let i = 0; i < hull.length; i += 2) {
      const j = (i + 2) % hull.length, x0 = hull[i], x1 = hull[j];
      if ((x0 - x) * (x1 - x) > 0 || x0 === x1) continue;
      wall = Math.max(wall, front * (hull[i + 1] + ((x - x0) / (x1 - x0)) * (hull[j + 1] - hull[i + 1])));
    }
    if (!Number.isFinite(wall)) continue;
    const cy = wall - (R - h), r = Math.sqrt(rr);
    for (let k = 0; k <= 32; k++) {
      const ang = (Math.PI * k) / 32, px = x + r * Math.cos(ang), py = cy + r * Math.sin(ang);
      if (py > wall) pts.push(px, front * py);
    }
  }
  return convexHull(pts);
}

export const bustWithCaps = (hull: Float32Array, front: 1 | -1, caps: Parameters<typeof withCaps>[2]) => perimeter(withCaps(hull, front, caps));
