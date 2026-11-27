// Breasts as two spherical caps on the chest wall: fitted to the normal-derived surface, placed on the bust section.
import { solve } from "./fit";
import { convexHull, mid2, type Pt } from "./measure";

/**
 * Left/right cap centres and base radius in image px, base radius in metres, height (projection) in metres; where
 * the height came from (surface normals, the body's outline in a turned view, or nothing: a guess), and how well
 * the data supported it (0–1).
 */
export type Caps = { l: Pt; r: Pt; a: number; aM: number; h: number; src: "normals" | "outline" | "prior"; conf: number };

/** Caps smaller than this (base radius, m) are noise: they don't place the bust. */
export const MIN_CAP_M = 0.015;

/** The caps' midpoint and the point a base radius below it (towards the hips), in image px. */
export function capLine(c: Caps) {
  const m = mid2(c.l, c.r), dx = c.r.x - c.l.x, dy = c.r.y - c.l.y, len = Math.hypot(dx, dy) || 1;
  let down = { x: -dy / len, y: dx / len };
  if (down.y < 0) down = { x: -down.x, y: -down.y };
  return { mid: m, below: { x: m.x + down.x * c.a, y: m.y + down.y * c.a } };
}

/**
 * Projection ÷ base radius used only as a weak prior (and for guesses): mid-range of breast base widths ~11–14 cm and
 * projections ~4–7 cm. Real ratios run ~0.6–1.0 and grow with size, so the data decides when it can.
 */
export const CAP_RATIO = 0.75;

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
 * Fit two caps (shared base radius a and projection h, each free) to d (NaN = not chest), by Levenberg–Marquardt with
 * a finite-difference Jacobian. The chest wall (a quadratic surface: the torso is round) is removed first, fitted
 * outside the starting circles only, so it can't flatten the breasts. h is pulled weakly towards CAP_RATIO × a.
 * `init` gives the starting centres and the largest base radius (px).
 */
export function fitCaps(g: Grid, init: { l: Pt; r: Pt; a: number }, iterations = 40): Caps | null {
  const G = g.grid, cell = g.side / G, toG = (p: Pt) => ({ x: (p.x - g.x0) / cell - 0.5, y: (p.y - g.y0) / cell - 0.5 });
  const cells: number[] = [];
  for (let i = 0; i < G * G; i++) if (Number.isFinite(g.d[i])) cells.push(i);
  if (cells.length < 50) return null;
  const aMax = init.a / cell, X = (i: number) => i % G, Y = (i: number) => (i / G) | 0;
  const basis = (i: number) => { const x = (X(i) - G / 2) / G, y = (Y(i) - G / 2) / G; return [1, x, y, x * x, x * y, y * y]; };
  const quadratic = (use: number[]) => {
    const rows = use.map(basis), n = 6;
    const A = Array.from({ length: n }, (_, u) => Array.from({ length: n }, (_, v) => rows.reduce((s, r) => s + r[u] * r[v], 0) + (u === v ? 1e-9 : 0)));
    const w = solve(A, Array.from({ length: n }, (_, u) => rows.reduce((s, r, k) => s + r[u] * g.d[use[k]], 0)));
    return (i: number) => basis(i).reduce((s, b, k) => s + b * w[k], 0);
  };
  // Where the breasts are: the two highest bumps left after taking the torso's overall tilt and curve out of the
  // whole chest, a breast's width apart and side by side. Weak bumps keep the skeleton's guess.
  const { L, R } = findBumps(g, cells, quadratic(cells), aMax, toG(init.l), toG(init.r));

  // The chest wall around the breasts, d ≈ quadratic in x, y (the torso is round: a plane isn't enough), fitted to
  // the cells outside both starting circles and taken out.
  const outside = cells.filter((i) => Math.min(Math.hypot(X(i) - L.x, Y(i) - L.y), Math.hypot(X(i) - R.x, Y(i) - R.y)) > aMax);
  const wall = outside.length >= 60 ? quadratic(outside) : () => 0;
  const dd = new Float32Array(G * G);
  for (const i of cells) dd[i] = g.d[i] - wall(i);

  // p = [lx, ly, rx, ry, a (cells), h (m)]
  const a0 = 0.6 * aMax;
  let p = [L.x, L.y, R.x, R.y, a0, CAP_RATIO * a0 * g.mPerCell];
  const both = (q: number[], i: number) => {
    const a = q[4] * g.mPerCell;
    return capHeight(Math.hypot(X(i) - q[0], Y(i) - q[1]) * g.mPerCell, a, q[5]) + capHeight(Math.hypot(X(i) - q[2], Y(i) - q[3]) * g.mPerCell, a, q[5]);
  };
  const prior = 0.02 * Math.sqrt(cells.length);   // weak: data across the caps outweighs it
  const res = (q: number[]) => [...cells.map((i) => both(q, i) - dd[i]), prior * (q[5] - CAP_RATIO * q[4] * g.mPerCell)];
  const cost = (r: number[]) => r.reduce((s, v) => s + v * v, 0);
  // Centres stay in their windows around the skeleton's guess (see inWindow).
  const L0 = toG(init.l), R0 = toG(init.r);
  const keep = (x: number, y: number, c0: Pt): [number, number] => {
    if (inWindow({ x, y }, c0, L0, R0, aMax)) return [x, y];
    // Pull back along the line to c0 until inside.
    let t = 1;
    while (t > 0 && !inWindow({ x: c0.x + (x - c0.x) * t, y: c0.y + (y - c0.y) * t }, c0, L0, R0, aMax)) t -= 0.05;
    return [c0.x + (x - c0.x) * Math.max(0, t), c0.y + (y - c0.y) * Math.max(0, t)];
  };
  const clamp = (q: number[]) => {
    [q[0], q[1]] = keep(q[0], q[1], L0); [q[2], q[3]] = keep(q[2], q[3], R0);
    q[4] = Math.min(aMax, Math.max(0.1 * aMax, q[4])); q[5] = Math.min(0.15, Math.max(0, q[5]));
    return q;
  };
  let r = res(p), c = cost(r), lambda = 1e-2;
  const steps = [0.5, 0.5, 0.5, 0.5, 0.5, 1e-3];
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
  // Confidence: how much of d's variation inside the circles the caps explain.
  const inside = cells.filter((i) => Math.min(Math.hypot(X(i) - p[0], Y(i) - p[1]), Math.hypot(X(i) - p[2], Y(i) - p[3])) < p[4]);
  const mean = inside.reduce((s, i) => s + dd[i], 0) / (inside.length || 1);
  const sst = inside.reduce((s, i) => s + (dd[i] - mean) ** 2, 0), sse = inside.reduce((s, i) => s + (both(p, i) - dd[i]) ** 2, 0);
  // Breasts sit side by side: if the pair leans more than 15° from the skeleton's left–right line, level them.
  {
    const sep = Math.hypot(R0.x - L0.x, R0.y - L0.y) || 1, ux = (R0.x - L0.x) / sep, uy = (R0.y - L0.y) / sep;
    const dx = p[2] - p[0], dy = p[3] - p[1], across = dx * ux + dy * uy, along = -dx * uy + dy * ux;
    if (Math.abs(along) > Math.tan((15 * Math.PI) / 180) * Math.abs(across)) {
      const half = along / 2;
      p[0] -= uy * half; p[1] += ux * half; p[2] += uy * half; p[3] -= ux * half;
    }
  }
  const px = (x: number, y: number): Pt => ({ x: g.x0 + (x + 0.5) * cell, y: g.y0 + (y + 0.5) * cell });
  return { l: px(p[0], p[1]), r: px(p[2], p[3]), a: p[4] * cell, aM: p[4] * g.mPerCell, h: p[5], src: "normals", conf: sst > 0 ? Math.max(0, 1 - sse / sst) : 0 };
}

/** The two strongest bumps of d − background (smoothed), ≥ 1.2 aMax apart and more side by side than stacked. */
function findBumps(g: Grid, cells: number[], background: (i: number) => number, aMax: number, L0: Pt, R0: Pt) {
  const G = g.grid, r = new Float32Array(G * G).fill(NaN);
  for (const i of cells) r[i] = g.d[i] - background(i);
  const k = Math.max(1, Math.round(aMax / 4)), sm = new Float32Array(G * G).fill(-Infinity);
  for (const i of cells) {
    const x = i % G, y = (i / G) | 0;
    let sum = 0, n = 0;
    for (let dy = -k; dy <= k; dy++) for (let dx = -k; dx <= k; dx++) {
      const xx = x + dx, yy = y + dy, v = xx >= 0 && yy >= 0 && xx < G && yy < G ? r[yy * G + xx] : NaN;
      if (Number.isFinite(v)) { sum += v; n++; }
    }
    if (n > k * k) sm[i] = sum / n;
  }
  // Each side on its own, only near where the skeleton puts that nipple: a bump on a sleeve or an arm can't pull
  // the circles apart.
  const best = (c0: Pt) => {
    let bi = -1;
    for (const i of cells) if (Number.isFinite(sm[i]) && inWindow({ x: i % G, y: (i / G) | 0 }, c0, L0, R0, aMax) && (bi < 0 || sm[i] > sm[bi])) bi = i;
    return bi >= 0 && sm[bi] >= MIN_BUMP ? { x: bi % G, y: (bi / G) | 0 } : null;
  };
  const l = best(L0), rr = best(R0);
  return l && rr ? { L: l, R: rr } : { L: L0, R: R0 };
}

/** Where a breast's centre may be: within 20 % of the nipples' spacing sideways and 0.6 aMax up or down of c0. */
function inWindow(p: Pt, c0: Pt, L0: Pt, R0: Pt, aMax: number) {
  const sep = Math.hypot(R0.x - L0.x, R0.y - L0.y) || 1, ux = (R0.x - L0.x) / sep, uy = (R0.y - L0.y) / sep;
  const dx = p.x - c0.x, dy = p.y - c0.y, across = dx * ux + dy * uy, along = -dx * uy + dy * ux;
  return Math.abs(across) <= 0.2 * sep && Math.abs(along) <= 0.6 * aMax;
}

/** A bump lower than this (m) above the torso's curve isn't taken as a breast's position. */
const MIN_BUMP = 0.004;

/**
 * A rest-pose section (hull: x across, y = front·forward) with the caps added where the level line crosses them.
 * Each cap sits on the chest wall facing the way the wall faces there (the torso is round, so caps out to the side
 * lean outwards). x: the cap's position across the section (m); dz: its centre's offset from the section plane (m).
 */
export function withCaps(hull: Float32Array, front: 1 | -1, caps: { x: number; dz: number; aM: number; h: number }[]) {
  const pts = Array.from(hull);
  for (const c of caps) {
    const sph = capSphere(hull, front, c.x, c.aM, c.h);
    if (!sph) continue;
    const rr = sph.R * sph.R - c.dz * c.dz;
    if (rr <= 0) continue;
    const r = Math.sqrt(rr);
    for (let k = 0; k < 64; k++) {
      const ang = (2 * Math.PI * k) / 64, px = sph.cx + r * Math.cos(ang), py = sph.cy + r * Math.sin(ang);
      if ((px - sph.wall.x) * sph.wall.nx + (py - sph.wall.y) * sph.wall.ny > 0) pts.push(px, py);
    }
  }
  return convexHull(pts);
}

/**
 * The sphere a cap is cut from, in section coordinates: on the front wall at x (the most forward hull edge crossing x)
 * facing the way the wall faces there, its centre R − h behind the wall.
 */
export function capSphere(hull: Float32Array, front: 1 | -1, x: number, aM: number, h: number) {
  if (h <= 0) return null;
  const n = hull.length / 2, R = (aM * aM + h * h) / (2 * h);
  let wall: { x: number; y: number; nx: number; ny: number } | null = null;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n, x0 = hull[i * 2], y0 = hull[i * 2 + 1], x1 = hull[j * 2], y1 = hull[j * 2 + 1];
    if ((x0 - x) * (x1 - x) > 0 || x0 === x1) continue;
    const y = y0 + ((x - x0) / (x1 - x0)) * (y1 - y0);
    if (wall && front * y <= front * wall.y) continue;
    const ex = x1 - x0, ey = y1 - y0, l = Math.hypot(ex, ey);
    let nx = ey / l, ny = -ex / l;
    if (front * ny < 0) { nx = -nx; ny = -ny; }
    wall = { x, y, nx, ny };
  }
  return wall && { wall, R, cx: wall.x - wall.nx * (R - h), cy: wall.y - wall.ny * (R - h) };
}
