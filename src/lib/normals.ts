// Surface normals → a height map, anchored to the fitted body: the chest's shape beyond what the body model has.
// Grids are G×G, row-major, y down. Heights point towards the camera (metres).

/** Nearest depth per grid cell of a triangle mesh, already projected: xy in grid units, z = distance (Infinity = none). */
export function rasterDepth(gx: Float32Array, gy: Float32Array, z: Float32Array, faces: Int32Array, G: number) {
  const out = new Float32Array(G * G).fill(Infinity);
  for (let f = 0; f < faces.length; f += 3) {
    const a = faces[f], b = faces[f + 1], c = faces[f + 2];
    const x0 = Math.max(0, Math.ceil(Math.min(gx[a], gx[b], gx[c]))), x1 = Math.min(G - 1, Math.floor(Math.max(gx[a], gx[b], gx[c])));
    const y0 = Math.max(0, Math.ceil(Math.min(gy[a], gy[b], gy[c]))), y1 = Math.min(G - 1, Math.floor(Math.max(gy[a], gy[b], gy[c])));
    if (x0 > x1 || y0 > y1) continue;
    const det = (gy[b] - gy[c]) * (gx[a] - gx[c]) + (gx[c] - gx[b]) * (gy[a] - gy[c]);
    if (Math.abs(det) < 1e-9) continue;
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const l1 = ((gy[b] - gy[c]) * (x - gx[c]) + (gx[c] - gx[b]) * (y - gy[c])) / det;
      const l2 = ((gy[c] - gy[a]) * (x - gx[c]) + (gx[a] - gx[c]) * (y - gy[c])) / det, l3 = 1 - l1 - l2;
      if (l1 < 0 || l2 < 0 || l3 < 0) continue;
      const d = l1 * z[a] + l2 * z[b] + l3 * z[c];
      if (d < out[y * G + x]) out[y * G + x] = d;
    }
  }
  return out;
}

/**
 * Height h on the masked cells minimising Σ (h[j] − h[i] − g)² over neighbouring cells + Σ λ (h − h0)², where g is
 * the slope the normals imply (n: x right, y up, z towards the camera; `s` = metres per cell) and h0 the body
 * model's height. Large λ pins a cell to the body, small λ lets the normals shape it. Conjugate gradients.
 */
export function integrateNormals(n: { x: Float32Array; y: Float32Array; z: Float32Array }, mask: Uint8Array, h0: Float32Array, lambda: Float32Array, G: number, s: number, iterations = 400) {
  const idx = new Int32Array(G * G).fill(-1), cells: number[] = [];
  for (let i = 0; i < G * G; i++) if (mask[i]) { idx[i] = cells.length; cells.push(i); }
  const N = cells.length, b = new Float64Array(N), diag = new Float64Array(N);
  // Slopes per cell (image y grows down, so dh/dy_img = +ny/nz).
  const sx = (i: number) => (-n.x[i] / Math.max(0.2, n.z[i])) * s, sy = (i: number) => (n.y[i] / Math.max(0.2, n.z[i])) * s;
  const edges: [number, number, number][] = [];   // (k, l, g): h_l − h_k ≈ g
  for (let k = 0; k < N; k++) {
    const i = cells[k], x = i % G;
    if (x + 1 < G && idx[i + 1] >= 0) edges.push([k, idx[i + 1], (sx(i) + sx(i + 1)) / 2]);
    if (i + G < G * G && idx[i + G] >= 0) edges.push([k, idx[i + G], (sy(i) + sy(i + G)) / 2]);
  }
  for (const [k, l, g] of edges) { diag[k]++; diag[l]++; b[k] -= g; b[l] += g; }
  for (let k = 0; k < N; k++) { diag[k] += lambda[cells[k]]; b[k] += lambda[cells[k]] * h0[cells[k]]; }
  const A = (v: Float64Array, out: Float64Array) => {
    for (let k = 0; k < N; k++) out[k] = diag[k] * v[k];
    for (const [k, l] of edges) { out[k] -= v[l]; out[l] -= v[k]; }
  };
  const h = new Float64Array(N);
  for (let k = 0; k < N; k++) h[k] = h0[cells[k]];
  const r = new Float64Array(N), p = new Float64Array(N), Ap = new Float64Array(N);
  A(h, Ap);
  for (let k = 0; k < N; k++) { r[k] = b[k] - Ap[k]; p[k] = r[k]; }
  let rr = r.reduce((a, v) => a + v * v, 0);
  for (let it = 0; it < iterations && rr > 1e-18; it++) {
    A(p, Ap);
    const alpha = rr / p.reduce((a, v, k) => a + v * Ap[k], 0);
    for (let k = 0; k < N; k++) { h[k] += alpha * p[k]; r[k] -= alpha * Ap[k]; }
    const rr2 = r.reduce((a, v) => a + v * v, 0);
    for (let k = 0; k < N; k++) p[k] = r[k] + (rr2 / rr) * p[k];
    rr = rr2;
  }
  const out = new Float32Array(G * G).fill(NaN);
  for (let k = 0; k < N; k++) out[cells[k]] = h[k];
  return out;
}
