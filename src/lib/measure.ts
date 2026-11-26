// Measurement logic shared by the worker and the UI. Pixel y grows downward.

type Mask = Uint8Array;
export type Vec3 = [number, number, number];

const LENGTH_CATEGORIES = [
  [0.4, "マイクロ"], [0.7, "ミニ"], [0.9, "膝上"], [1.15, "膝丈"],
  [1.6, "ミディ"], [1.9, "ロング"], [Infinity, "マキシ"],
] as const;

/** Largest 4-connected component of a w×h mask. */
export function largestComponent(mask: Mask, w: number, h: number): { mask: Mask; size: number } {
  const label = new Int32Array(w * h), stack: number[] = [];
  let best = 0, bestSize = 0, next = 0;
  for (let i = 0; i < w * h; i++) {
    if (!mask[i] || label[i]) continue;
    let size = 0;
    label[i] = ++next;
    stack.push(i);
    while (stack.length) {
      const p = stack.pop()!, x = p % w, y = (p / w) | 0;
      size++;
      for (const q of [x > 0 ? p - 1 : -1, x < w - 1 ? p + 1 : -1, y > 0 ? p - w : -1, y < h - 1 ? p + w : -1])
        if (q >= 0 && mask[q] && !label[q]) { label[q] = next; stack.push(q); }
    }
    if (size > bestSize) { bestSize = size; best = next; }
  }
  const out = new Uint8Array(w * h);
  if (best) for (let i = 0; i < w * h; i++) out[i] = label[i] === best ? 1 : 0;
  return { mask: out, size: bestSize };
}

export function extent(mask: Mask, w: number, h: number) {
  let x0 = w, x1 = -1, y0 = h, y1 = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
    if (mask[y * w + x]) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  if (x1 < 0) return null;
  return { x0, x1, y0, y1, touchTop: y0 <= 1, touchBottom: y1 >= h - 2 };
}

const median = (a: number[]) => [...a].sort((x, y) => x - y)[a.length >> 1] ?? NaN;

/** Waistband and hem: median top and bottom edges over the central 70% of the garment's columns. */
export function garmentLines(mask: Mask, w: number, h: number) {
  const e = extent(mask, w, h);
  if (!e) return null;
  const tops: number[] = [], bottoms: number[] = [], m = (e.x1 - e.x0) * 0.15;
  for (let x = Math.ceil(e.x0 + m); x <= e.x1 - m; x++) {
    let t = -1, b = -1;
    for (let y = e.y0; y <= e.y1; y++) if (mask[y * w + x]) { if (t < 0) t = y; b = y; }
    if (t >= 0) { tops.push(t); bottoms.push(b); }
  }
  return { waistY: median(tops), hemY: median(bottoms) };
}

/** The run of mask pixels on row y nearest to column cx: the body's width there. */
export function rowSpan(mask: Mask, w: number, h: number, y: number, cx: number) {
  const r = Math.min(h - 1, Math.max(0, Math.round(y))) * w;
  let best: { x0: number; x1: number } | null = null, bestD = Infinity;
  for (let x = 0; x < w; x++) {
    if (!mask[r + x]) continue;
    const x0 = x;
    while (x + 1 < w && mask[r + x + 1]) x++;
    const d = Math.max(0, x0 - cx, cx - x);
    if (d < bestD) { bestD = d; best = { x0, x1: x }; }
  }
  return best;
}

/** Hem position u on a hip (0) – knee (1) – ankle (2) scale, and its category. */
export function lengthCategory(hemY: number, hipY: number, kneeY: number, ankleY: number) {
  const u = hemY <= kneeY ? (hemY - hipY) / (kneeY - hipY) : 1 + (hemY - kneeY) / (ankleY - kneeY);
  return { u, label: LENGTH_CATEGORIES.find(([lim]) => u < lim)![1] };
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: Vec3): Vec3 => { const l = Math.hypot(...a); return [a[0] / l, a[1] / l, a[2] / l]; };

/** Convex hull of 2D points [x0, y0, x1, y1, …]. */
export function convexHull(pts: number[]): Float32Array {
  const p: [number, number][] = [];
  for (let i = 0; i < pts.length; i += 2) p.push([pts[i], pts[i + 1]]);
  p.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return new Float32Array(p.flat());
  const turn = (o: number[], a: number[], b: number[]) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (arr: [number, number][]) => {
    const hh: [number, number][] = [];
    for (const q of arr) { while (hh.length >= 2 && turn(hh[hh.length - 2], hh[hh.length - 1], q) <= 0) hh.pop(); hh.push(q); }
    hh.pop();
    return hh;
  };
  return new Float32Array([...half(p), ...half([...p].reverse())].flat());
}

/** Perimeter of a closed polygon, with x scaled by sx. */
export function perimeter(poly: Float32Array, sx = 1): number {
  let per = 0;
  const n = poly.length / 2;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    per += Math.hypot((poly[i * 2] - poly[j * 2]) * sx, poly[i * 2 + 1] - poly[j * 2 + 1]);
  }
  return per;
}

/** armContact > 0: an arm lay against the body at this height. */
export type Section = { t: number; hull: Float32Array; armContact: number };
/** Capsule around an arm bone, removed from sections. */
type Cutout = { a: Vec3; b: Vec3; r: number };

const segDist = (p: Vec3, { a, b }: Cutout) => {
  const ab = sub(b, a), s = Math.max(0, Math.min(1, dot(sub(p, a), ab) / dot(ab, ab)));
  return Math.hypot(p[0] - a[0] - ab[0] * s, p[1] - a[1] - ab[1] * s, p[2] - a[2] - ab[2] * s);
};

/**
 * Mesh cross-section at height t along `up`: edge–plane intersections minus the arm capsules, the cluster nearest
 * the body axis, its convex hull (x along `side`, y along up × side). `link` bridges the gap an arm cutout leaves.
 * ponytail: the convex hull overestimates concave sections by a few %.
 */
export function sectionAt(verts: Float32Array, faces: Int32Array, origin: Vec3, up: Vec3, side: Vec3, t: number, cutouts: Cutout[] = [], link = 0.1): Omit<Section, "t"> {
  const fwd = cross(up, side), n = verts.length / 3;
  const hgt = new Float32Array(n), a = new Float32Array(n), b = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const d = sub([verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2]], origin);
    hgt[i] = dot(d, up) - t; a[i] = dot(d, side); b[i] = dot(d, fwd);
  }
  const pts: number[] = [], removed: number[] = [];
  for (let f = 0; f < faces.length; f += 3) for (let e = 0; e < 3; e++) {
    const p = faces[f + e], q = faces[f + ((e + 1) % 3)];
    if (p < q && hgt[p] < 0 !== hgt[q] < 0) {
      const s = hgt[p] / (hgt[p] - hgt[q]);
      const v = [0, 1, 2].map((k) => verts[p * 3 + k] + s * (verts[q * 3 + k] - verts[p * 3 + k])) as Vec3;
      (cutouts.some((c) => segDist(v, c) < c.r) ? removed : pts).push(a[p] + s * (a[q] - a[p]), b[p] + s * (b[q] - b[p]));
    }
  }
  const m = pts.length / 2;
  if (m < 3) return { hull: new Float32Array(), armContact: 0 };

  // Flood fill from the point nearest the axis, neighbours looked up in a grid of link-sized cells.
  const cell = (v: number) => Math.floor(v / link), grid = new Map<string, number[]>();
  for (let i = 0; i < m; i++) { const k = `${cell(pts[i * 2])},${cell(pts[i * 2 + 1])}`; (grid.get(k) ?? grid.set(k, []).get(k)!).push(i); }
  let seed = 0, bestD = Infinity;
  for (let i = 0; i < m; i++) { const d = Math.hypot(pts[i * 2], pts[i * 2 + 1]); if (d < bestD) { bestD = d; seed = i; } }
  const inComp = new Uint8Array(m), stack = [seed];
  inComp[seed] = 1;
  while (stack.length) {
    const i = stack.pop()!, ax = pts[i * 2], ay = pts[i * 2 + 1], cx = cell(ax), cy = cell(ay);
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++)
      for (const j of grid.get(`${cx + dx},${cy + dy}`) ?? [])
        if (!inComp[j] && Math.hypot(ax - pts[j * 2], ay - pts[j * 2 + 1]) < link) { inComp[j] = 1; stack.push(j); }
  }

  // Count removed arm points within 2 cm of the body.
  const comp: number[] = [], near = new Set<string>(), C = 0.02;
  for (let i = 0; i < m; i++) if (inComp[i]) { comp.push(pts[i * 2], pts[i * 2 + 1]); near.add(`${Math.floor(pts[i * 2] / C)},${Math.floor(pts[i * 2 + 1] / C)}`); }
  let armContact = 0;
  for (let i = 0; i < removed.length; i += 2) {
    const cx = Math.floor(removed[i] / C), cy = Math.floor(removed[i + 1] / C);
    if ([-1, 0, 1].some((dx) => [-1, 0, 1].some((dy) => near.has(`${cx + dx},${cy + dy}`)))) armContact++;
  }
  return { hull: convexHull(comp), armContact };
}

// MHR70 joint indices.
export const J = {
  nose: 0, lShoulder: 5, rShoulder: 6, lElbow: 7, rElbow: 8, lHip: 9, rHip: 10, lKnee: 11, rKnee: 12,
  lAnkle: 13, rAnkle: 14, rWrist: 41, lWrist: 62, neck: 69,
} as const;
const FINGER_TIPS = { l: [42, 46, 50, 54, 58], r: [21, 25, 29, 33, 37] };
const ARM_BONES: [number, number, number][] = [   // from, to, capsule radius (m)
  [J.lShoulder, J.lElbow, 0.055], [J.lElbow, J.lWrist, 0.045], [J.rShoulder, J.rElbow, 0.055], [J.rElbow, J.rWrist, 0.045],
  ...FINGER_TIPS.l.map((tip): [number, number, number] => [J.lWrist, tip, 0.04]),
  ...FINGER_TIPS.r.map((tip): [number, number, number] => [J.rWrist, tip, 0.04]),
];

const joint = (joints: Float32Array, i: number): Vec3 => [joints[i * 3], joints[i * 3 + 1], joints[i * 3 + 2]];

export const armCutouts = (joints: Float32Array): Cutout[] => ARM_BONES.map(([i, k, r]) => ({ a: joint(joints, i), b: joint(joints, k), r }));

/** Origin at mid-hip, up towards the neck, side = camera x made orthogonal; front = which way the body faces along up × side. */
export function bodyFrame(joints: Float32Array) {
  const j = (i: number) => joint(joints, i);
  const origin: Vec3 = [0, 1, 2].map((k) => (j(J.lHip)[k] + j(J.rHip)[k]) / 2) as Vec3;
  const toNeck = sub(j(J.neck), origin), up = norm(toNeck);
  const x: Vec3 = [1, 0, 0], side = norm(sub(x, up.map((u) => u * dot(x, up)) as Vec3));
  const front: 1 | -1 = dot(sub(j(J.nose), j(J.neck)), cross(up, side)) >= 0 ? 1 : -1;
  return { origin, up, side, torso: Math.hypot(...toNeck), front };
}

export type Level = "bust" | "underbust" | "waist" | "hip";
export const LEVELS: Level[] = ["bust", "underbust", "waist", "hip"];
export const LEVEL_LABEL: Record<Level, string> = { bust: "バスト", underbust: "アンダーバスト", waist: "ウエスト", hip: "ヒップ" };

/**
 * Section index for each girth. Bust: furthest forward; underbust: most receded 5–13 cm below it.
 * Waist/hip: min/max girth, preferring sections no arm touched.
 */
export function autoLevels(sections: Section[], torso: number, front: 1 | -1 = 1): Record<Level, number> {
  const per = sections.map((s) => (s.hull.length >= 6 ? perimeter(s.hull) : NaN));
  const reach = sections.map((s) => {
    if (s.hull.length < 6) return NaN;
    let m = -Infinity;
    for (let i = 1; i < s.hull.length; i += 2) m = Math.max(m, front * s.hull[i]);
    return m;
  });
  const pick = (score: number[], lo: number, hi: number, better: (a: number, b: number) => boolean, preferFree = false) => {
    for (const free of preferFree ? [true, false] : [false]) {
      let best = -1;
      sections.forEach((s, i) => {
        if (s.t >= lo && s.t <= hi && Number.isFinite(score[i]) && (!free || !s.armContact) && (best < 0 || better(score[i], score[best]))) best = i;
      });
      if (best >= 0) return best;
    }
    return -1;
  };
  const bust = pick(reach, 0.55 * torso, 0.9 * torso, (a, b) => a > b);
  const bt = bust >= 0 ? sections[bust].t : 0.7 * torso;
  return {
    bust,
    underbust: pick(reach, bt - 0.13, bt - 0.05, (a, b) => a < b),
    waist: pick(per, 0.25 * torso, 0.55 * torso, (a, b) => a < b, true),
    hip: pick(per, -0.06, 0.12, (a, b) => a > b, true),
  };
}

/** Width of a section along x. */
export function sectionWidth(hull: Float32Array): number {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < hull.length; i += 2) { lo = Math.min(lo, hull[i]); hi = Math.max(hi, hull[i]); }
  return hi - lo;
}
