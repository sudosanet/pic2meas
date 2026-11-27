import { expect, it } from "vitest";
import type { Analysis } from "@/infer/types";
import { jisBraSize } from "./cup";
import { decodeSimcc, mergeCoco } from "./keypoints";
import { arcTo, armCutouts, autoLevels, bodyFrame, convexHull, insideRange, onPolyline, perimeter, sectionAt, sectionLine2d, spanAcross, type Pt, type Vec3 } from "./measure";
import { autoLandmarks, computeResults } from "./results";

it("positions along a polyline and across a mask", () => {
  const line = [{ x: 0, y: 0 }, { x: 0, y: 10 }, { x: 10, y: 10 }];
  const pos = onPolyline({ x: 4, y: 12 }, line);
  expect(pos).toEqual({ seg: 1, frac: 0.4 });
  expect(arcTo(line, pos)).toBe(14);

  // A 20 px wide band tilted 45°: its width across the band is 20 wherever we measure it.
  const w = 100, m = new Uint8Array(w * w);
  for (let y = 0; y < w; y++) for (let x = 0; x < w; x++) if (Math.abs(x - y) / Math.SQRT2 <= 10) m[y * w + x] = 1;
  const s = spanAcross(m, w, w, { x: 50, y: 50 }, { x: 1, y: -1 })!;
  expect(s.b - s.a).toBeGreaterThan(18);
  expect(s.b - s.a).toBeLessThan(22);
  expect(insideRange(m, w, w, [{ x: 0, y: 50 }, { x: 99, y: 50 }])!.first.x).toBeCloseTo(36, 0);
});

// A tiny synthetic analysis: skeleton from 2D points, a straight rest body, a rectangle as both person and skirt.
function fakeAnalysis(rot: number): { a: Analysis; P: (p: Pt) => Pt } {
  const c = Math.cos(rot), s = Math.sin(rot), P = (p: Pt) => ({ x: 200 + (p.x - 200) * c - (p.y - 200) * s, y: 200 + (p.x - 200) * s + (p.y - 200) * c });
  const j2 = new Float32Array(140), set = (i: number, p: Pt) => { const q = P(p); j2[i * 2] = q.x; j2[i * 2 + 1] = q.y; };
  set(69, { x: 200, y: 100 }); set(0, { x: 200, y: 80 });
  set(9, { x: 190, y: 200 }); set(10, { x: 210, y: 200 });
  set(11, { x: 190, y: 280 }); set(12, { x: 210, y: 280 });
  set(13, { x: 190, y: 360 }); set(14, { x: 210, y: 360 });
  set(67, { x: 180, y: 105 }); set(68, { x: 220, y: 105 }); set(5, { x: 182, y: 110 }); set(6, { x: 218, y: 110 });
  set(7, { x: 175, y: 160 }); set(8, { x: 225, y: 160 }); set(62, { x: 172, y: 205 }); set(41, { x: 228, y: 205 });
  const up = new Float32Array(70);
  up[69] = 0.5; up[9] = up[10] = 0; up[11] = up[12] = -0.4; up[13] = up[14] = -0.8;
  const hull = convexHull(Array.from({ length: 32 }, (_, k) => [0.15 * Math.cos(k / 5), 0.1 * Math.sin(k / 5)]).flat());
  // Each section's image line: across the spine at its height (mid-hip y 200, neck y 100), 60 px wide.
  const sections = Array.from({ length: 60 }, (_, i) => {
    const t = -0.1 + i / 100, c = P({ x: 200, y: 200 - (t / 0.5) * 100 });
    return { t, hull, armContact: 0, line2d: { c, dir: { x: Math.cos(rot), y: Math.sin(rot) }, a: -30, b: 30 } };
  });
  const found = {
    headTop: P({ x: 200, y: 50 }), chin: P({ x: 200, y: 90 }), sole: { l: P({ x: 190, y: 380 }), r: P({ x: 210, y: 380 }) },
    waist: P({ x: 200, y: 180 }), hem: { l: P({ x: 190, y: 240 }), r: P({ x: 210, y: 240 }) },
  };
  return {
    P,
    a: {
      w: 400, h: 400, backend: "wasm", ms: { segment: 0, body: 0, breast: 0, sections: 0 }, transparent: false,
      breast: { d: new Float32Array(0), x0: 0, y0: 0, side: 1, grid: 0, mPerCell: 0, caps: { l: { x: 190, y: 130 }, r: { x: 210, y: 130 }, a: 8, aM: 0.04, h: 0 } },
      person: new Uint8Array(160000).fill(1), garment: new Uint8Array(160000), garmentKind: "skirt", cut: false, joints2d: j2,
      found, art: { joints2d: j2, found },
      meshPoints: new Float32Array(), meshMask: new Uint8Array(),
      rest: { statureM: 1.6, floorM: -0.9, torsoM: 0.5, front: 1, jointUp: up, joints: new Float32Array(210), crotchF: 0.2, girthM: { neck: 0.33, thigh: 0.55, calf: 0.35 } },
      sections, autoLevels: { bust: 40, underbust: 35, waist: 30, hip: 10 }, meshAnchor: { neck: P({ x: 200, y: 100 }), hip: P({ x: 200, y: 200 }) },
    },
  };
}

it("skirt length and category don't depend on how the body is tilted", () => {
  for (const mode of ["photo", "illust"] as const) {
    const out = [0, 0.5, -1.2].map((rot) => {
      const { a } = fakeAnalysis(rot), lm = autoLandmarks(a);
      return computeResults(a, lm, lm, { heightCm: 160, mode, calibrate2D: false }, { girth: {} });
    });
    for (const r of out) {
      expect(r.skirt!.cm).toBeCloseTo(out[0].skirt!.cm, 3);
      expect(r.skirt!.category).toBe("ミニ");
      expect(r.headRatio).toBeCloseTo(out[0].headRatio, 3);
      for (const [g, group] of r.details.entries()) for (const [i, d] of group.items.entries()) expect(d.value).toBeCloseTo(out[0].details[g].items[i].value, 2);
    }
  }
  // Photo: waist 20 % above mid-hip (0.1 m) + hem halfway to the knee (0.2 m) = 0.3 m, × 160 / 1.6.
  const { a } = fakeAnalysis(0.3), lm = autoLandmarks(a);
  expect(computeResults(a, lm, lm, { heightCm: 160, mode: "photo", calibrate2D: false }, { girth: {} }).skirt!.cm).toBeCloseTo(30, 3);
});

// Vertical cylinder meshes (rings at y0 and y1).
function cylinders(specs: { cx: number; r: number; y0: number; y1: number; n: number }[]) {
  const v: number[] = [], f: number[] = [];
  for (const { cx, r, y0, y1, n } of specs) {
    const o = v.length / 3;
    for (const y of [y0, y1]) for (let k = 0; k < n; k++) v.push(cx + r * Math.cos((2 * Math.PI * k) / n), y, r * Math.sin((2 * Math.PI * k) / n));
    for (let k = 0; k < n; k++) { const k2 = (k + 1) % n; f.push(o + k, o + k2, o + n + k, o + k2, o + n + k2, o + n + k); }
  }
  return { verts: new Float32Array(v), faces: new Int32Array(f) };
}
const TORSO = 2 * Math.PI * 0.15;

it("arms touching the torso are cut out and flagged", () => {
  const { verts, faces } = cylinders([{ cx: 0, r: 0.15, y0: 0, y1: 0.6, n: 64 }, { cx: 0.19, r: 0.04, y0: 0.05, y1: 0.55, n: 16 }, { cx: -0.19, r: 0.04, y0: 0.05, y1: 0.55, n: 16 }]);
  const joints = new Float32Array(70 * 3), set = (i: number, p: Vec3) => joints.set(p, i * 3);
  set(9, [0.1, 0, 0]); set(10, [-0.1, 0, 0]); set(69, [0, 0.6, 0]); set(0, [0, 0.75, 0]);
  set(5, [0.19, 0.55, 0]); set(7, [0.19, 0.3, 0]); set(62, [0.19, 0.05, 0]);
  set(6, [-0.19, 0.55, 0]); set(8, [-0.19, 0.3, 0]); set(41, [-0.19, 0.05, 0]);
  const { origin, up, side } = bodyFrame(joints);
  expect(perimeter(sectionAt(verts, faces, origin, up, side, 0.3).hull)).toBeGreaterThan(TORSO * 1.1);
  const s = sectionAt(verts, faces, origin, up, side, 0.3, armCutouts(joints));
  expect(Math.abs(perimeter(s.hull) - TORSO) / TORSO).toBeLessThan(0.03);
  expect(s.armContact).toBeGreaterThan(0);

  // In the image (posed mesh projected: rotated 0.5 rad, 100 px/m) the section's line is the torso's alone.
  const proj = new Float32Array((verts.length / 3) * 2), c = Math.cos(0.5), sn = Math.sin(0.5);
  for (let i = 0; i < verts.length / 3; i++) { const x = verts[i * 3] * 100, y = -verts[i * 3 + 1] * 100; proj.set([x * c - y * sn, x * sn + y * c], i * 2); }
  const line = sectionLine2d(s.edges, proj)!;
  expect(line.b - line.a).toBeGreaterThan(28);   // the capsules trim the torso's sides a little where arms touch
  expect(line.b - line.a).toBeLessThan(31);      // arms included would be 46
  expect(Math.atan2(line.dir.y, line.dir.x)).toBeCloseTo(0.5, 2);
});

it("2D-width calibration stretches x", () => {
  const pts: number[] = [];
  for (let k = 0; k < 256; k++) pts.push(0.15 * Math.cos((2 * Math.PI * k) / 256), 0.1 * Math.sin((2 * Math.PI * k) / 256));
  const a = 0.3, b = 0.1, h = ((a - b) / (a + b)) ** 2;
  expect(perimeter(convexHull(pts), 2)).toBeCloseTo(Math.PI * (a + b) * (1 + (3 * h) / (10 + Math.sqrt(4 - 3 * h))), 3);
});

it("levels: bust near its prior height, underbust below it, always in order", () => {
  const ring = (bulge: number) => { const pts: number[] = []; for (let k = 0; k < 64; k++) { const y = 0.1 * Math.sin((2 * Math.PI * k) / 64); pts.push(0.15 * Math.cos((2 * Math.PI * k) / 64), y < 0 ? y - bulge : y); } return convexHull(pts); };
  // Breasts peaking at t = 0.4 (inside the prior window 0.66 × 0.6 ± 0.036).
  const breasts = Array.from({ length: 71 }, (_, i) => ({ t: i / 100, hull: ring(0.03 * Math.exp(-(((i / 100 - 0.4) / 0.03) ** 2))), armContact: 0 }));
  const lv = autoLevels(breasts, 0.6, -1);
  expect(breasts[lv.bust].t).toBeCloseTo(0.4, 2);
  expect(breasts[lv.underbust].t).toBeCloseTo(0.4 - 0.17 * 0.6, 1);
  // A flat body: the levels still come out ordered with ≥ 3 cm between them.
  const flat = Array.from({ length: 71 }, (_, i) => ({ t: -0.1 + i / 100, hull: ring(0), armContact: 0 }));
  const f = autoLevels(flat, 0.5, -1), t = (k: keyof typeof f) => flat[f[k]].t;
  expect(t("bust") - t("underbust")).toBeGreaterThanOrEqual(0.03 - 1e-9);
  expect(t("underbust") - t("waist")).toBeGreaterThanOrEqual(0.03 - 1e-9);
  expect(t("waist") - t("hip")).toBeGreaterThanOrEqual(0.03 - 1e-9);
});

it("JIS bra size: cups every 2.5 cm (±1.25), band in 5 cm steps", () => {
  expect(jisBraSize(83.74, 70).cup).toBe("B");
  expect(jisBraSize(83.76, 70).cup).toBe("C");
  expect(jisBraSize(76, 70).cup).toBe("AA未満");
  expect(jisBraSize(102, 70).cup).toBe("I超");
  expect(jisBraSize(85, 72.6).label).toBe("B75");
});

it("RTMPose SimCC decoding and merging into MHR joints", () => {
  const sx = new Float32Array(2 * 384), sy = new Float32Array(2 * 512);
  sx[100] = 0.9; sy[300] = 0.8; sx[384 + 10] = 0.2; sy[512 + 20] = 0.9;
  expect(decodeSimcc(sx, sy, 2)).toEqual([{ x: 50, y: 150, score: expect.closeTo(0.8, 5) }, { x: 5, y: 10, score: expect.closeTo(0.2, 5) }]);

  const coco = Array.from({ length: 17 }, () => ({ x: 0, y: 0, score: 0.9 }));
  coco[5] = { x: 90, y: 100, score: 0.9 }; coco[6] = { x: 110, y: 100, score: 0.9 };
  coco[11] = { x: 95, y: 200, score: 0.9 }; coco[12] = { x: 105, y: 200, score: 0.9 }; coco[13] = { x: 1, y: 2, score: 0.1 };
  const out = mergeCoco(new Float32Array(140).fill(7), coco, 0.1);
  expect([out[9 * 2], out[9 * 2 + 1]]).toEqual([95, 200]);   // left hip from COCO 11
  expect([out[11 * 2], out[11 * 2 + 1]]).toEqual([7, 7]);    // left knee kept: low score
  expect([out[69 * 2], out[69 * 2 + 1]]).toEqual([100, 90]); // neck 10 % of shoulder→hip above the shoulders
});

it("proportion fitting recovers known skeleton scales from ratios", async () => {
  const { fitScales, FIT_PARAMS, RATIO_KEYS } = await import("./fit");
  // A fake body: each ratio responds (non-linearly) to its own scale parameter, with some cross-talk.
  const base = { head: 0.15, torso: 0.33, shoulders: 0.2, upperArm: 0.17, hips: 0.1, thigh: 0.25, shin: 0.25 };
  const model = async (p: Float32Array) => Object.fromEntries(RATIO_KEYS.map((k, i) => {
    const own = p[FIT_PARAMS[i]], next = p[FIT_PARAMS[(i + 1) % 7]];
    return [k, base[k] * Math.exp(0.2 * own) * (1 + 0.02 * next)];
  })) as typeof base;
  const truth = new Float32Array(204);
  [0.8, -0.5, 0.3, 1.2, -1.0, 0.6, -0.4].forEach((v, i) => (truth[FIT_PARAMS[i]] = v));
  const fit = await fitScales(model, await model(truth), new Float32Array(204), { prior: 0 });
  FIT_PARAMS.forEach((i) => expect(fit.params[i]).toBeCloseTo(truth[i], 2));
  expect(fit.residual).toBeLessThan(1e-4);
});

it("normal integration recovers a spherical cap on a flat body", async () => {
  const { integrateNormals, rasterDepth } = await import("./normals");
  const G = 100, s = 0.003, a = 0.06, H = 0.04, R = (a * a + H * H) / (2 * H);
  const n = { x: new Float32Array(G * G), y: new Float32Array(G * G), z: new Float32Array(G * G).fill(1) };
  const lambda = new Float32Array(G * G), mask = new Uint8Array(G * G).fill(1), h0 = new Float32Array(G * G);
  for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) {
    const i = y * G + x, dx = (x - 50) * s, dy = (y - 50) * s, rho = Math.hypot(dx, dy);
    if (rho < a) { const hz = Math.sqrt(R * R - rho * rho); n.x[i] = dx / R; n.y[i] = -dy / R; n.z[i] = hz / R; }
    lambda[i] = rho > 0.09 ? 1 : 1e-4;
  }
  const h = integrateNormals(n, mask, h0, lambda, G, s, 2000);
  expect(Math.abs(h[50 * G + 50] - H) / H).toBeLessThan(0.05);
  expect(Math.abs(h[5 * G + 5])).toBeLessThan(0.002);

  // A square of two triangles at depth 2 covers its cells.
  const d = rasterDepth(new Float32Array([1, 8, 8, 1]), new Float32Array([1, 1, 8, 8]), new Float32Array([2, 2, 2, 2]), new Int32Array([0, 1, 2, 0, 2, 3]), 10);
  expect(d[4 * 10 + 4]).toBe(2);
  expect(d[0]).toBe(Infinity);
});

it("breast caps: fitted through a tilt, and added to the bust section", async () => {
  const { capHeight, fitCaps, withCaps } = await import("./breast");
  const G = 80, m = 0.004, d = new Float32Array(G * G);
  for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) {
    const tilt = 0.0005 * (y - 40);   // the drawing leans relative to the body model
    d[y * G + x] = tilt + capHeight(Math.hypot(x - 25, y - 35) * m, 0.06, 0.035) + capHeight(Math.hypot(x - 55, y - 35) * m, 0.06, 0.035);
  }
  // Image px = grid cells × 2 (side 160 px, origin 0).
  const caps = fitCaps({ d, x0: 0, y0: 0, side: 160, grid: G, mPerCell: m }, { l: { x: 45, y: 65 }, r: { x: 115, y: 65 }, a: 30 })!;
  expect(caps.h).toBeCloseTo(0.035, 2);
  expect(caps.aM).toBeCloseTo(0.06, 2);
  expect(caps.l.x).toBeCloseTo(51, -1);

  // A 30 cm wide, 20 cm deep box chest; caps of 6 cm base and 4 cm height at ±8 cm add their arcs over the chords.
  const box = new Float32Array([-0.15, -0.1, 0.15, -0.1, 0.15, 0.1, -0.15, 0.1]);
  const { perimeter } = await import("./measure");
  const caps2 = [-0.08, 0.08].map((x) => ({ x, dz: 0, aM: 0.06, h: 0.04 }));
  const grown = perimeter(withCaps(box, 1, caps2)) - perimeter(box);
  expect(grown).toBeGreaterThan(0.02);
  expect(grown).toBeLessThan(0.08);
});
