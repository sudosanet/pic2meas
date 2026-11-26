import { expect, it } from "vitest";
import { jisBraSize } from "./cup";
import { armCutouts, autoLevels, bodyFrame, convexHull, garmentLines, largestComponent, lengthCategory, perimeter, rowSpan, sectionAt, type Vec3 } from "./measure";

it("garment lines ignore specks; row span picks the run nearest the centre", () => {
  const w = 100, h = 100, m = new Uint8Array(w * h);
  for (let y = 20; y < 50; y++) for (let x = 30; x < 70; x++) m[y * w + x] = 1;
  m[90 * w + 5] = 1;
  expect(garmentLines(largestComponent(m, w, h).mask, w, h)).toEqual({ waistY: 20, hemY: 49 });

  const row = new Uint8Array(20);
  for (const x of [1, 2, 6, 7, 8, 9, 15, 16]) row[x] = 1;
  expect(rowSpan(row, 20, 1, 0, 8)).toEqual({ x0: 6, x1: 9 });
  expect(lengthCategory(105, 0, 100, 200).label).toBe("膝丈");
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
});

it("an arm crossing the plane at an angle is cut out", () => {
  const { verts, faces } = cylinders([{ cx: 0, r: 0.15, y0: 0, y1: 0.6, n: 64 }]);
  const v = [...verts], f = [...faces], n = 16, o = v.length / 3, dir: Vec3 = [Math.sin(Math.PI / 3), Math.cos(Math.PI / 3), 0], a0: Vec3 = [0.12, 0.2, 0];
  for (const s of [0, 0.4]) for (let k = 0; k < n; k++) {
    const c = Math.cos((2 * Math.PI * k) / n) * 0.04;
    v.push(a0[0] + dir[0] * s + c * dir[1], a0[1] + dir[1] * s - c * dir[0], Math.sin((2 * Math.PI * k) / n) * 0.04);
  }
  for (let k = 0; k < n; k++) { const k2 = (k + 1) % n; f.push(o + k, o + k2, o + n + k, o + k2, o + n + k2, o + n + k); }
  const arm = { a: a0, b: [a0[0] + dir[0] * 0.4, a0[1] + dir[1] * 0.4, 0] as Vec3, r: 0.055 };
  const p = perimeter(sectionAt(new Float32Array(v), new Int32Array(f), [0, 0, 0], [0, 1, 0], [1, 0, 0], 0.3, [arm]).hull);
  expect(Math.abs(p - TORSO) / TORSO).toBeLessThan(0.02);
});

it("2D-width calibration stretches x", () => {
  const pts: number[] = [];
  for (let k = 0; k < 256; k++) pts.push(0.15 * Math.cos((2 * Math.PI * k) / 256), 0.1 * Math.sin((2 * Math.PI * k) / 256));
  const a = 0.3, b = 0.1, h = ((a - b) / (a + b)) ** 2;
  expect(perimeter(convexHull(pts), 2)).toBeCloseTo(Math.PI * (a + b) * (1 + (3 * h) / (10 + Math.sqrt(4 - 3 * h))), 3);
});

it("bust = furthest forward, underbust = most receded below it", () => {
  const bulge = (t: number) => 0.03 * Math.exp(-(((t - 0.4) / 0.03) ** 2)) - 0.01 * Math.exp(-(((t - 0.33) / 0.015) ** 2));
  const sections = Array.from({ length: 61 }, (_, i) => {
    const t = i / 100, pts: number[] = [];
    for (let k = 0; k < 64; k++) { const y = 0.1 * Math.sin((2 * Math.PI * k) / 64); pts.push(0.15 * Math.cos((2 * Math.PI * k) / 64), y < 0 ? y - bulge(t) : y); }
    return { t, hull: convexHull(pts), armContact: 0 };
  });
  const lv = autoLevels(sections, 0.5, -1);
  expect(sections[lv.bust].t).toBeCloseTo(0.4, 2);
  expect(sections[lv.underbust].t).toBeCloseTo(0.33, 2);
});

it("JIS bra size: cups every 2.5 cm (±1.25), band in 5 cm steps", () => {
  expect(jisBraSize(83.74, 70).cup).toBe("B");
  expect(jisBraSize(83.76, 70).cup).toBe("C");
  expect(jisBraSize(76, 70).cup).toBe("AA未満");
  expect(jisBraSize(102, 70).cup).toBe("I超");
  expect(jisBraSize(85, 72.6).label).toBe("B75");
});
