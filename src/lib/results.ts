// Landmarks (automatic or hand-corrected) → measurements.
import type { Analysis, Pt } from "@/infer/types";
import { jisBraSize, type BraSize } from "./cup";
import { J, LEVELS, LEVEL_LABEL, lengthCategory, perimeter, rowSpan, sectionWidth, type Level } from "./measure";

export type Mode = "photo" | "illust";
export type Side = { l: Pt; r: Pt };
export type LevelLine = { y: number; x0: number; x1: number };
export type Landmarks = {
  headY: number; soleY: number; chinY: number;
  waistY: number; hemY: number;
  hip: Side; knee: Side; ankle: Side;
  levels: Record<Level, LevelLine>;
};
type Settings = { heightCm: number; mode: Mode; calibrate2D: boolean };
export type Overrides = { skirtCm?: number; girth: Partial<Record<Level, number>> };
export type Source = "auto" | "edited" | "manual";

const pt = (a: Analysis, i: number): Pt => ({ x: a.joints2d[i * 2], y: a.joints2d[i * 2 + 1] });

function nearestSection(a: Analysis, y: number) {
  let best = a.sections[0], d = Infinity;
  for (const s of a.sections) if (s.hull.length >= 6 && Math.abs(s.y - y) < d) { d = Math.abs(s.y - y); best = s; }
  return best;
}

/** Head/sole: the projected mesh for photos (barefoot body), the silhouette for illustrations. */
export function autoLandmarks(a: Analysis, mode: Mode): Landmarks {
  const { headY, soleY } = mode === "photo" ? a.meshSpan : { headY: a.silhouette.y0, soleY: a.silhouette.y1 };
  const side = (l: number, r: number): Side => ({ l: pt(a, l), r: pt(a, r) });
  const hip = side(J.lHip, J.rHip), shoulderY = (pt(a, J.lShoulder).y + pt(a, J.rShoulder).y) / 2;
  const hipY = (hip.l.y + hip.r.y) / 2;
  const levels = Object.fromEntries(LEVELS.map((lv) => {
    const s = a.sections[a.autoLevels[lv]] ?? nearestSection(a, hipY);
    const span = rowSpan(a.person, a.w, a.h, s.y, s.cx) ?? { x0: s.cx - 50, x1: s.cx + 50 };
    return [lv, { y: s.y, ...span }];
  })) as Record<Level, LevelLine>;
  return {
    headY, soleY,
    chinY: a.chinY ?? headY + (soleY - headY) / 7,
    // Dress or no garment: waist from the skeleton; no garment: hem at the knees as a placeholder.
    waistY: a.garmentKind === "skirt" && a.garmentLines ? a.garmentLines.waistY : shoulderY + 0.6 * (hipY - shoulderY),
    hemY: a.garmentLines?.hemY ?? (pt(a, J.lKnee).y + pt(a, J.rKnee).y) / 2,
    hip, knee: side(J.lKnee, J.rKnee), ankle: side(J.lAnkle, J.rAnkle), levels,
  };
}

export type Results = {
  skirt: { cm: number; ratio: number; category: string; u: number; source: Source } | null;
  girth: Record<Level, { cm: number; source: Source }>;
  bra: BraSize | null;
  headRatio: number;
  warnings: string[];
};

const avgY = (s: Side) => (s.l.y + s.r.y) / 2;
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

export function computeResults(a: Analysis, lm: Landmarks, auto: Landmarks, st: Settings, ov: Overrides): Results {
  const warnings: string[] = [];
  const span = lm.soleY - lm.headY, pxPerCm = span / st.heightCm;
  const spanEdited = lm.headY !== auto.headY || lm.soleY !== auto.soleY;
  const cut = st.mode === "photo" ? a.meshSpan.headY < 0 || a.meshSpan.soleY > a.h : a.silhouette.touchTop || a.silhouette.touchBottom;
  if (cut && !spanEdited) warnings.push("全身が写っていない可能性があります。頭頂と足裏の線を確認してください。");
  if (a.garmentKind === "none") warnings.push("スカート/ワンピースが見つかりませんでした。ウエストと裾の線は仮の位置です。ドラッグで合わせてください。");

  // Mesh scaled to the given height; 2D calibration stretches each section to the silhouette width.
  const k = st.heightCm / a.meshStatureM;   // cm per mesh metre
  const girth = Object.fromEntries(LEVELS.map((lv) => {
    if (ov.girth[lv] != null) return [lv, { cm: ov.girth[lv]!, source: "manual" }];
    const line = lm.levels[lv], s = nearestSection(a, line.y);
    const sx = st.calibrate2D ? ((line.x1 - line.x0) / pxPerCm) / (sectionWidth(s.hull) * k) : 1;
    if (s.armContact) warnings.push(`${LEVEL_LABEL[lv]}の高さで腕が体に触れているため、誤差が大きくなります。`);
    return [lv, { cm: perimeter(s.hull, sx) * k, source: same(line, auto.levels[lv]) ? "auto" : "edited" }];
  })) as Results["girth"];
  const bra = girth.bust.cm > 0 && girth.underbust.cm > 0 ? jisBraSize(girth.bust.cm, girth.underbust.cm) : null;

  const c = lengthCategory(lm.hemY, avgY(lm.hip), avgY(lm.knee), avgY(lm.ankle));
  const edited = lm.waistY !== auto.waistY || lm.hemY !== auto.hemY || !same(lm.knee, auto.knee) || !same(lm.hip, auto.hip);
  const px = lm.hemY - lm.waistY;
  // No garment found: no length until the user places the lines.
  const skirt: Results["skirt"] = ov.skirtCm != null
    ? { cm: ov.skirtCm, ratio: ov.skirtCm / st.heightCm, category: c.label, u: c.u, source: "manual" }
    : a.garmentKind === "none" && !edited ? null
    : { cm: px / pxPerCm, ratio: px / span, category: c.label, u: c.u, source: edited ? "edited" : "auto" };

  return { skirt, girth, bra, headRatio: span / (lm.chinY - lm.headY), warnings };
}
