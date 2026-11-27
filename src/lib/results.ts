// Landmarks (automatic or hand-corrected) → measurements. Everything is measured along the skeleton, and photos
// use the unposed standing body, so leaning, bent or sitting poses measure like upright ones.
import type { Analysis } from "@/infer/types";
import { capLine, capSphere, capVolume, MIN_CAP_M, withCaps, type Caps } from "./breast";
import { jisBraSize, type BraSize } from "./cup";
import { FLAGS } from "./flags";
import { arcTo, bodyFrame, J, lengthCategory, lerp2, LEVEL_LABEL, LEVEL_PRIOR, LEVELS, mid2, onPolyline, perimeter, sectionWidth, type Level, type Pt, type Vec3 } from "./measure";

export type Mode = "photo" | "illust";
export type Side = { l: Pt; r: Pt };
/** A girth level: height t (m above mid-hip, rest pose) and its ends a < 0 < b (px across the body, from levelAt's centre). */
export type LevelLine = { t: number; a: number; b: number };
export type Landmarks = {
  headTop: Pt; chin: Pt; sole: Side;
  waist: Pt; hem: Side; crotch: Pt;
  hip: Side; knee: Side; ankle: Side;
  levels: Record<Level, LevelLine>;
  /** Breasts as spherical caps (image px; aM and h in metres), or null when not estimated. */
  breast: Caps | null;
};
type Settings = { heightCm: number; mode: Mode; calibrate2D: boolean };
export type Overrides = { skirtCm?: number; girth: Partial<Record<Level, number>> };
export type Source = "auto" | "edited" | "manual";

/** The analysis as a mode sees it: illustrations use the joints and landmarks from the drawing-trained keypoints. */
export const forMode = (a: Analysis, mode: Mode): Analysis => (mode === "illust" && FLAGS.art ? { ...a, joints2d: a.art.joints2d, found: a.art.found } : a);

const joint = (a: Analysis, i: number): Pt => ({ x: a.joints2d[i * 2], y: a.joints2d[i * 2 + 1] });
const side = (a: Analysis, l: number, r: number): Side => ({ l: joint(a, l), r: joint(a, r) });

/** The spine in the image (neck → mid-hip), where rest height t falls on it, and the direction across it. */
export function spine(a: Analysis, lm: Pick<Landmarks, "hip">) {
  const neck = joint(a, J.neck), hip = mid2(lm.hip.l, lm.hip.r);
  const len = Math.hypot(neck.x - hip.x, neck.y - hip.y) || 1;
  const across = { x: -(neck.y - hip.y) / len, y: (neck.x - hip.x) / len };
  return { neck, hip, len, across, at: (t: number) => lerp2(hip, neck, t / a.rest.torsoM) };
}

/**
 * Where rest height t lies in the image: the nearest section's projected line, carried from the mesh's projection
 * onto the current spine (a similarity mapping the mesh's neck and mid-hip onto the landmarks'), so it follows the
 * drawing's keypoints and the user's hip handles.
 */
/**
 * The posed mesh's projection carried onto the current spine: a similarity mapping the mesh's neck and mid-hip onto
 * the landmarks' (identity for photos left as found; illustrations follow the drawing's keypoints and the user's hip
 * handles). toMesh is its inverse; scale is image px per projected-mesh px.
 */
export function meshMap(a: Analysis, lm: Pick<Landmarks, "hip">) {
  const sp = spine(a, lm), { neck: n0, hip: h0 } = a.meshAnchor;
  // z' = A z + B on complex numbers.
  const d0 = { x: n0.x - h0.x, y: n0.y - h0.y }, d1 = { x: sp.neck.x - sp.hip.x, y: sp.neck.y - sp.hip.y }, q = d0.x ** 2 + d0.y ** 2 || 1;
  const A = { x: (d1.x * d0.x + d1.y * d0.y) / q, y: (d1.y * d0.x - d1.x * d0.y) / q }, scale = Math.hypot(A.x, A.y) || 1;
  const rot = (p: Pt) => ({ x: A.x * p.x - A.y * p.y, y: A.y * p.x + A.x * p.y });
  const unrot = (p: Pt) => ({ x: (A.x * p.x + A.y * p.y) / scale ** 2, y: (A.x * p.y - A.y * p.x) / scale ** 2 });
  return {
    scale, rot,
    toImage: (p: Pt) => { const r = rot({ x: p.x - h0.x, y: p.y - h0.y }); return { x: r.x + sp.hip.x, y: r.y + sp.hip.y }; },
    toMesh: (p: Pt) => { const r = unrot({ x: p.x - sp.hip.x, y: p.y - sp.hip.y }); return { x: r.x + h0.x, y: r.y + h0.y }; },
  };
}

/**
 * Where rest height t lies in the image: the nearest section's projected line, carried by meshMap so it follows the
 * drawing's keypoints and the user's hip handles.
 */
export function levelAt(a: Analysis, lm: Pick<Landmarks, "hip">, t: number) {
  const mm = meshMap(a, lm);
  let best: Analysis["sections"][number] | null = null;
  for (const s of a.sections) if (s.line2d && (!best || Math.abs(s.t - t) < Math.abs(best.t - t))) best = s;
  if (!best?.line2d) return null;
  const L = best.line2d, r = mm.rot(L.dir);
  return { t: best.t, c: mm.toImage(L.c), dir: { x: r.x / mm.scale, y: r.y / mm.scale }, a: L.a * mm.scale, b: L.b * mm.scale };
}

/** The section whose image line passes nearest to p (for dragging a level along the body). */
export function levelNear(a: Analysis, lm: Pick<Landmarks, "hip">, p: Pt) {
  let best: ReturnType<typeof levelAt> = null, bestD = Infinity;
  for (const s of a.sections) {
    const l = s.line2d && levelAt(a, lm, s.t);
    if (!l) continue;
    const d = Math.abs((p.x - l.c.x) * -l.dir.y + (p.y - l.c.y) * l.dir.x);   // distance across the line
    if (d < bestD) { bestD = d; best = l; }
  }
  return best;
}

/**
 * The caps on the unposed body: each circle's centre, taken along and across the bust's image line, gives its place
 * across the section (the line's ends map to the section's x range) and its height offset from the section plane at
 * t. The measurement, the circles and the 3D view all use these.
 */
export function capsIn3d(a: Analysis, lm: Pick<Landmarks, "hip">, caps: Caps, t: number) {
  const place = levelAt(a, lm, t), hull = nearestSection(a, t).hull;
  if (!place || caps.a <= 0) return [];
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < hull.length; i += 2) { lo = Math.min(lo, hull[i]); hi = Math.max(hi, hull[i]); }
  const mPerPx = caps.aM / caps.a;
  return [caps.l, caps.r].map((c) => {
    const dx = c.x - place.c.x, dy = c.y - place.c.y, u = dx * place.dir.x + dy * place.dir.y;
    const f = (u - place.a) / (place.b - place.a || 1);
    return { x: lo + f * (hi - lo), dz: -(dx * -place.dir.y + dy * place.dir.x) * mPerPx, aM: caps.aM, h: caps.h };
  });
}

/**
 * The bust, the same for photos and drawings: a blend of two estimates of bust − underbust, weighed on published
 * sizes (eval/; mean cup error, Blue Archive 118 / photos 55: shape alone 2.67 / 1.98, cup model alone 2.14 / 2.25,
 * blend 2.19 / 2.04; the 0.3 weight minimises both).
 * - shape: the chest wall + caps as measured, against the corrected underbust;
 * - cup model: from the caps' projection (cm) and the hip girth (cm, uncorrected), fitted to the drawings' cups (as a
 *   profile site estimated them from the art: the game publishes none). Fuller figures get rounder bodies from the
 *   body model, so the hip carries much of it.
 * underbust: the body model's runs large (photos: cup bias −3.4 → 0 at 0.89).
 */
export const BUST = { shapeWeight: 0.3, underbust: 0.89, cup: { c0: -28.10, h: 0.258, hip: 0.326 } };

/**
 * Corrections shared by both modes, least squares against published sizes: waist and hip from 13 full-body photos
 * (0.787, 0.857; leave-one-out 15 → 4 cm) agree with 40 anime characters (0.771, 0.838).
 */
export const GIRTH_SCALE = { waist: 0.79, hip: 0.85 };


/**
 * A cap grown by f: the projection first (h × f), the base only once the projection passes a hemisphere's (h ≤ base
 * radius), so a bigger bust stands out rather than spreading into a wide disc.
 */
export function grownCap<T extends { aM: number; h: number }>(c: T, f: number): T {
  const h = Math.max(c.h, 0.002) * f;
  return { ...c, h, aM: Math.max(c.aM * Math.min(f, 1), h) };
}

/** Girth growth of the rib cage from the underbust to the bust line, without breasts (calibrated in eval/). */
export const RIB_GROWTH = 0.06;

function nearestSection(a: Analysis, t: number) {
  let s = a.sections[0];
  for (const c of a.sections) if (c.hull.length >= 6 && Math.abs(c.t - t) < Math.abs(s.t - t)) s = c;
  return s;
}

export function autoLandmarks(a: Analysis): Landmarks {
  const f = a.found, hip = side(a, J.lHip, J.rHip), knee = side(a, J.lKnee, J.rKnee);
  const sp = spine(a, { hip });
  const levels = Object.fromEntries(LEVELS.map((lv) => {
    const t = a.sections[a.autoLevels[lv]]?.t ?? 0, l = levelAt(a, { hip }, t);
    return [lv, { t, a: l?.a ?? -40, b: l?.b ?? 40 }];
  })) as Record<Level, LevelLine>;
  // No garment: the skirt's waist starts at the body's (narrowest) waist.
  const naturalWaist = levelAt(a, { hip }, levels.waist.t)?.c ?? lerp2(sp.neck, sp.hip, 0.62);
  const lm: Landmarks = {
    headTop: f.headTop,
    chin: f.chin ?? lerp2(f.headTop, sp.neck, 0.8),
    sole: f.sole,
    waist: f.waist ?? naturalWaist,
    // A hem missing on one leg mirrors the other; with no garment it starts at the knees.
    hem: { l: f.hem.l ?? f.hem.r ?? knee.l, r: f.hem.r ?? f.hem.l ?? knee.r },
    crotch: lerp2(sp.hip, mid2(knee.l, knee.r), a.rest.crotchF),
    hip, knee, ankle: side(a, J.lAnkle, J.rAnkle), levels,
    breast: FLAGS.breast ? a.breast.caps : null,
  };
  return followWaist(a, followCaps(a, lm));
}

/** A level line at the section nearest image point p (its own ends), or the current line if there is none. */
const lineAt = (a: Analysis, lm: Pick<Landmarks, "hip">, p: Pt, fallback: LevelLine): LevelLine => {
  const l = levelNear(a, lm, p);
  return l ? { t: l.t, a: l.a, b: l.b } : fallback;
};
const lineAtT = (a: Analysis, lm: Pick<Landmarks, "hip">, t: number, fallback: LevelLine): LevelLine => {
  const l = levelAt(a, lm, t);
  return l ? { t: l.t, a: l.a, b: l.b } : fallback;
};

/** The bust line through the caps' centres and the underbust a base radius below them (≥ 3 cm lower). */
export function followCaps(a: Analysis, lm: Landmarks): Landmarks {
  if (!lm.breast || lm.breast.aM < MIN_CAP_M) return lm;
  const { mid, below } = capLine(lm.breast), bust = lineAt(a, lm, mid, lm.levels.bust);
  let underbust = lineAt(a, lm, below, lm.levels.underbust);
  if (underbust.t > bust.t - LEVEL_PRIOR.gap) underbust = lineAtT(a, lm, bust.t - LEVEL_PRIOR.gap, underbust);
  return { ...lm, levels: { ...lm.levels, bust, underbust } };
}

/** How far (m) the skirt's top may be from the body's narrowest waist and still be the same waist. */
export const WAIST_LINK = 0.06;

/**
 * The waist girth at the waist point (the one the skirt is measured from) when that point is the body's waist,
 * i.e. near its narrowest section; a low or high skirt top keeps the girth at the narrowest section. Kept ≥ 3 cm
 * below the underbust.
 */
export function followWaist(a: Analysis, lm: Landmarks): Landmarks {
  const natural = a.sections[a.autoLevels.waist]?.t ?? lm.levels.waist.t;
  let waist = lineAt(a, lm, lm.waist, lm.levels.waist);
  if (Math.abs(waist.t - natural) > WAIST_LINK) waist = lineAtT(a, lm, natural, lm.levels.waist);
  const top = lm.levels.underbust.t - LEVEL_PRIOR.gap;
  if (waist.t > top) waist = lineAtT(a, lm, top, waist);
  return { ...lm, levels: { ...lm.levels, waist } };
}

/** Whether the waist point lies on the waist girth line (else it is just the skirt's top). */
export function waistLinked(a: Analysis, lm: Landmarks) {
  const l = levelNear(a, lm, lm.waist);
  return !!l && Math.abs(l.t - lm.levels.waist.t) < 0.015;
}

export type Detail = { label: string; value: number; unit: "cm" | "%" | "cc" | "" };
export type Results = {
  skirt: { cm: number; ratio: number; category: string; source: Source } | null;
  girth: Record<Level, { cm: number; source: Source }>;
  bra: BraSize | null;
  /** The breast circles as drawn: centres from the caps, radius from the cup (or the user's circle once edited). */
  breastCircles: { l: Pt; r: Pt; a: number } | null;
  /** Factor on the caps' size (base and projection) at which their shape gives the reported bust. */
  capScale: number;
  headRatio: number;
  details: { title: string; items: Detail[] }[];
  warnings: string[];
};

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
const mean = (x: number, y: number) => (x + y) / 2;
const dist = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y);

export function computeResults(a: Analysis, lm: Landmarks, auto: Landmarks, st: Settings, ov: Overrides): Results {
  const warnings: string[] = [];
  if (a.cut) warnings.push("体の一部が画像の端で切れています。頭と足元の点を確認してください。");
  if (a.garmentKind === "none") warnings.push("スカート/ワンピースが見つかりませんでした。ウエストと裾の点は仮の位置です。");

  const sp = spine(a, lm), up = a.rest.jointUp, k = st.heightCm / a.rest.statureM;   // k: cm per rest-mesh metre
  const legs = { l: [lm.hip.l, lm.knee.l, lm.ankle.l, lm.sole.l], r: [lm.hip.r, lm.knee.r, lm.ankle.r, lm.sole.r] };
  const legJoints = { l: [J.lHip, J.lKnee, J.lAnkle], r: [J.rHip, J.rKnee, J.rAnkle] };
  const waistF = onPolyline(lm.waist, [sp.neck, sp.hip]).frac;   // 0 at the neck, 1 at mid-hip
  const crotchF = onPolyline(lm.crotch, [sp.hip, mid2(lm.knee.l, lm.knee.r)]).frac;   // 0 at mid-hip, 1 at the knees
  const hemPos = { l: onPolyline(lm.hem.l, legs.l), r: onPolyline(lm.hem.r, legs.r) };
  const uOf = (s: "l" | "r") => Math.min(2.5, hemPos[s].seg + hemPos[s].frac);   // hip 0, knee 1, ankle 2
  const j2 = (i: number): Pt => ({ x: a.joints2d[i * 2], y: a.joints2d[i * 2 + 1] });

  // Heights above the floor (cm) of the body's landmarks, and limb lengths.
  let pxPerCm: number, ht: Record<"neck" | "shoulder" | "waist" | "hip" | "crotch" | "knee" | "ankle" | "hem", number>;
  let arm: { upper: number; fore: number; shoulders: number };
  if (st.mode === "photo") {
    // The unposed body: heights straight from its joints.
    const h = (u: number) => (u - a.rest.floorM) * k, avg = (i: number, j: number) => mean(up[i], up[j]);
    const hip = avg(J.lHip, J.rHip), knee = avg(J.lKnee, J.rKnee);
    const hemUp = (s: "l" | "r") => {
      const j = legJoints[s], u = uOf(s), i = Math.min(Math.floor(u), 1);
      return up[j[i]] + (up[j[i + 1]] - up[j[i]]) * (u - i);
    };
    const r3 = a.rest.joints, d3 = (i: number, j: number) => Math.hypot(r3[i * 3] - r3[j * 3], r3[i * 3 + 1] - r3[j * 3 + 1], r3[i * 3 + 2] - r3[j * 3 + 2]) * k;
    pxPerCm = sp.len / ((up[J.neck] - hip) * k);
    ht = {
      neck: h(up[J.neck]), shoulder: h(avg(67, 68)), waist: h(up[J.neck] + (hip - up[J.neck]) * waistF), hip: h(hip),
      crotch: h(hip + (knee - hip) * crotchF), knee: h(knee), ankle: h(avg(J.lAnkle, J.rAnkle)), hem: h(mean(hemUp("l"), hemUp("r"))),
    };
    arm = { upper: mean(d3(J.lShoulder, J.lElbow), d3(J.rShoulder, J.rElbow)), fore: mean(d3(J.lElbow, J.lWrist), d3(J.rElbow, J.rWrist)), shoulders: d3(67, 68) };
  } else {
    // Illustrations keep the drawing's own proportions: lengths along the skeleton in the image.
    const legLen = (s: "l" | "r") => arcTo(legs[s], { seg: 2, frac: 1 });
    pxPerCm = (dist(lm.headTop, sp.neck) + sp.len + mean(legLen("l"), legLen("r"))) / st.heightCm;
    const c = (px: number) => px / pxPerCm, seg = (i: number) => c(mean(dist(legs.l[i], legs.l[i + 1]), dist(legs.r[i], legs.r[i + 1])));
    const ankle = seg(2), knee = ankle + seg(1), hip = knee + seg(0), shoulders = mid2(j2(67), j2(68));
    ht = {
      neck: hip + c(sp.len), shoulder: hip + c(((shoulders.x - sp.hip.x) * (sp.neck.x - sp.hip.x) + (shoulders.y - sp.hip.y) * (sp.neck.y - sp.hip.y)) / sp.len),
      waist: hip + c((1 - waistF) * sp.len), hip, crotch: hip - crotchF * seg(0), knee, ankle,
      hem: hip - c(mean(arcTo(legs.l, hemPos.l), arcTo(legs.r, hemPos.r))),
    };
    arm = {
      upper: c(mean(dist(j2(J.lShoulder), j2(J.lElbow)), dist(j2(J.rShoulder), j2(J.rElbow)))),
      fore: c(mean(dist(j2(J.lElbow), j2(J.lWrist)), dist(j2(J.rElbow), j2(J.rWrist)))), shoulders: c(dist(j2(67), j2(68))),
    };
  }

  // Caps the image couldn't support are a placeholder: no cup from them until the user shapes them.
  const breastEdited = !!lm.breast && !same(lm.breast, auto.breast);
  const girth = Object.fromEntries(LEVELS.map((lv) => {
    if (ov.girth[lv] != null) return [lv, { cm: ov.girth[lv]!, source: "manual" }];
    const line = lm.levels[lv];
    const s = nearestSection(a, line.t);
    // 2D calibration stretches the section sideways to the silhouette width.
    const sx = st.calibrate2D ? (line.b - line.a) / pxPerCm / (sectionWidth(s.hull) * k) : 1;
    if (s.armContact) warnings.push(`${LEVEL_LABEL[lv]}の高さで腕が体に触れているため、誤差が大きくなります。`);
    // With the breast model, the bust is the chest wall (the underbust section, widened by the rib cage's growth up to
    // the bust: the body model's own chest is a man's) plus the caps.
    let hull = s.hull;
    if (lv === "bust" && FLAGS.breast && lm.breast) {
      const ub = nearestSection(a, lm.levels.underbust.t).hull.map((v) => v * (1 + RIB_GROWTH));
      hull = withCaps(ub, a.rest.front, capsIn3d(a, lm, lm.breast, line.t));
    }
    const edited = !same(line, auto.levels[lv]) || (lv === "bust" && !same(lm.breast, auto.breast));
    return [lv, { cm: perimeter(hull, sx) * k, source: edited ? "edited" : "auto" }];
  })) as Results["girth"];

  // Bust (BUST): blend the measured shape with the cup model; then size the caps so their shape gives that bust, which
  // is what the circles and the 3D view show.
  const hipRaw = girth.hip.cm;
  if (girth.underbust.source !== "manual") girth.underbust = { ...girth.underbust, cm: girth.underbust.cm * BUST.underbust };
  for (const lv of ["waist", "hip"] as const) if (girth[lv].source !== "manual") girth[lv] = { ...girth[lv], cm: girth[lv].cm * GIRTH_SCALE[lv] };
  let capScale = 1;
  if (FLAGS.breast && lm.breast && ov.girth.bust == null) {
    const hCm = lm.breast.src === "prior" && !breastEdited ? 0 : lm.breast.h * k;
    const model = 7.5 + 2.5 * (BUST.cup.c0 + BUST.cup.h * hCm + BUST.cup.hip * hipRaw);
    // Caps shaped by hand are taken as they are: the shape alone.
    const w = breastEdited ? 1 : BUST.shapeWeight;
    const delta = Math.max(0, w * (girth.bust.cm - girth.underbust.cm) + (1 - w) * model);
    girth.bust = { cm: girth.underbust.cm + delta, source: girth.bust.source };
    // Caps scale f whose shape gives that bust (0 if even no caps measure more).
    const wall = nearestSection(a, lm.levels.underbust.t).hull.map((v) => v * (1 + RIB_GROWTH));
    const placed = capsIn3d(a, lm, lm.breast, lm.levels.bust.t);
    const shapeAt = (f: number) => perimeter(withCaps(wall, a.rest.front, placed.map((c) => grownCap(c, f)))) * k;
    const want = girth.bust.cm;
    let lo = 0, hi = 4;
    if (shapeAt(lo) >= want) hi = 0;
    else for (let it = 0; it < 30; it++) { const mid = (lo + hi) / 2; if (shapeAt(mid) < want) lo = mid; else hi = mid; }
    capScale = breastEdited ? 1 : hi;
  }

  const skirtEdited = !same([lm.waist, lm.hem, lm.hip, lm.knee, lm.ankle], [auto.waist, auto.hem, auto.hip, auto.knee, auto.ankle]);
  const category = lengthCategory(mean(uOf("l"), uOf("r")));
  const skirtCm = ht.waist - ht.hem;
  const skirt: Results["skirt"] = ov.skirtCm != null
    ? { cm: ov.skirtCm, ratio: ov.skirtCm / st.heightCm, category, source: "manual" }
    : a.garmentKind === "none" && !skirtEdited ? null   // placeholders until the user places them
    : { cm: skirtCm, ratio: skirtCm / st.heightCm, category, source: skirtEdited ? "edited" : "auto" };

  const H = st.heightCm, headCm = dist(lm.headTop, lm.chin) / pxPerCm, pct = (v: number) => (v / H) * 100;
  const cm = (label: string, value: number): Detail => ({ label, value, unit: "cm" });
  const above = (label: [string, string], v: number) => cm(v >= 0 ? label[0] : label[1], Math.abs(v));
  const details: Results["details"] = [
    ...(skirt ? [{ title: "スカートの位置", items: [
      above(["裾は膝の上", "裾は膝の下"], ht.hem - ht.knee),
      above(["裾は股下より上", "裾は股下から下"], ht.hem - ht.crotch),
      cm("床から裾まで", ht.hem),
      cm("床からウエストまで", ht.waist),
    ] }] : []),
    { title: "高さと長さ", items: [
      cm("身長", H), cm("座高（推定）", H - ht.crotch), cm("股下", ht.crotch), cm("脚の長さ（床〜股関節）", ht.hip),
      cm("太ももの長さ", ht.hip - ht.knee), cm("すねの長さ", ht.knee - ht.ankle), cm("膝の高さ", ht.knee),
      cm("首の付け根の高さ", ht.neck), cm("肩の高さ", ht.shoulder), cm("頭の長さ（頭頂〜あご）", headCm),
    ] },
    { title: "周囲長を測った高さ（床から）", items: LEVELS.map((lv) => cm(LEVEL_LABEL[lv], ht.hip + (lm.levels[lv].t / a.rest.torsoM) * (ht.neck - ht.hip))) },
    ...(lm.breast ? [{ title: "胸（球冠モデル）", items: [
      cm("胸の出っぱり", grownCap(lm.breast, capScale).h * k),
      cm("胸の底面の直径", 2 * grownCap(lm.breast, capScale).aM * k),
      { label: "胸の体積（片側）", value: capVolume((grownCap(lm.breast, capScale).aM * k) / 100, (grownCap(lm.breast, capScale).h * k) / 100) * 1e6, unit: "cc" as const },
    ] }] : []),
    { title: "腕と肩", items: [cm("肩幅", arm.shoulders), cm("腕の長さ", arm.upper + arm.fore), cm("上腕", arm.upper), cm("前腕", arm.fore)] },
    { title: "その他の周囲長（推定）", items: [cm("首", a.rest.girthM.neck * k), cm("太もも", a.rest.girthM.thigh * k), cm("ふくらはぎ", a.rest.girthM.calf * k)] },
    { title: "比率", items: [
      { label: "股下 ÷ 身長", value: pct(ht.crotch), unit: "%" }, { label: "脚の長さ ÷ 身長", value: pct(ht.hip), unit: "%" },
      { label: "座高 ÷ 身長", value: pct(H - ht.crotch), unit: "%" }, { label: "ウエストの高さ ÷ 身長", value: pct(ht.waist), unit: "%" },
    ] },
  ];

  const bra = girth.bust.cm > 0 && girth.underbust.cm > 0 ? jisBraSize(girth.bust.cm, girth.underbust.cm) : null;
  // The circles: the caps' bases at the size that gives the bust (see capScale).
  const breastCircles = lm.breast && { l: lm.breast.l, r: lm.breast.r, a: lm.breast.a * (grownCap(lm.breast, capScale).aM / (lm.breast.aM || 1)) };
  return {
    skirt, girth, details, bra, breastCircles, capScale,
    headRatio: H / headCm, warnings,
  };
}

/**
 * What the 3D view shows, from the same functions as the measurement: the unposed body, the girth sections (the bust
 * with its caps) and the caps' spheres, at the bust's cap scale. Coordinates: x across, y up (above mid-hip),
 * z forward, in metres.
 */
export function view3d(a: Analysis, lm: Landmarks, capScale: number) {
  const f = bodyFrame(a.rest.joints), v = a.rest.verts;
  const fwd: Vec3 = [f.up[1] * f.side[2] - f.up[2] * f.side[1], f.up[2] * f.side[0] - f.up[0] * f.side[2], f.up[0] * f.side[1] - f.up[1] * f.side[0]];
  const along = (d: Vec3, u: Vec3) => d[0] * u[0] + d[1] * u[1] + d[2] * u[2];
  const verts = new Float32Array(v.length);
  for (let i = 0; i < v.length; i += 3) {
    const d: Vec3 = [v[i] - f.origin[0], v[i + 1] - f.origin[1], v[i + 2] - f.origin[2]];
    verts.set([along(d, f.side), along(d, f.up), f.front * along(d, fwd)], i);
  }
  // Section points (x across, y = forward along up × side) at height t → view coordinates.
  const ring = (hull: Float32Array, t: number) => Array.from({ length: hull.length / 2 + 1 }, (_, k) => {
    const i = (k % (hull.length / 2)) * 2;
    return [hull[i], t, f.front * hull[i + 1]] as Vec3;
  });
  const caps = lm.breast && FLAGS.breast ? capsIn3d(a, lm, lm.breast, lm.levels.bust.t).map((c) => grownCap(c, capScale)) : [];
  const wall = nearestSection(a, lm.levels.underbust.t).hull.map((x) => x * (1 + RIB_GROWTH));
  const rings = LEVELS.map((lv) => {
    const t = lm.levels[lv].t, hull = lv === "bust" && caps.length ? withCaps(wall, a.rest.front, caps) : nearestSection(a, t).hull;
    return { level: lv, points: ring(hull, t) };
  });
  const spheres = caps.flatMap((c) => {
    const t = lm.levels.bust.t + c.dz, s = capSphere(nearestSection(a, t).hull, a.rest.front, c.x, c.aM, c.h);
    return s ? [{ center: [s.cx, t, f.front * s.cy] as Vec3, radius: s.R, normal: [s.wall.nx, 0, f.front * s.wall.ny] as Vec3, h: c.h }] : [];
  });
  return { verts, faces: a.mesh.faces, rings, spheres };
}
