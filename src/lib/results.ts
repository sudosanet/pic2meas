// Landmarks (automatic or hand-corrected) → measurements. Everything is measured along the skeleton, and photos
// use the unposed standing body, so leaning, bent or sitting poses measure like upright ones.
import type { Analysis } from "@/infer/types";
import { jisBraSize, type BraSize } from "./cup";
import { FLAGS } from "./flags";
import { arcTo, J, lengthCategory, lerp2, LEVEL_LABEL, LEVELS, mid2, onPolyline, perimeter, sectionWidth, spanAcross, type Level, type Pt } from "./measure";

export type Mode = "photo" | "illust";
export type Side = { l: Pt; r: Pt };
/** A girth level: height t (m above mid-hip, rest pose) and the silhouette edges a < 0 < b (px across the spine). */
export type LevelLine = { t: number; a: number; b: number };
export type Landmarks = {
  headTop: Pt; chin: Pt; sole: Side;
  waist: Pt; hem: Side; crotch: Pt;
  hip: Side; knee: Side; ankle: Side;
  levels: Record<Level, LevelLine>;
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

export function autoLandmarks(a: Analysis): Landmarks {
  const f = a.found, hip = side(a, J.lHip, J.rHip), knee = side(a, J.lKnee, J.rKnee);
  const sp = spine(a, { hip });
  const levels = Object.fromEntries(LEVELS.map((lv) => {
    const t = a.sections[a.autoLevels[lv]]?.t ?? 0;
    return [lv, { t, ...(spanAcross(a.person, a.w, a.h, sp.at(t), sp.across) ?? { a: -40, b: 40 }) }];
  })) as Record<Level, LevelLine>;
  return {
    headTop: f.headTop,
    chin: f.chin ?? lerp2(f.headTop, sp.neck, 0.8),
    sole: f.sole,
    waist: f.waist ?? lerp2(sp.neck, sp.hip, 0.62),
    // A hem missing on one leg mirrors the other; with no garment it starts at the knees.
    hem: { l: f.hem.l ?? f.hem.r ?? knee.l, r: f.hem.r ?? f.hem.l ?? knee.r },
    crotch: lerp2(sp.hip, mid2(knee.l, knee.r), a.rest.crotchF),
    hip, knee, ankle: side(a, J.lAnkle, J.rAnkle), levels,
  };
}

export type Detail = { label: string; value: number; unit: "cm" | "%" | "" };
export type Results = {
  skirt: { cm: number; ratio: number; category: string; source: Source } | null;
  girth: Record<Level, { cm: number; source: Source }>;
  bra: BraSize | null;
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

  const girth = Object.fromEntries(LEVELS.map((lv) => {
    if (ov.girth[lv] != null) return [lv, { cm: ov.girth[lv]!, source: "manual" }];
    const line = lm.levels[lv];
    let s = a.sections[0];
    for (const c of a.sections) if (c.hull.length >= 6 && Math.abs(c.t - line.t) < Math.abs(s.t - line.t)) s = c;
    // 2D calibration stretches the section sideways to the silhouette width.
    const sx = st.calibrate2D ? (line.b - line.a) / pxPerCm / (sectionWidth(s.hull) * k) : 1;
    if (s.armContact) warnings.push(`${LEVEL_LABEL[lv]}の高さで腕が体に触れているため、誤差が大きくなります。`);
    return [lv, { cm: perimeter(s.hull, sx) * k, source: same(line, auto.levels[lv]) ? "auto" : "edited" }];
  })) as Results["girth"];

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
    { title: "腕と肩", items: [cm("肩幅", arm.shoulders), cm("腕の長さ", arm.upper + arm.fore), cm("上腕", arm.upper), cm("前腕", arm.fore)] },
    { title: "その他の周囲長（推定）", items: [cm("首", a.rest.girthM.neck * k), cm("太もも", a.rest.girthM.thigh * k), cm("ふくらはぎ", a.rest.girthM.calf * k)] },
    { title: "比率", items: [
      { label: "股下 ÷ 身長", value: pct(ht.crotch), unit: "%" }, { label: "脚の長さ ÷ 身長", value: pct(ht.hip), unit: "%" },
      { label: "座高 ÷ 身長", value: pct(H - ht.crotch), unit: "%" }, { label: "ウエストの高さ ÷ 身長", value: pct(ht.waist), unit: "%" },
    ] },
  ];

  return {
    skirt, girth, details,
    bra: girth.bust.cm > 0 && girth.underbust.cm > 0 ? jisBraSize(girth.bust.cm, girth.underbust.cm) : null,
    headRatio: H / headCm, warnings,
  };
}
