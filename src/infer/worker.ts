/// <reference lib="webworker" />
// SegFormer (clothes) + InstantHMR (body) on onnxruntime-web, then mesh cross-sections.
import * as Comlink from "comlink";
import * as ort from "onnxruntime-web/webgpu";
import { armCutouts, autoLevels, bodyFrame, edgeAlong, extent, insideRange, J, largestComponent, lerp2, mid2, perimeter, sectionAt, type Pt, type Vec3 } from "@/lib/measure";
import { fitScales, ratios3d, type Ratios } from "@/lib/fit";
import { decodeSimcc, mergeCoco, POSE_H, POSE_W } from "@/lib/keypoints";
import { configureOrt, MODELS } from "@/lib/models";
import type { Analysis, Backend, InferApi, Progress } from "./types";

configureOrt();

const MAX_SIDE = 1600, SEG = 512, CROP = 288;
const LABEL = { background: 0, skirt: 5, dress: 7, face: 11 } as const;
const CACHE = "pic2meas-models";

async function fetchModel(url: string, onProgress: (p: Progress) => void): Promise<Uint8Array> {
  const label = url.split("/").pop()!;
  try {
    const hit = await (await caches.open(CACHE)).match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch { /* no Cache API */ }
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`${label} の取得に失敗しました（${res.status}）`);
  const total = Number(res.headers.get("Content-Length")) || undefined, chunks: Uint8Array[] = [];
  let loaded = 0;
  for (const reader = res.body.getReader(); ;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); loaded += value.length;
    onProgress({ stage: "download", loaded, total, label });
  }
  const bytes = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) { bytes.set(c, o); o += c.length; }
  try { await (await caches.open(CACHE)).put(url, new Response(bytes)); } catch { /* quota */ }
  return bytes;
}

// Triangles are in mhr-body.onnx's metadata ("mesh_faces", base64 int32), which onnxruntime-web doesn't expose:
// find the key, then read the protobuf string that follows (0x12, varint length, value).
function meshFaces(bytes: Uint8Array): Int32Array {
  const key = new TextEncoder().encode("mesh_faces");
  let i = 0;
  find: for (; i < bytes.length - key.length; i++) { for (let k = 0; k < key.length; k++) if (bytes[i + k] !== key[k]) continue find; break; }
  let p = i + key.length;
  if (bytes[p++] !== 0x12) throw new Error("mesh_faces not found in mhr-body.onnx");
  let len = 0, shift = 0, c: number;
  do { c = bytes[p++]; len += (c & 0x7f) * 2 ** shift; shift += 7; } while (c & 0x80);
  const bin = atob(new TextDecoder().decode(bytes.subarray(p, p + len)));
  return new Int32Array(Uint8Array.from(bin, (ch) => ch.charCodeAt(0)).buffer);
}

/** The last analysis' body parameters, for refitting its proportions to a drawing. */
let last: { params: Float32Array; shape: Float32Array } | null = null;

type Models = { seg: ort.InferenceSession; hmr: ort.InferenceSession; body: ort.InferenceSession; pose: ort.InferenceSession; faces: Int32Array; backend: Backend };
let models: Promise<Models> | undefined;

async function load(want: Backend | "auto", onProgress: (p: Progress) => void): Promise<Models> {
  const gpu = want !== "wasm" && !!(await navigator.gpu?.requestAdapter());
  let backend: Backend = gpu ? "webgpu" : "wasm";
  const bytes = {
    seg: await fetchModel(MODELS.seg, onProgress), hmr: await fetchModel(MODELS.hmr, onProgress),
    body: await fetchModel(MODELS.body, onProgress), pose: await fetchModel(MODELS.pose, onProgress),
  };
  const create = async (k: keyof typeof bytes) => {
    onProgress({ stage: "session", label: k });
    try { return await ort.InferenceSession.create(bytes[k], { executionProviders: [backend], logSeverityLevel: 3 }); }
    catch (e) {
      if (backend === "wasm") throw e;
      console.warn(`${k}: WebGPU failed, falling back to WASM`, e);
      backend = "wasm";
      return ort.InferenceSession.create(bytes[k], { executionProviders: ["wasm"] });
    }
  };
  // One after another: creating them concurrently hung WebGPU.
  const seg = await create("seg"), hmr = await create("hmr"), body = await create("body"), pose = await create("pose");
  return { seg, hmr, body, pose, faces: meshFaces(bytes.body), backend };
}

async function loadImage(blob: Blob) {
  const bmp = await createImageBitmap(blob);
  const s = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
  const w = Math.round(bmp.width * s), h = Math.round(bmp.height * s);
  const raw = new OffscreenCanvas(w, h).getContext("2d", { willReadFrequently: true })!;
  raw.drawImage(bmp, 0, 0, w, h);
  const rgba = raw.getImageData(0, 0, w, h).data;
  let transparent = 0;
  for (let i = 3; i < rgba.length; i += 4) if (rgba[i] < 128) transparent++;
  let alpha: Uint8Array | null = null;
  if (transparent > w * h * 0.05) { alpha = new Uint8Array(w * h); for (let i = 0; i < w * h; i++) alpha[i] = rgba[i * 4 + 3] >= 128 ? 1 : 0; }
  const canvas = new OffscreenCanvas(w, h), ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, w, h); ctx.drawImage(bmp, 0, 0, w, h);
  bmp.close();
  return { canvas, w, h, alpha };
}
type Img = Awaited<ReturnType<typeof loadImage>>;

/** The image rotated by `angle` about the centre of `box`, and the person's box in it. */
function rotated(img: Img, person: Uint8Array, angle: number, box: { x0: number; x1: number; y0: number; y1: number }) {
  const c = new OffscreenCanvas(img.w, img.h), ctx = c.getContext("2d", { willReadFrequently: true })!;
  const cx = (box.x0 + box.x1) / 2, cy = (box.y0 + box.y1) / 2, cos = Math.cos(angle), sin = Math.sin(angle);
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, img.w, img.h);
  ctx.translate(cx, cy); ctx.rotate(angle); ctx.translate(-cx, -cy);
  ctx.drawImage(img.canvas, 0, 0);
  const b = { x0: Infinity, x1: -Infinity, y0: Infinity, y1: -Infinity };
  for (let i = 0; i < person.length; i++) if (person[i]) {
    const dx = (i % img.w) - cx, dy = ((i / img.w) | 0) - cy, x = cx + dx * cos - dy * sin, y = cy + dx * sin + dy * cos;
    b.x0 = Math.min(b.x0, x); b.x1 = Math.max(b.x1, x); b.y0 = Math.min(b.y0, y); b.y1 = Math.max(b.y1, y);
  }
  return [{ ...img, canvas: c }, b] as const;
}

// SegFormer on a source rectangle of the image (scaled to 512×512): 18 class logits at 128×128, upsampled
// bilinearly, argmax per pixel. Pixels outside the rectangle stay 0 (background).
type Rect = { x: number; y: number; sw: number; sh: number };
async function segment(m: Models, { canvas, w, h }: Img, r: Rect = { x: 0, y: 0, sw: w, sh: h }): Promise<Uint8Array> {
  const t = new OffscreenCanvas(SEG, SEG).getContext("2d", { willReadFrequently: true })!;
  t.fillStyle = "#fff"; t.fillRect(0, 0, SEG, SEG);
  t.drawImage(canvas, r.x, r.y, r.sw, r.sh, 0, 0, SEG, SEG);
  const d = t.getImageData(0, 0, SEG, SEG).data, x = new Float32Array(3 * SEG * SEG);
  const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
  for (let i = 0; i < SEG * SEG; i++) for (let c = 0; c < 3; c++) x[c * SEG * SEG + i] = (d[i * 4 + c] / 255 - mean[c]) / std[c];
  const out = await m.seg.run({ [m.seg.inputNames[0]]: new ort.Tensor("float32", x, [1, 3, SEG, SEG]) });
  const lg = out[m.seg.outputNames[0]], [, C, H, W] = lg.dims as number[], L = lg.data as Float32Array;
  const labels = new Uint8Array(w * h);
  for (let py = Math.max(0, Math.floor(r.y)); py < Math.min(h, r.y + r.sh); py++) {
    const fy = Math.min(H - 1, Math.max(0, ((py + 0.5 - r.y) * H) / r.sh - 0.5)), y0 = fy | 0, y1 = Math.min(H - 1, y0 + 1), ay = fy - y0;
    for (let px = Math.max(0, Math.floor(r.x)); px < Math.min(w, r.x + r.sw); px++) {
      const fx = Math.min(W - 1, Math.max(0, ((px + 0.5 - r.x) * W) / r.sw - 0.5)), x0 = fx | 0, x1 = Math.min(W - 1, x0 + 1), ax = fx - x0;
      let best = -Infinity, arg = 0;
      for (let c = 0; c < C; c++) {
        const b = c * H * W;
        const v = (L[b + y0 * W + x0] * (1 - ax) + L[b + y0 * W + x1] * ax) * (1 - ay) + (L[b + y1 * W + x0] * (1 - ax) + L[b + y1 * W + x1] * ax) * ay;
        if (v > best) { best = v; arg = c; }
      }
      labels[py * w + px] = arg;
    }
  }
  return labels;
}

// InstantHMR fit2: two square crops (0.9 / 0.7 of the box), then the MHR body graph. Same preprocessing and
// uncalibrated camera (focal 1.05 × diagonal) as upstream.
async function recoverBody(m: Models, { canvas, w, h }: Img, box: { x0: number; x1: number; y0: number; y1: number }) {
  const f = 1.05 * Math.hypot(w, h);
  const cx = (box.x0 + box.x1) / 2, cy = (box.y0 + box.y1) / 2;
  const bw = (box.x1 - box.x0) / 0.866, bh = (box.y1 - box.y0) / 0.866;   // tight box → the box the model was trained on
  const image = new Uint8Array(2 * CROP * CROP * 3), cliff = new Float32Array(6), cropBox = new Float32Array(8);
  [0.9, 0.7].forEach((s, k) => {
    const size = Math.max(bw, bh) * s, side = size * 1.2;
    const ix1 = Math.floor(cx - side / 2), iy1 = Math.floor(cy - side / 2);
    const iw = Math.ceil(cx - side / 2 + side) - ix1, ih = Math.ceil(cy - side / 2 + side) - iy1;
    cliff.set([Math.atan((cx - w / 2) / f), Math.atan((cy - h / 2) / f), size / f], k * 3);
    cropBox.set([ix1, iy1, iw, ih], k * 4);
    const t = new OffscreenCanvas(CROP, CROP).getContext("2d", { willReadFrequently: true })!;
    t.fillStyle = "#000"; t.fillRect(0, 0, CROP, CROP);
    t.drawImage(canvas, ix1, iy1, iw, ih, 0, 0, CROP, CROP);
    const d = t.getImageData(0, 0, CROP, CROP).data, o = k * CROP * CROP * 3;
    for (let i = 0; i < CROP * CROP; i++) { image[o + i * 3] = d[i * 4]; image[o + i * 3 + 1] = d[i * 4 + 1]; image[o + i * 3 + 2] = d[i * 4 + 2]; }
  });
  const r = await m.hmr.run({
    image: new ort.Tensor("uint8", image, [2, CROP, CROP, 3]), cliff_cond: new ort.Tensor("float32", cliff, [2, 3]),
    crop_box: new ort.Tensor("float32", cropBox, [2, 4]), cam: new ort.Tensor("float32", new Float32Array([f, w / 2, h / 2]), [1, 3]),
  });
  const params = new Float32Array(r.mhr_params.data as Float32Array), shape = new Float32Array(r.shape_params.data as Float32Array);
  const b = await m.body.run({ mhr_params: r.mhr_params, shape_params: r.shape_params });
  const tr = r.cam_trans.data as Float32Array;
  // Camera frame, for projecting the posed mesh onto the image.
  const toCam = (a: Float32Array) => { const o = new Float32Array(a.length); for (let i = 0; i < a.length; i += 3) { o[i] = a[i] + tr[0]; o[i + 1] = a[i + 1] + tr[1]; o[i + 2] = a[i + 2] + tr[2]; } return o; };
  const project = (p: Vec3): [number, number] => [(f * p[0]) / p[2] + w / 2, (f * p[1]) / p[2] + h / 2];
  return { joints2d: new Float32Array(r.joints_2d.data as Float32Array), verts: toCam(b.vertices.data as Float32Array), project, params, shape };
}

// RTMPose (trained on Human-Art, so it reads drawings): the person's box padded to 192:256, ×1.25.
async function artKeypoints(m: Models, { canvas }: Img, box: { x0: number; x1: number; y0: number; y1: number }) {
  const cx = (box.x0 + box.x1) / 2, cy = (box.y0 + box.y1) / 2;
  let bw = box.x1 - box.x0, bh = box.y1 - box.y0;
  if (bw / bh > POSE_W / POSE_H) bh = (bw * POSE_H) / POSE_W; else bw = (bh * POSE_W) / POSE_H;
  bw *= 1.25; bh *= 1.25;
  const t = new OffscreenCanvas(POSE_W, POSE_H).getContext("2d", { willReadFrequently: true })!;
  t.fillStyle = "#000"; t.fillRect(0, 0, POSE_W, POSE_H);
  t.drawImage(canvas, cx - bw / 2, cy - bh / 2, bw, bh, 0, 0, POSE_W, POSE_H);
  const d = t.getImageData(0, 0, POSE_W, POSE_H).data, x = new Float32Array(3 * POSE_W * POSE_H), n = POSE_W * POSE_H;
  const mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375];
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) x[c * n + i] = (d[i * 4 + c] - mean[c]) / std[c];
  const r = await m.pose.run({ input: new ort.Tensor("float32", x, [1, 3, POSE_H, POSE_W]) });
  return decodeSimcc(r.simcc_x.data as Float32Array, r.simcc_y.data as Float32Array, 17)
    .map((p) => ({ x: cx - bw / 2 + (p.x / POSE_W) * bw, y: cy - bh / 2 + (p.y / POSE_H) * bh, score: p.score }));
}

// The same body unposed: MHR's first 136 parameters are global transform and joint angles, the last 68 skeleton
// scales (as SAM 3D Body assembles them). Zeroing the pose leaves an A-pose with the person's proportions (used for
// the skeleton and the neck); a copy with the shoulders raised (params 31 right, 41 left) keeps the arms clear of
// the chest for the torso and leg girths.
async function restBody(m: Models, params: Float32Array, shape: Float32Array) {
  const run = async (raise: number) => {
    const p = params.slice();
    p.fill(0, 0, 136);
    p[31] = p[41] = raise;
    const b = await m.body.run({ mhr_params: new ort.Tensor("float32", p, [1, 204]), shape_params: new ort.Tensor("float32", shape, [1, 45]) });
    return { joints: b.joints_3d.data as Float32Array, verts: b.vertices.data as Float32Array };
  };
  return { neutral: await run(0), raised: await run(0.7) };
}

// Everything measured on the unposed body: sections, heights, crotch and limb girths.
function restGeometry(m: Models, { neutral, raised }: Awaited<ReturnType<typeof restBody>>): Pick<Analysis, "sections" | "rest" | "autoLevels"> {
  const frame = bodyFrame(neutral.joints), { origin, up } = frame;
  const upOf = (a: Float32Array, i: number) => (a[i * 3] - origin[0]) * up[0] + (a[i * 3 + 1] - origin[1]) * up[1] + (a[i * 3 + 2] - origin[2]) * up[2];
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < neutral.verts.length / 3; i++) { const u = upOf(neutral.verts, i); lo = Math.min(lo, u); hi = Math.max(hi, u); }
  const sections: Analysis["sections"] = [], cutouts = armCutouts(raised.joints);
  for (let t = -0.15; t <= frame.torso + 0.05; t += 0.01) sections.push({ t, ...sectionAt(raised.verts, m.faces, origin, up, frame.side, t, cutouts) });
  const jointUp = Float32Array.from({ length: 70 }, (_, i) => upOf(neutral.joints, i));
  const ju = (i: number) => jointUp[i], avg = (a: number, b: number) => (ju(a) + ju(b)) / 2;

  // Crotch: the lowest point of the body's midline below the hips.
  let crotch = 0;
  for (let i = 0; i < neutral.verts.length / 3; i++) {
    const v = neutral.verts, d: Vec3 = [v[i * 3] - origin[0], v[i * 3 + 1] - origin[1], v[i * 3 + 2] - origin[2]];
    if (Math.abs(d[0] * frame.side[0] + d[1] * frame.side[1] + d[2] * frame.side[2]) < 0.015) crotch = Math.min(crotch, upOf(v, i));
  }
  const hipUp = avg(J.lHip, J.rHip), kneeUp = avg(J.lKnee, J.rKnee);

  // Limb girths on one leg (sections seeded at its joint, with a short link so the other leg stays out);
  // the neck is the narrowest section above its base.
  const j3 = (i: number): Vec3 => [raised.joints[i * 3], raised.joints[i * 3 + 1], raised.joints[i * 3 + 2]];
  const girth = (verts: Float32Array, o: Vec3, t: number) => perimeter(sectionAt(verts, m.faces, o, up, frame.side, t, [], 0.03).hull);
  const range = (from: number, n: number, step: number) => Array.from({ length: n }, (_, i) => from + i * step);
  const perLeg = (f: (hip: number, knee: number) => number) => (f(J.lHip, J.lKnee) + f(J.rHip, J.rKnee)) / 2;
  const thigh = perLeg((hp) => girth(raised.verts, j3(hp), crotch - ju(hp) - 0.02));
  const calf = perLeg((_, kn) => Math.max(...range(-0.06, 15, -0.01).map((t) => girth(raised.verts, j3(kn), t))));
  const neck = Math.min(...range(ju(J.neck) + 0.02, 11, 0.01).map((t) => girth(neutral.verts, origin, t)).filter((g) => g > 0));

  return {
    sections,
    rest: {
      statureM: hi - lo, floorM: lo, torsoM: frame.torso, jointUp, joints: neutral.joints,
      crotchF: (hipUp - crotch) / (hipUp - kneeUp), girthM: { neck, thigh, calf },
    },
    autoLevels: autoLevels(sections, frame.torso, frame.front),
  };
}

// Projected mesh silhouette. Triangles get the same winding so front and back faces don't cancel, and are
// filled in batches because one huge path rasterises slowly.
function meshSilhouette(verts: Float32Array, faces: Int32Array, project: (p: Vec3) => [number, number], w: number, h: number) {
  const ctx = new OffscreenCanvas(w, h).getContext("2d", { willReadFrequently: true })!;
  const p = new Float32Array((verts.length / 3) * 2);
  for (let i = 0; i < verts.length / 3; i++) p.set(project([verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2]]), i * 2);
  for (let f = 0; f < faces.length; f += 3 * 256) {
    ctx.beginPath();
    for (let g = f; g < Math.min(faces.length, f + 3 * 256); g += 3) {
      const a = faces[g];
      let b = faces[g + 1], c = faces[g + 2];
      if ((p[b * 2] - p[a * 2]) * (p[c * 2 + 1] - p[a * 2 + 1]) < (p[c * 2] - p[a * 2]) * (p[b * 2 + 1] - p[a * 2 + 1])) [b, c] = [c, b];
      ctx.moveTo(p[a * 2], p[a * 2 + 1]); ctx.lineTo(p[b * 2], p[b * 2 + 1]); ctx.lineTo(p[c * 2], p[c * 2 + 1]); ctx.closePath();
    }
    ctx.fill();
  }
  const px = ctx.getImageData(0, 0, w, h).data, mask = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) mask[i] = px[i * 4 + 3] > 0 ? 1 : 0;
  return mask;
}

/** The unposed body's segment ratios for a parameter vector (see lib/fit.ts). */
async function bodyRatios(m: Models, params: Float32Array, shape: Float32Array): Promise<Ratios> {
  const p = params.slice();
  p.fill(0, 0, 136);
  const b = await m.body.run({ mhr_params: new ort.Tensor("float32", p, [1, 204]), shape_params: new ort.Tensor("float32", shape, [1, 45]) });
  const joints = b.joints_3d.data as Float32Array, verts = b.vertices.data as Float32Array, { up } = bodyFrame(joints);
  let top = -Infinity, sole = Infinity;
  for (let i = 0; i < verts.length; i += 3) { const h = verts[i] * up[0] + verts[i + 1] * up[1] + verts[i + 2] * up[2]; top = Math.max(top, h); sole = Math.min(sole, h); }
  return ratios3d(joints, up, top, sole);
}

const api: InferApi = {
  // Illustrations: refit the last body's skeleton scales to the drawing's proportions and re-measure.
  async refit(target) {
    const m = await (models ??= load("auto", () => {}));
    if (!last) throw new Error("no analysis to refit");
    const { shape } = last;
    const fit = await fitScales((p) => bodyRatios(m, p, shape), target, last.params);
    const geo = restGeometry(m, await restBody(m, fit.params, shape));
    return Comlink.transfer({ ...geo, residual: fit.residual }, geo.sections.map((s) => s.hull.buffer));
  },

  async init(backend, onProgress) {
    return (await (models ??= load(backend, onProgress))).backend;
  },

  async analyse(blob, onProgress) {
    const m = await (models ??= load("auto", onProgress));
    const img = await loadImage(blob), { w, h } = img;
    const time = async <T,>(f: () => Promise<T> | T): Promise<[T, number]> => { const t = performance.now(); const r = await f(); return [r, Math.round(performance.now() - t)]; };

    onProgress({ stage: "segment", label: "服と人物の切り抜き" });
    const [labels, msSeg] = await time(async () => {
      // Whole image first to find the person, then again on a square crop around them for sharper edges.
      const coarse = await segment(m, img);
      const box = extent(img.alpha ?? coarse.map((l) => (l === LABEL.background ? 0 : 1)), w, h);
      if (!box) return coarse;
      const side = Math.max(box.x1 - box.x0, box.y1 - box.y0) * 1.15;
      return segment(m, img, { x: (box.x0 + box.x1 - side) / 2, y: (box.y0 + box.y1 - side) / 2, sw: side, sh: side });
    });
    const person = largestComponent(img.alpha ?? labels.map((l) => (l === LABEL.background ? 0 : 1)), w, h).mask;
    const box = extent(person, w, h);
    if (!box) throw new Error("人物が見つかりませんでした");

    onProgress({ stage: "body", label: "骨格と3D体型" });
    const [[body, rest], msBody] = await time(async () => {
      const posed = await recoverBody(m, img, box);
      // The body model reads shape best from an upright person: if the spine leans, run it again on the image
      // rotated upright and take the body shape from there.
      const j = posed.joints2d, hipX = (j[J.lHip * 2] + j[J.rHip * 2]) / 2, hipY = (j[J.lHip * 2 + 1] + j[J.rHip * 2 + 1]) / 2;
      const lean = Math.atan2(j[J.neck * 2] - hipX, hipY - j[J.neck * 2 + 1]);
      const shapeFrom = Math.abs(lean) > (10 * Math.PI) / 180 ? await recoverBody(m, ...rotated(img, person, -lean, box)) : posed;
      last = { params: shapeFrom.params, shape: shapeFrom.shape };
      return [posed, await restBody(m, shapeFrom.params, shapeFrom.shape)] as const;
    });

    const comp = (cls: number) => largestComponent(labels.map((l) => (l === cls ? 1 : 0)), w, h);
    const skirt = comp(LABEL.skirt), dress = comp(LABEL.dress);
    const g = dress.size > skirt.size ? { ...dress, kind: "dress" as const } : { ...skirt, kind: "skirt" as const };
    const hasGarment = g.size > w * h * 0.002;
    const garment = hasGarment ? g.mask : new Uint8Array(w * h);
    const face = comp(LABEL.face).mask, kind = hasGarment ? g.kind : "none";
    const found = findLandmarks(body.joints2d, person, garment, kind, face, w, h);
    // Illustrations: the same landmarks from RTMPose's keypoints. The neck sits above the shoulders by the
    // unposed body's neck-to-shoulder rise over its shoulder-to-hip height.
    const ru = rest.neutral.joints, upY = (i: number) => -ru[i * 3 + 1];   // MHR's y points down in its rest frame
    const sh = (upY(J.lShoulder) + upY(J.rShoulder)) / 2, hp = (upY(J.lHip) + upY(J.rHip)) / 2;
    const joints2dArt = mergeCoco(body.joints2d, await artKeypoints(m, img, box), (upY(J.neck) - sh) / (sh - hp));
    const foundArt = findLandmarks(joints2dArt, person, garment, kind, face, w, h);

    onProgress({ stage: "sections", label: "断面の計算" });
    const [geo, msSec] = await time(() => {
      const meshPoints = new Float32Array(Math.ceil(body.verts.length / 3 / 6) * 2);
      for (let i = 0, k = 0; i < body.verts.length; i += 18) meshPoints.set(body.project([body.verts[i], body.verts[i + 1], body.verts[i + 2]]), (k++) * 2);
      return { meshPoints, meshMask: meshSilhouette(body.verts, m.faces, body.project, w, h), ...restGeometry(m, rest) };
    });

    const result: Analysis = {
      w, h, backend: m.backend, ms: { segment: msSeg, body: msBody, sections: msSec },
      transparent: img.alpha != null, person, garment, garmentKind: hasGarment ? g.kind : "none",
      cut: box.touchTop || box.touchBottom, joints2d: body.joints2d, found, art: { joints2d: joints2dArt, found: foundArt }, ...geo,
    };
    return Comlink.transfer(result, [person.buffer, garment.buffer, body.joints2d.buffer, joints2dArt.buffer, geo.meshPoints.buffer, geo.meshMask.buffer, ...geo.sections.map((s) => s.hull.buffer)]);
  },
};

// Head top, chin, soles, waist and hems, found by walking the masks along the skeleton, so tilted or bent
// poses work the same as upright ones.
function findLandmarks(j2: Float32Array, person: Uint8Array, garment: Uint8Array, kind: Analysis["garmentKind"], face: Uint8Array, w: number, h: number): Analysis["found"] {
  const P = (i: number): Pt => ({ x: j2[i * 2], y: j2[i * 2 + 1] });
  const neck = P(J.neck), nose = P(J.nose), hip = mid2(P(J.lHip), P(J.rHip)), down = { x: hip.x - neck.x, y: hip.y - neck.y };
  let chin: Pt | null = null, best = -Infinity;
  for (let i = 0; i < face.length; i++) if (face[i]) {
    const p = { x: i % w, y: (i / w) | 0 }, s = (p.x - neck.x) * down.x + (p.y - neck.y) * down.y;
    if (s > best) { best = s; chin = p; }
  }
  const leg = (hp: number, kn: number, an: number) => {
    const sole = edgeAlong(person, w, h, P(an), { x: P(an).x - P(kn).x, y: P(an).y - P(kn).y });
    return { sole, hem: insideRange(garment, w, h, [P(hp), P(kn), P(an), sole])?.last ?? null };
  };
  const l = leg(J.lHip, J.lKnee, J.lAnkle), r = leg(J.rHip, J.rKnee, J.rAnkle);
  const waist = kind === "skirt" ? insideRange(garment, w, h, [neck, hip, lerp2(neck, hip, 1.4)])?.first ?? null
    : kind === "dress" ? lerp2(neck, hip, 0.62) : null;
  return { headTop: edgeAlong(person, w, h, nose, { x: nose.x - neck.x, y: nose.y - neck.y }), chin, sole: { l: l.sole, r: r.sole }, waist, hem: { l: l.hem, r: r.hem } };
}

Comlink.expose(api);
