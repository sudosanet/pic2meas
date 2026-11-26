/// <reference lib="webworker" />
// SegFormer (clothes) + InstantHMR (body) on onnxruntime-web, then mesh cross-sections.
import * as Comlink from "comlink";
import * as ort from "onnxruntime-web/webgpu";
import { armCutouts, autoLevels, bodyFrame, extent, garmentLines, largestComponent, sectionAt, type Vec3 } from "@/lib/measure";
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

type Models = { seg: ort.InferenceSession; hmr: ort.InferenceSession; body: ort.InferenceSession; faces: Int32Array; backend: Backend };
let models: Promise<Models> | undefined;

async function load(want: Backend | "auto", onProgress: (p: Progress) => void): Promise<Models> {
  const gpu = want !== "wasm" && !!(await navigator.gpu?.requestAdapter());
  let backend: Backend = gpu ? "webgpu" : "wasm";
  const bytes = { seg: await fetchModel(MODELS.seg, onProgress), hmr: await fetchModel(MODELS.hmr, onProgress), body: await fetchModel(MODELS.body, onProgress) };
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
  const seg = await create("seg"), hmr = await create("hmr"), body = await create("body");
  return { seg, hmr, body, faces: meshFaces(bytes.body), backend };
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

// SegFormer: 512×512 in, 18 class logits at 128×128 out, upsampled bilinearly, argmax per pixel.
async function segment(m: Models, { canvas, w, h }: Img): Promise<Uint8Array> {
  const t = new OffscreenCanvas(SEG, SEG).getContext("2d", { willReadFrequently: true })!;
  t.drawImage(canvas, 0, 0, SEG, SEG);
  const d = t.getImageData(0, 0, SEG, SEG).data, x = new Float32Array(3 * SEG * SEG);
  const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
  for (let i = 0; i < SEG * SEG; i++) for (let c = 0; c < 3; c++) x[c * SEG * SEG + i] = (d[i * 4 + c] / 255 - mean[c]) / std[c];
  const out = await m.seg.run({ [m.seg.inputNames[0]]: new ort.Tensor("float32", x, [1, 3, SEG, SEG]) });
  const lg = out[m.seg.outputNames[0]], [, C, H, W] = lg.dims as number[], L = lg.data as Float32Array;
  const labels = new Uint8Array(w * h);
  for (let py = 0; py < h; py++) {
    const fy = Math.min(H - 1, Math.max(0, ((py + 0.5) * H) / h - 0.5)), y0 = fy | 0, y1 = Math.min(H - 1, y0 + 1), ay = fy - y0;
    for (let px = 0; px < w; px++) {
      const fx = Math.min(W - 1, Math.max(0, ((px + 0.5) * W) / w - 0.5)), x0 = fx | 0, x1 = Math.min(W - 1, x0 + 1), ax = fx - x0;
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
  const b = await m.body.run({ mhr_params: r.mhr_params, shape_params: r.shape_params });
  const tr = r.cam_trans.data as Float32Array;
  // Into the camera frame, so the body's side axis is the image horizontal.
  const toCam = (a: Float32Array) => { const o = new Float32Array(a.length); for (let i = 0; i < a.length; i += 3) { o[i] = a[i] + tr[0]; o[i + 1] = a[i + 1] + tr[1]; o[i + 2] = a[i + 2] + tr[2]; } return o; };
  const project = (p: Vec3): [number, number] => [(f * p[0]) / p[2] + w / 2, (f * p[1]) / p[2] + h / 2];
  return { joints2d: new Float32Array(r.joints_2d.data as Float32Array), joints: toCam(b.joints_3d.data as Float32Array), verts: toCam(b.vertices.data as Float32Array), project };
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

const api: InferApi = {
  async init(backend, onProgress) {
    return (await (models ??= load(backend, onProgress))).backend;
  },

  async analyse(blob, onProgress) {
    const m = await (models ??= load("auto", onProgress));
    const img = await loadImage(blob), { w, h } = img;
    const time = async <T,>(f: () => Promise<T> | T): Promise<[T, number]> => { const t = performance.now(); const r = await f(); return [r, Math.round(performance.now() - t)]; };

    onProgress({ stage: "segment", label: "服と人物の切り抜き" });
    const [labels, msSeg] = await time(() => segment(m, img));
    const person = largestComponent(img.alpha ?? labels.map((l) => (l === LABEL.background ? 0 : 1)), w, h).mask;
    const sil = extent(person, w, h);
    if (!sil) throw new Error("人物が見つかりませんでした");

    onProgress({ stage: "body", label: "骨格と3D体型" });
    const [body, msBody] = await time(() => recoverBody(m, img, sil));

    const comp = (cls: number) => largestComponent(labels.map((l) => (l === cls ? 1 : 0)), w, h);
    const skirt = comp(LABEL.skirt), dress = comp(LABEL.dress);
    const g = dress.size > skirt.size ? { ...dress, kind: "dress" as const } : { ...skirt, kind: "skirt" as const };
    const hasGarment = g.size > w * h * 0.002;
    const face = comp(LABEL.face);
    const faceExt = face.size > 0 ? extent(face.mask, w, h) : null;

    onProgress({ stage: "sections", label: "断面の計算" });
    const [geo, msSec] = await time(() => {
      const frame = bodyFrame(body.joints), { origin, up } = frame;
      const meshPoints = new Float32Array(Math.ceil(body.verts.length / 3 / 6) * 2);
      let headY = Infinity, soleY = -Infinity, lo = Infinity, hi = -Infinity;
      for (let i = 0, k = 0; i < body.verts.length; i += 3) {
        const p: Vec3 = [body.verts[i], body.verts[i + 1], body.verts[i + 2]];
        const [u, v] = body.project(p);
        headY = Math.min(headY, v); soleY = Math.max(soleY, v);
        const s = (p[0] - origin[0]) * up[0] + (p[1] - origin[1]) * up[1] + (p[2] - origin[2]) * up[2];
        lo = Math.min(lo, s); hi = Math.max(hi, s);
        if ((i / 3) % 6 === 0) { meshPoints[k++] = u; meshPoints[k++] = v; }
      }
      const sections: Analysis["sections"] = [], cutouts = armCutouts(body.joints);
      for (let t = -0.15; t <= frame.torso + 0.05; t += 0.01) {
        const [cx, y] = body.project([origin[0] + up[0] * t, origin[1] + up[1] * t, origin[2] + up[2] * t]);
        sections.push({ t, y, cx, ...sectionAt(body.verts, m.faces, origin, up, frame.side, t, cutouts) });
      }
      const meshMask = meshSilhouette(body.verts, m.faces, body.project, w, h);
      return { meshPoints, meshMask, meshSpan: { headY, soleY }, meshStatureM: hi - lo, sections, autoLevels: autoLevels(sections, frame.torso, frame.front) };
    });

    const result: Analysis = {
      w, h, backend: m.backend, ms: { segment: msSeg, body: msBody, sections: msSec },
      transparent: img.alpha != null, person, garment: hasGarment ? g.mask : new Uint8Array(w * h), garmentKind: hasGarment ? g.kind : "none",
      garmentLines: hasGarment ? garmentLines(g.mask, w, h) : null,
      silhouette: sil, chinY: faceExt?.y1 ?? null, joints2d: body.joints2d, ...geo,
    };
    return Comlink.transfer(result, [person.buffer, result.garment.buffer, body.joints2d.buffer, geo.meshPoints.buffer, geo.meshMask.buffer, ...geo.sections.map((s) => s.hull.buffer)]);
  },
};

Comlink.expose(api);
