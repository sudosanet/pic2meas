// Evaluation runner (dev only; needs `pnpm dev` running).
//   pnpm eval render [dir]  render every <dir>/*.vrm (default eval/vrm) into eval/data/vrm/<name>-<pose>/
//   pnpm eval               run the app on eval/data/<set>/*/ and write eval/results/<date>.md (+ cases.csv)
// A case is image.png + truth.json: { heightCm, crotchCm?, girthCm?: { bust?, underbust?, waist?, hip? }, skirtCm?,
// bra? (full JIS label), cup? (cup letter alone) }.
// EVAL_DATA / EVAL_OUT move the data and results (e.g. outside the repo for images you may not commit);
// EVAL_CONFIGS picks configurations by name (comma-separated).
import { mkdirSync, readdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { basename, join } from "node:path";
import { chromium, type Page } from "playwright";
import { jisBraSize } from "../src/lib/cup";

const BASE = process.env.EVAL_URL ?? "http://localhost:5173";
const EP = process.env.EVAL_EP ?? "auto";
const ALL_CONFIGS: Record<string, string> = { "現行": "", "胸の推定なし": "nobreast" };
const CONFIGS = Object.fromEntries(Object.entries(ALL_CONFIGS).filter(([k]) => !process.env.EVAL_CONFIGS || process.env.EVAL_CONFIGS.split(",").includes(k)));
const DATA = process.env.EVAL_DATA ?? "eval/data", OUT = process.env.EVAL_OUT ?? "eval/results";
const CUPS = ["AA", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"];
const cupIndex = (c: string) => (c === "AA未満" ? -1 : c === "I超" ? 10 : CUPS.indexOf(c));
const LEVEL_JA = { bust: "バスト", underbust: "アンダーバスト", waist: "ウエスト", hip: "ヒップ" } as const;
const GIRTHS = ["bust", "underbust", "waist", "hip"] as const;

type Truth = { heightCm: number; crotchCm?: number; girthCm?: Partial<Record<(typeof GIRTHS)[number], number>>; levelCm?: Partial<Record<(typeof GIRTHS)[number], number>>; bra?: string; cup?: string; skirtCm?: number | null };
type Out = { skirt: { lengthCm: number } | null; girthCm: Record<string, { value: number }>; bra: { label: string; cup: string } | null; details: Record<string, Record<string, string>> };

// Headless Chromium falls back to a software WebGPU unless pointed at the GPU (Metal on macOS).
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=metal", "--enable-features=Vulkan,WebGPU"] });

if (process.argv[2] === "render") {
  const dir = process.argv[3] ?? "eval/vrm", page = await browser.newPage();
  await page.goto(`${BASE}/eval/vrm.html`);
  await page.waitForFunction(() => "renderCase" in window);
  for (const file of readdirSync(dir).filter((f) => f.endsWith(".vrm"))) {
    const url = `/@fs${join(process.cwd(), dir, file)}`;
    const r = await page.evaluate((u) => (window as any).renderCase(u), url) as { truth: Truth; images: Record<string, string> };
    for (const [pose, png] of Object.entries(r.images)) {
      const out = join(DATA, "vrm", `${basename(file, ".vrm")}-${pose}`);
      mkdirSync(out, { recursive: true });
      writeFileSync(join(out, "image.png"), Buffer.from(png.split(",")[1], "base64"));
      writeFileSync(join(out, "truth.json"), JSON.stringify(r.truth, null, 2));
    }
    console.log(file, r.truth);
  }
  await browser.close();
  process.exit(0);
}

const storeState = (page: Page) => page.evaluate(async () => {
  const url = performance.getEntriesByType("resource").map((e) => e.name).find((n) => n.includes("/src/store.ts"))!;
  const st = (await import(url)).useStore.getState();
  return { busy: st.busy as boolean, analysis: st.analysis as object | null, fit: st.fitResidual as number | null, n: (window as any).__evalN as number | undefined };
});

// One page per configuration (loading the models once); each image replaces the last.
let loadedFlags: string | null = null;
async function measure(page: Page, image: string, heightCm: number, flags: string, mode: "photo" | "illust"): Promise<Out> {
  if (loadedFlags !== flags) {
    await page.goto(`${BASE}/?ep=${EP}${flags && `&${flags}`}`);
    await page.getByRole("button", { name: "同意して使う" }).click();
    loadedFlags = flags;
  }
  await page.locator("#height").fill(String(heightCm));
  // Mark the old analysis so a new one can be told apart.
  await page.evaluate(async () => {
    const url = performance.getEntriesByType("resource").map((e) => e.name).find((n) => n.includes("/src/store.ts"))!;
    (await import(url)).useStore.setState({ analysis: null, fitResidual: null });
  });
  await page.locator('input[type="file"]').setInputFiles(image);
  for (let t = 0; ; t += 250) {
    const s = await storeState(page);
    if (s.analysis && !s.busy) break;
    if (t > 240_000) throw new Error("analysis timed out");
    await page.waitForTimeout(250);
  }
  await page.getByRole("radio", { name: mode === "illust" ? "イラスト" : "写真" }).click();
  if (mode === "illust" && !flags.includes("nofit")) for (let t = 0; (await storeState(page)).fit == null; t += 250) { if (t > 60_000) throw new Error("refit timed out"); await page.waitForTimeout(250); }
  const json = page.getByRole("button", { name: "JSON" });
  const [download] = await Promise.all([page.waitForEvent("download"), json.click()]);
  return JSON.parse(readFileSync(await download.path(), "utf8"));
}

const num = (s?: string) => (s ? parseFloat(s) : NaN);
const detail = (o: Out, label: string) => num(Object.values(o.details).map((g) => g[label]).find(Boolean));

function errors(t: Truth, o: Out) {
  const e: Record<string, number> = {};
  if (t.crotchCm != null) e["股下"] = detail(o, "股下") - t.crotchCm;
  for (const g of GIRTHS) if (t.girthCm?.[g] != null) e[g] = o.girthCm[g].value - t.girthCm[g]!;
  for (const g of GIRTHS) if (t.levelCm?.[g] != null) e[`${g}の高さ`] = detail(o, LEVEL_JA[g]) - t.levelCm[g]!;
  if (t.skirtCm != null && o.skirt) e["スカート丈"] = o.skirt.lengthCm - t.skirtCm;
  return e;
}

const cases = (existsSync(DATA) ? readdirSync(DATA) : []).flatMap((set) => {
  const root = join(DATA, set);
  return existsSync(root) ? readdirSync(root).filter((id) => existsSync(join(root, id, "truth.json"))).map((id) => ({ set, id, dir: join(root, id) })) : [];
});
if (!cases.length) throw new Error(`no cases under ${DATA}/<set>/<id>/{image.png,truth.json}`);

const page = await browser.newPage({ acceptDownloads: true });
type Row = { set: string; id: string; config: string; err: Record<string, number>; cup: boolean | null; cupDiff: number | null; inner: Record<string, number | string> };
const rows: Row[] = [];
for (const c of cases) {
  const truth: Truth = JSON.parse(readFileSync(join(c.dir, "truth.json"), "utf8"));
  const tb = truth.girthCm?.bust, tu = truth.girthCm?.underbust;
  const labelTruth = truth.bra ?? (tb != null && tu != null ? jisBraSize(tb, tu).label : null);
  const cupTruth = truth.cup ?? (labelTruth ? labelTruth.replace(/\d+$/, "") : null);
  for (const [config, flags] of Object.entries(CONFIGS)) {
    try {
      const out = await measure(page, join(c.dir, "image.png"), truth.heightCm, flags, c.set === "photo" ? "photo" : "illust");
      const cupDiff = cupTruth && out.bra ? cupIndex(out.bra.cup) - cupIndex(cupTruth) : null;
      rows.push({ set: c.set, id: c.id, config, err: errors(truth, out), cup: cupDiff == null ? null : cupDiff === 0, cupDiff, inner: { ...(await internals(page, truth.heightCm)), ...Object.fromEntries(Object.entries(out.girthCm).map(([k, v]) => [`${k}Cm`, v.value])) } });
      console.log(c.set, c.id, config, rows.at(-1)!.err, "cup", cupDiff);
    } catch (e) {
      console.warn(c.set, c.id, config, "failed:", (e as Error).message.split("\n")[0]);
    }
  }
}
await browser.close();

// Markdown: mean absolute error (mean signed error) per metric and configuration, per set; cup steps likewise.
const metrics = [...new Set(rows.flatMap((r) => Object.keys(r.err)))];
const stat = (v: number[]) => (v.length ? `${(v.reduce((s, x) => s + Math.abs(x), 0) / v.length).toFixed(1)} (${(v.reduce((s, x) => s + x, 0) / v.length >= 0 ? "+" : "")}${(v.reduce((s, x) => s + x, 0) / v.length).toFixed(1)})` : "–");
const lines = [`# 評価結果 ${new Date().toISOString().slice(0, 10)}`, "", `症例 ${cases.length}・推論 ${EP}。値は平均絶対誤差（括弧内は平均の符号付き誤差、+ は大きく出る）。カップは段数。`];
for (const set of [...new Set(rows.map((r) => r.set))]) {
  lines.push("", `## ${set}`, "", `| 構成 | ${metrics.join(" | ")} | カップ段差 | カップ一致 / ±1 |`, `|---|${metrics.map(() => "---:").join("|")}|---:|---:|`);
  for (const config of Object.keys(CONFIGS)) {
    const rs = rows.filter((r) => r.set === set && r.config === config);
    const diffs = rs.map((r) => r.cupDiff).filter((x): x is number => x != null);
    const cups = diffs.length ? `${diffs.filter((d) => d === 0).length}/${diffs.filter((d) => Math.abs(d) <= 1).length}/${diffs.length}` : "–";
    lines.push(`| ${config} | ${metrics.map((m) => stat(rs.map((r) => r.err[m]).filter(Number.isFinite))).join(" | ")} | ${stat(diffs)} | ${cups} |`);
  }
}
mkdirSync(OUT, { recursive: true });
const stamp = new Date().toISOString().slice(0, 10), file = join(OUT, `${stamp}.md`);
writeFileSync(file, lines.join("\n") + "\n");
// Per case, for calibrating offline.
const keys = [...new Set(rows.flatMap((r) => [...Object.keys(r.err).map((k) => `err:${k}`), ...Object.keys(r.inner)]))];
writeFileSync(join(OUT, `${stamp}-cases.csv`), [["set", "id", "config", "cupDiff", ...keys].join(","), ...rows.map((r) => [r.set, r.id, r.config, r.cupDiff ?? "", ...keys.map((k) => (k.startsWith("err:") ? r.err[k.slice(4)] : r.inner[k]) ?? "")].join(","))].join("\n") + "\n");
console.log(lines.join("\n"), `\n→ ${file}`);

/** Intermediate values from the app's store (breast caps, girths in cm), for calibrating offline. */
async function internals(page: Page, heightCm: number) {
  return page.evaluate(async (H): Promise<Record<string, number | string>> => {
    const url = performance.getEntriesByType("resource").map((e) => e.name).find((n) => n.includes("/src/store.ts"))!;
    const st = (await import(url)).useStore.getState(), a = st.analysis, b = st.landmarks?.breast, k = H / a.rest.statureM;   // cm per body-model metre
    return { cut: a.cut ? 1 : 0, ...(b ? { capSrc: b.src, capConf: +b.conf.toFixed(2), capACm: +(b.aM * k).toFixed(2), capHCm: +(b.h * k).toFixed(2) } : {}) };
  }, heightCm);
}
