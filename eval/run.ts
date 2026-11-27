// Evaluation runner (dev only; needs `pnpm dev` running).
//   pnpm eval render [dir]  render every <dir>/*.vrm (default eval/vrm) into eval/data/vrm/<name>-<pose>/
//   pnpm eval               run the app on eval/data/{vrm,profile}/*/ and write eval/results/<date>.md
// A case is image.png + truth.json: { heightCm, crotchCm?, girthCm?: { bust?, underbust?, waist?, hip? }, skirtCm? }.
import { mkdirSync, readdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { basename, join } from "node:path";
import { chromium, type Page } from "playwright";
import { jisBraSize } from "../src/lib/cup";

const BASE = process.env.EVAL_URL ?? "http://localhost:5173";
const EP = process.env.EVAL_EP ?? "wasm";   // headless WebGPU is unreliable
const CONFIGS = { "B1+A2": "", "B1": "nofit", "A2": "noart", "なし": "noart&nofit" };
const GIRTHS = ["bust", "underbust", "waist", "hip"] as const;

type Truth = { heightCm: number; crotchCm?: number; girthCm?: Partial<Record<(typeof GIRTHS)[number], number>>; skirtCm?: number | null };
type Out = { skirt: { lengthCm: number } | null; girthCm: Record<string, { value: number }>; bra: { label: string } | null; details: Record<string, Record<string, string>> };

const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu"] });

if (process.argv[2] === "render") {
  const dir = process.argv[3] ?? "eval/vrm", page = await browser.newPage();
  await page.goto(`${BASE}/eval/vrm.html`);
  await page.waitForFunction(() => "renderCase" in window);
  for (const file of readdirSync(dir).filter((f) => f.endsWith(".vrm"))) {
    const url = `/@fs${join(process.cwd(), dir, file)}`;
    const r = await page.evaluate((u) => (window as any).renderCase(u), url) as { truth: Truth; images: Record<string, string> };
    for (const [pose, png] of Object.entries(r.images)) {
      const out = join("eval/data/vrm", `${basename(file, ".vrm")}-${pose}`);
      mkdirSync(out, { recursive: true });
      writeFileSync(join(out, "image.png"), Buffer.from(png.split(",")[1], "base64"));
      writeFileSync(join(out, "truth.json"), JSON.stringify(r.truth, null, 2));
    }
    console.log(file, r.truth);
  }
  await browser.close();
  process.exit(0);
}

async function measure(page: Page, image: string, heightCm: number, flags: string): Promise<Out> {
  await page.goto(`${BASE}/?ep=${EP}${flags && `&${flags}`}`);
  await page.getByRole("button", { name: "同意して使う" }).click();
  await page.locator("#height").fill(String(heightCm));
  await page.locator('input[type="file"]').setInputFiles(image);
  const json = page.getByRole("button", { name: "JSON" });
  await json.waitFor({ timeout: 180_000 });
  await page.getByRole("radio", { name: "イラスト" }).click();
  if (!flags.includes("nofit")) await page.getByText("体型の比率を絵に合わせています").waitFor({ timeout: 60_000 });
  const [download] = await Promise.all([page.waitForEvent("download"), json.click()]);
  return JSON.parse(readFileSync(await download.path(), "utf8"));
}

const num = (s?: string) => (s ? parseFloat(s) : NaN);
const detail = (o: Out, label: string) => num(Object.values(o.details).map((g) => g[label]).find(Boolean));

function errors(t: Truth, o: Out) {
  const e: Record<string, number> = {};
  if (t.crotchCm != null) e["股下"] = detail(o, "股下") - t.crotchCm;
  for (const g of GIRTHS) if (t.girthCm?.[g] != null) e[g] = o.girthCm[g].value - t.girthCm[g]!;
  if (t.skirtCm != null && o.skirt) e["スカート丈"] = o.skirt.lengthCm - t.skirtCm;
  return e;
}

const cases = ["vrm", "profile"].flatMap((set) => {
  const root = join("eval/data", set);
  return existsSync(root) ? readdirSync(root).filter((id) => existsSync(join(root, id, "truth.json"))).map((id) => ({ set, id, dir: join(root, id) })) : [];
});
if (!cases.length) throw new Error("no cases under eval/data/{vrm,profile}/<id>/{image.png,truth.json}");

const page = await browser.newPage({ acceptDownloads: true });
const rows: { set: string; config: string; err: Record<string, number>; cup: boolean | null }[] = [];
for (const c of cases) {
  const truth: Truth = JSON.parse(readFileSync(join(c.dir, "truth.json"), "utf8"));
  const tb = truth.girthCm?.bust, tu = truth.girthCm?.underbust;
  for (const [config, flags] of Object.entries(CONFIGS)) {
    try {
      const out = await measure(page, join(c.dir, "image.png"), truth.heightCm, flags);
      const cup = tb != null && tu != null && out.bra ? out.bra.label === jisBraSize(tb, tu).label : null;
      rows.push({ set: c.set, config, err: errors(truth, out), cup });
      console.log(c.set, c.id, config, rows.at(-1)!.err);
    } catch (e) {
      console.warn(c.set, c.id, config, "failed:", (e as Error).message.split("\n")[0]);
    }
  }
}
await browser.close();

// Markdown: mean absolute error (cm) per metric and configuration, per set.
const metrics = [...new Set(rows.flatMap((r) => Object.keys(r.err)))];
const lines = [`# 評価結果 ${new Date().toISOString().slice(0, 10)}`, "", `症例 ${cases.length}・推論 ${EP}。値は平均絶対誤差（cm）。`];
for (const set of [...new Set(rows.map((r) => r.set))]) {
  lines.push("", `## ${set}`, "", `| 構成 | ${metrics.join(" | ")} | カップ一致 |`, `|---|${metrics.map(() => "---:").join("|")}|---:|`);
  for (const config of Object.keys(CONFIGS)) {
    const rs = rows.filter((r) => r.set === set && r.config === config);
    const mae = metrics.map((m) => { const v = rs.map((r) => r.err[m]).filter(Number.isFinite); return v.length ? (v.reduce((s, x) => s + Math.abs(x), 0) / v.length).toFixed(1) : "–"; });
    const cups = rs.map((r) => r.cup).filter((x) => x != null);
    lines.push(`| ${config} | ${mae.join(" | ")} | ${cups.length ? `${cups.filter(Boolean).length}/${cups.length}` : "–"} |`);
  }
}
mkdirSync("eval/results", { recursive: true });
const file = `eval/results/${new Date().toISOString().slice(0, 10)}.md`;
writeFileSync(file, lines.join("\n") + "\n");
console.log(lines.join("\n"), `\n→ ${file}`);
