import { Copy, FileJson, ImageDown, TriangleAlert } from "lucide-react";
import { toast } from "sonner";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { LEVELS, LEVEL_LABEL, type Level } from "@/lib/measure";
import { computeResults, forMode, type Results, type Source } from "@/lib/results";
import { useStore } from "@/store";
import { editorStage } from "@/lib/editorStage";

const SOURCE: Record<Source, { label: string; variant: "secondary" | "outline" | "default" }> = {
  auto: { label: "自動", variant: "secondary" }, edited: { label: "補正", variant: "outline" }, manual: { label: "手入力", variant: "default" },
};
const f1 = (v: number) => (Number.isFinite(v) ? v.toFixed(1) : "—");

function Row({ label, value, unit, source, override, onOverride }: {
  label: string; value: number; unit: string; source: Source; override?: number; onOverride: (v: number | undefined) => void;
}) {
  return (
    <div className="grid grid-cols-[1fr_auto_5.5rem] items-center gap-2 text-sm">
      <span>{label} <Badge variant={SOURCE[source].variant} className="ml-1 align-middle">{SOURCE[source].label}</Badge></span>
      <span className="text-right font-semibold tabular-nums">{f1(value)} {unit}</span>
      <Input aria-label={`${label}を手入力`} type="number" step={0.1} placeholder="実測値" className="h-7 text-right" value={override ?? ""}
        onChange={(e) => onOverride(e.target.value === "" ? undefined : Number(e.target.value))} />
    </div>
  );
}

export function ResultPanel() {
  const { analysis, landmarks, auto, heightCm, mode, calibrate2D, overrides, setOverride, setLandmarks, file, fitResidual } = useStore();
  if (!analysis || !landmarks || !auto) return null;
  const r = computeResults(forMode(analysis, mode), landmarks, auto, { heightCm, mode, calibrate2D }, overrides);

  return (
    <Card size="sm">
      <CardHeader><CardTitle>結果</CardTitle></CardHeader>
      <CardContent className="space-y-4">
        {mode === "illust" && fitResidual != null && (
          <p className="text-xs text-muted-foreground">体型の比率を絵に合わせています（ずれ {(fitResidual * 100).toFixed(1)}%）</p>
        )}
        {r.warnings.map((w) => (
          <Alert key={w} variant="destructive"><TriangleAlert /><AlertDescription>{w}</AlertDescription></Alert>
        ))}

        <section className="space-y-2">
          <h3 className="text-xs font-medium text-muted-foreground">スカート / ワンピース（{analysis.garmentKind === "dress" ? "ワンピース" : analysis.garmentKind === "skirt" ? "スカート" : "未検出"}）</h3>
          {r.skirt ? (
            <>
              <Row label="丈" value={r.skirt.cm} unit="cm" source={r.skirt.source} override={overrides.skirtCm} onOverride={(v) => setOverride("skirtCm", v)} />
              <div className="flex justify-between text-sm"><span>丈の区分</span><span className="font-semibold">{r.skirt.category}</span></div>
              <div className="flex justify-between text-sm text-muted-foreground"><span>丈 / 身長</span><span className="tabular-nums">{f1(r.skirt.ratio * 100)} %</span></div>
            </>
          ) : <p className="text-sm text-muted-foreground">—</p>}
        </section>

        <section className="space-y-2">
          <h3 className="text-xs font-medium text-muted-foreground">周囲長（推定）{calibrate2D && "・シルエット幅で補正"}</h3>
          {LEVELS.map((lv: Level) => (
            <Row key={lv} label={LEVEL_LABEL[lv]} value={r.girth[lv].cm} unit="cm" source={r.girth[lv].source}
              override={overrides.girth[lv]} onOverride={(v) => setOverride(lv, v)} />
          ))}
          {landmarks.breast && (() => {
            const k = heightCm / analysis.rest.statureM, h = landmarks.breast.h * k;   // cm per body-model metre
            return (
              <div className="grid grid-cols-[1fr_auto_5.5rem] items-center gap-2 text-sm">
                <span>胸の出っぱり <Badge variant="outline" className="ml-1 align-middle">{same(landmarks.breast, auto.breast) ? "自動" : "修正"}</Badge></span>
                <span className="text-right font-semibold tabular-nums">{f1(h)} cm</span>
                <Input aria-label="胸の出っぱりを入力" type="number" step={0.1} min={0} className="h-7 text-right" value={f1(h)}
                  onChange={(e) => { const v = Number(e.target.value); if (Number.isFinite(v) && v >= 0) setLandmarks((l) => ({ ...l, breast: { ...l.breast!, h: v / k } })); }} />
              </div>
            );
          })()}
        </section>

        <section className="grid grid-cols-2 gap-2">
          <div className="rounded-lg bg-muted p-3 text-center">
            <div className="text-xs text-muted-foreground">ブラサイズ（JIS）</div>
            <div className="text-2xl font-bold">{r.bra?.label ?? "—"}</div>
            {r.bra && <div className="text-xs text-muted-foreground tabular-nums">差 {f1(r.bra.diff)} cm</div>}
          </div>
          <div className="rounded-lg bg-muted p-3 text-center">
            <div className="text-xs text-muted-foreground">頭身</div>
            <div className="text-2xl font-bold tabular-nums">{f1(r.headRatio)}</div>
          </div>
        </section>

        {r.details.map((g) => (
          <section key={g.title} className="space-y-1">
            <h3 className="text-xs font-medium text-muted-foreground">{g.title}</h3>
            {g.items.map((d) => (
              <div key={d.label} className="flex justify-between text-sm">
                <span>{d.label}</span><span className="tabular-nums">{f1(d.value)}{d.unit && ` ${d.unit}`}</span>
              </div>
            ))}
          </section>
        ))}

        <div className="flex gap-2">
          <Button variant="outline" size="sm" className="flex-1" onClick={exportPng}><ImageDown />画像</Button>
          <Button variant="outline" size="sm" className="flex-1" onClick={() => exportJson(r, file?.name)}><FileJson />JSON</Button>
          <Button variant="outline" size="sm" className="flex-1" onClick={() => copyText(r)}><Copy />コピー</Button>
        </div>
        <p className="text-xs text-muted-foreground">
          {analysis.backend === "webgpu" ? "WebGPU" : "WASM"}・切り抜き {analysis.ms.segment}ms・体型 {analysis.ms.body}ms・胸 {analysis.ms.breast}ms・断面 {analysis.ms.sections}ms
        </p>
      </CardContent>
    </Card>
  );
}

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

function download(href: string, name: string) {
  const a = document.createElement("a");
  a.href = href; a.download = name; a.click();
}

function exportPng() {
  const s = editorStage.current, img = editorStage.image;
  if (!s || !img) return;
  // The image area at working resolution, whatever the zoom.
  const t = s.getAbsoluteTransform().copy(), p = t.point({ x: 0, y: 0 }), q = t.point({ x: img.w, y: img.h });
  download(s.toDataURL({ x: p.x, y: p.y, width: q.x - p.x, height: q.y - p.y, pixelRatio: img.w / (q.x - p.x) }), "pic2meas.png");
}

function summary(r: Results) {
  return {
    skirt: r.skirt && { lengthCm: +r.skirt.cm.toFixed(1), ratio: +r.skirt.ratio.toFixed(3), category: r.skirt.category, source: r.skirt.source },
    girthCm: Object.fromEntries(LEVELS.map((lv) => [lv, { value: +r.girth[lv].cm.toFixed(1), source: r.girth[lv].source }])),
    bra: r.bra && { label: r.bra.label, cup: r.bra.cup, band: r.bra.band, diffCm: +r.bra.diff.toFixed(1) },
    headRatio: +r.headRatio.toFixed(2),
    details: Object.fromEntries(r.details.map((g) => [g.title, Object.fromEntries(g.items.map((d) => [d.label, `${f1(d.value)}${d.unit && ` ${d.unit}`}`]))])),
    warnings: r.warnings,
  };
}

function exportJson(r: Results, name?: string) {
  const blob = new Blob([JSON.stringify({ image: name, ...summary(r) }, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  download(url, "pic2meas.json");
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function copyText(r: Results) {
  const lines = [
    r.skirt ? `スカート丈 ${f1(r.skirt.cm)} cm（${r.skirt.category}）` : null,
    ...LEVELS.map((lv) => `${LEVEL_LABEL[lv]} ${f1(r.girth[lv].cm)} cm`),
    r.bra ? `ブラサイズ ${r.bra.label}` : null,
    `頭身 ${f1(r.headRatio)}`,
    ...r.details.flatMap((g) => g.items.map((d) => `${d.label} ${f1(d.value)}${d.unit && ` ${d.unit}`}`)),
    "※ 画像1枚からの推定値",
  ].filter(Boolean);
  await navigator.clipboard.writeText(lines.join("\n"));
  toast.success("結果をコピーしました");
}
