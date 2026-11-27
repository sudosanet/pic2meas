import { Cpu, Zap } from "lucide-react";
import { lazy, Suspense, useEffect, useState } from "react";
import { toast } from "sonner";
import { ImageDrop } from "@/components/ImageDrop";
import { InputPanel } from "@/components/InputPanel";
import { ResultPanel } from "@/components/ResultPanel";
import { TermsDialog } from "@/components/TermsDialog";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Toaster } from "@/components/ui/sonner";
import { analyseImage, initInference, refitBody } from "@/infer/client";
import { ratios2d } from "@/lib/fit";
import { FLAGS } from "@/lib/flags";
import { J } from "@/lib/measure";
import { forMode, spine } from "@/lib/results";
import type { Progress as P } from "@/infer/types";
import { useStore } from "@/store";

// Konva loads only once an image is in.
const Editor = lazy(() => import("@/components/Editor").then((m) => ({ default: m.Editor })));
const Body3D = lazy(() => import("@/components/Body3D").then((m) => ({ default: m.Body3D })));

const pct = (p: P | null) => (p?.total ? Math.round(((p.loaded ?? 0) / p.total) * 100) : null);

export default function App() {
  const { bitmap, analysis, landmarks, mode, busy, progress, backend, setImage, setAnalysis, applyRefit, set } = useStore();
  const onProgress = (p: P) => set({ progress: p });

  // Start loading the models right away.
  useEffect(() => {
    set({ busy: true });
    initInference(onProgress)
      .then((backend) => set({ backend }))
      .catch((e: Error) => toast.error(`モデルを読み込めませんでした: ${e.message}`))
      .finally(() => set({ busy: false, progress: null }));
  }, []);

  // Illustrations: fit the 3D body's proportions to the drawing's skeleton, after analysis and after the user
  // moves points (debounced). Girth levels are excluded from the key so a refit doesn't trigger itself.
  const skeletonKey = landmarks && JSON.stringify([landmarks.headTop, landmarks.hip, landmarks.knee, landmarks.ankle, landmarks.sole]);
  useEffect(() => {
    if (mode !== "illust" || !FLAGS.fit || !analysis || !landmarks) return;
    const a = forMode(analysis, mode), j = (i: number) => ({ x: a.joints2d[i * 2], y: a.joints2d[i * 2 + 1] });
    const target = ratios2d({
      ...landmarks, neck: spine(a, landmarks).neck,
      shoulder: { l: j(J.lShoulder), r: j(J.rShoulder) }, elbow: { l: j(J.lElbow), r: j(J.rElbow) },
    });
    const timer = setTimeout(() => refitBody(target).then((r) => applyRefit(analysis, r)).catch((e: Error) => console.warn("refit", e)), 500);
    return () => clearTimeout(timer);
  }, [mode, analysis?.joints2d, skeletonKey]);

  useEffect(() => {
    const dark = matchMedia("(prefers-color-scheme: dark)");
    const apply = () => document.documentElement.classList.toggle("dark", dark.matches);
    apply();
    dark.addEventListener("change", apply);
    return () => dark.removeEventListener("change", apply);
  }, []);

  const onFile = async (file: File) => {
    setImage(file, await createImageBitmap(file));
    set({ busy: true });
    try {
      const a = await analyseImage(file, onProgress);
      if (useStore.getState().file === file) setAnalysis(a);   // ignore if another image was chosen meanwhile
    }
    catch (e) { toast.error(`解析できませんでした: ${(e as Error).message}`); }
    finally { set({ busy: false, progress: null }); }
  };

  const [view, setView] = useState<"image" | "3d">("image");
  return (
    <>
      <div className="flex h-dvh flex-col bg-background text-foreground">
        <header className="flex items-center gap-3 border-b px-4 py-2">
          <h1 className="text-lg font-bold">pic2meas</h1>
          <span className="hidden text-sm text-muted-foreground sm:inline">画像1枚からスカート丈・周囲長・カップを推定</span>
          <div className="ml-auto flex items-center gap-2">
            {backend && <Badge variant="secondary">{backend === "webgpu" ? <><Zap />WebGPU</> : <><Cpu />WASM</>}</Badge>}
            <a href="/licenses" className="text-xs text-muted-foreground underline-offset-2 hover:underline">ライセンス</a>
          </div>
        </header>
        <div className="flex min-h-0 flex-1 flex-col md:flex-row">
          <section className="relative min-h-[60vh] flex-1 bg-muted/40 md:min-h-0">
            {bitmap ? <Suspense>{view === "3d" && analysis ? <Body3D /> : <Editor />}</Suspense> : <ImageDrop large onFile={onFile} />}
            {analysis && (
              <ToggleGroup type="single" variant="outline" size="sm" value={view} onValueChange={(v) => v && setView(v as "image" | "3d")} className="absolute top-2 right-2 bg-background">
                <ToggleGroupItem value="image">画像</ToggleGroupItem>
                <ToggleGroupItem value="3d">3D</ToggleGroupItem>
              </ToggleGroup>
            )}
            {busy && progress && (
              <div className="absolute inset-x-0 top-0 space-y-1 bg-background/90 px-4 py-2 text-sm backdrop-blur">
                <div className="flex justify-between"><span>{label(progress)}</span><span className="tabular-nums">{pct(progress) != null && `${pct(progress)}%`}</span></div>
                <Progress value={pct(progress) ?? undefined} />
              </div>
            )}
          </section>
          <aside className="w-full space-y-4 overflow-y-auto border-l p-4 md:w-[400px]">
            {bitmap && <ImageDrop onFile={onFile} />}
            <InputPanel />
            {analysis && <ResultPanel />}
          </aside>
        </div>
      </div>
      <TermsDialog />
      <Toaster />
    </>
  );
}

function label(p: P) {
  if (p.stage === "download") return `モデルをダウンロード中（初回のみ）: ${p.label}`;
  if (p.stage === "session") return `モデルを準備中: ${p.label}`;
  return p.label;
}
