import { Cpu, Zap } from "lucide-react";
import { lazy, Suspense, useEffect } from "react";
import { toast } from "sonner";
import { ImageDrop } from "@/components/ImageDrop";
import { InputPanel } from "@/components/InputPanel";
import { ResultPanel } from "@/components/ResultPanel";
import { TermsDialog } from "@/components/TermsDialog";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Toaster } from "@/components/ui/sonner";
import { analyseImage, initInference } from "@/infer/client";
import type { Progress as P } from "@/infer/types";
import { useStore } from "@/store";

// Konva loads only once an image is in.
const Editor = lazy(() => import("@/components/Editor").then((m) => ({ default: m.Editor })));

const pct = (p: P | null) => (p?.total ? Math.round(((p.loaded ?? 0) / p.total) * 100) : null);

export default function App() {
  const { bitmap, analysis, busy, progress, backend, setImage, setAnalysis, set } = useStore();
  const onProgress = (p: P) => set({ progress: p });

  // Start loading the models right away.
  useEffect(() => {
    set({ busy: true });
    initInference(onProgress)
      .then((backend) => set({ backend }))
      .catch((e: Error) => toast.error(`モデルを読み込めませんでした: ${e.message}`))
      .finally(() => set({ busy: false, progress: null }));
  }, []);

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
            {bitmap ? <Suspense><Editor /></Suspense> : <ImageDrop large onFile={onFile} />}
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
