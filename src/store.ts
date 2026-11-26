import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Analysis, Backend, Progress } from "@/infer/types";
import type { Level } from "@/lib/measure";
import { autoLandmarks, type Landmarks, type Mode, type Overrides } from "@/lib/results";

export type Display = "fill" | "outline" | "off";
type Layers = { mesh: Display; garment: Display; levels: boolean };

type State = {
  file: File | null;
  bitmap: ImageBitmap | null;
  analysis: Analysis | null;
  auto: Landmarks | null;
  landmarks: Landmarks | null;
  overrides: Overrides;
  heightCm: number;
  mode: Mode;
  calibrate2D: boolean;
  layers: Layers;
  busy: boolean;
  progress: Progress | null;
  backend: Backend | null;

  setImage: (file: File, bitmap: ImageBitmap) => void;
  setAnalysis: (a: Analysis) => void;
  setMode: (m: Mode) => void;
  setLandmarks: (f: (l: Landmarks) => Landmarks) => void;
  resetLandmarks: () => void;
  setOverride: (key: "skirtCm" | Level, v: number | undefined) => void;
  set: (p: Partial<Pick<State, "heightCm" | "calibrate2D" | "busy" | "progress" | "backend" | "layers">>) => void;
};

export const useStore = create<State>()(
  persist(
    (set) => ({
      file: null, bitmap: null, analysis: null, auto: null, landmarks: null, overrides: { girth: {} },
      heightCm: 160, mode: "photo", calibrate2D: false,
      layers: { mesh: "fill", garment: "fill", levels: true },
      busy: false, progress: null, backend: null,

      setImage: (file, bitmap) => set((s) => {
        s.bitmap?.close();
        return { file, bitmap, analysis: null, auto: null, landmarks: null, overrides: { girth: {} } };
      }),
      setAnalysis: (analysis) => set((s) => {
        // Guess: PNG = illustration (calibrated by silhouette width), otherwise photo.
        const mode: Mode = analysis.transparent || s.file?.type === "image/png" ? "illust" : "photo";
        const auto = autoLandmarks(analysis, mode);
        return { analysis, mode, calibrate2D: mode === "illust", auto, landmarks: structuredClone(auto) };
      }),
      setMode: (mode) => set((s) => {
        if (!s.analysis) return { mode, calibrate2D: mode === "illust" };
        const auto = autoLandmarks(s.analysis, mode);
        return { mode, calibrate2D: mode === "illust", auto, landmarks: structuredClone(auto) };
      }),
      setLandmarks: (f) => set((s) => (s.landmarks ? { landmarks: f(s.landmarks) } : {})),
      resetLandmarks: () => set((s) => ({ landmarks: s.auto && structuredClone(s.auto), overrides: { girth: {} } })),
      setOverride: (key, v) => set((s) => (key === "skirtCm"
        ? { overrides: { ...s.overrides, skirtCm: v } }
        : { overrides: { ...s.overrides, girth: { ...s.overrides.girth, [key]: v } } })),
      set: (p) => set(p),
    }),
    { name: "pic2meas-settings", partialize: (s) => ({ heightCm: s.heightCm, layers: s.layers }) },
  ),
);
