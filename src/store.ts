import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Analysis, Backend, Progress, Refit } from "@/infer/types";
import type { Level } from "@/lib/measure";
import { autoLandmarks, forMode, type Landmarks, type Mode, type Overrides } from "@/lib/results";

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
  /** Illustrations: how far the body's proportions are from the drawing's after fitting (RMS of the ratios). */
  fitResidual: number | null;

  setImage: (file: File, bitmap: ImageBitmap) => void;
  setAnalysis: (a: Analysis) => void;
  setMode: (m: Mode) => void;
  setLandmarks: (f: (l: Landmarks) => Landmarks) => void;
  applyRefit: (forAnalysis: Analysis, r: Refit) => void;
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
      busy: false, progress: null, backend: null, fitResidual: null,

      setImage: (file, bitmap) => set((s) => {
        s.bitmap?.close();
        return { file, bitmap, analysis: null, auto: null, landmarks: null, overrides: { girth: {} }, fitResidual: null };
      }),
      setAnalysis: (analysis) => set((s) => {
        // Guess: PNG = illustration, otherwise photo.
        const mode: Mode = analysis.transparent || s.file?.type === "image/png" ? "illust" : "photo";
        const auto = autoLandmarks(forMode(analysis, mode));
        return { analysis, mode, auto, landmarks: structuredClone(auto) };
      }),
      // Each mode places its own landmarks (illustrations use drawing-trained keypoints), so switching resets them.
      setMode: (mode) => set((s) => {
        if (!s.analysis) return { mode, fitResidual: null };
        const auto = autoLandmarks(forMode(s.analysis, mode));
        return { mode, auto, landmarks: structuredClone(auto), fitResidual: null };
      }),
      setLandmarks: (f) => set((s) => (s.landmarks ? { landmarks: f(s.landmarks) } : {})),
      // A refit replaces the unposed body; girth levels follow it unless the user moved them.
      applyRefit: (forAnalysis, r) => set((s) => {
        if (s.analysis !== forAnalysis || !s.auto || !s.landmarks) return {};
        const analysis = { ...s.analysis, rest: r.rest, sections: r.sections, autoLevels: r.autoLevels };
        const levels = autoLandmarks(forMode(analysis, s.mode)).levels;
        const moved = JSON.stringify(s.landmarks.levels) !== JSON.stringify(s.auto.levels);
        return {
          analysis, fitResidual: r.residual, auto: { ...s.auto, levels },
          landmarks: moved ? s.landmarks : { ...s.landmarks, levels: structuredClone(levels) },
        };
      }),
      resetLandmarks: () => set((s) => ({ landmarks: s.auto && structuredClone(s.auto), overrides: { girth: {} } })),
      setOverride: (key, v) => set((s) => (key === "skirtCm"
        ? { overrides: { ...s.overrides, skirtCm: v } }
        : { overrides: { ...s.overrides, girth: { ...s.overrides.girth, [key]: v } } })),
      set: (p) => set(p),
    }),
    { name: "pic2meas-settings", partialize: (s) => ({ heightCm: s.heightCm, layers: s.layers }) },
  ),
);
