import type Konva from "konva";

/** The editor's stage, for the PNG export (in its own module so Konva stays lazily loaded). */
export const editorStage: { current: Konva.Stage | null; image: { w: number; h: number } | null } = { current: null, image: null };
