import type Konva from "konva";
import { useEffect, useMemo, useRef, useState } from "react";
import { Circle, Group, Image as KImage, Layer, Line, Stage, Text } from "react-konva";
import { editorStage } from "@/lib/editorStage";
import { LEVELS, LEVEL_LABEL, type Level } from "@/lib/measure";
import type { Landmarks, LevelLine, Side } from "@/lib/results";
import { useStore } from "@/store";


const COLORS = { head: "#64748b", chin: "#94a3b8", waist: "#16a34a", hem: "#dc2626", level: "#d97706", joint: "#eab308" };

export function Editor() {
  const { bitmap, analysis, landmarks, layers, setLandmarks } = useStore();
  const box = useRef<HTMLDivElement>(null), stageRef = useRef<Konva.Stage>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 });
  const w = analysis?.w ?? bitmap?.width ?? 1, h = analysis?.h ?? bitmap?.height ?? 1;

  // Fit the image to the panel.
  useEffect(() => {
    const el = box.current!;
    const fit = () => {
      const r = el.getBoundingClientRect(), s = Math.min(r.width / w, r.height / h) * 0.96;
      setSize({ w: r.width, h: r.height });
      setView({ scale: s, x: (r.width - w * s) / 2, y: (r.height - h * s) / 2 });
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(el);
    return () => ro.disconnect();
  }, [w, h]);

  useEffect(() => { editorStage.current = stageRef.current; editorStage.image = { w, h }; });

  // Garment and mesh overlays as one bitmap.
  const overlay = useMemo(() => {
    if (!analysis) return null;
    const { w: W, h: H } = analysis, c = new OffscreenCanvas(W, H), ctx = c.getContext("2d")!;
    const img = ctx.createImageData(W, H), px = img.data, edge = Math.max(1, Math.round(W / 500));
    const paint = (mask: Uint8Array, rgba: [number, number, number, number], outline: boolean) => {
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = y * W + x;
        if (!mask[i]) continue;
        // Outline: keep only pixels within `edge` of the mask boundary.
        if (outline) {
          let inner = true;
          for (let d = -edge; d <= edge && inner; d++)
            if (!mask[Math.min(H - 1, Math.max(0, y + d)) * W + x] || !mask[y * W + Math.min(W - 1, Math.max(0, x + d))]) inner = false;
          if (inner) continue;
        }
        px.set(rgba, i * 4);
      }
    };
    if (layers.garment !== "off") paint(analysis.garment, [236, 72, 153, layers.garment === "outline" ? 255 : 90], layers.garment === "outline");
    if (layers.mesh === "outline") paint(analysis.meshMask, [14, 165, 233, 255], true);
    ctx.putImageData(img, 0, 0);
    if (layers.mesh === "fill") {
      ctx.fillStyle = "rgba(14,165,233,0.55)";
      const d = Math.max(2, W / 400);
      for (let i = 0; i < analysis.meshPoints.length; i += 2) ctx.fillRect(analysis.meshPoints[i], analysis.meshPoints[i + 1], d, d);
    }
    return c;
  }, [analysis, layers.garment, layers.mesh]);

  const onWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const p = stageRef.current!.getPointerPosition()!, k = e.evt.deltaY < 0 ? 1.1 : 1 / 1.1;
    setView((v) => ({ scale: v.scale * k, x: p.x - (p.x - v.x) * k, y: p.y - (p.y - v.y) * k }));
  };

  const u = 1 / view.scale;   // one screen pixel in image units
  const set = (f: (l: Landmarks) => Landmarks) => setLandmarks(f);

  return (
    <div ref={box} className="absolute inset-0">
      {size.w > 0 && bitmap && (
        <Stage ref={stageRef} width={size.w} height={size.h} scaleX={view.scale} scaleY={view.scale} x={view.x} y={view.y}
          draggable onWheel={onWheel}
          onDragEnd={(e) => { if (e.target === stageRef.current) setView((v) => ({ ...v, x: e.target.x(), y: e.target.y() })); }}>
          <Layer>
            <KImage image={bitmap} width={w} height={h} />
            {overlay && <KImage image={overlay} width={w} height={h} listening={false} />}
          </Layer>
          {landmarks && (
            <Layer>
              <HLine y={landmarks.headY} w={w} u={u} color={COLORS.head} label="頭頂" onChange={(y) => set((l) => ({ ...l, headY: y }))} />
              <HLine y={landmarks.chinY} w={w} u={u} color={COLORS.chin} label="あご（頭身）" onChange={(y) => set((l) => ({ ...l, chinY: y }))} />
              <HLine y={landmarks.soleY} w={w} u={u} color={COLORS.head} label="足裏" onChange={(y) => set((l) => ({ ...l, soleY: y }))} />
              {<HLine y={landmarks.waistY} w={w} u={u} color={COLORS.waist} label="ウエスト（丈の起点）" onChange={(y) => set((l) => ({ ...l, waistY: y }))} />}
              {<HLine y={landmarks.hemY} w={w} u={u} color={COLORS.hem} label="裾" onChange={(y) => set((l) => ({ ...l, hemY: y }))} />}
              {layers.levels && LEVELS.map((lv) => (
                <WidthLine key={lv} line={landmarks.levels[lv]} u={u} label={LEVEL_LABEL[lv]}
                  onChange={(patch) => set((l) => ({ ...l, levels: { ...l.levels, [lv]: { ...l.levels[lv], ...patch } } as Record<Level, LevelLine> }))} />
              ))}
              {(["hip", "knee", "ankle"] as const).map((k) => (
                <Joints key={k} side={landmarks[k]} u={u} onChange={(lr, p) => set((l) => ({ ...l, [k]: { ...l[k], [lr]: p } }))} />
              ))}
            </Layer>
          )}
        </Stage>
      )}
    </div>
  );
}

const cursor = (e: Konva.KonvaEventObject<MouseEvent>, c: string) => { const s = e.target.getStage(); if (s) s.container().style.cursor = c; };

function HLine({ y, w, u, color, label, onChange }: { y: number; w: number; u: number; color: string; label: string; onChange: (y: number) => void }) {
  return (
    <Group y={y} draggable onDragMove={(e) => { e.target.x(0); onChange(e.target.y()); }}
      onMouseEnter={(e) => cursor(e, "ns-resize")} onMouseLeave={(e) => cursor(e, "")}>
      <Line points={[0, 0, w, 0]} stroke={color} strokeWidth={2 * u} hitStrokeWidth={14 * u} />
      <Text text={label} x={6 * u} y={-18 * u} fontSize={13 * u} fill={color} fontStyle="bold" />
    </Group>
  );
}

/** Girth level: drag the line for height, its ends for the silhouette width. */
function WidthLine({ line, u, label, onChange }: { line: LevelLine; u: number; label: string; onChange: (patch: Partial<LevelLine>) => void }) {
  const end = (key: "x0" | "x1") => (
    <Circle x={line[key]} y={line.y} radius={6 * u} fill="#fff" stroke={COLORS.level} strokeWidth={2 * u} draggable
      onMouseEnter={(e) => cursor(e, "ew-resize")} onMouseLeave={(e) => cursor(e, "")}
      onDragMove={(e) => { e.target.y(line.y); onChange({ [key]: e.target.x() }); }} />
  );
  return (
    <>
      <Group y={line.y} draggable onDragMove={(e) => { e.target.x(0); onChange({ y: e.target.y() }); }}
        onMouseEnter={(e) => cursor(e, "ns-resize")} onMouseLeave={(e) => cursor(e, "")}>
        <Line points={[line.x0, 0, line.x1, 0]} stroke={COLORS.level} strokeWidth={2 * u} dash={[8 * u, 4 * u]} hitStrokeWidth={14 * u} />
        <Text text={label} x={line.x1 + 10 * u} y={-7 * u} fontSize={12 * u} fill={COLORS.level} fontStyle="bold" />
      </Group>
      {end("x0")}
      {end("x1")}
    </>
  );
}

function Joints({ side, u, onChange }: { side: Side; u: number; onChange: (lr: "l" | "r", p: { x: number; y: number }) => void }) {
  return (["l", "r"] as const).map((k) => (
    <Circle key={k} x={side[k].x} y={side[k].y} radius={6 * u} fill={COLORS.joint} stroke="#000" strokeWidth={u} draggable
      onMouseEnter={(e) => cursor(e, "move")} onMouseLeave={(e) => cursor(e, "")}
      onDragMove={(e) => onChange(k, { x: e.target.x(), y: e.target.y() })} />
  ));
}
