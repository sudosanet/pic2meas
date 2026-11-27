import type Konva from "konva";
import { useEffect, useMemo, useRef, useState } from "react";
import { Circle, Group, Image as KImage, Layer, Line, Stage, Text } from "react-konva";
import { editorStage } from "@/lib/editorStage";
import { LEVELS, LEVEL_LABEL, mid2, type Level, type Pt } from "@/lib/measure";
import { forMode, levelAt, levelNear, spine, type Landmarks, type LevelLine } from "@/lib/results";
import { useStore } from "@/store";

const COLORS = { breast: "#db2777", head: "#64748b", crotch: "#7c3aed", waist: "#16a34a", hem: "#dc2626", level: "#d97706", joint: "#eab308", bone: "rgba(234,179,8,0.6)" };

export function Editor() {
  const { bitmap, analysis, landmarks, layers, mode, setLandmarks } = useStore();
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
          {analysis && landmarks && (() => {
            const am = forMode(analysis, mode), lm = landmarks, sp = spine(am, lm);
            const pt = (key: "headTop" | "chin" | "waist" | "crotch", label: string, color: string) => (
              <Handle p={lm[key]} u={u} color={color} label={label} onChange={(p) => set((l) => ({ ...l, [key]: p }))} />
            );
            const pair = (key: "hip" | "knee" | "ankle" | "hem" | "sole", label: string, color: string) => (["l", "r"] as const).map((s) => (
              <Handle key={key + s} p={lm[key][s]} u={u} color={color} label={label} onChange={(p) => set((l) => ({ ...l, [key]: { ...l[key], [s]: p } }))} />
            ));
            const flat = (ps: Pt[]) => ps.flatMap((p) => [p.x, p.y]);
            return (
              <Layer>
                {/* The skeleton everything is measured along. */}
                {[[sp.neck, sp.hip], ...(["l", "r"] as const).map((s) => [mid2(lm.hip.l, lm.hip.r), lm.hip[s], lm.knee[s], lm.ankle[s], lm.sole[s]])].map((ps, i) => (
                  <Line key={i} points={flat(ps)} stroke={COLORS.bone} strokeWidth={2 * u} dash={[6 * u, 4 * u]} listening={false} />
                ))}
                {layers.levels && LEVELS.map((lv) => (
                  <Level key={lv} line={lm.levels[lv]} u={u} label={LEVEL_LABEL[lv]}
                    at={(t) => levelAt(am, lm, t)} near={(p) => levelNear(am, lm, p)}
                    onChange={(patch) => set((l) => ({ ...l, levels: { ...l.levels, [lv]: { ...l.levels[lv], ...patch } } as Record<Level, LevelLine> }))} />
                ))}
                {layers.levels && lm.breast && (["l", "r"] as const).map((k) => (
                  <Cap key={k} c={lm.breast![k]} a={lm.breast!.a} u={u}
                    onMove={(c) => set((l) => ({ ...l, breast: { ...l.breast!, [k]: c } }))}
                    onRadius={(r) => set((l) => ({ ...l, breast: { ...l.breast!, a: r, aM: (l.breast!.aM / l.breast!.a) * r } }))} />
                ))}
                {pt("headTop", "頭頂", COLORS.head)}
                {pt("chin", "あご", COLORS.head)}
                {mode === "illust" && pair("sole", "足裏", COLORS.head)}
                {pair("hip", "", COLORS.joint)}
                {pair("knee", "", COLORS.joint)}
                {pair("ankle", "", COLORS.joint)}
                {pt("crotch", "股下", COLORS.crotch)}
                {pt("waist", "ウエスト", COLORS.waist)}
                {pair("hem", "裾", COLORS.hem)}
              </Layer>
            );
          })()}
        </Stage>
      )}
    </div>
  );
}

const cursor = (e: Konva.KonvaEventObject<MouseEvent>, c: string) => { const s = e.target.getStage(); if (s) s.container().style.cursor = c; };

function Handle({ p, u, color, label, onChange }: { p: Pt; u: number; color: string; label: string; onChange: (p: Pt) => void }) {
  return (
    <Group x={p.x} y={p.y} draggable onDragMove={(e) => onChange({ x: e.target.x(), y: e.target.y() })}
      onMouseEnter={(e) => cursor(e, "move")} onMouseLeave={(e) => cursor(e, "")}>
      <Circle radius={6 * u} fill={color} stroke="#fff" strokeWidth={1.5 * u} />
      {label && <Text text={label} x={9 * u} y={-7 * u} fontSize={12 * u} fill={color} fontStyle="bold" />}
    </Group>
  );
}

/** Girth level: the centre slides along the body (snapping to the sections, re-measuring the width), the ends adjust it. */
function Level({ line, u, label, at: placeAt, near, onChange }: {
  line: LevelLine; u: number; label: string;
  at: (t: number) => ReturnType<typeof levelAt>; near: (p: Pt) => ReturnType<typeof levelAt>; onChange: (patch: Partial<LevelLine>) => void;
}) {
  const place = placeAt(line.t);
  if (!place) return null;
  const { c, dir: d } = place, at = (s: number) => ({ x: c.x + d.x * s, y: c.y + d.y * s });
  const A = at(line.a), B = at(line.b);
  const across = (q: Pt) => (q.x - c.x) * d.x + (q.y - c.y) * d.y;
  const end = (key: "a" | "b", p: Pt) => (
    <Circle x={p.x} y={p.y} radius={6 * u} fill="#fff" stroke={COLORS.level} strokeWidth={2 * u} draggable
      onMouseEnter={(e) => cursor(e, "ew-resize")} onMouseLeave={(e) => cursor(e, "")}
      onDragMove={(e) => { const s = across(e.target.position()); e.target.position(at(s)); onChange({ [key]: s }); }} />
  );
  return (
    <>
      <Line points={[A.x, A.y, B.x, B.y]} stroke={COLORS.level} strokeWidth={2 * u} dash={[8 * u, 4 * u]} listening={false} />
      <Text text={label} x={B.x + 10 * u} y={B.y - 7 * u} fontSize={12 * u} fill={COLORS.level} fontStyle="bold" listening={false} />
      <Circle x={c.x} y={c.y} radius={5 * u} fill={COLORS.level} draggable
        onMouseEnter={(e) => cursor(e, "ns-resize")} onMouseLeave={(e) => cursor(e, "")}
        onDragMove={(e) => { const l = near(e.target.position()); if (!l) return; e.target.position(l.c); onChange({ t: l.t, a: l.a, b: l.b }); }} />
      {end("a", A)}
      {end("b", B)}
    </>
  );
}

/** One breast cap's base: drag the centre to move it, the ring's right-hand handle to resize both. */
function Cap({ c, a, u, onMove, onRadius }: { c: Pt; a: number; u: number; onMove: (c: Pt) => void; onRadius: (r: number) => void }) {
  return (
    <>
      <Circle x={c.x} y={c.y} radius={a} stroke={COLORS.breast} strokeWidth={1.5 * u} dash={[5 * u, 4 * u]} listening={false} />
      <Circle x={c.x} y={c.y} radius={5 * u} fill={COLORS.breast} draggable
        onMouseEnter={(e) => cursor(e, "move")} onMouseLeave={(e) => cursor(e, "")}
        onDragMove={(e) => onMove({ x: e.target.x(), y: e.target.y() })} />
      <Circle x={c.x + a} y={c.y} radius={5 * u} fill="#fff" stroke={COLORS.breast} strokeWidth={2 * u} draggable
        onMouseEnter={(e) => cursor(e, "ew-resize")} onMouseLeave={(e) => cursor(e, "")}
        onDragMove={(e) => { const r = Math.max(4, e.target.x() - c.x); e.target.position({ x: c.x + r, y: c.y }); onRadius(r); }} />
    </>
  );
}
