import { Line, OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useMemo } from "react";
import * as THREE from "three";
import { LEVEL_LABEL } from "@/lib/measure";
import { computeResults, forMode, view3d } from "@/lib/results";
import { useStore } from "@/store";

const RING_COLOR = { bust: "#db2777", underbust: "#d97706", waist: "#16a34a", hip: "#7c3aed" } as const;

/** The unposed body as measured: girth sections and the breast caps, from the same geometry as the numbers. */
export function Body3D() {
  const { analysis, landmarks, auto, heightCm, calibrate2D, overrides, mode } = useStore();
  const view = useMemo(() => {
    if (!analysis || !landmarks || !auto) return null;
    const am = forMode(analysis, mode), r = computeResults(am, landmarks, auto, { heightCm, mode, calibrate2D }, overrides);
    return view3d(am, landmarks, r.capScale);
  }, [analysis, landmarks, auto, heightCm, calibrate2D, overrides, mode]);
  const geometry = useMemo(() => {
    if (!view) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(view.verts, 3));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(view.faces), 1));
    g.computeVertexNormals();
    return g;
  }, [view]);
  if (!view || !geometry) return null;
  const top = Math.max(...view.rings.map((r) => r.points[0][1])) + 0.3;

  return (
    <div className="absolute inset-0">
      <Canvas camera={{ position: [0, top * 0.6, 1.6], fov: 35 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[1, 2, 2]} intensity={1.4} />
        <mesh geometry={geometry}>
          <meshStandardMaterial color="#cbd5e1" transparent opacity={0.55} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
        {view.spheres.map((s, i) => {
          // Only the part outside the chest wall: the spherical cap around the wall's normal.
          const n = new THREE.Vector3(...s.normal).normalize(), q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), n);
          return (
            <mesh key={i} position={s.center} quaternion={q}>
              <sphereGeometry args={[s.radius, 48, 24, 0, Math.PI * 2, 0, Math.acos(Math.max(-1, Math.min(1, (s.radius - s.h) / s.radius)))]} />
              <meshStandardMaterial color={RING_COLOR.bust} transparent opacity={0.6} side={THREE.DoubleSide} />
            </mesh>
          );
        })}
        {view.rings.map((r) => (
          <Line key={r.level} points={r.points} color={RING_COLOR[r.level]} lineWidth={3} />
        ))}
        <OrbitControls target={[0, top * 0.5, 0]} />
      </Canvas>
      <div className="pointer-events-none absolute bottom-3 left-3 space-y-0.5 rounded-md bg-background/80 px-2 py-1 text-xs">
        {view.rings.map((r) => <div key={r.level} style={{ color: RING_COLOR[r.level] }}>━ {LEVEL_LABEL[r.level]}</div>)}
        <div className="text-muted-foreground">ドラッグで回転・ホイールで拡大</div>
      </div>
    </div>
  );
}
