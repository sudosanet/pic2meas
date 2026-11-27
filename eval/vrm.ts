// Dev-only page for the evaluation: renders a VRM in several poses and measures its ground truth with the same
// section code the app uses. Driven by eval/run.ts through window.renderCase().
import { VRMLoaderPlugin, VRMUtils, type VRM, type VRMHumanBoneName } from "@pixiv/three-vrm";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { armCutouts, autoLevels, bodyFrame, J, perimeter, sectionAt, type Section } from "@/lib/measure";

const canvas = document.getElementById("c") as HTMLCanvasElement;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setClearColor(0xffffff, 1);
const scene = new THREE.Scene();
const sun = new THREE.DirectionalLight(0xffffff, 1.5);
sun.position.set(0.5, 1, 2);
scene.add(new THREE.AmbientLight(0xffffff, 1.2), sun);

type Mesh = { verts: Float32Array; faces: Int32Array };

/** Skinned world-space geometry of the primitives whose material name matches. */
function collect(vrm: VRM, test: RegExp): Mesh {
  const v: number[] = [], f: number[] = [], p = new THREE.Vector3();
  vrm.scene.updateMatrixWorld(true);
  vrm.scene.traverse((o) => {
    const m = o as THREE.Mesh;
    if (!m.isMesh || !test.test([m.material].flat()[0].name)) return;   // [material, outline]
    const g = m.geometry, base = v.length / 3, n = g.attributes.position.count;
    for (let i = 0; i < n; i++) { m.getVertexPosition(i, p); p.applyMatrix4(m.matrixWorld); v.push(p.x, p.y, p.z); }
    const idx = g.index ? g.index.array : Array.from({ length: n }, (_, i) => i);
    for (let i = 0; i < idx.length; i++) f.push(base + idx[i]);
  });
  return { verts: new Float32Array(v), faces: new Int32Array(f) };
}

// MHR70 joint slots filled from VRM humanoid bones (enough for bodyFrame, armCutouts and the leg joints).
const BONES: [number, VRMHumanBoneName][] = [
  [J.lShoulder, "leftUpperArm"], [J.rShoulder, "rightUpperArm"], [J.lElbow, "leftLowerArm"], [J.rElbow, "rightLowerArm"],
  [J.lWrist, "leftHand"], [J.rWrist, "rightHand"], [J.lHip, "leftUpperLeg"], [J.rHip, "rightUpperLeg"],
  [J.lKnee, "leftLowerLeg"], [J.rKnee, "rightLowerLeg"], [J.lAnkle, "leftFoot"], [J.rAnkle, "rightFoot"], [J.neck, "neck"],
];
function joints(vrm: VRM) {
  const out = new Float32Array(70 * 3), p = new THREE.Vector3();
  for (const [i, b] of BONES) { vrm.humanoid.getRawBoneNode(b)!.getWorldPosition(p); out.set([p.x, p.y, p.z], i * 3); }
  vrm.humanoid.getRawBoneNode("head")!.getWorldPosition(p);
  out.set([p.x, p.y + 0.05, p.z + 0.1], J.nose * 3);   // in front of the head: VRM faces +Z after rotateVRM0
  for (const tip of [42, 46, 50, 54, 58]) out.set(out.slice(J.lWrist * 3, J.lWrist * 3 + 3), tip * 3);
  for (const tip of [21, 25, 29, 33, 37]) out.set(out.slice(J.rWrist * 3, J.rWrist * 3 + 3), tip * 3);
  return out;
}

/** Ground truth in the rest (T) pose, in cm. */
function truth(vrm: VRM) {
  vrm.humanoid.resetNormalizedPose(); vrm.update(0);
  // Tops count as body: VRoid deletes the skin hidden under clothes unless that export option is turned off.
  const skin = collect(vrm, /SKIN/), torso = collect(vrm, /SKIN|Tops/), bottoms = collect(vrm, /Bottoms|Onepiece|Skirt/i), j = joints(vrm);
  const frame = bodyFrame(j), { origin, up, side } = frame;
  const ys = (m: Mesh, keep: (x: number, z: number) => boolean) => {
    const out: number[] = [];
    for (let i = 0; i < m.verts.length; i += 3) if (keep(m.verts[i], m.verts[i + 2])) out.push(m.verts[i + 1]);
    return out;
  };
  const all = ys(skin, () => true), floor = Math.min(...all), top = Math.max(...all);
  const sections: Section[] = [];
  for (let t = -0.15; t <= frame.torso + 0.05; t += 0.01) sections.push({ t, ...sectionAt(torso.verts, torso.faces, origin, up, side, t, armCutouts(j)) });
  const lv = autoLevels(sections, frame.torso, frame.front);
  const g = (i: number) => perimeter(sections[i].hull) * 100;
  const crotch = Math.min(...ys(skin, (x) => Math.abs(x - origin[0]) < 0.015).filter((y) => y < origin[1]));
  const front = ys(bottoms, (x, z) => Math.abs(x - origin[0]) < 0.05 && z > origin[2]);
  return {
    heightCm: (top - floor) * 100, crotchCm: (crotch - floor) * 100,
    girthCm: { bust: g(lv.bust), underbust: g(lv.underbust), waist: g(lv.waist), hip: g(lv.hip) },
    skirtCm: front.length ? (Math.max(...front) - Math.min(...front)) * 100 : null,
  };
}

// Poses for the images: [bone, x, y, z] rotations (radians) on normalized bones, and a turn of the whole body.
const POSES: Record<string, { bones: [VRMHumanBoneName, number, number, number][]; turn: number }> = {
  stand: { bones: [["leftUpperArm", 0, 0, -1.2], ["rightUpperArm", 0, 0, 1.2]], turn: 0 },
  contra: { bones: [["leftUpperArm", 0, 0, -1.1], ["rightUpperArm", 0, 0, 0.9], ["rightLowerArm", 0, 0, 0.8], ["hips", 0, 0, 0.08], ["leftUpperLeg", 0, 0, -0.08], ["rightUpperLeg", -0.2, 0, 0.1], ["rightLowerLeg", 0.4, 0, 0]], turn: 0 },
  turn: { bones: [["leftUpperArm", 0, 0, -1.2], ["rightUpperArm", 0, 0, 1.2]], turn: 0.45 },
  kick: { bones: [["leftUpperArm", 0, 0, -0.8], ["rightUpperArm", 0, 0, 0.8], ["leftUpperLeg", -0.9, 0, 0], ["leftLowerLeg", 1.0, 0, 0], ["spine", 0.1, 0, 0]], turn: 0.2 },
};

function render(vrm: VRM, pose: keyof typeof POSES) {
  vrm.humanoid.resetNormalizedPose();
  for (const [b, x, y, z] of POSES[pose].bones) vrm.humanoid.getNormalizedBoneNode(b)?.rotation.set(x, y, z);
  vrm.scene.rotation.y = POSES[pose].turn;
  vrm.update(0);
  const box = new THREE.Box3().setFromObject(vrm.scene), c = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3());
  const cam = new THREE.PerspectiveCamera(20, canvas.width / canvas.height, 0.1, 50);
  const dist = (Math.max(s.y, s.x * (canvas.height / canvas.width)) * 1.15) / 2 / Math.tan((10 * Math.PI) / 180);
  cam.position.set(c.x, c.y, c.z + dist); cam.lookAt(c);
  renderer.render(scene, cam);
  vrm.scene.rotation.y = 0;
  return canvas.toDataURL("image/png");
}

Object.assign(window, {
  async renderCase(url: string) {
    const gltf = await new GLTFLoader().register((p) => new VRMLoaderPlugin(p)).loadAsync(url);
    const vrm = gltf.userData.vrm as VRM;
    VRMUtils.rotateVRM0(vrm);
    scene.add(vrm.scene);
    const t = truth(vrm), images = Object.fromEntries(Object.keys(POSES).map((p) => [p, render(vrm, p)]));
    scene.remove(vrm.scene);
    return { truth: t, images };
  },
});
