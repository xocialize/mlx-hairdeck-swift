// Load a hair item .glb with three.js and @pixiv/three-vrm's spring-bone plugin, and check that it is a valid
// skinned mesh whose chains swing under a head turn.
//
//   NODE_MODULES=/path/to/node_modules node tools/check_hair_item.mjs HAIR.glb
// (any project with three and @pixiv/three-vrm installed; written against three 0.162, three-vrm 3.5.5)
// Head turn: yaw 50 degrees in 0.35 s, then held, 2.5 s at 60 fps (HairDeck's 'turn' motion).
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const nm = process.env.NODE_MODULES;
const file = process.argv[2];
if (!nm || !file) { console.error('usage: NODE_MODULES=... node check_hair_item.mjs HAIR.glb'); process.exit(2); }
const imp = (p) => import(pathToFileURL(path.join(nm, p)).href);
const THREE = await imp('three/build/three.module.js');
const { GLTFLoader } = await imp('three/examples/jsm/loaders/GLTFLoader.js');
const { VRMSpringBoneLoaderPlugin } = await imp('@pixiv/three-vrm-springbone/lib/three-vrm-springbone.module.js');

const buf = fs.readFileSync(file);
const loader = new GLTFLoader();
loader.register((parser) => new VRMSpringBoneLoaderPlugin(parser));
const gltf = await new Promise((res, rej) => loader.parse(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), '', res, rej));
const scene = gltf.scene;
let mesh = null;
scene.traverse((o) => { if (o.isSkinnedMesh) mesh = o; });
if (!mesh) throw new Error('no skinned mesh');
const sw = mesh.geometry.getAttribute('skinWeight'), si = mesh.geometry.getAttribute('skinIndex');
let worst = 0, badIndex = 0;
for (let i = 0; i < sw.count; i++) {
  worst = Math.max(worst, Math.abs(sw.getX(i) + sw.getY(i) + sw.getZ(i) + sw.getW(i) - 1));
  for (const j of [si.getX(i), si.getY(i), si.getZ(i), si.getW(i)]) if (j >= mesh.skeleton.bones.length) badIndex++;
}
const mgr = gltf.userData.vrmSpringBoneManager;
if (!mgr) throw new Error('the spring-bone plugin found no VRMC_springBone');
const joints = [...mgr.joints];
const head = scene.children.find((o) => o.isBone || o.children.some((c) => c.isBone)) ?? mesh.skeleton.bones[0];
const ends = [];
scene.traverse((o) => { if (o.name.endsWith('_end')) ends.push(o); });

scene.updateMatrixWorld(true);
mgr.setInitState();
mgr.reset();
// rest position of every chain end in the head's own space
const inv = new THREE.Matrix4();
const local = (o) => { inv.copy(head.matrixWorld).invert(); return new THREE.Vector3().setFromMatrixPosition(o.matrixWorld).applyMatrix4(inv); };
const rest = ends.map(local);
// the skinned position of the vertex most displaced, to see that the mesh follows its bones
const v = new THREE.Vector3();
const restVerts = [];
const sample = [];
for (let i = 0; i < sw.count; i += Math.max(1, Math.floor(sw.count / 400))) sample.push(i);
for (const i of sample) restVerts.push(mesh.getVertexPosition(i, v).clone().applyMatrix4(mesh.matrixWorld));

let peak = 0, peakMesh = 0, nan = false;
const FPS = 60, NF = 150;
for (let f = 1; f <= NF; f++) {
  const x = Math.min(Math.max(f / FPS / 0.35, 0), 1);
  head.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), (50 * Math.PI / 180) * x * x * (3 - 2 * x));
  scene.updateMatrixWorld(true);
  mgr.update(1 / FPS);
  scene.updateMatrixWorld(true);
  ends.forEach((o, i) => { const d = local(o).distanceTo(rest[i]); if (!Number.isFinite(d)) nan = true; peak = Math.max(peak, d); });
  if (f % 10 === 0) {
    sample.forEach((i, s) => {
      // displacement of the skinned vertex from where a rigid head turn would put it
      const rigid = restVerts[s].clone().sub(head.position).applyQuaternion(head.quaternion).add(head.position);
      peakMesh = Math.max(peakMesh, mesh.getVertexPosition(i, v).clone().applyMatrix4(mesh.matrixWorld).distanceTo(rigid));
    });
  }
}
const final = Math.max(...ends.map((o, i) => local(o).distanceTo(rest[i])));
const out = {
  file: path.basename(file), vertices: sw.count, bones: mesh.skeleton.bones.length, springJoints: joints.length, chainEnds: ends.length,
  weightSumMaxError: +worst.toExponential(2), skinIndexOutOfRange: badIndex,
  headBone: head.name, peakEndLag_m: +peak.toFixed(4), endOffsetAfter2_5s_m: +final.toFixed(4), peakVertexLag_m: +peakMesh.toFixed(4), nan,
};
console.log(JSON.stringify(out, null, 1));
const ok = !nan && badIndex === 0 && worst < 1e-5 && joints.length > 0 && peak > 0.005 && peakMesh > 0.005 && final < peak;
console.log(ok ? 'CHECK PASSED' : 'CHECK FAILED');
process.exit(ok ? 0 : 1);
