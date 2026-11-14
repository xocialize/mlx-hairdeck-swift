// Merge a hair item .glb into a body .vrm with Item Studio's mergeVrm and check the result: the item's springs
// are in the merged file, the hair is still where the item put it, and it swings under a head turn.
//
//   NODE_MODULES=/path/to/node_modules node tools/check_merged_vrm.mjs VRMMERGE.js BODY.vrm HAIR.glb OUT.vrm
// VRMMERGE.js: Item Studio's web/fitting/vrmMerge.js. The three.js part loads a copy of the merged file with its
// textures stripped (Node has no image decoder); OUT.vrm itself is the untouched merge.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const nm = process.env.NODE_MODULES;
const [mergePath, bodyPath, hairPath, outPath] = process.argv.slice(2);
if (!nm || !outPath) { console.error('usage: NODE_MODULES=... node check_merged_vrm.mjs VRMMERGE.js BODY.vrm HAIR.glb OUT.vrm'); process.exit(2); }
const imp = (p) => import(pathToFileURL(path.join(nm, p)).href);
const THREE = await imp('three/build/three.module.js');
const { GLTFLoader } = await imp('three/examples/jsm/loaders/GLTFLoader.js');
const { VRMSpringBoneLoaderPlugin } = await imp('@pixiv/three-vrm-springbone/lib/three-vrm-springbone.module.js');
const { mergeVrm, parseGlb, buildGlb } = await import(pathToFileURL(path.resolve(mergePath)).href);

const body = new Uint8Array(fs.readFileSync(bodyPath)), hair = new Uint8Array(fs.readFileSync(hairPath));
const before = parseGlb(body).json, item = parseGlb(hair).json;
const merged = mergeVrm(body, [{ name: 'hair', bytes: hair }]);
fs.writeFileSync(outPath, merged.bytes);
const { json: J, bin } = parseGlb(merged.bytes);
const sbB = before.extensions?.VRMC_springBone?.springs?.length || 0, sbI = item.extensions.VRMC_springBone.springs.length;
const sb = J.extensions.VRMC_springBone;
const names = (s) => s.joints.map((j) => J.nodes[j.node].name);
const hairSprings = sb.springs.filter((s) => names(s)[0].startsWith('HairGen_'));
// every hair spring: a chain of parent → child nodes ending in its _end node, the first under the head bone
const parent = new Map();
J.nodes.forEach((n, i) => (n.children || []).forEach((c) => parent.set(c, i)));
let chainsOK = true;
for (const s of hairSprings) {
  const js = s.joints.map((j) => j.node);
  for (let k = 1; k < js.length; k++) if (parent.get(js[k]) !== js[k - 1]) chainsOK = false;
  if (J.nodes[parent.get(js[0])].name !== item.nodes[0].name || !J.nodes[js[js.length - 1]].name.endsWith('_end')) chainsOK = false;
}
const headBones = J.nodes.filter((n) => n.name === item.nodes[0].name).length;

// three.js: textures stripped
const strip = (o) => { if (o && typeof o === 'object') for (const k of Object.keys(o)) { if (/Texture$/i.test(k)) delete o[k]; else strip(o[k]); } };
const T = JSON.parse(JSON.stringify(J));
strip(T.materials);
delete T.textures; delete T.images; delete T.samplers;
const loader = new GLTFLoader();
loader.register((p) => new VRMSpringBoneLoaderPlugin(p));
const glb = buildGlb(T, bin);
const gltf = await new Promise((res, rej) => loader.parse(glb.buffer.slice(glb.byteOffset, glb.byteOffset + glb.byteLength), '', res, rej));
const scene = gltf.scene;
let mesh = null, head = null;
const ends = [];
scene.traverse((o) => {
  if (o.isSkinnedMesh && o.name.startsWith('Hair_gen')) mesh = o;
  if (o.name === item.nodes[0].name) head = o;
  if (o.name.startsWith('HairGen_') && o.name.endsWith('_end')) ends.push(o);
});
scene.updateMatrixWorld(true);
// at rest the skinned hair must be where the item's vertices are
const pos = mesh.geometry.getAttribute('position'), v = new THREE.Vector3();
let restErr = 0;
for (let i = 0; i < pos.count; i += 37) restErr = Math.max(restErr, mesh.getVertexPosition(i, v).applyMatrix4(mesh.matrixWorld).distanceTo(new THREE.Vector3().fromBufferAttribute(pos, i)));
const mgr = gltf.userData.vrmSpringBoneManager;
mgr.setInitState();
mgr.reset();
const inv = new THREE.Matrix4();
const local = (o) => { inv.copy(head.matrixWorld).invert(); return new THREE.Vector3().setFromMatrixPosition(o.matrixWorld).applyMatrix4(inv); };
const rest = ends.map(local);
const q0 = head.quaternion.clone();
let peak = 0, nan = false, at2_5 = 0, over2_5 = 0;
// 0.35 s turn, then rest. A long soft chain (stiffness near 0.14 over 0.37 m) is still coming back at 2.5 s and is at
// rest by 5 s, so the settle condition is read at 5 s; the 2.5 s figure is reported, not judged.
for (let f = 1; f <= 300; f++) {
  const x = Math.min(Math.max(f / 60 / 0.35, 0), 1);
  head.quaternion.copy(q0).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), (50 * Math.PI / 180) * x * x * (3 - 2 * x)));
  scene.updateMatrixWorld(true);
  mgr.update(1 / 60);
  scene.updateMatrixWorld(true);
  ends.forEach((o, i) => { const d = local(o).distanceTo(rest[i]); if (!Number.isFinite(d)) nan = true; peak = Math.max(peak, d); });
  if (f === 150) { const ds = ends.map((o, i) => local(o).distanceTo(rest[i])); at2_5 = Math.max(...ds); over2_5 = ds.filter((d) => d > 0.01).length; }
}
const final = Math.max(...ends.map((o, i) => local(o).distanceTo(rest[i])));
const out = {
  report: merged.report, bytes: merged.bytes.length, springsBody: sbB, springsItem: sbI, springsMerged: sb.springs.length, hairSprings: hairSprings.length,
  hairChainsIntact: chainsOK, headBonesNamedLikeItem: headBones, hairVertices: pos.count, restPositionError_m: +restErr.toExponential(2),
  springJointsLoaded: [...mgr.joints].length, peakEndLag_m: +peak.toFixed(4), endOffsetAfter2_5s_m: +at2_5.toFixed(4), endsOver1cmAt2_5s: over2_5, endOffsetAfter5s_m: +final.toFixed(4), nan,
};
console.log(JSON.stringify(out, null, 1));
const ok = sb.springs.length === sbB + sbI && hairSprings.length === sbI && chainsOK && headBones === 1 && restErr < 1e-4 && !nan && peak > 0.005 && final < 0.01;
console.log(ok ? 'CHECK PASSED' : 'CHECK FAILED');
process.exit(ok ? 0 : 1);
