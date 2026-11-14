import Foundation

/// A hair item as a binary glTF (`.glb`): the skinned hair mesh of a `HairRig.Rig` on its own small
/// skeleton, with a `VRMC_springBone` block. It replaces the Blender step of HairDeck's VRM export
/// (`blender/vrm_hair.py: add_generated_hair`) and is laid out for a merger that matches bones by
/// name: the root node carries the name of the body's head bone at the head joint, every other bone
/// is new.
///
///   nodes      head bone; per chain c the bones `HairGen_cCC_KK` at joint k (the first under the head
///              bone, each next under the one before) and a leaf `HairGen_cCC_end` at the last joint;
///              the mesh node
///   skin       head bone + every chain bone; a vertex weighs head 1 − w, its segment w · α and the
///              neighbour segment w · (1 − α) (HairDeck's binding); bones have no rotation, so the
///              inverse bind matrices are translations
///   mesh       POSITION, NORMAL (area-weighted), TEXCOORD_0 (v flipped to glTF's top-left origin),
///              JOINTS_0, WEIGHTS_0, 32-bit indices; one double-sided material in the hair colour
///              with a `VRMC_materials_mtoon` block (shade colour 0.6 × the colour)
///   springs    one per chain, joints from the chain's first bone to its `_end` node; stiffness as
///              predicted, drag 0.4, gravity 0, hit radius 0.02 (HairDeck's values); no colliders —
///              those belong to the body
///
/// Axes are those of the rig (VRM 1.0: Y up, face toward +Z, metres). `vrm0Axes` writes the same
/// geometry turned half a turn about Y (−x, y, −z), the axes of a VRM 0.x file.
public enum HairItem {
    public static let drag = 0.4, gravityPower = 0.0, hitRadius = 0.02

    public struct Summary {
        public var vertices = 0, triangles = 0, bones = 0, springs = 0, threeInfluences = 0, headOnly = 0
    }

    public static func glb(_ rig: HairRig.Rig, headBone: String, headJoint: HairRig.V3, colour: [Double] = [0.35, 0.22, 0.14],
                           name: String = "Hair_gen", vrm0Axes: Bool = false) -> (data: Data, summary: Summary) {
        typealias V3 = HairRig.V3
        let ax: (V3) -> V3 = { vrm0Axes ? V3(-$0.x, $0.y, -$0.z) : $0 }
        var sum = Summary()

        // ---- nodes
        var nodes: [[String: Any]] = [["name": headBone, "translation": [ax(headJoint).x, ax(headJoint).y, ax(headJoint).z]]]
        var world: [V3] = [ax(headJoint)]            // rest position of every node
        var children: [Int: [Int]] = [:]
        var chainBones: [[Int]] = []                 // per chain: node of the bone at joint k
        var chainEnd: [Int] = []
        for (c, ch) in rig.chains.enumerated() {
            var parent = 0
            var bones: [Int] = []
            for (k, j) in ch.joints.enumerated() {
                let last = k == ch.joints.count - 1
                let p = ax(j), rel = p - world[parent]
                let i = nodes.count
                nodes.append(["name": String(format: last ? "HairGen_c%02d_end" : "HairGen_c%02d_%02d", c, k), "translation": [rel.x, rel.y, rel.z]])
                world.append(p)
                children[parent, default: []].append(i)
                if last { chainEnd.append(i) } else { bones.append(i) }
                parent = i
            }
            chainBones.append(bones)
        }
        for (p, cs) in children { nodes[p]["children"] = cs }
        let joints = [0] + chainBones.flatMap { $0 }
        var jointIndex = [Int: Int]()
        for (i, n) in joints.enumerated() { jointIndex[n] = i }
        sum.bones = joints.count - 1

        // ---- vertex attributes
        let n = rig.vertices.count
        let P = rig.vertices.map(ax)
        var N = [V3](repeating: V3(0, 0, 0), count: n)
        for f in rig.faces {
            let fn = HairRig.cross(P[f[1]] - P[f[0]], P[f[2]] - P[f[0]])   // length = 2 × area
            for i in f { N[i] += fn }
        }
        var bin = Data()
        var views: [[String: Any]] = [], accessors: [[String: Any]] = []
        func view(_ d: Data, target: Int?) -> Int {
            while bin.count % 4 != 0 { bin.append(0) }
            var v: [String: Any] = ["buffer": 0, "byteOffset": bin.count, "byteLength": d.count]
            if let target { v["target"] = target }
            bin.append(d)
            views.append(v)
            return views.count - 1
        }
        func floats(_ a: [Float]) -> Data { a.withUnsafeBufferPointer { Data(buffer: $0) } }

        var pos = [Float](), nrm = [Float](), uv = [Float](), jo = [UInt16](), we = [Float]()
        var lo = [Float](repeating: .infinity, count: 3), hi = [Float](repeating: -.infinity, count: 3)
        for i in 0..<n {
            let p = [Float(P[i].x), Float(P[i].y), Float(P[i].z)]
            for a in 0..<3 { lo[a] = min(lo[a], p[a]); hi[a] = max(hi[a], p[a]) }
            pos += p
            let l = HairRig.norm(N[i])
            let u = l > 0 ? N[i] / l : V3(0, 1, 0)
            nrm += [Float(u.x), Float(u.y), Float(u.z)]
            uv += [Float(rig.uv[i].x), Float(1 - rig.uv[i].y)]
            // head 1 − w, segment w · α, neighbour segment w · (1 − α)
            var inf: [(Int, Double)] = []
            let c = rig.chain[i], w = rig.weight[i]
            if c < 0 || w <= 0 {
                inf = [(0, 1)]
                sum.headOnly += 1
            } else {
                let two = rig.segment2[i] != rig.segment[i] && rig.alpha[i] < 1
                inf.append((jointIndex[chainBones[c][rig.segment[i]]]!, w * (two ? rig.alpha[i] : 1)))
                if two { inf.append((jointIndex[chainBones[c][rig.segment2[i]]]!, w * (1 - rig.alpha[i]))) }
                if w < 1 { inf.append((0, 1 - w)) }
                if two && w < 1 { sum.threeInfluences += 1 }
            }
            let total = inf.reduce(0) { $0 + $1.1 }
            for s in 0..<4 {
                jo.append(s < inf.count ? UInt16(inf[s].0) : 0)
                we.append(s < inf.count ? Float(inf[s].1 / total) : 0)
            }
        }
        func accessor(_ v: Int, _ type: String, _ comp: Int, _ count: Int, min: [Float]? = nil, max: [Float]? = nil) -> Int {
            var a: [String: Any] = ["bufferView": v, "componentType": comp, "count": count, "type": type]
            if let min, let max { a["min"] = min; a["max"] = max }
            accessors.append(a)
            return accessors.count - 1
        }
        let aPos = accessor(view(floats(pos), target: 34962), "VEC3", 5126, n, min: lo, max: hi)
        let aNrm = accessor(view(floats(nrm), target: 34962), "VEC3", 5126, n)
        let aUV = accessor(view(floats(uv), target: 34962), "VEC2", 5126, n)
        let aJo = accessor(view(jo.withUnsafeBufferPointer { Data(buffer: $0) }, target: 34962), "VEC4", 5123, n)
        let aWe = accessor(view(floats(we), target: 34962), "VEC4", 5126, n)
        let idx = rig.faces.flatMap { $0.map { UInt32($0) } }
        let aIdx = accessor(view(idx.withUnsafeBufferPointer { Data(buffer: $0) }, target: 34963), "SCALAR", 5125, idx.count)
        var ibm = [Float]()
        for j in joints {
            let p = world[j]
            ibm += [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, Float(-p.x), Float(-p.y), Float(-p.z), 1]
        }
        let aIBM = accessor(view(floats(ibm), target: nil), "MAT4", 5126, joints.count)
        sum.vertices = n
        sum.triangles = rig.faces.count

        // ---- springs: joints from the chain's first bone to its end node
        var springs: [[String: Any]] = []
        for (c, ch) in rig.chains.enumerated() {
            var js: [[String: Any]] = chainBones[c].map {
                ["node": $0, "hitRadius": hitRadius, "stiffness": ch.stiffness, "gravityPower": gravityPower, "gravityDir": [0, -1, 0], "dragForce": drag]
            }
            js.append(["node": chainEnd[c]])
            springs.append(["name": "HairGen chain \(c)", "joints": js])
        }
        sum.springs = springs.count

        let meshNode = nodes.count
        nodes.append(["name": name, "mesh": 0, "skin": 0])
        let material: [String: Any] = [
            "name": "\(name)_MToon", "doubleSided": true,
            "pbrMetallicRoughness": ["baseColorFactor": colour + [1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0],
            "extensions": ["VRMC_materials_mtoon": ["specVersion": "1.0", "shadeColorFactor": colour.map { 0.6 * $0 }]],
        ]
        var gltf: [String: Any] = [
            "asset": ["version": "2.0", "generator": "mlx-hairdeck-swift"],
            "scene": 0, "scenes": [["nodes": [0, meshNode]]], "nodes": nodes,
            "meshes": [["name": name, "primitives": [["attributes": ["POSITION": aPos, "NORMAL": aNrm, "TEXCOORD_0": aUV, "JOINTS_0": aJo, "WEIGHTS_0": aWe],
                                                       "indices": aIdx, "material": 0, "mode": 4]]]],
            "skins": [["name": name, "joints": joints, "inverseBindMatrices": aIBM, "skeleton": 0]],
            "materials": [material], "accessors": accessors, "bufferViews": views,
            "extensionsUsed": springs.isEmpty ? ["VRMC_materials_mtoon"] : ["VRMC_materials_mtoon", "VRMC_springBone"],
        ]
        if !springs.isEmpty { gltf["extensions"] = ["VRMC_springBone": ["specVersion": "1.0", "springs": springs]] }
        while bin.count % 4 != 0 { bin.append(0) }
        gltf["buffers"] = [["byteLength": bin.count]]

        var json = try! JSONSerialization.data(withJSONObject: gltf, options: [.sortedKeys])
        while json.count % 4 != 0 { json.append(0x20) }
        var out = Data()
        func u32(_ v: Int) { var x = UInt32(v).littleEndian; out.append(Data(bytes: &x, count: 4)) }
        u32(0x46546C67); u32(2); u32(12 + 8 + json.count + 8 + bin.count)
        u32(json.count); u32(0x4E4F534A); out.append(json)
        u32(bin.count); u32(0x004E4942); out.append(bin)
        return (out, sum)
    }
}
