import Foundation
import simd

/// A body description read from a VRM file, for placing head-frame hair on it without HairDeck's
/// Blender-made fit.
///
/// HairDeck's head frame maps every head onto a canonical one (`hairdeck/data/head_frame.py`): the unit
/// is the width of the rest-pose face mesh, the origin its bounding box's centre in x and z and its
/// top in y, so p_head = k · p + b with k = W / w and b = O − k · anchor (W, O: the canonical head of
/// the released models, `data/head_frame_train.npz`). Here w and the anchor come from the POSITION
/// bounds of the face mesh in the file (a VRoid model keeps its skinned meshes in rest-pose world
/// coordinates), and the head bone's name and rest transform from the humanoid map and the node
/// hierarchy. A VRM 0.x file stores the model half a turn about Y; the description is always in
/// VRM 1.0 axes (Y up, face toward +Z) and `isVRM0` says which the file uses.
public struct VRMBody {
    /// The canonical head of the released HairDeck models (head frame).
    public static let canonicalFaceWidth = 0.12759823777166274
    public static let canonicalAnchor = SIMD3<Double>(-2.228594313152571e-08, 0.4790549478264301, 0.025801452799668806)

    public var isVRM0: Bool
    public var headBone: String
    /// Rest position and world rotation of the head bone, VRM 1.0 axes.
    public var headPosition: SIMD3<Double>
    public var headRotation: simd_quatd
    /// Bounding box of the face mesh, VRM 1.0 axes.
    public var faceMin: SIMD3<Double>
    public var faceMax: SIMD3<Double>
    public var faceMesh: String

    public var faceWidth: Double { faceMax.x - faceMin.x }
    public var anchor: SIMD3<Double> { SIMD3((faceMin.x + faceMax.x) / 2, faceMax.y, (faceMin.z + faceMax.z) / 2) }

    /// Hair with no body: the model's own head frame, as an item on a bone named `Head` at the head
    /// joint of the canonical head (HairDeck's avatar's head joint taken into the head frame).
    public static var canonicalPlacement: BodyPlacement {
        BodyPlacement(k: 1, b: SIMD3(0, 0, 0), s: 1, t: SIMD3(0, 0, 0), faceWidthDataset: canonicalFaceWidth,
                      headJointDataset: SIMD3(0, 0.3622, 0.0015), headBone: "Head")
    }

    /// The placement of head-frame hair on this body: dataset frame = the body's world.
    public var placement: BodyPlacement {
        let k = Self.canonicalFaceWidth / faceWidth
        return BodyPlacement(k: k, b: Self.canonicalAnchor - k * anchor, s: 1, t: SIMD3(0, 0, 0), faceWidthDataset: faceWidth,
                             headJointDataset: headPosition, headBone: headBone, headRotation: headRotation)
    }

    public enum VRMError: Error, CustomStringConvertible {
        case bad(String)
        public var description: String { if case .bad(let s) = self { return s }; return "" }
    }

    /// The JSON chunk of a `.vrm` / `.glb`.
    public static func json(glb path: String) throws -> [String: Any] {
        guard let d = FileManager.default.contents(atPath: path) else { throw VRMError.bad("cannot read \(path)") }
        return try json(glb: d, label: path)
    }

    public static func json(glb d: Data, label path: String = "<data>") throws -> [String: Any] {
        guard d.count > 20 else { throw VRMError.bad("\(path): too short for a binary glTF") }
        let u32: (Int) -> Int = { o in Int(d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: UInt32.self) }) }
        guard u32(0) == 0x46546C67, u32(16) == 0x4E4F534A else { throw VRMError.bad("\(path): not a binary glTF") }
        guard let o = try JSONSerialization.jsonObject(with: d.subdata(in: 20..<(20 + u32(12)))) as? [String: Any] else { throw VRMError.bad("\(path): bad JSON chunk") }
        return o
    }

    public init(vrm path: String, faceMeshPattern: String = "face") throws {
        guard let d = FileManager.default.contents(atPath: path) else { throw VRMError.bad("cannot read \(path)") }
        try self.init(glb: d, label: path, faceMeshPattern: faceMeshPattern)
    }

    /// From the bytes of a `.vrm` / `.glb`.
    public init(glb data: Data, label path: String = "<data>", faceMeshPattern: String = "face") throws {
        let g = try Self.json(glb: data, label: path)
        let ext = g["extensions"] as? [String: Any] ?? [:]
        guard let nodes = g["nodes"] as? [[String: Any]] else { throw VRMError.bad("\(path): no nodes") }
        let head: Int
        if let v1 = ext["VRMC_vrm"] as? [String: Any] {
            isVRM0 = false
            guard let hb = ((v1["humanoid"] as? [String: Any])?["humanBones"] as? [String: Any])?["head"] as? [String: Any],
                  let n = (hb["node"] as? NSNumber)?.intValue else { throw VRMError.bad("\(path): no humanoid head bone") }
            head = n
        } else if let v0 = ext["VRM"] as? [String: Any] {
            isVRM0 = true
            guard let hb = ((v0["humanoid"] as? [String: Any])?["humanBones"] as? [[String: Any]])?.first(where: { ($0["bone"] as? String) == "head" }),
                  let n = (hb["node"] as? NSNumber)?.intValue else { throw VRMError.bad("\(path): no humanoid head bone") }
            head = n
        } else { throw VRMError.bad("\(path): neither VRMC_vrm nor VRM") }
        headBone = nodes[head]["name"] as? String ?? "head"

        // world transform of the head bone: translation / rotation / scale up the hierarchy
        var parent = [Int: Int]()
        for (i, n) in nodes.enumerated() { for c in (n["children"] as? [NSNumber] ?? []) { parent[c.intValue] = i } }
        func nums(_ a: Any?) -> [Double]? { (a as? [NSNumber])?.map { $0.doubleValue } }
        var m = matrix_identity_double4x4
        var i: Int? = head
        var q = simd_quatd(ix: 0, iy: 0, iz: 0, r: 1)
        while let n = i {
            guard nodes[n]["matrix"] == nil else { throw VRMError.bad("\(path): node \(n) uses a matrix; expected translation / rotation / scale") }
            let t = nums(nodes[n]["translation"]) ?? [0, 0, 0], r = nums(nodes[n]["rotation"]) ?? [0, 0, 0, 1], s = nums(nodes[n]["scale"]) ?? [1, 1, 1]
            let rq = simd_quatd(ix: r[0], iy: r[1], iz: r[2], r: r[3])
            var local = simd_double4x4(rq) * simd_double4x4(diagonal: SIMD4(s[0], s[1], s[2], 1))
            local.columns.3 = SIMD4(t[0], t[1], t[2], 1)
            m = local * m
            q = rq * q
            i = parent[n]
        }
        var pos = SIMD3(m.columns.3.x, m.columns.3.y, m.columns.3.z)

        // the face mesh: POSITION bounds over its primitives
        guard let meshes = g["meshes"] as? [[String: Any]], let accessors = g["accessors"] as? [[String: Any]] else { throw VRMError.bad("\(path): no meshes") }
        guard let fm = meshes.first(where: { ($0["name"] as? String ?? "").lowercased().contains(faceMeshPattern) }) else {
            throw VRMError.bad("\(path): no mesh named like '\(faceMeshPattern)' (meshes: \(meshes.map { $0["name"] as? String ?? "?" }))")
        }
        faceMesh = fm["name"] as? String ?? ""
        var lo = SIMD3<Double>(repeating: .infinity), hi = SIMD3<Double>(repeating: -.infinity)
        for p in fm["primitives"] as? [[String: Any]] ?? [] {
            guard let a = ((p["attributes"] as? [String: Any])?["POSITION"] as? NSNumber)?.intValue,
                  let mn = nums(accessors[a]["min"]), let mx = nums(accessors[a]["max"]) else { throw VRMError.bad("\(path): face mesh without POSITION bounds") }
            lo = simd_min(lo, SIMD3(mn[0], mn[1], mn[2]))
            hi = simd_max(hi, SIMD3(mx[0], mx[1], mx[2]))
        }
        if isVRM0 {
            // half a turn about Y into VRM 1.0 axes
            (lo, hi) = (SIMD3(-hi.x, lo.y, -hi.z), SIMD3(-lo.x, hi.y, -lo.z))
            pos = SIMD3(-pos.x, pos.y, -pos.z)
            q = simd_quatd(ix: 0, iy: 1, iz: 0, r: 0) * q * simd_quatd(ix: 0, iy: 1, iz: 0, r: 0).inverse
        }
        faceMin = lo
        faceMax = hi
        headPosition = pos
        headRotation = q
    }
}
