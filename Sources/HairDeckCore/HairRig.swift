import Foundation

/// Cards and chains → one skinned hair mesh, without Blender: a port of `hairdeck/cards/frames.py`,
/// `hairdeck/cards/mesh.py` (card mesh, faces, UVs) and `hairdeck/vrm/prep.py` with the geometric
/// skinning of `hairdeck/spring/sim.py`. Plain Swift in Double, as the numpy reference.
///
/// A card (n × 8: xyz, width, thickness, root width direction) becomes n rings of 4 vertices; the
/// width direction is the Bishop transport of the root direction along the 5-point-stencil tangents.
/// A card follows the chain nearest to the lower 60 % of its centreline if that mean distance is
/// below `assign` face widths, else it stays with the head; its vertices bind to the nearest segment
/// of that chain with a weight ramping from 0 at the chain root to 1 over `ramp` face widths of arc
/// length, blended between adjacent segments around each joint.
public enum HairRig {
    public typealias V3 = SIMD3<Double>
    public static let assign = 0.4, ramp = 0.05, centrelinePoints = 16

    static func norm(_ v: V3) -> Double { (v.x * v.x + v.y * v.y + v.z * v.z).squareRoot() }
    static func dot(_ a: V3, _ b: V3) -> Double { a.x * b.x + a.y * b.y + a.z * b.z }
    static func cross(_ a: V3, _ b: V3) -> V3 { V3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x) }

    // MARK: frames.py

    static let stencil: [Double] = [-1.0 / 12, 2.0 / 3, 0, -2.0 / 3, 1.0 / 12]

    /// Unit tangents from the 5-point stencil (pointing from tip towards root); cards with fewer than
    /// 5 points use the root → tip chord for every point.
    static func tangents(_ p: [V3]) -> [V3] {
        let n = p.count
        if n < 5 {
            let t = p[n - 1] - p[0]
            return [V3](repeating: t / (norm(t) + 1e-12), count: n)
        }
        return (0..<n).map { i in
            let lo = min(max(i - 2, 0), n - 5)
            var t = V3(0, 0, 0)
            for j in 0..<5 { t += stencil[j] * p[lo + j] }
            return t / (norm(t) + 1e-12)
        }
    }

    static func perp(_ v: V3, _ t: V3) -> V3 {
        let w = v - dot(v, t) * t
        return w / (norm(w) + 1e-12)
    }

    /// Rotation-minimising transport of the root width direction n0 along the tangents.
    static func bishop(_ n0: V3, _ T: [V3]) -> [V3] {
        var out = [perp(n0, T[0])]
        for i in 1..<T.count {
            let a = T[i - 1], b = T[i]
            let ax = cross(a, b)
            let s = norm(ax), c = min(max(dot(a, b), -1), 1)
            var n = out[i - 1]
            if s > 1e-9 {
                let k = ax / s
                n = n * c + cross(k, n) * s + k * dot(k, n) * (1 - c)
            }
            out.append(perp(n, b))
        }
        return out
    }

    // MARK: mesh.py

    /// (vertices 4 per control point, faces) of one card (`card_mesh_vf`).
    public static func cardMesh(_ seq: [[Double]], triangle: Bool) -> (v: [V3], f: [[Int]]) {
        let p = seq.map { V3($0[0], $0[1], $0[2]) }
        let T = tangents(p)
        let N = bishop(V3(seq[0][5], seq[0][6], seq[0][7]), T)
        let (up, down): (Double, Double) = triangle ? (3, 6) : (2, 2)
        var verts: [V3] = []
        for i in 0..<seq.count {
            let w = seq[i][3], t = seq[i][4]
            let n = N[i] / (norm(N[i]) + 1e-12)
            var g = T[i] - dot(T[i], n) * n
            g = g / (norm(g) + 1e-12)
            let b = cross(n, g)
            verts += [p[i] + w / 2 * n, p[i] + t / up * b, p[i] - w / 2 * n, p[i] - t / down * b]
        }
        var f: [[Int]] = [[0, 2, 1], [0, 3, 2]]
        for i in 1..<max(seq.count, 1) {
            let cur = 4 * i, prev = 4 * i - 4
            for k in 0..<4 { f.append([cur + k, prev + k, cur + (k + 1) % 4]) }
            for k in [0, 3, 2, 1] { f.append([prev + (k + 1) % 4, cur + (k + 1) % 4, prev + k]) }
            f += [[cur, cur + 1, cur + 2], [cur, cur + 2, cur + 3]]
        }
        return (verts, f)
    }

    static func arcLengths(_ p: [V3]) -> [Double] {
        var s = [0.0]
        for i in 1..<max(p.count, 1) { s.append(s[i - 1] + norm(p[i] - p[i - 1])) }
        return s
    }

    /// `card_uv`: u across the width (0, 0.5, 1, 0.5 around a ring), v the arc-length fraction from the root.
    public static func cardUV(_ seq: [[Double]]) -> [SIMD2<Double>] {
        let p = seq.map { V3($0[0], $0[1], $0[2]) }
        let s = arcLengths(p)
        let total = s[s.count - 1]
        var out: [SIMD2<Double>] = []
        for i in 0..<p.count {
            let v = total > 0 ? s[i] / total : (p.count > 1 ? Double(i) / Double(p.count - 1) : 0)
            for u in [0.0, 0.5, 1.0, 0.5] { out.append(SIMD2(u, v)) }
        }
        return out
    }

    // MARK: sim.py

    static func interp(_ t: Double, _ xs: [Double], _ ys: [Double]) -> Double {
        if t <= xs[0] { return ys[0] }
        if t >= xs[xs.count - 1] { return ys[ys.count - 1] }
        var j = 1
        while xs[j] < t { j += 1 }
        if xs[j] == xs[j - 1] { return ys[j - 1] }
        return ys[j - 1] + (ys[j] - ys[j - 1]) * (t - xs[j - 1]) / (xs[j] - xs[j - 1])
    }

    /// `centreline`: n arc-length samples of a card's control polyline.
    static func centreline(_ p: [V3], n: Int = centrelinePoints) -> [V3] {
        let d = arcLengths(p)
        let stop = d[d.count - 1], step = stop / Double(n - 1)
        return (0..<n).map { i in
            let t = i == n - 1 ? stop : Double(i) * step
            return V3(interp(t, d, p.map { $0.x }), interp(t, d, p.map { $0.y }), interp(t, d, p.map { $0.z }))
        }
    }

    /// `clean`: a joint polyline with zero-length segments removed.
    static func clean(_ J: [V3]) -> [V3] {
        var out: [V3] = []
        for (i, p) in J.enumerated() where i == 0 || norm(p - J[i - 1]) > 1e-6 { out.append(p) }
        return out
    }

    /// `project`: per point the distance to polyline J, the nearest segment (first minimum) and the
    /// arc length from J[0], clipped to the line.
    static func project(_ P: [V3], _ J: [V3]) -> (dist: [Double], seg: [Int], arc: [Double], u: [Double]) {
        let ns = J.count - 1
        var l2 = [Double](), cum = [0.0]
        for k in 0..<ns {
            let ab = J[k + 1] - J[k]
            l2.append(dot(ab, ab))
            cum.append(cum[k] + l2[k].squareRoot())
        }
        var dist = [Double](), seg = [Int](), arc = [Double](), us = [Double]()
        for p in P {
            var best = Double.infinity, bk = 0, bu = 0.0
            for k in 0..<ns {
                let ab = J[k + 1] - J[k]
                let u = min(max(dot(p - J[k], ab) / l2[k], 0), 1)
                let d = norm(p - (J[k] + u * ab))
                if d < best { best = d; bk = k; bu = u }
            }
            dist.append(best)
            seg.append(bk)
            arc.append(cum[bk] + bu * l2[bk].squareRoot())
            us.append(bu)
        }
        return (dist, seg, arc, us)
    }

    /// Mean distance of the lower 60 % of the points to a chain (the assignment test of `skin`).
    static func lowerMean(_ P: [V3], _ J: [V3]) -> Double {
        let lo = Array(P[Int(0.4 * Double(P.count))...])
        let d = project(lo, J).dist
        return d.reduce(0, +) / Double(d.count)
    }

    // MARK: prep.py

    public struct Chain {
        /// Joint positions, root first.
        public var joints: [V3]
        public var stiffness: Double
    }

    /// The hair mesh and its binding, in the body's world (metres, Y up).
    public struct Rig {
        public var vertices: [V3] = []
        public var faces: [[Int]] = []
        public var uv: [SIMD2<Double>] = []
        public var card: [Int] = []
        /// Per vertex: chain (-1 = rigid with the head), segment, weight, second segment, blend factor.
        public var chain: [Int] = [], segment: [Int] = [], weight: [Double] = [], segment2: [Int] = [], alpha: [Double] = []
        /// The chains that bind at least one card.
        public var chains: [Chain] = []
        public var cardsOnChain = 0, bindMismatch = 0, chainsDegenerate = 0, chainsDropped = 0
    }

    /// cards (8 columns) and chains in the head frame, a body placement → the rig (`prepare` + `prep`,
    /// or `prep_no_chains` when no chain is usable).
    public static func build(cards: [HairCard], chains: [(joints: [[Double]], stiffness: Double)], body: BodyPlacement) -> Rig {
        var rig = Rig()
        let W = body.faceWidthWorld
        // head frame → the body's dataset frame (`head_to_body`), then joints to the world
        let kept = chains.map { c in Chain(joints: clean(c.joints.map { body.headToDataset(V3($0[0], $0[1], $0[2])) }), stiffness: c.stiffness) }
            .filter { $0.joints.count >= 2 }
        rig.chainsDegenerate = chains.count - kept.count
        let world = kept.map { Chain(joints: $0.joints.map(body.datasetToWorld), stiffness: $0.stiffness) }
        var cardChain: [Int] = []
        for (ci, card) in cards.enumerated() {
            let seq = card.seq.map { r -> [Double] in
                let p = body.headToDataset(V3(r[0], r[1], r[2]))
                return [p.x, p.y, p.z, r[3] / body.k, r[4] / body.k, r[5], r[6], r[7]]
            }
            let (v, f) = cardMesh(seq, triangle: card.triangle)
            let vb = v.map(body.datasetToWorld)
            let cl = centreline(seq.map { V3($0[0], $0[1], $0[2]) }).map(body.datasetToWorld)
            // card → chain: the nearest within `assign` face widths
            var c = -1, bd = assign * W
            for (i, ch) in world.enumerated() {
                let dm = lowerMean(cl, ch.joints)
                if dm < bd { c = i; bd = dm }
            }
            let n = vb.count
            var seg = [Int](repeating: 0, count: n), w = [Double](repeating: 0, count: n)
            var seg2 = [Int](repeating: 0, count: n), al = [Double](repeating: 1, count: n)
            if c >= 0 {
                let J = world[c].joints
                if lowerMean(vb, J) < assign * W {
                    let pr = project(vb, J)
                    let nseg = J.count - 1
                    for i in 0..<n {
                        seg[i] = pr.seg[i]
                        seg2[i] = pr.seg[i]
                        w[i] = min(max(pr.arc[i] / (ramp * W), 0), 1)
                        let k = pr.seg[i], u = pr.u[i]
                        if u < 0.5 && k > 0 { seg2[i] = k - 1; al[i] = 0.5 + u } else if u > 0.5 && k < nseg - 1 { seg2[i] = k + 1; al[i] = 1.5 - u }
                    }
                } else {
                    // the vertex-based test disagrees with the centreline: keep the card rigid
                    rig.bindMismatch += 1
                    c = -1
                }
            }
            let off = rig.vertices.count
            rig.vertices += vb
            rig.uv += cardUV(seq)
            rig.faces += f.map { $0.map { $0 + off } }
            rig.card += [Int](repeating: ci, count: n)
            rig.chain += [Int](repeating: c, count: n)
            rig.segment += seg
            rig.weight += w
            rig.segment2 += seg2
            rig.alpha += al
            cardChain.append(c)
        }
        // drop chains that bind no card
        let used = Array(Set(cardChain.filter { $0 >= 0 })).sorted()
        var remap = [Int: Int]()
        for (i, c) in used.enumerated() { remap[c] = i }
        rig.chain = rig.chain.map { $0 < 0 ? -1 : remap[$0]! }
        rig.chains = used.map { world[$0] }
        rig.chainsDropped = kept.count - used.count
        rig.cardsOnChain = cardChain.filter { $0 >= 0 }.count
        return rig
    }
}

/// Where a body's head is: the two similarities that take head-frame hair onto it, as HairDeck's body
/// file gives them (`hairdeck/vrm/body.py`): p_dataset = (p_head − b) / k, then p_world =
/// (p_dataset − t) / s (the axes of the dataset frame are those of a VRM 1.0 file: Y up, face
/// toward +Z, metres).
public struct BodyPlacement {
    public typealias V3 = SIMD3<Double>
    public var k: Double
    public var b: V3
    public var s: Double
    public var t: V3
    /// Face-mesh width in the dataset frame; the skinning thresholds are in face widths.
    public var faceWidthDataset: Double
    /// The head joint in the dataset frame and the name of the body's head bone.
    public var headJointDataset: V3
    public var headBone: String

    public var faceWidthWorld: Double { faceWidthDataset / s }
    public func headToDataset(_ p: V3) -> V3 { (p - b) / k }
    public func datasetToWorld(_ p: V3) -> V3 { (p - t) / s }
    public var headJointWorld: V3 { datasetToWorld(headJointDataset) }

    /// From a body description as `tools/dump_p3_fixtures.py` writes it: {"k", "b", "fit": {"s", "t",
    /// "sx", "sz", "w", "head_joint"}, "head_bone"}. Only the axis signs of the released data (sx 1,
    /// sz −1) are supported.
    public init(json path: String) throws {
        let o = try HDIO.jsonObject(path)
        guard let fit = o["fit"] as? [String: Any], let name = o["head_bone"] as? String,
              (fit["sx"] as? NSNumber)?.intValue == 1, (fit["sz"] as? NSNumber)?.intValue == -1
        else { throw HDIO.IOError.bad("\(path): not a body description with sx 1, sz -1") }
        func v3(_ a: Any?) -> V3 { let x = (a as! [NSNumber]).map { $0.doubleValue }; return V3(x[0], x[1], x[2]) }
        k = (o["k"] as! NSNumber).doubleValue
        b = v3(o["b"])
        s = (fit["s"] as! NSNumber).doubleValue
        t = v3(fit["t"])
        faceWidthDataset = (fit["w"] as! NSNumber).doubleValue
        headJointDataset = v3(fit["head_joint"])
        headBone = name
    }
}
