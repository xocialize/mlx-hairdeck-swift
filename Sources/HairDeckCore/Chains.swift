import Foundation
import MLX

/// The spring-bone chain generator (`hairdeck/spring/chaindit.py`): the set DiT at dim 384, depth 8,
/// 6 heads over 48 chain tokens of 27 values, conditioned on the hairstyle's 192 × 40 card tokens in
/// the head frame (`card_proj(card_norm(cards)) + card_pos`). Weights `ckpt/chain_model/model.safetensors`.
///
/// Chain token, normalised by `data/spring/chain_stats_head.json`: 8 arc-length samples × 3, joint
/// count, stiffness, then the existence flag (±1).
public struct ChainDiT {
    public static let slots = 48
    public static let pathPoints = 8
    public let dit: SetDiT
    /// `chain_stats_head.json`: mean and standard deviation of the 26 token values.
    public let mu: [Double]
    public let sd: [Double]
    public var tokenDim: Int { mu.count + 1 }

    public init(weights: [String: MLXArray], mu: [Double], sd: [Double]) {
        dit = SetDiT(weights: weights, depth: 8, heads: 6)
        self.mu = mu
        self.sd = sd
    }

    /// cards [B, 192, 40] (head-frame card tokens) → ctx [B, 192, 384].
    public func encodeCond(_ cards: MLXArray) -> MLXArray {
        HDOps.lin(HDOps.ln(cards, dit.w, "card_norm"), dit.w, "card_proj") + dit.w["card_pos"]!
    }

    /// One decoded chain, in the frame of the card tokens (the head frame).
    public struct Chain {
        public var path: [[Double]]
        public var joints: [[Double]]
        public var stiffness: Double
        public var json: [String: Any] { ["path": path, "joints": joints, "n_joints": joints.count, "stiffness": stiffness] }
    }

    static func interp(_ t: Double, _ xs: [Double], _ ys: [Double]) -> Double {
        // numpy.interp for increasing xs
        if t <= xs[0] { return ys[0] }
        if t >= xs[xs.count - 1] { return ys[ys.count - 1] }
        var j = 1
        while xs[j] < t { j += 1 }
        let x0 = xs[j - 1], x1 = xs[j]
        if x1 == x0 { return ys[j - 1] }
        return ys[j - 1] + (ys[j] - ys[j - 1]) * (t - x0) / (x1 - x0)
    }

    /// `decode`: chain tokens of one hairstyle (N × 27, row-major) → the chains of its existing slots:
    /// the 8-point path resampled by arc length to the predicted joint count (2 to 20).
    public func decode(_ tokens: [Float]) -> [Chain] {
        let D = tokenDim, M = Self.pathPoints
        var out: [Chain] = []
        for s in 0..<(tokens.count / D) {
            let t = (0..<D).map { Double(tokens[s * D + $0]) }
            guard t[D - 1] > 0 else { continue }
            let v = (0..<(D - 1)).map { t[$0] * sd[$0] + mu[$0] }
            let path = (0..<M).map { i in [v[3 * i], v[3 * i + 1], v[3 * i + 2]] }
            let nj = Int(min(max(v[3 * M].rounded(.toNearestOrEven), 2), 20))
            var d = [0.0]
            for i in 1..<M {
                let a = path[i], b = path[i - 1]
                d.append(d[i - 1] + ((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])).squareRoot())
            }
            // numpy.linspace(0, d[-1], nj): the last value is the stop exactly
            let step = d[M - 1] / Double(nj - 1)
            let tt = (0..<nj).map { $0 == nj - 1 ? d[M - 1] : Double($0) * step }
            let joints = tt.map { x in (0..<3).map { k in Self.interp(x, d, path.map { $0[k] }) } }
            out.append(Chain(path: path, joints: joints, stiffness: min(max(v[3 * M + 1], 0), 1)))
        }
        return out
    }
}

/// Head-frame card tokens of generated cards, the condition of the chain DiT
/// (`hairdeck/testset/chains.py: pointcloud_run_cards`): with a head fit (k, b), root → k · root + b,
/// the root-relative card scaled by k and encoded again by the card VAE, the root normals signed away
/// from the mean root, standardised with the head-frame statistics.
public enum HeadFrame {
    /// `pad_seqs`: cards (n × ≥5 rows) → root-relative padded features [C, L, 5] in fp32 and lengths.
    /// Cards longer than L are refused (the reference resamples them; generated cards never are).
    public static func padSeqs(_ cards: [[[Double]]], L: Int, scale k: Float = 1) -> (x: [Float], lens: [Int]) {
        var x = [Float](repeating: 0, count: cards.count * L * 5)
        for (j, s) in cards.enumerated() {
            precondition(s.count <= L, "card with more than \(L) points")
            for (i, row) in s.enumerated() {
                for c in 0..<5 {
                    // float32 store, the root subtracted in float64, float32 again; then × k in float32
                    let v = c < 3 ? Float(Double(Float(row[c])) - s[0][c]) : Float(row[c])
                    x[(j * L + i) * 5 + c] = v * k
                }
            }
        }
        return (x, cards.map { $0.count })
    }

    /// `arc_u`: arc-length fraction of every point (0 at the root, 1 at the tip, 0 for padding), fp32.
    public static func arcU(_ x: [Float], lens: [Int], L: Int) -> [Float] {
        var u = [Float](repeating: 0, count: lens.count * L)
        for (j, n) in lens.enumerated() {
            var s = [Float](repeating: 0, count: n)
            for i in 1..<max(n, 1) {
                let a = (j * L + i) * 5, b = a - 5
                let dx = x[a] - x[b], dy = x[a + 1] - x[b + 1], dz = x[a + 2] - x[b + 2]
                s[i] = s[i - 1] + (dx * dx + dy * dy + dz * dz).squareRoot()
            }
            let tot = max(s[n - 1], 1e-9)
            for i in 0..<n { u[j * L + i] = s[i] / tot }
        }
        return u
    }

    /// cards of one hairstyle (dataset frame, 8 columns) → card tokens [192 × 40] row-major, at most
    /// the first 192 cards with at least 3 points.
    public static func cardTokens(_ cards: [HairCard], k: Double, b: [Double], vae: CardVAE, tokenizer: CardTokenizer,
                                  slots: Int = 192) -> [Float] {
        let cs = Array(cards.filter { $0.seq.count >= 3 }.prefix(slots))
        let D = tokenizer.dim, zdim = tokenizer.zdim
        var tok = [Float](repeating: 0, count: slots * D)
        for s in 0..<slots { tok[s * D + D - 1] = -1 }
        guard !cs.isEmpty else { return tok }
        let (x, lens) = padSeqs(cs.map { $0.seq }, L: vae.L, scale: Float(k))
        let lat = vae.encode(x, lens: lens).asArray(Float.self)
        // sign rule of the training tokens: flip a root normal that points towards the mean card root
        var mean = [0.0, 0.0, 0.0]
        for c in cs { for a in 0..<3 { mean[a] += c.seq[0][a] / Double(cs.count) } }
        for (j, c) in cs.enumerated() {
            let r = c.seq[0]
            var n = [r[5], r[6], r[7]]
            if n[0] * (r[0] - mean[0]) + n[1] * (r[1] - mean[1]) + n[2] * (r[2] - mean[2]) < 0 { n = n.map { -$0 } }
            for a in 0..<3 {
                tok[j * D + a] = (Float(k * r[a] + b[a]) - tokenizer.rootMu[a]) / tokenizer.rootSd[a]
                tok[j * D + 3 + zdim + a] = Float(n[a])
            }
            for z in 0..<zdim { tok[j * D + 3 + z] = (lat[j * zdim + z] - tokenizer.latentMu[z]) / tokenizer.latentSd[z] }
            tok[j * D + 6 + zdim] = c.triangle ? 1 : -1
            tok[j * D + 7 + zdim] = 1
        }
        return tok
    }
}
