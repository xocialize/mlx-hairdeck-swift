import Foundation
import MLX

/// Dora-VAE 1.1 shape encoder (Seed3D/Dora, code and weights Apache-2.0), the point-cloud condition of
/// HairDeck: a 1:1 port of `hairdeck/encoders/dora.py` (`ShapeVAE.encode`, kind 'dora'), which is
/// itself a clean forward-pass re-implementation written from the checkpoint key layout.
///
/// encode(points [B, N, 6]) → pre-KL tokens [B, 512, 768]. The released condition feeds the same
/// points to both branches (`sharp = coarse`), t = 256 query points per branch, so the two FPS index
/// sets are equal. Weights: `dora_vae_1_1_shape_model.safetensors`, keys `encoder.*`, fp32.
///
/// Numerics: Fourier frequencies 2^0..2^7 without π, embed = cat[xyz, sin, cos] coordinate-major;
/// qkv / q / kv projections carry no bias; MLP uses erf GELU; LayerNorm eps 1e-5; 12 heads; the fused
/// projections are laid out per head as [q | k | v] (split heads first, then the parts).
public enum Dora {
    public static let heads = 12
    public static let selfBlocks = 8
    public static let tokens = 512

    /// [..., 3] → [..., 51]: xyz, then sin, then cos of every coordinate times 2^k (k = 0..7).
    public static func fourier(_ xyz: MLXArray, numFreqs: Int = 8) -> MLXArray {
        let f = MLXArray((0..<numFreqs).map { Float(pow(2.0, Double($0))) })
        let s = xyz.shape
        let ang = (xyz.expandedDimensions(axis: -1) * f).reshaped(Array(s.dropLast()) + [s.last! * numFreqs])
        return concatenated([xyz, HDOps.sinR(ang), HDOps.cosR(ang)], axis: -1)
    }

    /// Greedy farthest point sampling on the CPU, as `farthest_point_sample` (fp32, start 0, first
    /// maximum on ties). xyz: N × 3 row-major → k indices.
    public static func farthestPointSample(_ xyz: [Float], count n: Int, k: Int, start: Int = 0) -> [Int32] {
        var out = [Int32](repeating: 0, count: k)
        var dist = [Float](repeating: .infinity, count: n)
        var cur = start
        xyz.withUnsafeBufferPointer { p in
            for i in 0..<k {
                out[i] = Int32(cur)
                let cx = p[3 * cur], cy = p[3 * cur + 1], cz = p[3 * cur + 2]
                var best: Float = -1
                var bestJ = 0
                for j in 0..<n {
                    let dx = p[3 * j] - cx, dy = p[3 * j + 1] - cy, dz = p[3 * j + 2] - cz
                    let d = dx * dx + dy * dy + dz * dz
                    if d < dist[j] { dist[j] = d }
                    if dist[j] > best { best = dist[j]; bestJ = j }
                }
                cur = bestJ
            }
        }
        return out
    }

    static func embed(_ pts: MLXArray, _ w: [String: MLXArray], _ p: String) -> MLXArray {
        let xyz = pts[.ellipsis, 0..<3], rest = pts[.ellipsis, 3...]
        return HDOps.lin(concatenated([fourier(xyz), rest], axis: -1), w, p)
    }

    static func mlp(_ x: MLXArray, _ w: [String: MLXArray], _ p: String) -> MLXArray {
        HDOps.lin(HDOps.geluErf(HDOps.lin(x, w, "\(p).c_fc")), w, "\(p).c_proj")
    }

    /// x + attn(ln_1 x → q, ln_2 data → [k | v] per head), then + mlp(ln_3 x).
    static func crossBlock(_ x: MLXArray, _ data: MLXArray, _ w: [String: MLXArray], _ p: String) -> MLXArray {
        let q = HDOps.splitHeads(HDOps.lin(HDOps.ln(x, w, "\(p).ln_1"), w, "\(p).attn.c_q"), heads: heads)
        let kv = HDOps.splitHeads(HDOps.lin(HDOps.ln(data, w, "\(p).ln_2"), w, "\(p).attn.c_kv"), heads: heads)
            .split(parts: 2, axis: -1)
        let y = x + HDOps.lin(HDOps.mergeHeads(HDOps.sdpa(q, kv[0], kv[1])), w, "\(p).attn.c_proj")
        return y + mlp(HDOps.ln(y, w, "\(p).ln_3"), w, "\(p).mlp")
    }

    /// x + attn(ln_1 x), then + mlp(ln_2 x); fused qkv projection, per head [q | k | v].
    static func selfBlock(_ x: MLXArray, _ w: [String: MLXArray], _ p: String) -> MLXArray {
        let qkv = HDOps.splitHeads(HDOps.lin(HDOps.ln(x, w, "\(p).ln_1"), w, "\(p).attn.c_qkv"), heads: heads)
            .split(parts: 3, axis: -1)
        let y = x + HDOps.lin(HDOps.mergeHeads(HDOps.sdpa(qkv[0], qkv[1], qkv[2])), w, "\(p).attn.c_proj")
        return y + mlp(HDOps.ln(y, w, "\(p).ln_2"), w, "\(p).mlp")
    }

    /// Intermediate activations of `encode`, for the parity gate.
    public struct Trace {
        public var query, afterCross, afterSelf0, tokens: MLXArray
    }

    /// points [B, N, 6] (xyz × 2, normals), idx [B, 256] FPS indices of the positions → tokens [B, 512, 768].
    public static func encode(_ pts: MLXArray, idx: MLXArray, weights w: [String: MLXArray]) -> Trace {
        let dc = embed(pts, w, "encoder.input_proj"), ds = embed(pts, w, "encoder.input_proj1")
        let gi = idx.expandedDimensions(axis: -1)
        let q = concatenated([takeAlong(dc, gi, axis: 1), takeAlong(ds, gi, axis: 1)], axis: 1)
        let cross = crossBlock(q, dc, w, "encoder.cross_attn") + crossBlock(q, ds, w, "encoder.cross_attn1")
        var x = cross
        var self0 = cross
        for i in 0..<selfBlocks {
            x = selfBlock(x, w, "encoder.self_attn.resblocks.\(i)")
            if i == 0 { self0 = x }
        }
        return Trace(query: q, afterCross: cross, afterSelf0: self0, tokens: HDOps.ln(x, w, "encoder.ln_post"))
    }

    /// The released condition of one point cloud: FPS on the CPU, then `encode`. points: N × 6 row-major.
    public static func condition(points: [Float], count n: Int, weights w: [String: MLXArray]) -> MLXArray {
        var xyz = [Float](repeating: 0, count: 3 * n)
        for i in 0..<n { for c in 0..<3 { xyz[3 * i + c] = points[6 * i + c] } }
        let idx = farthestPointSample(xyz, count: n, k: tokens / 2)
        return encode(MLXArray(points, [1, n, 6]), idx: MLXArray(idx, [1, idx.count]), weights: w).tokens
    }
}
