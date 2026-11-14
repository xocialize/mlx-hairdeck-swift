import Foundation
import MLX

/// The image path's generators (`hairdeck/models/imgmodel.py`, `models/colormodel.py`,
/// `cards/tokens.py`): the image set DiT and the colour model, both the set DiT of `SetDiT` with the
/// DINOv3 tokens of the 448 px model input as the cross-attention context.
public struct ImageDiT {
    public let dit: SetDiT

    public init(weights: [String: MLXArray]) { dit = SetDiT(weights: weights) }

    /// DINOv3 tokens [B, 789, 1024] → ctx [B, 789, 768]: `cond_proj(cond_norm(tokens)) + ctx_pos`.
    public func encodeCond(_ tokens: MLXArray) -> MLXArray {
        HDOps.lin(HDOps.ln(tokens, dit.w, "cond_norm"), dit.w, "cond_proj") + dit.w["ctx_pos"]!
    }
}

/// Per-card hair colour: the image model's network fine-tuned to read the colours of the generated
/// cards off the image. Input: the clean card tokens at t = 0, the DINOv3 tokens of the crop, and per
/// 16 px patch its mean RGB, the matte fraction and the mean RGB under the matte, added to the patch
/// tokens of the context. Output: per card 16 CIELAB colours (D65) from root to tip.
public struct ColorDiT {
    public static let bins = 16
    public let image: ImageDiT
    let w: [String: MLXArray]
    let mu: MLXArray, sd: MLXArray

    /// weights: `ckpt/color_model/model.safetensors` (the image DiT under `dit.`, the colour layers,
    /// and the Lab standardisation `mu`, `sd`).
    public init(weights: [String: MLXArray]) {
        var dit = [String: MLXArray]()
        for (k, v) in weights where k.hasPrefix("dit.") { dit[String(k.dropFirst(4))] = v }
        image = ImageDiT(weights: dit)
        w = weights
        mu = weights["mu"]!
        sd = weights["sd"]!
    }

    /// tokens [B, 789, 1024], rgb [B, res, res, 3] as floats 0..255, matte [B, res, res] (0 / 1) → ctx.
    public func encodeCtx(_ tokens: MLXArray, rgb: MLXArray, matte: MLXArray) -> MLXArray {
        let ctx = image.encodeCond(tokens)
        let B = rgb.shape[0], n = rgb.shape[1] / 16
        func pool(_ x: MLXArray) -> MLXArray {   // [B, res, res, C] → mean over 16 × 16 patches [B, n · n, C]
            let c = x.shape[3]
            return mean(x.reshaped([B, n, 16, n, 16, c]), axes: [2, 4]).reshaped([B, n * n, c])
        }
        let x = rgb / 255, m = matte.expandedDimensions(axis: -1)
        let frac = pool(m)
        let st = concatenated([pool(x), frac, pool(x * m) / maximum(frac, MLXArray(Float(1e-3)))], axis: -1)
        return concatenated([ctx[0..., 0..<5], ctx[0..., 5...] + HDOps.lin(st, w, "patch")], axis: 1)
    }

    /// card tokens [B, N, 40] → the standardised colours [B, N, 48].
    public func raw(_ x: MLXArray, _ ctx: MLXArray) -> MLXArray {
        let dit = image.dit
        let c = dit.conditioning([Float](repeating: 0, count: x.shape[0]), ctx)
        var h = HDOps.lin(x, dit.w, "inp")
        for i in 0..<dit.depth { h = dit.block(i, h, c, ctx) }
        let m = HDOps.lin(HDOps.siLU(c), w, "col_ada").split(parts: 2, axis: -1)
        return HDOps.lin(SetDiT.modulate(HDOps.layerNormPlain(h, eps: 1e-6), shift: m[0], scale: m[1]), w, "col_out")
    }

    /// card tokens [B, N, 40] → CIELAB [B, N, 16, 3].
    public func callAsFunction(_ x: MLXArray, _ ctx: MLXArray) -> MLXArray {
        let y = raw(x, ctx)
        return y.reshaped([y.shape[0], y.shape[1], Self.bins, 3]) * sd + mu
    }

    /// CIELAB (D65) → sRGB 0..1, clipped, as `skimage.color.lab2rgb`.
    public static func labToSRGB(_ lab: [Double]) -> [Double] {
        let fy = (lab[0] + 16) / 116
        let fx = lab[1] / 500 + fy
        let fz = max(fy - lab[2] / 200, 0)
        func inv(_ f: Double) -> Double { f > 0.2068966 ? f * f * f : (f - 16.0 / 116) / 7.787 }
        let x = inv(fx) * 0.95047, y = inv(fy), z = inv(fz) * 1.08883
        let lin = [3.2404813432005266 * x - 1.5371515162713185 * y - 0.4985363261688878 * z,
                   -0.9692549499965682 * x + 1.8759900014898907 * y + 0.0415559265582928 * z,
                   0.0556466391351772 * x - 0.2040413383665112 * y + 1.0573110696453443 * z]
        return lin.map { v in min(max(v > 0.0031308 ? 1.055 * pow(v, 1 / 2.4) - 0.055 : 12.92 * v, 0), 1) }
    }
}

extension CardTokenizer {
    /// `clean_tokens`: generated tokens in the training-time format — slots with existence ≤ 0 zeroed,
    /// existence and card class snapped to ±1, the root normal unit length. tokens: N × 40 row-major.
    public func clean(_ tokens: [Float]) -> [Float] {
        let D = dim, z = zdim
        var y = tokens
        for s in 0..<(tokens.count / D) {
            let o = s * D
            guard tokens[o + D - 1] > 0 else {
                for a in 0..<(D - 1) { y[o + a] = 0 }
                y[o + D - 1] = -1
                continue
            }
            let n = (0..<3).map { tokens[o + 3 + z + $0] }
            let len = max((n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).squareRoot(), 1e-6)
            for a in 0..<3 { y[o + 3 + z + a] = n[a] / len }
            let c = tokens[o + 6 + z]
            y[o + 6 + z] = c > 0 ? 1 : (c < 0 ? -1 : 1)
            y[o + D - 1] = 1
        }
        return y
    }

    /// `sign_token_normals`: cleaned tokens with the root normal of every existing card signed by the
    /// rule of the training tokens (flipped where it points towards the mean card root).
    public func signNormals(_ tokens: [Float]) -> [Float] {
        let D = dim, z = zdim
        var y = tokens
        let slots = (0..<(tokens.count / D)).filter { tokens[$0 * D + D - 1] > 0 }
        guard !slots.isEmpty else { return y }
        let roots = slots.map { s in (0..<3).map { Double(tokens[s * D + $0] * rootSd[$0] + rootMu[$0]) } }
        var mean = [0.0, 0.0, 0.0]
        for r in roots { for a in 0..<3 { mean[a] += r[a] / Double(roots.count) } }
        for (i, s) in slots.enumerated() {
            var d = 0.0
            for a in 0..<3 { d += Double(tokens[s * D + 3 + z + a]) * (roots[i][a] - mean[a]) }
            if d < 0 { for a in 0..<3 { y[s * D + 3 + z + a] = -tokens[s * D + 3 + z + a] } }
        }
        return y
    }
}
