import Foundation
import MLX

/// The card-level set DiT of HairDeck's point-cloud model: a 1:1 port of `hairdeck/models/setdit.py`
/// (`SetDiT.forward`) with `models/pc_dit.py` (`GenericCondDiT.encode_cond`), dict-functional on the
/// released weights (`ckpt/pointcloud_model/model.safetensors`, 235 keys, fp32).
///
/// Tokens x [B, 192, 40], no positional embedding. The condition is a token set ctx [B, 512, 768]
/// read by cross-attention in every block; its mean, through `pool_mlp`, is added to the timestep
/// embedding that drives adaLN. Attention: q and kv projections without bias, kv laid out as
/// [k | v] halves, RMSNorm on the per-head q and k (eps 1e-6); LayerNorms without affine, eps 1e-6;
/// tanh GELU; ada chunks (shift, scale, gate) × (self, cross, mlp).
public struct SetDiT {
    public let w: [String: MLXArray]
    public let depth: Int
    public let heads: Int

    public init(weights: [String: MLXArray], depth: Int = 12, heads: Int = 12) {
        self.w = weights
        self.depth = depth
        self.heads = heads
    }

    /// `timestep_embedding(t, 256)`: cat[cos, sin] of t · 1000 · exp(-ln(10000) · i / 128), in fp32 as
    /// the reference, evaluated on the CPU (arguments reach 1000 rad).
    public static func timestepEmbedding(_ t: [Float], dim: Int = 256) -> MLXArray {
        let half = dim / 2
        var out = [Float](repeating: 0, count: t.count * dim)
        for (b, tb) in t.enumerated() {
            for i in 0..<half {
                let f = expf(-logf(10000) * Float(i) / Float(half))
                let a = tb * 1000 * f
                out[b * dim + i] = cosf(a)
                out[b * dim + half + i] = sinf(a)
            }
        }
        return MLXArray(out, [t.count, dim])
    }

    func mlp2(_ x: MLXArray, _ p: String) -> MLXArray {
        HDOps.lin(HDOps.siLU(HDOps.lin(x, w, "\(p).0")), w, "\(p).2")
    }

    /// cond [B, 512, 768] (the Dora tokens) → ctx [B, 512, 768].
    public func encodeCond(_ cond: MLXArray) -> MLXArray {
        HDOps.lin(HDOps.ln(cond, w, "cond_norm"), w, "cond_proj")
    }

    /// The learned null condition, for classifier-free guidance: [1, 512, 768].
    public var nullCtx: MLXArray { w["null_ctx"]!.expandedDimensions(axis: 0) }

    func attention(_ x: MLXArray, _ ctx: MLXArray, _ p: String) -> MLXArray {
        let q = HDOps.splitHeads(HDOps.linear(x, w["\(p).q.weight"]!), heads: heads)
        let kv = HDOps.linear(ctx, w["\(p).kv.weight"]!).split(parts: 2, axis: -1)
        let k = HDOps.splitHeads(kv[0], heads: heads), v = HDOps.splitHeads(kv[1], heads: heads)
        let o = HDOps.sdpa(HDOps.rmsNorm(q, w["\(p).qn.w"]!), HDOps.rmsNorm(k, w["\(p).kn.w"]!), v)
        return HDOps.lin(HDOps.mergeHeads(o), w, "\(p).o")
    }

    static func modulate(_ x: MLXArray, shift: MLXArray, scale: MLXArray) -> MLXArray {
        x * (1 + scale.expandedDimensions(axis: 1)) + shift.expandedDimensions(axis: 1)
    }

    public func block(_ i: Int, _ x: MLXArray, _ c: MLXArray, _ ctx: MLXArray) -> MLXArray {
        let p = "blocks.\(i)"
        let m = HDOps.lin(HDOps.siLU(c), w, "\(p).ada").split(parts: 9, axis: -1)
        func gate(_ g: MLXArray) -> MLXArray { g.expandedDimensions(axis: 1) }
        var h = x
        let a = Self.modulate(HDOps.layerNormPlain(h, eps: 1e-6), shift: m[0], scale: m[1])
        h = h + gate(m[2]) * attention(a, a, "\(p).attn")
        h = h + gate(m[5]) * attention(Self.modulate(HDOps.layerNormPlain(h, eps: 1e-6), shift: m[3], scale: m[4]), ctx, "\(p).xattn")
        let f = Self.modulate(HDOps.layerNormPlain(h, eps: 1e-6), shift: m[6], scale: m[7])
        return h + gate(m[8]) * HDOps.lin(HDOps.geluTanh(HDOps.lin(f, w, "\(p).mlp.0")), w, "\(p).mlp.2")
    }

    /// c = t_mlp(timestep embedding) + pool_mlp(mean of ctx): [B, 768].
    public func conditioning(_ t: [Float], _ ctx: MLXArray) -> MLXArray {
        mlp2(Self.timestepEmbedding(t), "t_mlp") + mlp2(mean(ctx, axis: 1), "pool_mlp")
    }

    /// x [B, N, 40], t [B], ctx [B, 512, 768] (already the null condition where wanted) → v [B, N, 40].
    public func callAsFunction(_ x: MLXArray, _ t: [Float], _ ctx: MLXArray) -> MLXArray {
        let c = conditioning(t, ctx)
        var h = HDOps.lin(x, w, "inp")
        for i in 0..<depth { h = block(i, h, c, ctx) }
        let m = HDOps.lin(HDOps.siLU(c), w, "f_ada").split(parts: 2, axis: -1)
        return HDOps.lin(Self.modulate(HDOps.layerNormPlain(h, eps: 1e-6), shift: m[0], scale: m[1]), w, "out")
    }
}

/// Rectified-flow sampler of `hairdeck/sampling/flow.py` (`fm_sample`, Euler): from noise (t = 1) to
/// data (t = 0) in `steps` uniform steps, v = v_null + cfg (v_cond − v_null), both in one doubled batch.
public enum Flow {
    /// `torch.linspace(1, 0, steps + 1)` in fp32 (first half from the start, second half from the end).
    public static func timesteps(_ steps: Int) -> [Float] {
        let n = steps + 1
        let step = (Float(0) - Float(1)) / Float(n - 1)
        let half = n / 2
        return (0..<n).map { i in i < half ? Float(1) + Float(i) * step : Float(0) - Float(n - 1 - i) * step }
    }

    /// noise [B, N, 40], ctx [B, 512, 768] → tokens [B, N, 40]. `onStep` runs after every step.
    public static func sample(_ model: SetDiT, ctx: MLXArray, noise: MLXArray, steps: Int = 50, cfg: Float = 2.0,
                              onStep: ((Int) -> Void)? = nil) -> MLXArray {
        let B = noise.shape[0]
        let ts = timesteps(steps)
        let both = concatenated([ctx, broadcast(model.nullCtx, to: ctx.shape)], axis: 0)
        var x = noise
        for i in 0..<steps {
            let t = ts[i], tn = ts[i + 1]
            var v: MLXArray
            if cfg == 1.0 {
                v = model(x, [Float](repeating: t, count: B), ctx)
            } else {
                let o = model(concatenated([x, x], axis: 0), [Float](repeating: t, count: 2 * B), both).split(parts: 2, axis: 0)
                v = o[1] + cfg * (o[0] - o[1])
            }
            x = x + (tn - t) * v
            eval(x)
            onStep?(i)
        }
        return x
    }
}
