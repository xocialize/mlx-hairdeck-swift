import Foundation
import MLX

/// DINOv3 ViT-L/16 image encoder (facebook/dinov3-vitl16-pretrain-lvd1689m, DINOv3 License; weights in
/// the transformers layout, loaded unchanged), as HairDeck's `DinoEncoder` runs it
/// (`hairdeck/models/imgmodel.py`). Lifted from mlx-trellis2-swift's `DINOv3.swift` (same author) and
/// changed where HairDeck's use of the model differs from TRELLIS.2's:
///   · the output is `forward_features`' normed tokens: the model's own final LayerNorm (with its
///     weight and bias) on CLS, the 4 storage tokens and the patch tokens;
///   · the RoPE periods are those of Meta's original checkpoint, rounded to bfloat16;
///   · the tokens are rounded to bfloat16, as the encoder hands them on;
///   · explicit ops instead of `MLXFast`, so the CPU-stream parity gate stays on the CPU.
///
/// forward: patch conv (16 × 16, stride 16) → [cls, 4 storage tokens, patches] → 24 blocks (affine
/// LayerNorm eps 1e-5 → attention with axial RoPE on the patch tokens, no key bias → LayerScale;
/// affine LayerNorm → erf-GELU MLP → LayerScale) → final LayerNorm. 448 px input: 789 × 1024 tokens.
public struct DINOv3 {
    public static let hidden = 1024, heads = 16, headDim = 64, layers = 24, patch = 16, prefix = 5
    static let mean: [Float] = [0.485, 0.456, 0.406], std: [Float] = [0.229, 0.224, 0.225]
    let w: [String: MLXArray]
    let patchW: MLXArray

    public init(weights: [String: MLXArray]) {
        w = weights
        // conv weight [Cout, Cin, kH, kW] → MLX conv2d's [Cout, kH, kW, Cin]
        patchW = weights["embeddings.patch_embeddings.weight"]!.transposed(0, 2, 3, 1)
    }

    /// The keys this encoder reads from the transformers file.
    public static var keys: Set<String> {
        var k = ["embeddings.cls_token", "embeddings.register_tokens", "embeddings.patch_embeddings.weight", "embeddings.patch_embeddings.bias",
                 "norm.weight", "norm.bias"]
        for i in 0..<layers {
            let p = "layer.\(i)."
            for n in ["norm1", "norm2", "attention.o_proj", "attention.q_proj", "attention.v_proj", "mlp.up_proj", "mlp.down_proj"] {
                k += [p + n + ".weight", p + n + ".bias"]
            }
            k += [p + "attention.k_proj.weight", p + "layer_scale1.lambda1", p + "layer_scale2.lambda1"]
        }
        return Set(k)
    }

    /// RoPE periods: 100^(2i / 32), i = 0..15, rounded to bfloat16 (the buffer of the original checkpoint).
    static var periods: [Float] {
        (0..<(headDim / 4)).map { i in
            let p = powf(100.0, 2 * Float(i) / Float(headDim / 2))
            // round to nearest bfloat16: keep the top 16 bits of the float32, ties to even
            let b = p.bitPattern
            let r = (b &+ 0x7FFF &+ ((b >> 16) & 1)) & 0xFFFF_0000
            return Float(bitPattern: r)
        }
    }

    /// (sin, cos) of the axial RoPE angles, each [nH · nW, 64]: per patch centre in [−1, 1] the y
    /// angles then the x angles (2π · coordinate / period), the 32 values repeated twice.
    public static func rope(_ nH: Int, _ nW: Int) -> (sin: MLXArray, cos: MLXArray) {
        let per = periods, nf = per.count
        var s = [Float](repeating: 0, count: nH * nW * headDim), c = s
        for h in 0..<nH {
            let cy = 2 * ((Float(h) + 0.5) / Float(nH)) - 1
            for x in 0..<nW {
                let cx = 2 * ((Float(x) + 0.5) / Float(nW)) - 1
                let o = (h * nW + x) * headDim
                for t in 0..<2 {
                    for f in 0..<nf {
                        let ay = 2 * Float.pi * cy / per[f], ax = 2 * Float.pi * cx / per[f]
                        s[o + t * 2 * nf + f] = sinf(ay); c[o + t * 2 * nf + f] = cosf(ay)
                        s[o + t * 2 * nf + nf + f] = sinf(ax); c[o + t * 2 * nf + nf + f] = cosf(ax)
                    }
                }
            }
        }
        return (MLXArray(s, [nH * nW, headDim]), MLXArray(c, [nH * nW, headDim]))
    }

    /// uint8 pixels as floats 0..255, [B, H, W, 3] → normalised [B, H, W, 3].
    public static func normalise(_ rgb: MLXArray) -> MLXArray {
        (rgb / 255 - MLXArray(mean)) / MLXArray(std)
    }

    /// Normalised pixels [B, H, W, 3] → [B, 5 + patches, 1024].
    public func embed(_ pixels: MLXArray) -> MLXArray {
        let B = pixels.shape[0], nH = pixels.shape[1] / Self.patch, nW = pixels.shape[2] / Self.patch
        let pe = conv2d(pixels, patchW, stride: IntOrPair(Self.patch), padding: IntOrPair(0)).reshaped([B, nH * nW, Self.hidden])
            + w["embeddings.patch_embeddings.bias"]!
        let cls = broadcast(w["embeddings.cls_token"]!.reshaped([1, 1, Self.hidden]), to: [B, 1, Self.hidden])
        let reg = broadcast(w["embeddings.register_tokens"]!.reshaped([1, 4, Self.hidden]), to: [B, 4, Self.hidden])
        return concatenated([cls, reg, pe], axis: 1)
    }

    /// RoPE on the patch tokens only: x [B, heads, L, 64].
    func applyRope(_ x: MLXArray, _ sin: MLXArray, _ cos: MLXArray) -> MLXArray {
        let L = x.shape[2], half = Self.headDim / 2
        let pre = x[0..., 0..., 0..<Self.prefix, 0...], p = x[0..., 0..., Self.prefix..<L, 0...]
        let rot = concatenated([-p[.ellipsis, half...], p[.ellipsis, 0..<half]], axis: -1)
        return concatenated([pre, p * cos + rot * sin], axis: 2)
    }

    public func block(_ i: Int, _ x: MLXArray, _ sin: MLXArray, _ cos: MLXArray) -> MLXArray {
        let p = "layer.\(i)."
        let h = HDOps.ln(x, w, p + "norm1")
        let q = applyRope(HDOps.splitHeads(HDOps.lin(h, w, p + "attention.q_proj"), heads: Self.heads), sin, cos)
        let k = applyRope(HDOps.splitHeads(HDOps.linear(h, w[p + "attention.k_proj.weight"]!), heads: Self.heads), sin, cos)
        let v = HDOps.splitHeads(HDOps.lin(h, w, p + "attention.v_proj"), heads: Self.heads)
        let y = x + HDOps.lin(HDOps.mergeHeads(HDOps.sdpa(q, k, v)), w, p + "attention.o_proj") * w[p + "layer_scale1.lambda1"]!
        let m = HDOps.lin(HDOps.geluErf(HDOps.lin(HDOps.ln(y, w, p + "norm2"), w, p + "mlp.up_proj")), w, p + "mlp.down_proj")
        return y + m * w[p + "layer_scale2.lambda1"]!
    }

    /// Normalised pixels [B, H, W, 3] → the normed tokens [B, 5 + patches, 1024], rounded to bfloat16.
    public func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let (s, c) = Self.rope(pixels.shape[1] / Self.patch, pixels.shape[2] / Self.patch)
        var h = embed(pixels)
        for i in 0..<Self.layers {
            h = block(i, h, s, c)
            if i % 6 == 5 { eval(h) }
        }
        return HDOps.ln(h, w, "norm").asType(.bfloat16).asType(.float32)
    }
}
