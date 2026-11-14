import Foundation
import MLX
import MLXNN

/// Shared low-level ops of the HairDeck port, lifted from mlx-skintokens-swift's `SkinOps` (MIT, same
/// author) and extended. Deliberately explicit (no `MLXFast.*`), so every op stays on the current
/// stream: the parity gates pin `.cpu`, where matmul + softmax + mean / rsqrt match PyTorch CPU fp32
/// closely; `MLXFast` attention and layer norm route to Metal even under a CPU default.
public enum HDOps {

    /// PyTorch `nn.Linear`: weight is `[out, in]`, so `y = x @ Wᵀ (+ b)`.
    public static func linear(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray? = nil) -> MLXArray {
        var y = matmul(x, w.swappedAxes(-1, -2))
        if let b { y = y + b }
        return y
    }

    /// `linear` with the weights of prefix `p` in a weights dictionary (`p.weight`, optional `p.bias`).
    public static func lin(_ x: MLXArray, _ w: [String: MLXArray], _ p: String) -> MLXArray {
        linear(x, w["\(p).weight"]!, w["\(p).bias"])
    }

    /// `nn.LayerNorm` over the last axis with weight and bias (PyTorch default eps 1e-5).
    public static func layerNorm(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray, eps: Float = 1e-5) -> MLXArray {
        layerNormPlain(x, eps: eps) * w + b
    }

    public static func ln(_ x: MLXArray, _ w: [String: MLXArray], _ p: String, eps: Float = 1e-5) -> MLXArray {
        layerNorm(x, w["\(p).weight"]!, w["\(p).bias"]!, eps: eps)
    }

    /// `nn.LayerNorm(elementwise_affine=False)`: biased variance over the last axis.
    public static func layerNormPlain(_ x: MLXArray, eps: Float) -> MLXArray {
        let mu = mean(x, axis: -1, keepDims: true)
        let xc = x - mu
        let v = mean(xc * xc, axis: -1, keepDims: true)
        return xc * rsqrt(v + eps)
    }

    /// HairDeck's `RMSNorm`: `x * rsqrt(mean(x², -1) + eps) * w`.
    public static func rmsNorm(_ x: MLXArray, _ w: MLXArray, eps: Float = 1e-6) -> MLXArray {
        x * rsqrt(mean(x * x, axis: -1, keepDims: true) + eps) * w
    }

    /// Exact (erf) GELU: `F.gelu` / `nn.GELU()` default.
    public static func geluErf(_ x: MLXArray) -> MLXArray { gelu(x) }

    /// `nn.GELU(approximate='tanh')`: 0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³))).
    public static func geluTanh(_ x: MLXArray) -> MLXArray {
        let c = Float((2.0 / Double.pi).squareRoot())
        return 0.5 * x * (1 + tanh(c * (x + 0.044715 * x * x * x)))
    }

    public static func siLU(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

    /// Scaled-dot-product attention. q: [B, H, Tq, Dh], k / v: [B, H, Tk, Dh]; scale 1/√Dh.
    /// `bias`: additive attention bias broadcastable to [B, H, Tq, Tk] (0 keep, -inf drop).
    public static func sdpa(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, bias: MLXArray? = nil) -> MLXArray {
        let scale = 1.0 / Float(q.shape.last!).squareRoot()
        var w = matmul(q, k.swappedAxes(-1, -2)) * scale
        if let bias { w = w + bias }
        return matmul(softmax(w, axis: -1), v)
    }

    /// x [B, T, C] → [B, H, T, C/H].
    public static func splitHeads(_ x: MLXArray, heads: Int) -> MLXArray {
        let s = x.shape
        return x.reshaped([s[0], s[1], heads, s[2] / heads]).swappedAxes(1, 2)
    }

    /// x [B, H, T, Dh] → [B, T, H*Dh].
    public static func mergeHeads(_ x: MLXArray) -> MLXArray {
        let s = x.shape
        return x.swappedAxes(1, 2).reshaped([s[0], s[2], s[1] * s[3]])
    }

    /// Range-reduced sin / cos (fp32): reduce the argument mod 2π first. High-frequency Fourier
    /// embedders push arguments past 100 rad, where MLX's fp32 range reduction diverges from PyTorch's.
    private static let twoPi: Float = 2.0 * .pi
    public static func sinR(_ x: MLXArray) -> MLXArray { sin(x - round(x / twoPi) * twoPi) }
    public static func cosR(_ x: MLXArray) -> MLXArray { cos(x - round(x / twoPi) * twoPi) }

    // MARK: parity metrics

    public static func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        let af = a.asType(.float32).reshaped([-1]), bf = b.asType(.float32).reshaped([-1])
        return (sum(af * bf) / (sqrt(sum(af * af)) * sqrt(sum(bf * bf)) + 1e-20)).item(Float.self)
    }

    public static func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }
}
