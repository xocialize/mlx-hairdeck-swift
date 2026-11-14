import Foundation
import MLX

/// Decoder side of HairDeck's card VAE (`hairdeck/cards/vae.py`: `CardVAE.decode` with the adaLN
/// decoder, `n_head`, and `LoadedVAE.decode_cards`), dict-functional on `ckpt/card_vae/model.safetensors`.
///
/// A card latent z (32) gives the card's point count (n_head, argmax + 1, at least 3) and, queried at
/// the uniform arc fractions u_j = j / (n − 1), its points as root-relative xyz, width and thickness in
/// the VAE's normalised units (`stats.json`). Released configuration: L 128, dim 256, depth 4, heads 8,
/// 6 Fourier frequencies (2^k π), `dec = ada`, `integ = false`.
public struct CardVAE {
    public let w: [String: MLXArray]
    public let L: Int
    public let depth: Int
    public let heads: Int
    public let numFreqs: Int
    /// `stats.json`: mean and standard deviation of the five card features.
    public let mu: [Float]
    public let sd: [Float]

    public init(weights: [String: MLXArray], mu: [Float], sd: [Float], L: Int = 128, depth: Int = 4, heads: Int = 8, numFreqs: Int = 6) {
        self.w = weights
        self.mu = mu
        self.sd = sd
        self.L = L
        self.depth = depth
        self.heads = heads
        self.numFreqs = numFreqs
    }

    /// `Fourier`: u [...] → [..., 1 + 2 nf] = u, sin(u 2^k π), cos(u 2^k π).
    func fourier(_ u: MLXArray) -> MLXArray {
        let f = MLXArray((0..<numFreqs).map { Float(pow(2.0, Double($0)) * Double.pi) })
        let a = u.expandedDimensions(axis: -1) * f
        return concatenated([u.expandedDimensions(axis: -1), HDOps.sinR(a), HDOps.cosR(a)], axis: -1)
    }

    func mlp2(_ x: MLXArray, _ p: String) -> MLXArray {
        HDOps.lin(HDOps.siLU(HDOps.lin(x, w, "\(p).0")), w, "\(p).2")
    }

    /// Point-count logits of every latent: z [C, 32] → [C, L].
    public func countLogits(_ z: MLXArray) -> MLXArray { mlp2(z, "n_head") }

    public func fromZ(_ z: MLXArray) -> MLXArray { mlp2(z, "from_z") }

    /// `AdaBlock`: pre-LN block whose LayerNorms (no affine, eps 1e-5) are modulated by the latent.
    func adaBlock(_ i: Int, _ x: MLXArray, _ c: MLXArray, _ bias: MLXArray) -> MLXArray {
        let p = "dec.blocks.\(i)"
        let m = HDOps.lin(HDOps.siLU(c), w, "\(p).ada").expandedDimensions(axis: 1).split(parts: 6, axis: -1)
        let (s1, sc1, g1, s2, sc2, g2) = (m[0], m[1], m[2], m[3], m[4], m[5])
        let qkv = HDOps.lin(HDOps.layerNormPlain(x, eps: 1e-5) * (1 + sc1) + s1, w, "\(p).qkv").split(parts: 3, axis: -1)
        let a = HDOps.sdpa(HDOps.splitHeads(qkv[0], heads: heads), HDOps.splitHeads(qkv[1], heads: heads),
                           HDOps.splitHeads(qkv[2], heads: heads), bias: bias)
        let y = x + g1 * HDOps.lin(HDOps.mergeHeads(a), w, "\(p).o")
        let f = HDOps.layerNormPlain(y, eps: 1e-5) * (1 + sc2) + s2
        return y + g2 * HDOps.lin(HDOps.geluErf(HDOps.lin(f, w, "\(p).mlp.0")), w, "\(p).mlp.2")
    }

    /// z [C, 32], u [C, L], mask [C, L] (1 for the card's points) → normalised features [C, L, 5].
    public func decode(_ z: MLXArray, u: MLXArray, mask: MLXArray) -> MLXArray {
        let c = fromZ(z)
        var h = HDOps.lin(fourier(u), w, "q_in") + c.expandedDimensions(axis: 1)
        // attention reads only the card's own points: additive bias 0 / -inf over the keys
        let keep = mask.expandedDimensions(axes: [1, 2])
        let bias = which(keep .> 0, MLXArray(Float(0)), MLXArray(-Float.infinity))
        for i in 0..<depth { h = adaBlock(i, h, c, bias) }
        return HDOps.lin(HDOps.ln(h, w, "norm"), w, "out")
    }

    /// Point counts and the decoder queries of `decode_cards`: n = max(argmax + 1, 3), u_j = j / (n − 1).
    public func queries(_ z: MLXArray) -> (ns: [Int], u: MLXArray, mask: MLXArray) {
        let ns = argMax(countLogits(z), axis: -1).asArray(Int32.self).map { max(Int($0) + 1, 3) }
        var u = [Float](repeating: 0, count: ns.count * L)
        var m = [Float](repeating: 0, count: ns.count * L)
        for (b, n) in ns.enumerated() {
            for j in 0..<n {
                u[b * L + j] = min(Float(j) / max(Float(n) - 1, 1), 1)
                m[b * L + j] = 1
            }
        }
        return (ns, MLXArray(u, [ns.count, L]), MLXArray(m, [ns.count, L]))
    }

    /// `LoadedVAE.decode_cards`: z [C, 32], roots (C × 3) → per card n × 5 rows (xyz in the frame of
    /// the roots, width, thickness; both clipped at 0), in Double as the reference.
    public func decodeCards(_ z: MLXArray, roots: [[Double]]) -> [[[Double]]] {
        let (ns, u, mask) = queries(z)
        let feat = decode(z, u: u, mask: mask) * MLXArray(sd) + MLXArray(mu)
        let flat = feat.asArray(Float.self)
        var out: [[[Double]]] = []
        for (b, n) in ns.enumerated() {
            var card: [[Double]] = []
            for j in 0..<n {
                let o = (b * L + j) * 5
                card.append([Double(flat[o]) + roots[b][0], Double(flat[o + 1]) + roots[b][1], Double(flat[o + 2]) + roots[b][2],
                             max(Double(flat[o + 3]), 0), max(Double(flat[o + 4]), 0)])
            }
            out.append(card)
        }
        return out
    }
}
