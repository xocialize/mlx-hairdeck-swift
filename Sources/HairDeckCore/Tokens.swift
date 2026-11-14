import Foundation
import MLX

/// Card tokens of the set DiTs (`hairdeck/models/tokenizer.py`, `hairdeck/cards/tokens.py`):
/// [root (3), card-VAE latent (32, standardised), root width direction (3), shape class (±1),
/// existence (±1)] = 40 values. Roots are standardised with `data/token_stats.json`, latents with
/// `ckpt/card_vae/latent_stats.json` (the dataset-frame statistics of the point-cloud model).
public struct CardTokenizer {
    public let rootMu: [Float], rootSd: [Float], latentMu: [Float], latentSd: [Float]
    public var zdim: Int { latentMu.count }
    public var dim: Int { 3 + zdim + 3 + 2 }

    public init(rootMu: [Float], rootSd: [Float], latentMu: [Float], latentSd: [Float]) {
        self.rootMu = rootMu
        self.rootSd = rootSd
        self.latentMu = latentMu
        self.latentSd = latentSd
    }

    public struct Decoded {
        /// [B, N, 3], [B, N, zdim], [B, N, 3]
        public var root, z, nrm: MLXArray
        /// B × N flags
        public var shape, exist: [[Bool]]
    }

    public func decode(_ tok: MLXArray) -> Decoded {
        let B = tok.shape[0], N = tok.shape[1]
        let root = tok[.ellipsis, 0..<3] * MLXArray(rootSd) + MLXArray(rootMu)
        let z = tok[.ellipsis, 3..<(3 + zdim)] * MLXArray(latentSd) + MLXArray(latentMu)
        let nrm = tok[.ellipsis, (3 + zdim)..<(6 + zdim)]
        let tail = tok[.ellipsis, (6 + zdim)...].asArray(Float.self)
        var shape = [[Bool]](), exist = [[Bool]]()
        for b in 0..<B {
            shape.append((0..<N).map { tail[(b * N + $0) * 2] > 0 })
            exist.append((0..<N).map { tail[(b * N + $0) * 2 + 1] > 0 })
        }
        return Decoded(root: root, z: z, nrm: nrm, shape: shape, exist: exist)
    }
}

/// One hair card: n rows of xyz, width, thickness and the unit root width direction (8 columns), and
/// its cross-section class.
public struct HairCard {
    public var seq: [[Double]]
    public var triangle: Bool
}

public enum Cards {
    /// `tokens_to_cards` (root_dir = true): tokens [B, N, 40] → per hairstyle the cards of its existing slots.
    public static func fromTokens(_ tok: MLXArray, tokenizer: CardTokenizer, vae: CardVAE) -> [[HairCard]] {
        let d = tokenizer.decode(tok)
        let N = tok.shape[1], zdim = tokenizer.zdim
        let root = d.root.asArray(Float.self), nrm = d.nrm.asArray(Float.self)
        var out: [[HairCard]] = []
        for b in 0..<tok.shape[0] {
            let js = (0..<N).filter { d.exist[b][$0] }
            guard !js.isEmpty else { out.append([]); continue }
            let zb = d.z[b].take(MLXArray(js.map { Int32($0) }), axis: 0).reshaped([js.count, zdim])
            let roots = js.map { j in (0..<3).map { Double(root[(b * N + j) * 3 + $0]) } }
            let seqs = vae.decodeCards(zb, roots: roots)
            var cards: [HairCard] = []
            for (k, j) in js.enumerated() {
                var n = (0..<3).map { Double(nrm[(b * N + j) * 3 + $0]) }
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).squareRoot() + 1e-12
                n = n.map { $0 / len }
                cards.append(HairCard(seq: seqs[k].map { $0 + n }, triangle: d.shape[b][j]))
            }
            out.append(cards)
        }
        return out
    }

    /// CHARM-style json: [{"shape": "square" | "triangle", "seq": rows}, ...] (`cards_to_json`).
    public static func json(_ cards: [HairCard]) throws -> Data {
        let obj = cards.map { ["shape": $0.triangle ? "triangle" : "square", "seq": $0.seq] as [String: Any] }
        return try JSONSerialization.data(withJSONObject: obj)
    }
}
