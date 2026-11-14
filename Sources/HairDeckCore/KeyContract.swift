import Foundation

/// The key contract of the three released checkpoints of the point-cloud path: the key sets this port
/// reads, generated from the architecture, against the actual safetensors headers (pure Foundation,
/// no MLX). A loader that finds a missing key throws instead of running with a partial model.
public enum KeyContract {
    public enum ContractError: Error, CustomStringConvertible {
        case mismatch(String)
        public var description: String { if case .mismatch(let s) = self { return s }; return "" }
    }

    static func block(_ p: String, lns: Int, attn: [String]) -> [String] {
        var k: [String] = []
        for i in 1...lns { k += ["\(p).ln_\(i).weight", "\(p).ln_\(i).bias"] }
        k += attn.map { "\(p).attn.\($0).weight" }
        k += ["\(p).attn.c_proj.weight", "\(p).attn.c_proj.bias"]
        for m in ["c_fc", "c_proj"] { k += ["\(p).mlp.\(m).weight", "\(p).mlp.\(m).bias"] }
        return k
    }

    /// `encoder.*` of Dora-VAE 1.1 (122 keys).
    public static var doraEncoder: Set<String> {
        var k: [String] = []
        for p in ["encoder.input_proj", "encoder.input_proj1", "encoder.ln_post"] { k += ["\(p).weight", "\(p).bias"] }
        for p in ["encoder.cross_attn", "encoder.cross_attn1"] { k += block(p, lns: 3, attn: ["c_q", "c_kv"]) }
        for i in 0..<Dora.selfBlocks { k += block("encoder.self_attn.resblocks.\(i)", lns: 2, attn: ["c_qkv"]) }
        return Set(k)
    }

    /// The point-cloud set DiT (235 keys).
    public static func setDiT(depth: Int = 12) -> Set<String> {
        var k = ["null_ctx"]
        for p in ["inp", "cond_norm", "cond_proj", "f_ada", "out", "t_mlp.0", "t_mlp.2", "pool_mlp.0", "pool_mlp.2"] {
            k += ["\(p).weight", "\(p).bias"]
        }
        for i in 0..<depth {
            let p = "blocks.\(i)"
            for a in ["attn", "xattn"] {
                k += ["\(p).\(a).q.weight", "\(p).\(a).kv.weight", "\(p).\(a).o.weight", "\(p).\(a).o.bias", "\(p).\(a).qn.w", "\(p).\(a).kn.w"]
            }
            for m in ["ada", "mlp.0", "mlp.2"] { k += ["\(p).\(m).weight", "\(p).\(m).bias"] }
        }
        return Set(k)
    }

    /// The keys of the card VAE that decoding reads (the encoder half of the file is not used).
    public static func cardVAEDecoder(depth: Int = 4) -> Set<String> {
        var k: [String] = []
        for p in ["from_z.0", "from_z.2", "n_head.0", "n_head.2", "q_in", "norm", "out"] { k += ["\(p).weight", "\(p).bias"] }
        for i in 0..<depth { for m in ["ada", "qkv", "o", "mlp.0", "mlp.2"] { k += ["dec.blocks.\(i).\(m).weight", "dec.blocks.\(i).\(m).bias"] } }
        return Set(k)
    }

    /// The keys of the card VAE that encoding reads.
    public static func cardVAEEncoder(depth: Int = 4) -> Set<String> {
        var k = ["cls", "pe_enc", "inp.weight", "inp.bias", "to_z.weight", "to_z.bias"]
        for i in 0..<depth {
            let p = "enc.layers.\(i)"
            k += ["\(p).self_attn.in_proj_weight", "\(p).self_attn.in_proj_bias"]
            for m in ["self_attn.out_proj", "linear1", "linear2", "norm1", "norm2"] { k += ["\(p).\(m).weight", "\(p).\(m).bias"] }
        }
        return Set(k)
    }

    /// The chain DiT: the set DiT's keys at depth 8 with the card condition; the file also holds the
    /// base class's unused `head_norm` / `head_proj`.
    public static var chainDiT: Set<String> {
        setDiT(depth: 8).union(["card_norm.weight", "card_norm.bias", "card_proj.weight", "card_proj.bias", "card_pos",
                                "head_norm.weight", "head_norm.bias", "head_proj.weight", "head_proj.bias"])
    }

    public static func check(chain: String, vae: String) throws {
        let c = try SafetensorsHeader.keys(chain)
        guard c == chainDiT else { throw ContractError.mismatch("chain DiT keys differ: \(c.symmetricDifference(chainDiT).sorted().prefix(6))") }
        let missing = cardVAEEncoder().subtracting(try SafetensorsHeader.keys(vae))
        guard missing.isEmpty else { throw ContractError.mismatch("card VAE encoder keys missing: \(missing.sorted().prefix(6))") }
    }

    /// Throws unless every key this port reads is in the files, the DiT file holds exactly the
    /// expected set, and the Dora file's `encoder.*` keys are exactly the expected set.
    public static func check(dora: String, dit: String, vae: String) throws {
        let d = try SafetensorsHeader.keys(dora).filter { $0.hasPrefix("encoder.") }
        guard d == doraEncoder else { throw ContractError.mismatch("Dora encoder keys differ: \(d.symmetricDifference(doraEncoder).sorted().prefix(6))") }
        let t = try SafetensorsHeader.keys(dit)
        guard t == setDiT() else { throw ContractError.mismatch("set DiT keys differ: \(t.symmetricDifference(setDiT()).sorted().prefix(6))") }
        let v = try SafetensorsHeader.keys(vae)
        let missing = cardVAEDecoder().subtracting(v)
        guard missing.isEmpty else { throw ContractError.mismatch("card VAE keys missing: \(missing.sorted().prefix(6))") }
    }
}
