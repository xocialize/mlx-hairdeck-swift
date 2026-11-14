import Foundation
import MLX

/// File inputs and outputs of the point-cloud path that are not model weights.
public enum HDIO {
    public enum IOError: Error, CustomStringConvertible {
        case bad(String)
        public var description: String { if case .bad(let s) = self { return s }; return "" }
    }

    /// `condition_points` for the released clouds: hair_reform.ply (binary little endian, double
    /// x y z nx ny nz + uchar rgb) → N × 6 row-major fp32 with positions × 2 (dataset frame → about
    /// [-1, 1]). The released condition takes exactly 10,000 points; other counts are refused here
    /// (the reference resamples them with numpy's PCG64, which this port does not reproduce).
    public static func conditionPoints(ply path: String, expect: Int = 10_000) throws -> (points: [Float], count: Int) {
        guard let raw = FileManager.default.contents(atPath: path) else { throw IOError.bad("cannot read \(path)") }
        let marker = Data("end_header\n".utf8)
        guard let r = raw.range(of: marker) else { throw IOError.bad("\(path): no PLY header") }
        let header = String(decoding: raw[..<r.upperBound], as: UTF8.self)
        guard header.contains("format binary_little_endian") else { throw IOError.bad("\(path): not binary little endian") }
        let lines = header.split(separator: "\n").map(String.init)
        let props = lines.filter { $0.hasPrefix("property") }.map { $0.split(separator: " ").map(String.init) }
        guard props.map({ $0.last! }) == ["x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"],
              props.prefix(6).allSatisfy({ $0[1] == "double" }), props.suffix(3).allSatisfy({ $0[1] == "uchar" })
        else { throw IOError.bad("\(path): unexpected PLY properties") }
        guard let vline = lines.first(where: { $0.hasPrefix("element vertex") }), let n = Int(vline.split(separator: " ").last!)
        else { throw IOError.bad("\(path): no vertex count") }
        guard n == expect else { throw IOError.bad("\(path): \(n) points, expected \(expect)") }
        let stride = 6 * 8 + 3
        let start = r.upperBound
        guard raw.count >= start + n * stride else { throw IOError.bad("\(path): truncated") }
        var out = [Float](repeating: 0, count: n * 6)
        raw.withUnsafeBytes { buf in
            for i in 0..<n {
                for c in 0..<6 {
                    let v = buf.loadUnaligned(fromByteOffset: start + i * stride + c * 8, as: Double.self)
                    out[i * 6 + c] = Float(c < 3 ? v * 2.0 : v)
                }
            }
        }
        return (out, n)
    }

    /// NumPy `.npy` v1.0 writer, float32, C order.
    public static func writeNPY(_ a: MLXArray, to path: String) throws {
        let shape = a.shape
        let vals = a.asType(.float32).asArray(Float.self)
        var dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (\(shape.map(String.init).joined(separator: ", "))\(shape.count == 1 ? "," : "")), }"
        let pad = 64 - ((10 + dict.utf8.count + 1) % 64)
        dict += String(repeating: " ", count: pad % 64) + "\n"
        var data = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 1, 0])
        let len = UInt16(dict.utf8.count)
        data.append(contentsOf: [UInt8(len & 0xff), UInt8(len >> 8)])
        data.append(Data(dict.utf8))
        vals.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        try data.write(to: URL(fileURLWithPath: path))
    }

    static func floats(_ any: Any?) -> [Float] { (any as? [Any] ?? []).map { ($0 as! NSNumber).floatValue } }

    static func jsonObject(_ path: String) throws -> [String: Any] {
        guard let d = FileManager.default.contents(atPath: path),
              let o = try JSONSerialization.jsonObject(with: d) as? [String: Any] else { throw IOError.bad("cannot read json \(path)") }
        return o
    }
}

/// The released files of the point-cloud path in a HairDeck workspace (the layout of
/// `scripts/setup/download_weights.py`), loaded once.
public struct PointCloudModel {
    public let dora: [String: MLXArray]
    public let dit: SetDiT
    public let vae: CardVAE
    public let tokenizer: CardTokenizer

    public static func paths(_ root: String) -> (dora: String, dit: String, vae: String, vaeStats: String, latentStats: String, tokenStats: String) {
        ("\(root)/third_party/weights/dora_vae_1_1/dora_vae_1_1_shape_model.safetensors",
         "\(root)/ckpt/pointcloud_model/model.safetensors", "\(root)/ckpt/card_vae/model.safetensors",
         "\(root)/ckpt/card_vae/stats.json", "\(root)/ckpt/card_vae/latent_stats.json", "\(root)/data/token_stats.json")
    }

    public init(workspace root: String) throws {
        let p = Self.paths(root)
        try KeyContract.check(dora: p.dora, dit: p.dit, vae: p.vae)
        // only the encoder half of the Dora checkpoint is used
        dora = try Weights.load(p.dora).filter { $0.key.hasPrefix("encoder.") }
        dit = SetDiT(weights: try Weights.load(p.dit))
        let vs = try HDIO.jsonObject(p.vaeStats), ls = try HDIO.jsonObject(p.latentStats), ts = try HDIO.jsonObject(p.tokenStats)
        vae = CardVAE(weights: try Weights.load(p.vae), mu: HDIO.floats(vs["mu"]), sd: HDIO.floats(vs["sd"]))
        tokenizer = CardTokenizer(rootMu: HDIO.floats(ts["root_mu"]), rootSd: HDIO.floats(ts["root_sd"]),
                                  latentMu: HDIO.floats(ls["mu"]), latentSd: HDIO.floats(ls["sd"]))
    }
}

/// The released files of the chain model in a HairDeck workspace: the chain DiT, the card VAE (both
/// halves) and the head-frame statistics.
public struct ChainModel {
    public let chain: ChainDiT
    public let vae: CardVAE
    /// The head-frame card tokenizer (`data/token_stats_head.json`, `latent_stats_head.json`).
    public let tokenizer: CardTokenizer

    public init(workspace root: String) throws {
        let ckpt = "\(root)/ckpt/chain_model/model.safetensors", vaePath = "\(root)/ckpt/card_vae/model.safetensors"
        try KeyContract.check(chain: ckpt, vae: vaePath)
        let cs = try HDIO.jsonObject("\(root)/data/spring/chain_stats_head.json")
        let doubles: (Any?) -> [Double] = { ($0 as? [Any] ?? []).map { ($0 as! NSNumber).doubleValue } }
        chain = ChainDiT(weights: try Weights.load(ckpt), mu: doubles(cs["mu"]), sd: doubles(cs["sd"]))
        let vs = try HDIO.jsonObject("\(root)/ckpt/card_vae/stats.json")
        vae = CardVAE(weights: try Weights.load(vaePath), mu: HDIO.floats(vs["mu"]), sd: HDIO.floats(vs["sd"]))
        let ts = try HDIO.jsonObject("\(root)/data/token_stats_head.json"), ls = try HDIO.jsonObject("\(root)/ckpt/card_vae/latent_stats_head.json")
        tokenizer = CardTokenizer(rootMu: HDIO.floats(ts["root_mu"]), rootSd: HDIO.floats(ts["root_sd"]),
                                  latentMu: HDIO.floats(ls["mu"]), latentSd: HDIO.floats(ls["sd"]))
    }
}

extension Cards {
    /// CHARM-format card json (`load_cards`: cards with at least 3 finite points) → cards.
    public static func load(json path: String) throws -> [HairCard] {
        guard let d = FileManager.default.contents(atPath: path),
              let arr = try JSONSerialization.jsonObject(with: d) as? [[String: Any]] else { throw HDIO.IOError.bad("cannot read cards \(path)") }
        var out: [HairCard] = []
        for c in arr {
            guard let seq = c["seq"] as? [[NSNumber]], seq.count >= 3 else { continue }
            let rows = seq.map { $0.map { $0.doubleValue } }
            guard rows.allSatisfy({ $0.allSatisfy { $0.isFinite } }) else { continue }
            out.append(HairCard(seq: rows, triangle: (c["shape"] as? String) == "triangle"))
        }
        return out
    }
}
