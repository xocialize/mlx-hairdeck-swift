import CoreGraphics
import Foundation
import ImageIO
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

    /// An image file as 8-bit RGBA rows, top to bottom (sRGB values as stored; alpha not premultiplied).
    public static func readRGBA(_ path: String) throws -> (pixels: [UInt8], width: Int, height: Int) {
        guard let d = FileManager.default.contents(atPath: path) else { throw IOError.bad("cannot read image \(path)") }
        return try readRGBA(data: d, label: path)
    }

    /// As `readRGBA(_:)`, from encoded image bytes (PNG, JPEG, …). The alpha comes back straight
    /// (un-premultiplied), so a cut-out picture's alpha can serve as its matte.
    public static func readRGBA(data: Data, label path: String = "<data>") throws -> (pixels: [UInt8], width: Int, height: Int) {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw IOError.bad("cannot read image \(path)") }
        let w = img.width, h = img.height
        var px = [UInt8](repeating: 0, count: w * h * 4)
        // draw into the image's own colour space, so the stored values come back unchanged
        let space = img.colorSpace.flatMap { $0.model == .rgb ? $0 : nil } ?? CGColorSpace(name: CGColorSpace.sRGB)!
        let ok = px.withUnsafeMutableBytes { buf -> Bool in
            guard let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: space,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            ctx.interpolationQuality = .none
            ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard ok else { throw IOError.bad("cannot decode image \(path)") }
        // the context holds premultiplied values: back to straight alpha
        for i in 0..<(w * h) {
            let a = Int(px[i * 4 + 3])
            if a != 0 && a != 255 { for c in 0..<3 { px[i * 4 + c] = UInt8(min(255, (Int(px[i * 4 + c]) * 255 + a / 2) / a)) } }
        }
        return (px, w, h)
    }

    static func floats(_ any: Any?) -> [Float] { (any as? [Any] ?? []).map { ($0 as! NSNumber).floatValue } }

    static func jsonObject(_ path: String) throws -> [String: Any] {
        guard let d = FileManager.default.contents(atPath: path),
              let o = try JSONSerialization.jsonObject(with: d) as? [String: Any] else { throw IOError.bad("cannot read json \(path)") }
        return o
    }
}

/// Where the released files are: a HairDeck workspace (the layout of its `scripts/setup/download_weights.py`)
/// or the consolidated snapshot of `tools/build_weights_repo.py` (flat files + `config.json`; DINOv3 is the
/// `dino.safetensors` of the TRELLIS.2 snapshot, given separately).
public enum HairDeckSource {
    case workspace(String)
    case snapshot(String, dino: String?)

    struct Files {
        var dora, pointCloud, image, color, chain, vae: String
        var dino: String?
    }

    var files: Files {
        switch self {
        case .workspace(let r):
            return Files(dora: "\(r)/third_party/weights/dora_vae_1_1/dora_vae_1_1_shape_model.safetensors",
                         pointCloud: "\(r)/ckpt/pointcloud_model/model.safetensors", image: "\(r)/ckpt/image_model/model.safetensors",
                         color: "\(r)/ckpt/color_model/model.safetensors", chain: "\(r)/ckpt/chain_model/model.safetensors",
                         vae: "\(r)/ckpt/card_vae/model.safetensors", dino: "\(r)/third_party/weights/dinov3/model.safetensors")
        case .snapshot(let d, let dino):
            return Files(dora: "\(d)/dora_encoder.safetensors", pointCloud: "\(d)/pointcloud_dit.safetensors", image: "\(d)/image_dit.safetensors",
                         color: "\(d)/color_dit.safetensors", chain: "\(d)/chain_dit.safetensors", vae: "\(d)/card_vae.safetensors", dino: dino)
        }
    }

    /// One of the statistics objects: `card_stats`, `latent_stats`, `latent_stats_head`, `token_stats`,
    /// `token_stats_head`, `chain_stats_head`.
    func stats(_ name: String) throws -> [String: Any] {
        switch self {
        case .workspace(let r):
            let paths = ["card_stats": "ckpt/card_vae/stats.json", "latent_stats": "ckpt/card_vae/latent_stats.json",
                         "latent_stats_head": "ckpt/card_vae/latent_stats_head.json", "token_stats": "data/token_stats.json",
                         "token_stats_head": "data/token_stats_head.json", "chain_stats_head": "data/spring/chain_stats_head.json"]
            return try HDIO.jsonObject("\(r)/\(paths[name]!)")
        case .snapshot(let d, _):
            guard let o = try HDIO.jsonObject("\(d)/config.json")[name] as? [String: Any] else { throw HDIO.IOError.bad("\(d)/config.json: no \(name)") }
            return o
        }
    }

    func vae() throws -> CardVAE {
        let vs = try stats("card_stats")
        return CardVAE(weights: try Weights.load(files.vae), mu: HDIO.floats(vs["mu"]), sd: HDIO.floats(vs["sd"]))
    }

    func tokenizer(head: Bool) throws -> CardTokenizer {
        let ts = try stats(head ? "token_stats_head" : "token_stats"), ls = try stats(head ? "latent_stats_head" : "latent_stats")
        return CardTokenizer(rootMu: HDIO.floats(ts["root_mu"]), rootSd: HDIO.floats(ts["root_sd"]), latentMu: HDIO.floats(ls["mu"]), latentSd: HDIO.floats(ls["sd"]))
    }
}

/// The released files of the point-cloud path in a HairDeck workspace (the layout of
/// `scripts/setup/download_weights.py`), loaded once.
public struct PointCloudModel {
    public let dora: [String: MLXArray]
    public let dit: SetDiT
    public let vae: CardVAE
    public let tokenizer: CardTokenizer

    public init(workspace root: String) throws { try self.init(source: .workspace(root)) }

    public init(source: HairDeckSource) throws {
        let f = source.files
        try KeyContract.check(dora: f.dora, dit: f.pointCloud, vae: f.vae)
        // only the encoder half of the Dora checkpoint is used
        dora = try Weights.load(f.dora).filter { $0.key.hasPrefix("encoder.") }
        dit = SetDiT(weights: try Weights.load(f.pointCloud))
        vae = try source.vae()
        tokenizer = try source.tokenizer(head: false)
    }
}

/// The released files of the chain model in a HairDeck workspace: the chain DiT, the card VAE (both
/// halves) and the head-frame statistics.
public struct ChainModel {
    public let chain: ChainDiT
    public let vae: CardVAE
    /// The head-frame card tokenizer (`data/token_stats_head.json`, `latent_stats_head.json`).
    public let tokenizer: CardTokenizer

    public init(workspace root: String) throws { try self.init(source: .workspace(root)) }

    public init(source: HairDeckSource) throws {
        let f = source.files
        try KeyContract.check(chain: f.chain, vae: f.vae)
        let cs = try source.stats("chain_stats_head")
        let doubles: (Any?) -> [Double] = { ($0 as? [Any] ?? []).map { ($0 as! NSNumber).doubleValue } }
        chain = ChainDiT(weights: try Weights.load(f.chain), mu: doubles(cs["mu"]), sd: doubles(cs["sd"]))
        vae = try source.vae()
        tokenizer = try source.tokenizer(head: true)
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

/// The released files of the image path in a HairDeck workspace: DINOv3, the image DiT, the colour
/// model, the chain model, the card VAE and the head-frame statistics.
public struct ImagePathModel {
    public let dino: DINOv3
    public let image: ImageDiT
    public let color: ColorDiT
    public let chains: ChainModel
    public var vae: CardVAE { chains.vae }
    public var tokenizer: CardTokenizer { chains.tokenizer }

    public init(workspace root: String) throws { try self.init(source: .workspace(root)) }

    public init(source: HairDeckSource) throws {
        let f = source.files
        guard let dinoPath = f.dino else { throw HDIO.IOError.bad("the image path needs the DINOv3 weights file") }
        let missing = DINOv3.keys.subtracting(try SafetensorsHeader.keys(dinoPath))
        guard missing.isEmpty else { throw KeyContract.ContractError.mismatch("DINOv3 keys missing: \(missing.sorted().prefix(6))") }
        dino = DINOv3(weights: try Weights.load(dinoPath).filter { DINOv3.keys.contains($0.key) })
        let want = KeyContract.setDiT().union(["ctx_pos"])
        let ik = try SafetensorsHeader.keys(f.image)
        guard ik == want else { throw KeyContract.ContractError.mismatch("image DiT keys differ: \(ik.symmetricDifference(want).sorted().prefix(6))") }
        let wantCol = Set(want.map { "dit." + $0 }).union(["patch.weight", "patch.bias", "col_ada.weight", "col_ada.bias", "col_out.weight", "col_out.bias", "mu", "sd"])
        let ck = try SafetensorsHeader.keys(f.color)
        guard ck == wantCol else { throw KeyContract.ContractError.mismatch("colour model keys differ: \(ck.symmetricDifference(wantCol).sorted().prefix(6))") }
        image = ImageDiT(weights: try Weights.load(f.image))
        color = ColorDiT(weights: try Weights.load(f.color))
        chains = try ChainModel(source: source)
    }
}
