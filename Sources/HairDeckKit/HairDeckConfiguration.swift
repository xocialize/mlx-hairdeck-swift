import Foundation
import MLXToolKit

/// The injected person-matting hook for opaque pictures: canonical `Image` in, canonical `Matte` out
/// (soft alpha at the picture's size, white = the character). The app or plugin composes the shipped
/// BiRefNet `matting` capability here (the Trellis2 pattern) so this package takes no dependency on
/// another package. A picture that already has transparency needs none: its alpha is the matte.
public typealias HairDeckMatting = @Sendable (Image) async throws -> Matte

/// Init-time configuration (C9) of the HairDeck `characterPart` package.
public struct HairDeckConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// The consolidated HairDeck snapshot (the released float32 tensors under flat names + config.json).
    public var weightsRepo: String
    public var revision: String?
    /// The repository whose `dino.safetensors` is DINOv3 ViT-L/16 in the transformers layout.
    public var dinoRepo: String
    public var dinoRevision: String?
    /// The released weights are float32 and load unchanged; there is one tier.
    public var quant: Quant
    public var modelsRootDirectory: URL?
    /// A folder that already holds the snapshot's files flat (and, for the image path, `dino.safetensors`
    /// beside them or under `<dinoRepo>/`): resolved first, never touching the network.
    public var weightsRootOverride: URL?
    /// Matting for opaque pictures; nil → an opaque picture is refused with the reason.
    public var matting: HairDeckMatting?

    public static let defaultWeightsRepo = "xocialize/hairdeck-mlx"
    public static let defaultDinoRepo = "xocialize/trellis2-mlx"

    public init(weightsRepo: String = HairDeckConfiguration.defaultWeightsRepo, revision: String? = "main",
                dinoRepo: String = HairDeckConfiguration.defaultDinoRepo, dinoRevision: String? = "main",
                quant: Quant = .fp32, modelsRootDirectory: URL? = nil, weightsRootOverride: URL? = nil,
                matting: HairDeckMatting? = nil) {
        self.weightsRepo = weightsRepo
        self.revision = revision
        self.dinoRepo = dinoRepo
        self.dinoRevision = dinoRevision
        self.quant = quant
        self.modelsRootDirectory = modelsRootDirectory
        self.weightsRootOverride = weightsRootOverride
        self.matting = matting
    }

    // Codable: everything except the matting closure (which decodes to nil).
    enum CodingKeys: String, CodingKey { case weightsRepo, revision, dinoRepo, dinoRevision, quant, modelsRootDirectory, weightsRootOverride }
}

// MARK: - Weight sources

extension HairDeckConfiguration: WeightSourcing {
    /// The files of the HairDeck snapshot the image path loads. The probe is strict: a half-finished
    /// download reports missing.
    public static let snapshotFiles = ["config.json", "image_dit.safetensors", "color_dit.safetensors", "chain_dit.safetensors", "card_vae.safetensors"]
    public static let dinoFile = "dino.safetensors"

    public var weightSources: [WeightSource] {
        [WeightSource(role: "snapshot", repo: weightsRepo, revision: revision, matching: Self.snapshotFiles),
         WeightSource(role: "dinov3", repo: dinoRepo, revision: dinoRevision, matching: [Self.dinoFile, "DINOv3_LICENSE.md"])]
    }

    /// Where each source's files are on this machine, or nil: the override folder first, then the
    /// engine's store folder of the repository.
    public func directory(for source: WeightSource, storeRoot: URL?) -> URL? {
        let files = source.role == "dinov3" ? [Self.dinoFile] : Self.snapshotFiles
        func has(_ dir: URL) -> Bool { files.allSatisfy { FileManager.default.fileExists(atPath: dir.appending(path: $0).path) } }
        if let o = weightsRootOverride {
            for dir in [o, o.appending(path: source.repo)] where has(dir) { return dir }
        }
        if let dir = ModelStore(root: storeRoot).directory(for: source.repo), has(dir) { return dir }
        return nil
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        weightSources.filter { directory(for: $0, storeRoot: storeRoot) == nil }
    }
}

extension HairDeckConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        weightSources.compactMap { directory(for: $0, storeRoot: modelsRootDirectory) }
    }
}
