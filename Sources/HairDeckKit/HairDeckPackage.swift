import Foundation
import HairDeckCore
import MLX
import MLXToolKit

public enum HairDeckPackageError: Error, Sendable, CustomStringConvertible {
    case weightsNotFound(String)
    public var description: String {
        switch self { case .weightsNotFound(let s): return "HairDeck weights not found: \(s)" }
    }
}

/// MLXToolKit `ModelPackage` conformer for HairDeck (`characterPart`, contract 1.63.0): one character
/// picture and the head box → hair cards with per-card colours and spring-bone chains, as a skinned
/// hair item GLB placed on the given body (or in the model's own head frame without one).
///
/// The pipeline is the engine-free `HairDeckCore.HairDeckPipeline`, the code the parity gates and the
/// CLI run; this type declares and dispatches (C13: the engine constructs, loads, drives and evicts
/// it). The point-cloud path of the port is not a surface here: a bare cloud has no head to hang
/// chains on or to place on a body, so it stays a library and CLI feature.
///
/// Licences: HairDeck's weights are Apache-2.0 and so is Dora; the image path's DINOv3 conditioner is
/// under Meta's DINOv3 License, the most restrictive component, so it governs the weight layer
/// (allowlisted; "Built with DINOv3" wherever shown). The port's code is MIT.
@InferenceActor
public final class HairDeckPackage: ModelPackage {
    public typealias Configuration = HairDeckConfiguration

    public nonisolated static let controls = CharacterPartControls(
        parts: [.hair], inputs: [.image],
        features: [.subjectBox, .bodyFit, .springBones, .colors],
        requiresSubjectBox: true)

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .dinov3, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "hyz317/HairDeck", revision: "cb20f70", tier: 3),
            requirements: RequirementsManifest(
                // Measured through the engine (`MEASURE=5 RunHairDeckEngine`, release, GPU, Mac Studio 128 GB,
                // mlx-swift 0.33.0, 2026-11-14; AB-R-0855) by the memory-harness method, on the one
                // workload the model has (a 448 px crop, 192 card tokens, colours and chains on):
                // resident 2.83 GB (MLX active memory with the cache cleared: DINOv3 1.21 GB, image and
                // colour DiTs 0.73 GB each, chain DiT 0.12 GB, card VAE 0.03 GB); peak over five runs
                // 4.36 GB, so 1.53 GB of activation; the process's phys_footprint peaked at 5.23 GB.
                // Declared: 2.9 GB resident + 2.6 GB activation = 5.5 GB, the harness's rule
                // (peak × 1.2 + 256 MB = 5.49 GB), which also covers the phys_footprint peak.
                footprints: [QuantFootprint(quant: .fp32, residentBytes: 2_900_000_000, peakActivationBytes: 2_600_000_000)],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: .base),
            specialties: [SpecialtyWeight(.characterHair, strength: 1.0)],
            surfaces: [CharacterPartContract.descriptor(
                name: "hairdeck",
                summary: "Anime-style hair for an avatar from one character picture: hair cards with per-card colours and "
                       + "spring-bone chains (VRMC_springBone), as a skinned GLB item. Needs the head box in the picture's "
                       + "pixels: the box around all of the hair and the face (the package squares it and adds a margin). With a body (a VRM or rigged GLB with a face mesh and a humanoid head bone) "
                       + "the item is placed on its head and skinned to that bone; without one it comes back in the model's "
                       + "own head frame. An opaque picture needs the host's matting; a cut-out picture's alpha is used as is.",
                controls: controls)])
    }

    public nonisolated static var registration: PackageRegistration { .of(HairDeckPackage.self) }

    private let configuration: Configuration
    private var model: ImagePathModel?   // host-owned lifecycle; no global singleton (C13)

    public nonisolated init(configuration: Configuration) { self.configuration = configuration }

    public func load() async throws {
        let root = configuration.modelsRootDirectory
        if let root {
            let missing = configuration.missingWeightSources(storeRoot: root)
            if !missing.isEmpty { try await WeightMaterializer.materialize(missing, into: root) }
        }
        let sources = configuration.weightSources
        guard let snap = configuration.directory(for: sources[0], storeRoot: root) else {
            throw HairDeckPackageError.weightsNotFound(sources[0].repo)
        }
        guard let dino = configuration.directory(for: sources[1], storeRoot: root) else {
            throw HairDeckPackageError.weightsNotFound("\(sources[1].repo)/\(Configuration.dinoFile)")
        }
        model = try ImagePathModel(source: .snapshot(snap.path, dino: dino.appending(path: Configuration.dinoFile).path))
    }

    public func unload() async {
        model = nil
        MLX.Memory.clearCache()
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // The cancellation checkpoint is the first act, before validation and the notLoaded guard.
        try Task.checkCancellation()
        guard request.capability == .characterPart, let req = request as? CharacterPartRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        if let why = Self.controls.refusal(for: req) { throw PackageError.unsupportedRequestFeature(why) }
        guard let model else { throw PackageError.notLoaded }
        guard let image = req.image, let box = req.subjectBox else {
            throw PackageError.unsupportedRequestFeature("image and subjectBox — this surface reads a picture and its head box")
        }
        guard image.format == .png || image.format == .jpeg else {
            throw PackageError.unsupportedRequestFeature("image.format(\(image.format.rawValue)) — send a PNG or JPEG")
        }
        let pic = try HDIO.readRGBA(data: image.data, label: "the request's image")
        let n = pic.width * pic.height
        var rgb = [UInt8](repeating: 0, count: n * 3)
        var matte = [UInt8](repeating: 255, count: n)
        var cutOut = false
        for i in 0..<n {
            for c in 0..<3 { rgb[i * 3 + c] = pic.pixels[i * 4 + c] }
            matte[i] = pic.pixels[i * 4 + 3]
            if matte[i] != 255 { cutOut = true }
        }
        if !cutOut {
            // an opaque picture: the host's matting, or a legible refusal
            guard let matting = configuration.matting else {
                throw PackageError.unsupportedRequestFeature(
                    "image — an opaque picture needs a person matte and this package was configured without matting "
                    + "(HairDeckConfiguration.matting); send a picture cut out with transparency, or configure the hook")
            }
            let m = try await matting(image)
            let mp = try HDIO.readRGBA(data: m.data, label: "the matte")
            guard mp.width == pic.width, mp.height == pic.height else {
                throw PackageError.unsupportedRequestFeature("matte — \(mp.width) × \(mp.height) for a \(pic.width) × \(pic.height) picture")
            }
            for i in 0..<n { matte[i] = mp.pixels[i * 4] }
        }
        var placement: BodyPlacement? = nil
        var vrm0 = false
        if let body = req.body {
            let b = try VRMBody(glb: body.data, label: "the request's body")
            placement = b.placement
            vrm0 = b.isVRM0
        }
        try Task.checkCancellation()
        // `subjectBox` is the box around the hair and the face. The model's input is the square around it
        // with an 8 % margin, which may reach past the picture (white there), as the reference crops.
        let crop = ModelInput.headBox(hair: box.map(Double.init), face: nil) ?? box.map(Double.init)
        let r = try HairDeckPipeline.imageHair(
            model, rgb: rgb, matte: matte, width: pic.width, height: pic.height, box: crop,
            body: placement, vrm0Axes: vrm0, seed: req.seed ?? UInt64.random(in: 0...UInt64(UInt32.max)),
            springs: req.springBones ?? true, colours: req.colors ?? true,
            checkpoint: { try Task.checkCancellation() })
        // a cancellation during the last stage surfaces here, before a response exists
        try Task.checkCancellation()
        return CharacterPartResponse(mesh: Mesh(format: .glb, data: r.glb, vertexCount: r.summary.vertices, faceCount: r.summary.triangles),
                                     part: .hair, fittedToBody: placement != nil)
    }
}
