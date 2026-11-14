// The offline conformance checks of `HairDeckPackage` (the `characterPart` conformer): static
// assertions on the manifest, the declaration, the configuration and the weight sources, and the
// refusals `run()` makes before it needs weights. No MLX kernels, no weights.

import Foundation
import MLXToolKit
import XCTest
@testable import HairDeckKit

final class ConformanceTests: XCTestCase {
    var manifest: PackageManifest { HairDeckPackage.manifest }

    /// C0 — the contract version it targets; `characterPart` landed at 1.63.0.
    func testC0_ContractVersion() {
        XCTAssertEqual(manifest.contractVersion, ContractVersion.current)
        XCTAssertGreaterThanOrEqual(manifest.contractVersion, SemanticVersion(major: 1, minor: 63, patch: 0))
    }

    /// C1 / C2 — one surface, `characterPart`, whose canonical output is a mesh.
    func testC1C2_Surface() {
        XCTAssertEqual(manifest.capabilities, [.characterPart])
        XCTAssertEqual(manifest.surfaces.count, 1)
        XCTAssertEqual(Capability.characterPart.canonicalOutput, .mesh)
    }

    /// C11 — the declaration is coherent, and the descriptor advertises exactly what it declares.
    func testC11_Declaration() {
        XCTAssertTrue(HairDeckPackage.controls.declarationProblems.isEmpty, "\(HairDeckPackage.controls.declarationProblems)")
        let names = Set(manifest.surfaces[0].parameters.map(\.name))
        XCTAssertTrue(names.isSuperset(of: ["image", "subjectBox", "body", "seed", "springBones", "colors"]), "\(names)")
        XCTAssertFalse(names.contains("pointCloud"))
        XCTAssertEqual(HairDeckPackage.controls.defaultPart, .hair)
    }

    /// C6 — the governed specialty.
    func testC6_Specialty() {
        XCTAssertEqual(manifest.specialties.map(\.specialty), [.characterHair])
    }

    /// C7 + C8 — DINOv3 governs the weight layer and is allowlisted; the port's code is MIT; the
    /// default policy admits the package.
    func testC7C8_Licences() {
        XCTAssertEqual(manifest.license.weightLicense, .dinov3)
        XCTAssertEqual(manifest.license.portCodeLicense, .mit)
        XCTAssertTrue(LicensePolicy.permissiveOnly.evaluate(manifest.license).isAdmitted)
    }

    /// C9 — the configuration is defaultable and survives a Codable round trip (without the hook).
    func testC9_Configuration() throws {
        var c = HairDeckConfiguration()
        c.matting = { _ in Matte(format: .png, data: Data(), kind: .softAlpha) }
        let back = try JSONDecoder().decode(HairDeckConfiguration.self, from: JSONEncoder().encode(c))
        XCTAssertEqual(back.weightsRepo, HairDeckConfiguration.defaultWeightsRepo)
        XCTAssertEqual(back.quant, .fp32)
        XCTAssertNil(back.matting)
    }

    /// C10 — a footprint for the one tier, the backend, the OS floor.
    func testC10_Requirements() {
        let fp = manifest.requirements.footprints
        XCTAssertEqual(fp.map(\.quant), [.fp32])
        XCTAssertGreaterThan(fp[0].residentBytes, 0)
        XCTAssertGreaterThan(fp[0].peakActivationBytes, 0)
        XCTAssertTrue(manifest.requirements.requiredBackends.contains(.metalGPU))
    }

    // MARK: weight sources

    func testWeightSourcesAndProbe() throws {
        let fm = FileManager.default
        let c = HairDeckConfiguration()
        XCTAssertEqual(c.weightSources.map(\.repo), ["xocialize/hairdeck-mlx", "xocialize/trellis2-mlx"])
        // nothing anywhere: both missing
        XCTAssertEqual(c.missingWeightSources(storeRoot: nil).count, 2)
        // a store with the files flat in each repository's folder satisfies it
        let root = fm.temporaryDirectory.appending(path: "hairdeck-kit-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: root) }
        let store = ModelStore(root: root)
        let snap = try XCTUnwrap(store.directory(for: c.weightsRepo)), dino = try XCTUnwrap(store.directory(for: c.dinoRepo))
        try fm.createDirectory(at: snap, withIntermediateDirectories: true)
        try fm.createDirectory(at: dino, withIntermediateDirectories: true)
        for f in HairDeckConfiguration.snapshotFiles.dropLast() { fm.createFile(atPath: snap.appending(path: f).path, contents: Data()) }
        fm.createFile(atPath: dino.appending(path: HairDeckConfiguration.dinoFile).path, contents: Data())
        // one snapshot file short: the snapshot still reports missing, DINOv3 does not
        XCTAssertEqual(c.missingWeightSources(storeRoot: root).map(\.role), ["snapshot"])
        fm.createFile(atPath: snap.appending(path: HairDeckConfiguration.snapshotFiles.last!).path, contents: Data())
        XCTAssertTrue(c.missingWeightSources(storeRoot: root).isEmpty)
        // an override folder with everything flat needs no store
        var o = HairDeckConfiguration(weightsRootOverride: snap)
        XCTAssertEqual(o.missingWeightSources(storeRoot: nil).map(\.role), ["dinov3"])
        fm.createFile(atPath: snap.appending(path: HairDeckConfiguration.dinoFile).path, contents: Data())
        o = HairDeckConfiguration(weightsRootOverride: snap)
        XCTAssertTrue(o.missingWeightSources(storeRoot: nil).isEmpty)
    }

    // MARK: run()'s refusals, without weights

    func testRunRefusals() async throws {
        let p = HairDeckPackage(configuration: HairDeckConfiguration())
        let png = Image(format: .png, data: Data())
        func refusal(_ r: CharacterPartRequest) async -> String? {
            do { _ = try await p.run(r); return nil } catch PackageError.unsupportedRequestFeature(let s) { return s } catch { return "other: \(error)" }
        }
        // a point cloud: not a source of this surface
        let cloud = await refusal(CharacterPartRequest(pointCloud: PointCloud(data: Data(), hasNormals: true)))
        XCTAssertTrue(cloud?.hasPrefix("pointCloud") == true, "\(cloud ?? "nil")")
        // a picture without the box
        let noBox = await refusal(CharacterPartRequest(image: png))
        XCTAssertTrue(noBox?.hasPrefix("subjectBox") == true, "\(noBox ?? "nil")")
        // another part
        let garment = await refusal(CharacterPartRequest(image: png, subjectBox: [0, 0, 10, 10], part: .garment))
        XCTAssertTrue(garment?.hasPrefix("part") == true, "\(garment ?? "nil")")
        // a well-formed request before load(): not loaded
        do { _ = try await p.run(CharacterPartRequest(image: png, subjectBox: [0, 0, 10, 10])); XCTFail("expected notLoaded") } catch PackageError.notLoaded {} catch { XCTFail("\(error)") }
        // another capability
        do { _ = try await p.run(MeshRigRequest(mesh: Mesh(format: .glb, data: Data()))); XCTFail("expected unsupportedCapability") } catch PackageError.unsupportedCapability {} catch { XCTFail("\(error)") }
    }

    /// A pre-cancelled run refuses before anything else.
    func testCancelledRunRefusesFirst() async {
        let p = HairDeckPackage(configuration: HairDeckConfiguration())
        let t = Task { () -> Bool in
            withUnsafeCurrentTask { $0?.cancel() }
            do { _ = try await p.run(CharacterPartRequest(image: Image(format: .png, data: Data()), subjectBox: [0, 0, 1, 1])); return false } catch is CancellationError { return true } catch { return false }
        }
        let cancelled = await t.value
        XCTAssertTrue(cancelled)
    }
}
