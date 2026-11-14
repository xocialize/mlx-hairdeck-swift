import Foundation
import HairDeckCore
import HairDeckKit
import MLX
import MLXServeCore
import MLXToolKit

// RunHairDeckEngine — HairDeck driven through MLXServeEngine: register (licence gate, device
// eligibility) → prepare (admission, load) → run → the hair item.
//
// Env: IMAGE (a cut-out PNG, or any picture with MATTE), BOX "x0,y0,x1,y1" (pixels: the box around the hair and the
//      face; the package squares it and adds its 8 % margin, unlike RunHairDeck image-hair's --box, which is the crop), BODY (a .vrm / .glb,
//      optional), OUT (the .glb), SEED (default 0), MATTE (a matte PNG for an opaque picture, standing in
//      for the host's matting), WEIGHTS_DIR (a folder with the snapshot's files and dino.safetensors) or
//      MODELS_ROOT (the engine's model store; prepare() downloads what is missing there).
//      MEASURE=<n>: after the first run, the footprint by the memory-harness method — resident (MLX active memory
//      with the cache cleared) and, over n more runs with other seeds, the peak above it; plus the process's
//      phys_footprint.

setvbuf(stdout, nil, _IONBF, 0)
func gb(_ b: UInt64) -> String { String(format: "%.2f GB", Double(b) / 1e9) }
let env = ProcessInfo.processInfo.environment
guard let imagePath = env["IMAGE"], let imageData = FileManager.default.contents(atPath: imagePath),
      let box = env["BOX"]?.split(separator: ",").compactMap({ Float($0) }), box.count == 4 else {
    fatalError("IMAGE=<picture> BOX=x0,y0,x1,y1 [BODY=<vrm>] [OUT=<glb>]")
}
let out = env["OUT"] ?? "hair_engine.glb"
let matteData = env["MATTE"].flatMap { FileManager.default.contents(atPath: $0) }
let engine = MLXServeEngine()
if let root = env["MODELS_ROOT"] { await engine.useModelStore(ModelStore(root: URL(fileURLWithPath: root, isDirectory: true))) }
do {
    var cfg = HairDeckConfiguration(weightsRootOverride: env["WEIGHTS_DIR"].map { URL(fileURLWithPath: $0, isDirectory: true) })
    if let matteData { cfg.matting = { _ in Matte(format: .png, data: matteData, kind: .softAlpha) } }
    let id = try await engine.register(HairDeckPackage.registration, configuration: cfg)
    print("[engine] registered \(id.description); budget \(gb(await engine.memory.budgetBytes))")
    var t = Date()
    try await engine.prepare(.characterPart, package: id)
    print(String(format: "[engine] prepare %.1f s", -t.timeIntervalSinceNow))
    let fmt: Image.Format = imagePath.lowercased().hasSuffix(".jpg") || imagePath.lowercased().hasSuffix(".jpeg") ? .jpeg : .png
    let body = env["BODY"].flatMap { FileManager.default.contents(atPath: $0) }.map { Mesh(format: .glb, data: $0) }
    let req = CharacterPartRequest(image: Image(format: fmt, data: imageData), subjectBox: box, body: body, seed: UInt64(env["SEED"] ?? "0") ?? 0)
    t = Date()
    guard let resp = try await engine.run(req, package: id) as? CharacterPartResponse else { fatalError("unexpected response type") }
    print(String(format: "[engine] run %.2f s → %d bytes, part %@, fitted to the body: %@, %d vertices", -t.timeIntervalSinceNow, resp.mesh.data.count,
                 resp.part.rawValue, String(resp.fittedToBody), resp.mesh.vertexCount ?? 0))
    try resp.mesh.data.write(to: URL(fileURLWithPath: out))
    if let n = env["MEASURE"].flatMap({ Int($0) }), n > 0 {
        func phys() -> UInt64 {
            var info = task_vm_info_data_t()
            var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
            let kr = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) } }
            return kr == KERN_SUCCESS ? info.phys_footprint : 0
        }
        MLX.Memory.clearCache()
        let resident = UInt64(MLX.Memory.activeMemory)
        var worst: UInt64 = 0, worstPhys = phys()
        for i in 1...n {
            MLX.Memory.clearCache()
            MLX.Memory.peakMemory = 0
            let r = CharacterPartRequest(image: Image(format: fmt, data: imageData), subjectBox: box, body: body, seed: UInt64(i))
            _ = try await engine.run(r, package: id)
            worst = max(worst, UInt64(MLX.Memory.peakMemory))
            worstPhys = max(worstPhys, phys())
        }
        print("[measure] resident (active, cache cleared) \(gb(resident)); peak over \(n) runs \(gb(worst)) → activation \(gb(worst > resident ? worst - resident : 0)); phys_footprint peak \(gb(worstPhys))")
    }
    print(String(format: "[engine] wrote %@ | peak MLX %.2f GiB", out, Double(GPU.peakMemory) / Double(1 << 30)))
} catch {
    print("[engine] failed: \(error)")
    exit(1)
}
