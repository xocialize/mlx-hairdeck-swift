import Foundation
import HairDeckCore
import MLX

// RunHairDeck — parity gates and the end-to-end point-cloud run of mlx-hairdeck-swift.
//
//   RunHairDeck gate --workspace HAIRDECK_ROOT --fixtures Fixtures/p1 [--gpu]
//       every P1 stage against the PyTorch CPU fp32 captures of tools/dump_p1_fixtures.py, on the CPU
//       stream. --gpu: on the GPU, where fp32 matmuls are TF32-class by default (AB-L-0175), so the
//       gate only passes there with MLX_ENABLE_TF32=0 in the environment.
//   RunHairDeck run --workspace HAIRDECK_ROOT --plys DIR --noise NOISE.npy --out RUN [--rows A:B] [--batch 10]
//       DIR/{idx:03d}.ply → RUN/JsonResults/{idx:03d}.json (CHARM-format cards), RUN/tokens.npy,
//       RUN/swift_timing.json. NOISE.npy: [rows, 192, 40], the reference's noise for the seed.

func err(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

var args = Array(CommandLine.arguments.dropFirst())
guard let mode = args.first else { err("usage: RunHairDeck gate|run ..."); exit(2) }
args.removeFirst()
func opt(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}
func flag(_ name: String) -> Bool { args.contains(name) }
func need(_ name: String) -> String {
    guard let v = opt(name) else { err("missing \(name)"); exit(2) }
    return v
}

func peakRSSGiB() -> Double {
    var u = rusage()
    getrusage(RUSAGE_SELF, &u)
    return Double(u.ru_maxrss) / Double(1 << 30)
}

func gate() throws -> Bool {
    let fx = need("--fixtures")
    let model = try PointCloudModel(workspace: need("--workspace"))
    func F(_ n: String) throws -> MLXArray { try NPYReader.load("\(fx)/\(n).npy") }
    var ok = true
    func check(_ name: String, _ got: MLXArray, _ want: MLXArray, tol: Float) {
        guard got.shape == want.shape else { print("FAIL \(name): shape \(got.shape) vs \(want.shape)"); ok = false; return }
        let m = HDOps.maxAbs(got, want), c = HDOps.cosine(got, want)
        let scale = abs(want).max().item(Float.self)
        let pass = m <= tol
        if !pass { ok = false }
        print(String(format: "%@ %-22@ max_abs %.3e (tol %.0e, |ref|max %.2f) cos %.8f", pass ? "ok  " : "FAIL", name as NSString, m, tol, scale, c))
    }

    // ---- Dora encoder
    let pts = try F("dora_points")
    let n = pts.shape[1]
    let xyz = pts[0, 0..., 0..<3].asArray(Float.self)
    let idx = Dora.farthestPointSample(xyz, count: n, k: 256)
    let wantIdx = try F("dora_fps_idx").asArray(Int32.self)
    let same = idx == wantIdx
    if !same { ok = false }
    print("\(same ? "ok  " : "FAIL") dora_fps_idx           \(zip(idx, wantIdx).filter { $0 != $1 }.count) of 256 differ")
    check("dora_fourier", Dora.fourier(pts[0..., 0..<64, 0..<3]), try F("dora_fourier"), tol: 1e-5)
    let tr = Dora.encode(pts, idx: MLXArray(wantIdx, [1, 256]), weights: model.dora)
    check("dora_query", tr.query, try F("dora_query"), tol: 1e-4)
    check("dora_after_cross", tr.afterCross, try F("dora_after_cross"), tol: 1e-3)
    check("dora_after_self0", tr.afterSelf0, try F("dora_after_self0"), tol: 1e-3)
    check("dora_tokens", tr.tokens, try F("dora_tokens"), tol: 1e-3)

    // ---- set DiT
    let dit = model.dit
    let ctx = dit.encodeCond(try F("dit_cond"))
    check("dit_ctx", ctx, try F("dit_ctx"), tol: 1e-4)
    let noise = try F("noise_seed0")
    // arguments reach 700 rad here: one ulp of the fp32 frequency is 4e-5 in the angle
    check("dit_temb", SetDiT.timestepEmbedding([0.7]), try F("dit_temb"), tol: 1e-4)
    let c = dit.conditioning([0.7], ctx)
    check("dit_c", c, try F("dit_c"), tol: 1e-4)
    check("dit_block0", dit.block(0, HDOps.lin(noise, dit.w, "inp"), c, ctx), try F("dit_block0"), tol: 1e-4)
    check("dit_v_cond", dit(noise, [0.7], ctx), try F("dit_v_cond"), tol: 1e-4)
    check("dit_v_null", dit(noise, [0.7], dit.nullCtx), try F("dit_v_null"), tol: 1e-4)
    check("sample_tokens_4step", Flow.sample(dit, ctx: ctx, noise: noise, steps: 4), try F("sample_tokens_4step"), tol: 1e-3)
    let t0 = Date()
    let tokens = Flow.sample(dit, ctx: ctx, noise: noise, steps: 50)
    check("sample_tokens", tokens, try F("sample_tokens"), tol: 1e-2)
    print(String(format: "     50 steps, CFG 2, one hairstyle: %.1f s", Date().timeIntervalSince(t0)))

    // ---- tokens → cards (on the reference's tokens, so this stage is gated by itself)
    let refTok = try F("sample_tokens")
    let d = model.tokenizer.decode(refTok)
    check("tok_root", d.root, try F("tok_root"), tol: 1e-6)
    check("tok_z", d.z, try F("tok_z"), tol: 1e-5)
    let exist = try F("tok_exist").asArray(Int32.self).map { $0 != 0 }
    let existSame = d.exist[0] == exist
    if !existSame { ok = false }
    print("\(existSame ? "ok  " : "FAIL") tok_exist              \(exist.filter { $0 }.count) cards")
    let js = (0..<exist.count).filter { exist[$0] }
    let z = d.z[0].take(MLXArray(js.map { Int32($0) }), axis: 0)
    check("vae_n_logits", model.vae.countLogits(z), try F("vae_n_logits"), tol: 1e-4)
    let q = model.vae.queries(z)
    let wantNs = try F("vae_ns").asArray(Int32.self).map { Int($0) }
    if q.ns != wantNs { ok = false }
    print("\(q.ns == wantNs ? "ok  " : "FAIL") vae_ns                 \(q.ns)")
    check("vae_u", q.u, try F("vae_u"), tol: 1e-7)
    check("vae_from_z", model.vae.fromZ(z), try F("vae_from_z"), tol: 1e-4)
    // compare the card's own points only: the reference's padded rows are attention-masked garbage too
    let raw = model.vae.decode(z, u: q.u, mask: q.mask) * q.mask.expandedDimensions(axis: -1)
    check("vae_decode_raw", raw, try F("vae_decode_raw") * q.mask.expandedDimensions(axis: -1), tol: 1e-4)

    // cards.json: every number of the decoded cards
    let cards = Cards.fromTokens(refTok, tokenizer: model.tokenizer, vae: model.vae)[0]
    let ref = try JSONSerialization.jsonObject(with: FileManager.default.contents(atPath: "\(fx)/cards.json")!) as! [[String: Any]]
    var worst = 0.0
    var shapeOK = cards.count == ref.count
    if shapeOK {
        for (a, b) in zip(cards, ref) {
            let seq = b["seq"] as! [[NSNumber]]
            if seq.count != a.seq.count || a.triangle != ((b["shape"] as! NSNumber).intValue == 1) { shapeOK = false; break }
            for (ra, rb) in zip(a.seq, seq) { for (x, y) in zip(ra, rb) { worst = max(worst, abs(x - y.doubleValue)) } }
        }
    }
    let cardsOK = shapeOK && worst <= 1e-5
    if !cardsOK { ok = false }
    print(String(format: "%@ cards.json             %d cards, counts and classes %@, max_abs %.3e (tol 1e-05)", cardsOK ? "ok  " : "FAIL", cards.count, shapeOK ? "equal" : "DIFFER", worst))
    print(String(format: "peak RSS %.2f GiB", peakRSSGiB()))
    return ok
}

func run() throws {
    let out = need("--out")
    let plys = need("--plys")
    let t0 = Date()
    let model = try PointCloudModel(workspace: need("--workspace"))
    let tLoad = Date().timeIntervalSince(t0)
    let noise = try NPYReader.load(need("--noise"))
    var lo = 0, hi = noise.shape[0]
    if let r = opt("--rows") { let p = r.split(separator: ":"); lo = Int(p[0])!; hi = Int(p[1])! }
    let batch = Int(opt("--batch") ?? "10")!
    try FileManager.default.createDirectory(atPath: "\(out)/JsonResults", withIntermediateDirectories: true)
    var all: [MLXArray] = []
    var tCond = 0.0, tSample = 0.0, tDecode = 0.0, nCards = 0
    for s in stride(from: lo, to: hi, by: batch) {
        let rows = Array(s..<min(s + batch, hi))
        var t = Date()
        var conds: [MLXArray] = []
        for i in rows {
            let p = try HDIO.conditionPoints(ply: String(format: "%@/%03d.ply", plys, i))
            // the released condition is stored as float16
            let c = Dora.condition(points: p.points, count: p.count, weights: model.dora).asType(.float16).asType(.float32)
            eval(c)
            conds.append(c)
        }
        tCond += Date().timeIntervalSince(t)
        t = Date()
        let ctx = model.dit.encodeCond(concatenated(conds, axis: 0))
        let tok = Flow.sample(model.dit, ctx: ctx, noise: noise[rows[0]..<(rows.last! + 1)])
        tSample += Date().timeIntervalSince(t)
        t = Date()
        let hairs = Cards.fromTokens(tok, tokenizer: model.tokenizer, vae: model.vae)
        for (i, cards) in zip(rows, hairs) {
            try Cards.json(cards).write(to: URL(fileURLWithPath: String(format: "%@/JsonResults/%03d.json", out, i)))
            nCards += cards.count
        }
        tDecode += Date().timeIntervalSince(t)
        all.append(tok)
        err("rows \(rows.first!)..\(rows.last!) done")
    }
    try HDIO.writeNPY(concatenated(all, axis: 0), to: "\(out)/tokens.npy")
    let n = Double(hi - lo)
    let timing: [String: Any] = ["rows": hi - lo, "batch": batch, "load_s": tLoad, "condition_s_per_hairstyle": tCond / n,
                                 "sample_s_per_hairstyle": tSample / n, "decode_s_per_hairstyle": tDecode / n,
                                 "n_cards_mean": Double(nCards) / n, "peak_rss_gib": peakRSSGiB(),
                                 "mlx_peak_gib": Double(GPU.peakMemory) / Double(1 << 30), "device": flag("--cpu") ? "cpu" : "gpu"]
    try JSONSerialization.data(withJSONObject: timing, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: "\(out)/swift_timing.json"))
    print(String(data: try JSONSerialization.data(withJSONObject: timing, options: [.prettyPrinted, .sortedKeys]), encoding: .utf8)!)
}

do {
    switch mode {
    case "gate":
        // parity gates run on the CPU stream unless --gpu
        let ok = flag("--gpu") ? try gate() : try Device.withDefaultDevice(.cpu) { try gate() }
        print(ok ? "GATE PASSED" : "GATE FAILED")
        exit(ok ? 0 : 1)
    case "run":
        if flag("--cpu") { try Device.withDefaultDevice(.cpu) { try run() } } else { try run() }
    default:
        err("unknown mode \(mode)"); exit(2)
    }
} catch {
    err("error: \(error)"); exit(1)
}
