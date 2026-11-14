import Foundation
import HairDeckCore
import CoreGraphics
import ImageIO
import MLX
import MLXRandom
import simd

// RunHairDeck — parity gates and the end-to-end point-cloud run of mlx-hairdeck-swift.
//
//   RunHairDeck gate --workspace HAIRDECK_ROOT --fixtures Fixtures/p1 [--gpu]
//       every P1 stage against the PyTorch CPU fp32 captures of tools/dump_p1_fixtures.py, on the CPU
//       stream. --gpu: on the GPU, where fp32 matmuls are TF32-class by default (AB-L-0175), so the
//       gate only passes there with MLX_ENABLE_TF32=0 in the environment.
//   RunHairDeck run --workspace HAIRDECK_ROOT --plys DIR --noise NOISE.npy --out RUN [--rows A:B] [--batch 10]
//       DIR/{idx:03d}.ply → RUN/JsonResults/{idx:03d}.json (CHARM-format cards), RUN/tokens.npy,
//       RUN/swift_timing.json. NOISE.npy: [rows, 192, 40], the reference's noise for the seed.

//   RunHairDeck gate2 --workspace HAIRDECK_ROOT --fixtures Fixtures/p2 [--gpu]
//       the chain DiT, chain decoding, the card VAE encoder and head-frame card tokens (P2)
//   RunHairDeck chains --workspace HAIRDECK_ROOT --noise NOISE.npy --out RUN (--tokens CTOK.npy --rows ROWS.json
//                      | --cards PC_RUN --fits FITS.json)
//       chains for card tokens given directly ([H, 192, 40], head frame; ROWS.json: the row id of each), or for
//       the cards of a point-cloud run with each row's head fit (FITS.json: [{"row", "k", "b"}]).
//       → RUN/chains_head.json {row: [chain]} in the head frame, RUN/swift_timing.json.

//   RunHairDeck gate3 --fixtures Fixtures/p3
//       the hair rig (card mesh, UVs, skinning) against hairdeck.vrm.prep on HairDeck's avatar body (P3)
//   RunHairDeck hair-item --cards CARDS.json --chains CHAINS.json (--body BODY.json | --vrm BODY.vrm) --out HAIR.glb
//                         [--vrm0] [--colour R,G,B]
//       head-frame cards and chains on a body → a skinned hair item .glb with VRMC_springBone. --body: HairDeck's
//       fit of a body; --vrm: the description read from the VRM itself (face-mesh bounds, head bone); the item is
//       written in the file's own axes (--vrm0 forces VRM 0.x axes with --body).
//   RunHairDeck body --vrm BODY.vrm [--body BODY.json]
//       the description read from a VRM, and how it differs from a HairDeck fit of the same body

//   RunHairDeck gate4 --workspace HAIRDECK_ROOT --fixtures Fixtures/p4 [--gpu]
//       DINOv3 tokens, the image DiT, the colour model and chains from image tokens (P4a)

//   RunHairDeck image-hair --workspace HAIRDECK_ROOT (--image IMAGE.png --matte MATTE.png --box X0,Y0,X1,Y1 | --crop MODEL_INPUT.png)
//                          (--vrm BODY.vrm | --body BODY.json) --out HAIR.glb
//                          [--seed 0] [--noise NOISE.npy] [--no-chains] [--no-colours] [--save DIR]
//       a picture with its person matte (same size, white = character) and a head box in pixels (the square around
//       the hair and the face, as ModelInput.headBox makes it; the caller supplies it) → the 448 px model input →
//       DINOv3 → image DiT (Euler 50, CFG 2) → cards, per-card colours and chains → a coloured, skinned hair item on
//       the body. --crop: a ready 448 x 448 model input instead (the character on white); its matte is --matte
//       (448 x 448) or, without one, "not white". --save: the model input, cards, chains and tokens as files.

//   RunHairDeck gate4b --fixtures Fixtures/p4b
//       the model input and matte crop from a picture, its matte and a head box, against the reference's (P4b)
//   RunHairDeck model-input --image IMAGE.png --matte MATTE.png --box X0,Y0,X1,Y1 --out INPUT.png [--matte-out M.png]

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

/// The weights: --workspace HAIRDECK_ROOT, or --snapshot DIR (the consolidated files) with --dino FILE for the image path.
func source() -> HairDeckSource {
    if let d = opt("--snapshot") { return .snapshot(d, dino: opt("--dino")) }
    return .workspace(need("--workspace"))
}

func peakRSSGiB() -> Double {
    var u = rusage()
    getrusage(RUSAGE_SELF, &u)
    return Double(u.ru_maxrss) / Double(1 << 30)
}

func gate() throws -> Bool {
    let fx = need("--fixtures")
    let model = try PointCloudModel(source: source())
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
    check("sample_tokens_4step", try Flow.sample(dit, ctx: ctx, noise: noise, steps: 4), try F("sample_tokens_4step"), tol: 1e-3)
    let t0 = Date()
    let tokens = try Flow.sample(dit, ctx: ctx, noise: noise, steps: 50)
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

func gate2() throws -> Bool {
    let fx = need("--fixtures")
    let model = try ChainModel(source: source())
    func F(_ n: String) throws -> MLXArray { try NPYReader.load("\(fx)/\(n).npy") }
    var ok = true
    func check(_ name: String, _ got: MLXArray, _ want: MLXArray, tol: Float) {
        guard got.shape == want.shape else { print("FAIL \(name): shape \(got.shape) vs \(want.shape)"); ok = false; return }
        let m = HDOps.maxAbs(got, want)
        if m > tol { ok = false }
        print(String(format: "%@ %-22@ max_abs %.3e (tol %.0e, |ref|max %.2f) cos %.8f", m <= tol ? "ok  " : "FAIL", name as NSString, m, tol,
                     abs(want).max().item(Float.self), HDOps.cosine(got, want)))
    }
    let chain = model.chain
    let ctx = chain.encodeCond(try F("chain_card_tokens"))
    check("chain_ctx", ctx, try F("chain_ctx"), tol: 1e-4)
    let noise = try F("chain_noise")
    let B = noise.shape[0]
    let t = [Float](repeating: 0.7, count: B)
    check("chain_v_cond", chain.dit(noise, t, ctx), try F("chain_v_cond"), tol: 1e-4)
    check("chain_v_null", chain.dit(noise, t, broadcast(chain.dit.nullCtx, to: ctx.shape)), try F("chain_v_null"), tol: 1e-4)
    let x = try Flow.sample(chain.dit, ctx: ctx, noise: noise)
    check("chain_sample", x, try F("chain_sample"), tol: 1e-3)

    // decoding, on the reference's tokens
    let refTok = try F("chain_sample")
    let ref = try JSONSerialization.jsonObject(with: FileManager.default.contents(atPath: "\(fx)/chains.json")!) as! [[[String: Any]]]
    var worst = 0.0, same = true
    let D = chain.tokenDim, N = ChainDiT.slots
    let flat = refTok.asArray(Float.self)
    for b in 0..<B {
        let got = chain.decode(Array(flat[(b * N * D)..<((b + 1) * N * D)]))
        if got.count != ref[b].count { same = false; continue }
        for (g, r) in zip(got, ref[b]) {
            let rj = r["joints"] as! [[NSNumber]]
            if rj.count != g.joints.count { same = false; continue }
            for (a, c) in zip(g.joints, rj) { for (p, q) in zip(a, c) { worst = max(worst, abs(p - q.doubleValue)) } }
            for (a, c) in zip(g.path, r["path"] as! [[NSNumber]]) { for (p, q) in zip(a, c) { worst = max(worst, abs(p - q.doubleValue)) } }
            worst = max(worst, abs(g.stiffness - (r["stiffness"] as! NSNumber).doubleValue))
        }
    }
    let decOK = same && worst <= 1e-9
    if !decOK { ok = false }
    print(String(format: "%@ chains.json            chain and joint counts %@, max_abs %.3e (tol 1e-09)", decOK ? "ok  " : "FAIL", same ? "equal" : "DIFFER", worst))

    // card VAE encoder and head-frame card tokens of generated cards
    let meta = try JSONSerialization.jsonObject(with: FileManager.default.contents(atPath: "\(fx)/meta.json")!) as! [String: Any]
    let rows = (meta["rows"] as! [NSNumber]).map { $0.intValue }
    let ks = (meta["k"] as! [NSNumber]).map { $0.doubleValue }
    let bs = (meta["b"] as! [[NSNumber]]).map { $0.map { $0.doubleValue } }
    let cards0 = try Cards.load(json: String(format: "%@/cards/%03d.json", fx, rows[0]))
    let (px, lens) = HeadFrame.padSeqs(Array(cards0.prefix(192)).map { $0.seq }, L: model.vae.L, scale: Float(ks[0]))
    check("enc_X", MLXArray(px, [lens.count, model.vae.L, 5]), try F("enc_X"), tol: 1e-8)
    check("enc_u", MLXArray(HeadFrame.arcU(px, lens: lens, L: model.vae.L), [lens.count, model.vae.L]), try F("enc_u"), tol: 1e-5)
    check("enc_latents", model.vae.encode(px, lens: lens), try F("enc_latents"), tol: 1e-4)
    var toks: [Float] = []
    for (i, r) in rows.enumerated() {
        toks += HeadFrame.cardTokens(try Cards.load(json: String(format: "%@/cards/%03d.json", fx, r)), k: ks[i], b: bs[i], vae: model.vae, tokenizer: model.tokenizer)
    }
    check("enc_card_tokens", MLXArray(toks, [rows.count, 192, 40]), try F("enc_card_tokens"), tol: 1e-4)
    print(String(format: "peak RSS %.2f GiB", peakRSSGiB()))
    return ok
}

/// A float64 `.npy` as Doubles (the P3 fixtures keep the reference's precision).
func npyDoubles(_ path: String) throws -> [Double] {
    let d = FileManager.default.contents(atPath: path)!
    let h = try NPYReader.parseHeader(d, label: path)
    precondition(h.descr == "<f8", "\(path): expected float64")
    return d.withUnsafeBytes { b in (0..<h.count).map { b.loadUnaligned(fromByteOffset: h.dataOffset + 8 * $0, as: Double.self) } }
}

func loadChains(_ path: String) throws -> [(joints: [[Double]], stiffness: Double)] {
    let arr = try JSONSerialization.jsonObject(with: FileManager.default.contents(atPath: path)!) as! [[String: Any]]
    return arr.map { (($0["joints"] as! [[NSNumber]]).map { $0.map { $0.doubleValue } }, ($0["stiffness"] as! NSNumber).doubleValue) }
}

func gate3() throws -> Bool {
    let fx = need("--fixtures")
    let body = try BodyPlacement(json: "\(fx)/body.json")
    let rig = HairRig.build(cards: try Cards.load(json: "\(fx)/cards_head.json"), chains: try loadChains("\(fx)/chains_head.json"), body: body)
    var ok = true
    func ints(_ n: String) throws -> [Int] { try NPYReader.load("\(fx)/\(n).npy").asArray(Int32.self).map { Int($0) } }
    func checkD(_ name: String, _ got: [Double], _ want: [Double], tol: Double) {
        let same = got.count == want.count
        let m = same ? zip(got, want).map { abs($0 - $1) }.max() ?? 0 : .infinity
        if !(m <= tol) { ok = false }
        print(String(format: "%@ %-12@ %d values, max_abs %.3e (tol %.0e)", m <= tol ? "ok  " : "FAIL", name as NSString, got.count, m, tol))
    }
    func checkI(_ name: String, _ got: [Int], _ want: [Int]) {
        let diff = got.count == want.count ? zip(got, want).filter { $0 != $1 }.count : -1
        if diff != 0 { ok = false }
        print(String(format: "%@ %-12@ %d values, %d differ", diff == 0 ? "ok  " : "FAIL", name as NSString, got.count, diff))
    }
    checkD("V", rig.vertices.flatMap { [$0.x, $0.y, $0.z] }, try npyDoubles("\(fx)/V.npy"), tol: 1e-9)
    checkI("F", rig.faces.flatMap { $0 }, try ints("F"))
    checkD("UV", rig.uv.flatMap { [$0.x, $0.y] }, try npyDoubles("\(fx)/UV.npy"), tol: 1e-12)
    checkI("vcard", rig.card, try ints("vcard"))
    checkI("vchain", rig.chain, try ints("vchain"))
    // A vertex exactly on the bisector plane of a joint is equally near both segments, and which one
    // wins is a rounding accident; the binding is the same either way (the two segments at α = 0.5).
    // Such a vertex may come out with (segment, segment2) swapped.
    let wantSeg = try ints("vseg"), wantSeg2 = try ints("vseg2")
    var swapped = 0, wrong = 0
    for i in 0..<min(rig.segment.count, wantSeg.count) where rig.segment[i] != wantSeg[i] || rig.segment2[i] != wantSeg2[i] {
        if rig.segment[i] == wantSeg2[i] && rig.segment2[i] == wantSeg[i] && abs(rig.alpha[i] - 0.5) < 1e-9 {
            swapped += 1
            print(String(format: "     vertex %d: segments (%d, %d) vs (%d, %d), alpha %.15f, weight %.6f — a tie", i, rig.segment[i], rig.segment2[i], wantSeg[i], wantSeg2[i], rig.alpha[i], rig.weight[i]))
        } else { wrong += 1 }
    }
    if wrong != 0 || rig.segment.count != wantSeg.count { ok = false }
    print(String(format: "%@ vseg, vseg2  %d values, %d differ, %d ties with the two segments swapped", wrong == 0 ? "ok  " : "FAIL", rig.segment.count, wrong, swapped))
    checkD("vw", rig.weight, try npyDoubles("\(fx)/vw.npy"), tol: 1e-9)
    checkD("valpha", rig.alpha, try npyDoubles("\(fx)/valpha.npy"), tol: 1e-9)
    checkD("chain_J", rig.chains.flatMap { $0.joints.flatMap { [$0.x, $0.y, $0.z] } }, try npyDoubles("\(fx)/chain_J.npy"), tol: 1e-9)
    checkD("stiffness", rig.chains.map { $0.stiffness }, try npyDoubles("\(fx)/stiffness.npy"), tol: 1e-12)
    let hj = body.headJointWorld
    checkD("head_joint", [hj.x, hj.y, hj.z], try npyDoubles("\(fx)/head_joint.npy"), tol: 1e-9)
    print("     cards on a chain \(rig.cardsOnChain), bind mismatches \(rig.bindMismatch), chains \(rig.chains.count), dropped \(rig.chainsDropped)")
    return ok
}

func gate4() throws -> Bool {
    let fx = need("--fixtures")
    let m = try ImagePathModel(source: source())
    func F(_ n: String) throws -> MLXArray { try NPYReader.load("\(fx)/\(n).npy") }
    var ok = true
    func check(_ name: String, _ got: MLXArray, _ want: MLXArray, tol: Float) {
        guard got.shape == want.shape else { print("FAIL \(name): shape \(got.shape) vs \(want.shape)"); ok = false; return }
        let d = HDOps.maxAbs(got, want)
        if d > tol { ok = false }
        print(String(format: "%@ %-20@ max_abs %.3e (tol %.0e, |ref|max %.2f) cos %.8f", d <= tol ? "ok  " : "FAIL", name as NSString, d, tol,
                     abs(want).max().item(Float.self), HDOps.cosine(got, want)))
    }
    // ---- DINOv3
    let rgb = try F("crop").asType(.float32).expandedDimensions(axis: 0)          // [1, 448, 448, 3], 0..255
    let px = DINOv3.normalise(rgb)
    check("dino_pixels", px, try F("dino_pixels").transposed(0, 2, 3, 1), tol: 1e-5)
    let (rs, rc) = DINOv3.rope(28, 28)
    check("dino_rope_sin", rs, try F("dino_rope_sin"), tol: 1e-5)
    check("dino_rope_cos", rc, try F("dino_rope_cos"), tol: 1e-5)
    let emb = m.dino.embed(px)
    check("dino_embed", emb, try F("dino_embed"), tol: 1e-4)
    // DINOv3 carries a few tokens of very large norm (|ref| max 1.6e5 here): the tolerance is relative, 1e-6
    let b0 = try F("dino_block0")
    check("dino_block0", m.dino.block(0, emb, rs, rc), b0, tol: 1e-6 * abs(b0).max().item(Float.self))
    let t0 = Date()
    let tokens = m.dino(px)
    eval(tokens)
    // the tokens are rounded to bfloat16 (steps of up to 1/128 of the value): a last-bit difference before the rounding is one step
    check("dino_tokens", tokens, try F("dino_tokens"), tol: 5e-2)
    print(String(format: "     DINOv3 forward %.1f s", Date().timeIntervalSince(t0)))

    // ---- image DiT, on the reference's tokens
    let refTok = try F("dino_tokens")
    let ctx = m.image.encodeCond(refTok)
    check("img_ctx", ctx, try F("img_ctx"), tol: 1e-4)
    let noise = try F("noise_seed0")
    check("img_v_cond", m.image.dit(noise, [0.7], ctx), try F("img_v_cond"), tol: 1e-4)
    check("img_v_null", m.image.dit(noise, [0.7], m.image.dit.nullCtx), try F("img_v_null"), tol: 1e-4)
    let x = try Flow.sample(m.image.dit, ctx: ctx, noise: noise)
    check("img_sample", x, try F("img_sample"), tol: 1e-3)
    let refX = try F("img_sample")
    let cards = Cards.fromTokens(refX, tokenizer: m.tokenizer, vae: m.vae)[0]
    let ref = try JSONSerialization.jsonObject(with: FileManager.default.contents(atPath: "\(fx)/cards.json")!) as! [[String: Any]]
    var worst = 0.0, same = cards.count == ref.count
    if same {
        for (a, b) in zip(cards, ref) {
            let seq = b["seq"] as! [[NSNumber]]
            if seq.count != a.seq.count || a.triangle != ((b["shape"] as! NSNumber).intValue == 1) { same = false; break }
            for (ra, rb) in zip(a.seq, seq) { for (p, q) in zip(ra, rb) { worst = max(worst, abs(p - q.doubleValue)) } }
        }
    }
    if !(same && worst <= 1e-5) { ok = false }
    print(String(format: "%@ cards.json           %d cards, counts and classes %@, max_abs %.3e (tol 1e-05)", same && worst <= 1e-5 ? "ok  " : "FAIL", cards.count, same ? "equal" : "DIFFER", worst))

    // ---- colour model
    let matte = try F("matte").asType(.float32).expandedDimensions(axis: 0)
    let cctx = m.color.encodeCtx(refTok, rgb: rgb, matte: matte)
    check("col_ctx", cctx, try F("col_ctx"), tol: 1e-4)
    let flat = refX.asArray(Float.self)
    let cleaned = m.tokenizer.clean(flat)
    check("chain_clean_tokens", MLXArray(cleaned, [1, 192, 40]), try F("chain_clean_tokens"), tol: 1e-6)
    let signed = MLXArray(m.tokenizer.signNormals(cleaned), [1, 192, 40])
    check("col_tokens", signed, try F("col_tokens"), tol: 1e-6)
    let refSigned = try F("col_tokens")
    check("col_raw", m.color.raw(refSigned, cctx), try F("col_raw"), tol: 1e-4)
    let lab = m.color(refSigned, cctx)
    check("col_lab", lab, try F("col_lab"), tol: 1e-3)
    // sRGB of the reference's Lab, for the existing slots, rounded to 4 decimals
    let labRef = try F("col_lab").asArray(Float.self)
    let want = try npyDoubles("\(fx)/col_srgb.npy")
    let slots = (0..<192).filter { cleaned[$0 * 40 + 39] > 0 }
    var sw = 0.0, n = 0
    for s in slots {
        for b in 0..<16 {
            let o = (s * 16 + b) * 3
            let c = ColorDiT.labToSRGB([Double(labRef[o]), Double(labRef[o + 1]), Double(labRef[o + 2])])
            for a in 0..<3 { sw = max(sw, abs((c[a] * 10000).rounded(.toNearestOrEven) / 10000 - want[n])); n += 1 }
        }
    }
    let sOK = n == want.count && sw <= 1.0001e-4
    if !sOK { ok = false }
    print(String(format: "%@ col_srgb             %d values, max_abs %.3e (tol 1e-04: one step of the 4-decimal rounding)", sOK ? "ok  " : "FAIL", n, sw))

    // ---- chains from the image-model tokens
    let cx = try Flow.sample(m.chains.chain.dit, ctx: m.chains.chain.encodeCond(try F("chain_clean_tokens")), noise: try F("chain_noise"))
    check("chain_sample", cx, try F("chain_sample"), tol: 1e-3)
    print(String(format: "peak RSS %.2f GiB", peakRSSGiB()))
    return ok
}

func imageHair() throws {
    let t0 = Date()
    let m = try ImagePathModel(source: source())
    let tLoad = Date().timeIntervalSince(t0)
    var rgb = [Float](repeating: 0, count: 448 * 448 * 3), matte = [Float](repeating: 0, count: 448 * 448)
    var inputBytes: [UInt8]
    if let ip = opt("--image") {
        let pic = try rgbOf(ip), person = try rgbOf(need("--matte"))
        guard person.w == pic.w, person.h == pic.h else { err("the matte must have the image's size"); exit(2) }
        let box = need("--box").split(separator: ",").map { Double($0)! }
        guard box.count == 4, box[2] > box[0], box[3] > box[1] else { err("--box needs X0,Y0,X1,Y1"); exit(2) }
        let pm = (0..<(pic.w * pic.h)).map { person.px[$0 * 3] }
        inputBytes = ModelInput.make(rgb: pic.px, matte: pm, width: pic.w, height: pic.h, box: box)
        matte = ModelInput.matteCrop(matte: pm, width: pic.w, height: pic.h, box: box).map { Float($0) }
    } else {
        let img = try rgbOf(need("--crop"))
        guard img.w == 448, img.h == 448 else { err("the model input must be 448 x 448, got \(img.w) x \(img.h)"); exit(2) }
        inputBytes = img.px
        for i in 0..<(448 * 448) { matte[i] = (0..<3).contains { img.px[i * 3 + $0] < 250 } ? 1 : 0 }
        if let mp = opt("--matte") {
            let mi = try rgbOf(mp)
            guard mi.w == 448, mi.h == 448 else { err("the matte must be 448 x 448"); exit(2) }
            for i in 0..<(448 * 448) { matte[i] = mi.px[i * 3] > 127 ? 1 : 0 }
        }
        if !matte.contains(1) { matte = [Float](repeating: 1, count: matte.count) }
    }
    _ = rgb
    var vrm0 = false
    var body: BodyPlacement? = nil
    if let vp = opt("--vrm") { let v = try VRMBody(vrm: vp); body = v.placement; vrm0 = v.isVRM0 } else if let bp = opt("--body") { body = try BodyPlacement(json: bp) }
    let seed = UInt64(opt("--seed") ?? "0")!
    let r = try HairDeckPipeline.imageHair(m, modelInput: inputBytes, matteCrop: matte, body: body, vrm0Axes: vrm0, seed: seed,
                                           noise: try opt("--noise").map { try NPYReader.load($0) }, springs: !flag("--no-chains"), colours: !flag("--no-colours"))
    if let dir = opt("--save") {
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        var obj = r.cards.map { ["shape": $0.triangle ? "triangle" : "square", "seq": $0.seq] as [String: Any] }
        if let c = r.colours { for i in 0..<obj.count { obj[i]["colors"] = c[i] } }
        try JSONSerialization.data(withJSONObject: obj).write(to: URL(fileURLWithPath: "\(dir)/cards_head.json"))
        try JSONSerialization.data(withJSONObject: r.chains.map { $0.json }).write(to: URL(fileURLWithPath: "\(dir)/chains_head.json"))
        try HDIO.writeNPY(r.tokens, to: "\(dir)/tokens.npy")
        try writePNG(r.modelInput, 448, 448, channels: 3, to: "\(dir)/model_input.png")
    }
    try r.glb.write(to: URL(fileURLWithPath: need("--out")))
    print("wrote \(need("--out")): \(r.glb.count) bytes, \(r.cards.count) cards, \(r.chains.count) chains generated (\(r.chainsUsed) used), \(r.summary.vertices) vertices, "
          + "\(r.summary.springs) springs, colours \(r.colours == nil ? "one" : "per card")\(body == nil ? ", no body: the model's head frame" : "")")
    print(String(format: "load %.1f s | DINOv3 %.2f s, image DiT %.2f s, decode + colours %.2f s, chains %.2f s, rig + glb %.2f s | peak MLX %.2f GiB, peak RSS %.2f GiB",
                 tLoad, r.seconds[0], r.seconds[1], r.seconds[2], r.seconds[3], r.seconds[4], Double(GPU.peakMemory) / Double(1 << 30), peakRSSGiB()))
}

func writePNG(_ px: [UInt8], _ w: Int, _ h: Int, channels: Int, to path: String) throws {
    var rgba = [UInt8](repeating: 255, count: w * h * 4)
    for i in 0..<(w * h) { for a in 0..<3 { rgba[i * 4 + a] = px[i * channels + min(a, channels - 1)] } }
    let img = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                      bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: CGDataProvider(data: Data(rgba) as CFData)!,
                      decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    let dst = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, "public.png" as CFString, 1, nil)!
    CGImageDestinationAddImage(dst, img, nil)
    guard CGImageDestinationFinalize(dst) else { throw HDIO.IOError.bad("cannot write \(path)") }
}

func rgbOf(_ path: String) throws -> (px: [UInt8], w: Int, h: Int) {
    let i = try HDIO.readRGBA(path)
    var out = [UInt8](repeating: 0, count: i.width * i.height * 3)
    for p in 0..<(i.width * i.height) { for a in 0..<3 { out[p * 3 + a] = i.pixels[p * 4 + a] } }
    return (out, i.width, i.height)
}

func gate4b() throws -> Bool {
    let fx = need("--fixtures")
    var ok = true
    for ex in ["girl"] {
        let info = try JSONSerialization.jsonObject(with: FileManager.default.contents(atPath: "\(fx)/\(ex)_box.json")!) as! [String: Any]
        let box = (info["box"] as! [NSNumber]).map { $0.doubleValue }
        func b(_ k: String) -> [Double]? { (info[k] as? [NSNumber])?.map { $0.doubleValue } }
        let hb = ModelInput.headBox(hair: b("hair_box"), face: b("face_box"))!
        let boxErr = zip(hb, box).map { abs($0 - $1) }.max()!
        let rgb = try rgbOf("\(fx)/\(ex)_rgb.png"), person = try rgbOf("\(fx)/\(ex)_person.png")
        let matte = (0..<(rgb.w * rgb.h)).map { person.px[$0 * 3] }
        let got = ModelInput.make(rgb: rgb.px, matte: matte, width: rgb.w, height: rgb.h, box: box)
        let want = try rgbOf("\(fx)/\(ex)_input.png")
        let diffs = zip(got, want.px).map { abs(Int($0) - Int($1)) }
        let nd = diffs.filter { $0 != 0 }.count, md = diffs.max() ?? 0
        let mc = ModelInput.matteCrop(matte: matte, width: rgb.w, height: rgb.h, box: box)
        let wm = try rgbOf("\(fx)/\(ex)_matte.png")
        let mdiff = (0..<(448 * 448)).filter { (mc[$0] == 1) != (wm.px[$0 * 3] > 127) }.count
        // the reference takes the mask boxes at its 1024 px working resolution, so its box can sit a few
        // hundredths of a pixel from the one made from the full-resolution mask boxes
        let pass = boxErr < 0.05 && md == 0 && mdiff == 0
        if !pass { ok = false }
        print(String(format: "%@ %@: head box from the hair and face boxes within %.1e px; model input %d of %d values differ (max %d); matte crop %d pixels differ",
                     pass ? "ok  " : "FAIL", ex, boxErr, nd, got.count, md, mdiff))
    }
    return ok
}

func modelInput() throws {
    let rgb = try rgbOf(need("--image")), person = try rgbOf(need("--matte"))
    guard person.w == rgb.w, person.h == rgb.h else { err("the matte must have the image's size"); exit(2) }
    let box = need("--box").split(separator: ",").map { Double($0)! }
    let matte = (0..<(rgb.w * rgb.h)).map { person.px[$0 * 3] }
    try writePNG(ModelInput.make(rgb: rgb.px, matte: matte, width: rgb.w, height: rgb.h, box: box), 448, 448, channels: 3, to: need("--out"))
    if let mo = opt("--matte-out") {
        try writePNG(ModelInput.matteCrop(matte: matte, width: rgb.w, height: rgb.h, box: box).map { $0 * 255 }, 448, 448, channels: 1, to: mo)
    }
    print("wrote \(need("--out"))")
}

func describeBody() throws {
    let v = try VRMBody(vrm: need("--vrm"))
    let p = v.placement
    print(String(format: "%@: VRM %@, head bone %@ at (%.4f, %.4f, %.4f), rest rotation (%.4f, %.4f, %.4f, %.4f)", need("--vrm"), v.isVRM0 ? "0.x" : "1.0",
                 v.headBone, v.headPosition.x, v.headPosition.y, v.headPosition.z, v.headRotation.imag.x, v.headRotation.imag.y, v.headRotation.imag.z, v.headRotation.real))
    print(String(format: "face mesh '%@': width %.4f m, anchor (%.4f, %.4f, %.4f); head frame → world scale %.5f", v.faceMesh, v.faceWidth, v.anchor.x, v.anchor.y, v.anchor.z, 1 / p.k))
    if let bp = opt("--body") {
        let h = try BodyPlacement(json: bp)
        // the same head-frame points through both placements
        var worst = 0.0
        for q in [SIMD3<Double>(0, 0.45, 0), SIMD3(0.1, 0.3, -0.1), SIMD3(-0.15, 0.1, 0.05), SIMD3(0, -0.2, -0.2)] {
            let a = p.datasetToWorld(p.headToDataset(q)), b = h.datasetToWorld(h.headToDataset(q))
            worst = max(worst, simd_length(a - b))
        }
        let hj = simd_length(p.headJointWorld - h.headJointWorld)
        print(String(format: "against %@: face width %.4f vs %.4f m, head joint differs by %.2e m, head-frame points land within %.2e m", bp, v.faceWidth, h.faceWidthWorld, hj, worst))
    }
}

func hairItem() throws {
    var vrm0 = flag("--vrm0")
    let body: BodyPlacement
    if let vp = opt("--vrm") {
        let v = try VRMBody(vrm: vp)
        body = v.placement
        vrm0 = v.isVRM0
    } else {
        body = try BodyPlacement(json: need("--body"))
    }
    let rig = HairRig.build(cards: try Cards.load(json: need("--cards")), chains: opt("--chains") != nil ? try loadChains(need("--chains")) : [], body: body)
    let colour = opt("--colour")?.split(separator: ",").map { Double($0)! } ?? [0.35, 0.22, 0.14]
    let (data, s) = HairItem.glb(rig, headBone: body.headBone, headJoint: body.headJointWorld, headRotation: body.headRotation, colour: colour, vrm0Axes: vrm0)
    try data.write(to: URL(fileURLWithPath: need("--out")))
    print("wrote \(need("--out")): \(data.count) bytes, \(s.vertices) vertices, \(s.triangles) triangles, \(s.bones) chain bones, \(s.springs) springs, "
          + "\(s.headOnly) vertices on the head only, \(s.threeInfluences) with three influences; cards on a chain \(rig.cardsOnChain)")
}

func chains() throws {
    let out = need("--out")
    let t0 = Date()
    let model = try ChainModel(source: source())
    let tLoad = Date().timeIntervalSince(t0)
    let noise = try NPYReader.load(need("--noise"))
    var rows: [Int] = []
    var ctok: MLXArray
    var tTok = 0.0
    if let tp = opt("--tokens") {
        ctok = try NPYReader.load(tp)
        rows = (try JSONSerialization.jsonObject(with: FileManager.default.contents(atPath: need("--rows"))!) as! [NSNumber]).map { $0.intValue }
    } else {
        let run = need("--cards")
        let fits = try JSONSerialization.jsonObject(with: FileManager.default.contents(atPath: need("--fits"))!) as! [[String: Any]]
        var flat: [Float] = []
        let t = Date()
        for f in fits {
            let r = (f["row"] as! NSNumber).intValue
            rows.append(r)
            flat += HeadFrame.cardTokens(try Cards.load(json: String(format: "%@/JsonResults/%03d.json", run, r)), k: (f["k"] as! NSNumber).doubleValue,
                                         b: (f["b"] as! [NSNumber]).map { $0.doubleValue }, vae: model.vae, tokenizer: model.tokenizer)
        }
        tTok = Date().timeIntervalSince(t)
        ctok = MLXArray(flat, [rows.count, 192, 40])
    }
    precondition(noise.shape[0] == rows.count && ctok.shape[0] == rows.count, "noise, tokens and rows differ in count")
    let t = Date()
    let x = try Flow.sample(model.chain.dit, ctx: model.chain.encodeCond(ctok), noise: noise)
    let tSample = Date().timeIntervalSince(t)
    let D = model.chain.tokenDim, N = ChainDiT.slots
    let flat = x.asArray(Float.self)
    var res: [String: Any] = [:]
    var n = 0
    for (j, r) in rows.enumerated() {
        let cs = model.chain.decode(Array(flat[(j * N * D)..<((j + 1) * N * D)]))
        n += cs.count
        res[String(r)] = cs.map { $0.json }
    }
    try FileManager.default.createDirectory(atPath: out, withIntermediateDirectories: true)
    try JSONSerialization.data(withJSONObject: res).write(to: URL(fileURLWithPath: "\(out)/chains_head.json"))
    let H = Double(rows.count)
    let timing: [String: Any] = ["rows": rows.count, "load_s": tLoad, "card_tokens_s_per_hairstyle": tTok / H, "sample_s_per_hairstyle": tSample / H,
                                 "n_chains_mean": Double(n) / H, "peak_rss_gib": peakRSSGiB(), "mlx_peak_gib": Double(GPU.peakMemory) / Double(1 << 30)]
    let td = try JSONSerialization.data(withJSONObject: timing, options: [.prettyPrinted, .sortedKeys])
    try td.write(to: URL(fileURLWithPath: "\(out)/swift_timing.json"))
    print(String(data: td, encoding: .utf8)!)
}

func run() throws {
    let out = need("--out")
    let plys = need("--plys")
    let t0 = Date()
    let model = try PointCloudModel(source: source())
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
        let tok = try Flow.sample(model.dit, ctx: ctx, noise: noise[rows[0]..<(rows.last! + 1)])
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
    case "gate2":
        let ok = flag("--gpu") ? try gate2() : try Device.withDefaultDevice(.cpu) { try gate2() }
        print(ok ? "GATE PASSED" : "GATE FAILED")
        exit(ok ? 0 : 1)
    case "chains":
        try chains()
    case "gate3":
        let ok = try gate3()
        print(ok ? "GATE PASSED" : "GATE FAILED")
        exit(ok ? 0 : 1)
    case "gate4":
        let ok = flag("--gpu") ? try gate4() : try Device.withDefaultDevice(.cpu) { try gate4() }
        print(ok ? "GATE PASSED" : "GATE FAILED")
        exit(ok ? 0 : 1)
    case "gate4b":
        let ok = try gate4b()
        print(ok ? "GATE PASSED" : "GATE FAILED")
        exit(ok ? 0 : 1)
    case "model-input":
        try modelInput()
    case "image-hair":
        try imageHair()
    case "hair-item":
        try hairItem()
    case "body":
        try describeBody()
    case "run":
        if flag("--cpu") { try Device.withDefaultDevice(.cpu) { try run() } } else { try run() }
    default:
        err("unknown mode \(mode)"); exit(2)
    }
} catch {
    err("error: \(error)"); exit(1)
}
