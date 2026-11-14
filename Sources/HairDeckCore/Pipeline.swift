import Foundation
import MLX
import MLXRandom

/// The image path end to end, shared by the CLI and the engine package: a picture, its person matte
/// and a head box → cards, per-card colours and chains → a skinned hair item on a body (or in the
/// model's own head frame when there is none).
public enum HairDeckPipeline {
    public struct Result {
        public var glb: Data
        public var cards: [HairCard]
        /// Per card 16 sRGB colours from root to tip, when the colour model ran.
        public var colours: [[[Double]]]?
        public var chains: [ChainDiT.Chain]
        public var tokens: MLXArray
        public var modelInput: [UInt8]
        public var summary: HairItem.Summary
        public var chainsUsed: Int
        /// Seconds: DINOv3, image DiT, decode + colours, chains, rig + glb.
        public var seconds: [Double]
    }

    /// rgb: h × w × 3; matte: h × w, 0..255 (white = the character); box: [x0, y0, x1, y1] in pixels of
    /// the picture. `noise` ([1, 192, 40]) replaces the generator's for a sample-for-sample comparison
    /// with the reference; otherwise both the card noise and the chain noise come from MLX's generator
    /// keyed by `seed`. `checkpoint` is called between stages and after every sampling step, and may throw.
    public static func imageHair(_ m: ImagePathModel, rgb: [UInt8], matte: [UInt8], width: Int, height: Int, box: [Double],
                                 body: BodyPlacement? = nil, vrm0Axes: Bool = false, seed: UInt64 = 0, noise: MLXArray? = nil,
                                 springs: Bool = true, colours wantColours: Bool = true,
                                 checkpoint: () throws -> Void = {}) throws -> Result {
        let input = ModelInput.make(rgb: rgb, matte: matte, width: width, height: height, box: box)
        let matteCrop = ModelInput.matteCrop(matte: matte, width: width, height: height, box: box)
        return try imageHair(m, modelInput: input, matteCrop: matteCrop.map { Float($0) }, body: body, vrm0Axes: vrm0Axes, seed: seed, noise: noise,
                             springs: springs, colours: wantColours, checkpoint: checkpoint)
    }

    /// From a ready 448 × 448 model input and its matte crop (448 × 448 of 0 / 1).
    public static func imageHair(_ m: ImagePathModel, modelInput: [UInt8], matteCrop: [Float], body: BodyPlacement? = nil, vrm0Axes: Bool = false,
                                 seed: UInt64 = 0, noise: MLXArray? = nil, springs: Bool = true, colours wantColours: Bool = true,
                                 checkpoint: () throws -> Void = {}) throws -> Result {
        var secs: [Double] = []
        var t = Date()
        func lap() { secs.append(Date().timeIntervalSince(t)); t = Date() }
        let rgbA = MLXArray(modelInput.map { Float($0) }, [1, 448, 448, 3])
        let tokens = m.dino(DINOv3.normalise(rgbA))
        eval(tokens)
        lap()
        try checkpoint()
        let x0 = noise ?? MLXRandom.normal([1, 192, 40], key: MLXRandom.key(seed))
        let x = try Flow.sample(m.image.dit, ctx: m.image.encodeCond(tokens), noise: x0) { _ in try checkpoint() }
        lap()
        let cards = Cards.fromTokens(x, tokenizer: m.tokenizer, vae: m.vae)[0]
        let cleaned = m.tokenizer.clean(x.asArray(Float.self))
        var colours: [[[Double]]]? = nil
        if wantColours && !cards.isEmpty {
            let lab = m.color(MLXArray(m.tokenizer.signNormals(cleaned), [1, 192, 40]),
                              m.color.encodeCtx(tokens, rgb: rgbA, matte: MLXArray(matteCrop, [1, 448, 448]))).asArray(Float.self)
            let slots = (0..<192).filter { cleaned[$0 * 40 + 39] > 0 }
            colours = slots.map { s in (0..<16).map { b in
                let o = (s * 16 + b) * 3
                return ColorDiT.labToSRGB([Double(lab[o]), Double(lab[o + 1]), Double(lab[o + 2])]).map { ($0 * 10000).rounded(.toNearestOrEven) / 10000 }
            } }
        }
        lap()
        try checkpoint()
        var chains: [ChainDiT.Chain] = []
        if springs {
            let cn = MLXRandom.normal([1, ChainDiT.slots, m.chains.chain.tokenDim], key: MLXRandom.key(seed))
            let cx = try Flow.sample(m.chains.chain.dit, ctx: m.chains.chain.encodeCond(MLXArray(cleaned, [1, 192, 40])), noise: cn) { _ in try checkpoint() }
            chains = m.chains.chain.decode(cx.asArray(Float.self))
        }
        lap()
        let place = body ?? VRMBody.canonicalPlacement
        let rig = HairRig.build(cards: cards, chains: chains.map { (joints: $0.joints, stiffness: $0.stiffness) }, body: place)
        let (data, summary) = HairItem.glb(rig, headBone: place.headBone, headJoint: place.headJointWorld, cardColours: colours,
                                           headRotation: place.headRotation, vrm0Axes: vrm0Axes)
        lap()
        return Result(glb: data, cards: cards, colours: colours, chains: chains, tokens: x, modelInput: modelInput, summary: summary,
                      chainsUsed: rig.chains.count, seconds: secs)
    }
}
