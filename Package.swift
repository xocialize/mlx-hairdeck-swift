// swift-tools-version:5.9
import PackageDescription

// mlx-hairdeck-swift — Swift-MLX port of hyz317/HairDeck (Apache-2.0): hair cards as tokens.
//
// P1 (bridge AB-T-0477, plan AB-P-0012): the point-cloud path
//   point cloud (10,000 xyz + normals) → Dora-VAE 1.1 encoder (512 × 768 tokens) → point-cloud set DiT
//   (Euler 50, CFG 2) → card tokens → card VAE decoder → hair cards.
//
// HairDeckCore: the ported blocks (pure MLX / Foundation; no MLXToolKit, no engine). Every stage is
//   gated against PyTorch CPU fp32 captures (tools/dump_p1_fixtures.py → Fixtures/p1/*.npy).
// RunHairDeck: the gate + end-to-end CLI. Gates are CLI modes, not `swift test` (the SPM test
//   product's metallib is unreliable).
// The engine contract. `characterPart` is contract 1.63.0 (mlx-engine-swift 0.78.0, AB-A-0277). The engine comes
// from GitHub; MLX_ENGINE_PATH is an optional override for building against a local checkout.
let enginePath = Context.environment["MLX_ENGINE_PATH"]

let package = Package(
    name: "mlx-hairdeck-swift",
    platforms: [.macOS("26.0")],
    products: [
        .library(name: "HairDeckCore", targets: ["HairDeckCore"]),
        .library(name: "HairDeckKit", targets: ["HairDeckKit"]),
    ],
    dependencies: [
        // The fleet's mlx-swift fork — the same URL and floor the engine uses. 0.33.1 is upstream 0.33.0 with the
        // split-head attention routes off (AB-D-0264); the fork's own 0.33.0 tag is unpatched upstream.
        .package(url: "https://github.com/xocialize/mlx-swift", from: "0.33.1"),
        enginePath.map { .package(path: $0) } ?? .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.82.1"),
        // First-run materialization of the declared weight sources into the engine's model store.
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.9.0"),
    ],
    targets: [
        .target(name: "HairDeckCore", dependencies: [
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "MLXRandom", package: "mlx-swift"),
        ]),
        .executableTarget(name: "RunHairDeck", dependencies: [
            "HairDeckCore",
            .product(name: "MLX", package: "mlx-swift"),
        ]),
        // The MLXToolKit layer: the `characterPart` ModelPackage conformer, its configuration and
        // first-run weight materialization. Offline conformance runs under `swift test`.
        .target(name: "HairDeckKit", dependencies: [
            "HairDeckCore",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            .product(name: "HuggingFace", package: "swift-huggingface"),
        ]),
        // Drives the package through MLXServeEngine: register → prepare → run.
        .executableTarget(name: "RunHairDeckEngine", dependencies: [
            "HairDeckKit",
            "HairDeckCore",
            .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            .product(name: "MLX", package: "mlx-swift"),
        ]),
        .testTarget(name: "HairDeckKitTests", dependencies: [
            "HairDeckKit",
            .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
        ]),
    ]
)
