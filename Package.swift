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
let package = Package(
    name: "mlx-hairdeck-swift",
    platforms: [.macOS("26.0")],
    products: [
        .library(name: "HairDeckCore", targets: ["HairDeckCore"]),
    ],
    dependencies: [
        // The fleet's pinned mlx-swift fork (AB-D-0208) — the same URL the engine uses.
        .package(url: "https://github.com/xocialize/mlx-swift", from: "0.32.4"),
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
    ]
)
