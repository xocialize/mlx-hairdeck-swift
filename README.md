# mlx-hairdeck-swift

Swift-MLX port of [HairDeck](https://github.com/hyz317/HairDeck) (Apache-2.0): hair as a set of card tokens for
anime-style avatars. Work in progress — plan AB-P-0012 on the bridge.

## Status

| Phase | What | State |
|---|---|---|
| P1 | point cloud → Dora-VAE 1.1 encoder → point-cloud set DiT (Euler 50, CFG 2) → card VAE decoder → cards | parity gate passed, Table 1 reproduced (2026-11-13) |
| P2 | spring-bone chains: chain DiT, chain decoding, card VAE encoder + head-frame card tokens | parity gate passed, Table 4 reproduced (2026-11-13) |
| P3 | hair item `.glb` writer (card mesh, skinning, `VRMC_springBone`), no Blender | rig equal to the reference; `.glb` loads and swings in three-vrm; body read from any VRoid-style VRM; merged into Spec by Item Studio's `mergeVrm` with its springs (2026-11-13) |
| P4 | image path: P4a generation from a model-input crop (DINOv3, image DiT, colour model, chains) — parity gate passed, coloured hair item end to end (2026-11-13); P4b preprocessing (matte, head crop) | P4b: picture + matte + head box → coloured hair item in one call, model input bit-exact; the head box comes from the caller (decision 2026-11-13), the matte from a matting model outside this package |
| P5 | MLXEngine `ModelPackage`, conformance, weights repo, Gaea plugin | `HairDeckKit`: the `characterPart` package (contract 1.63.0) — offline conformance green, engine-driven run byte-identical to the CLI, weights download into a clean store (2026-11-13). Open: the engine's 0.78.0 tag is not on GitHub yet (build with `MLX_ENGINE_PATH`), the footprint through the memory harness, the Gaea plugin (after `mesh.generate_part`, AB-T-0482) |

## P1 results

Parity against the PyTorch reference (CPU, fp32; `tools/dump_p1_fixtures.py` → `Fixtures/p1`), CPU stream, max abs:
Dora tokens 2.9e-5, one DiT forward 4.0e-6, 50-step sample 7.3e-6, decoded cards 6.0e-8 (point counts and
classes equal). On the GPU the gate passes with `MLX_ENABLE_TF32=0`; with MLX's default TF32-class fp32 matmuls
(AB-L-0175) the 50-step tokens differ by 0.10 max abs (cosine 0.99997).

Table 1 on CHARM's 100 test point clouds, mean of seeds 0–2, same noise as the reference, evaluated with the
reference's own code (`tools/eval_p1.py`); Swift on the GPU with default TF32:

| | CD | EMD | HD | IoU | Orient. |
|---|---|---|---|---|---|
| Paper | 0.0066 | 0.0084 | 0.0242 | 0.867 | 16.8° |
| PyTorch on this Mac (MPS, bf16) | 0.00654 | 0.00837 | 0.02375 | 0.8666 | 16.74° |
| Swift-MLX | 0.00654 | 0.00836 | 0.02372 | 0.8655 | 16.66° |

Mac Studio (128 GB), release build, batch 10: condition 0.02 s, sampling 0.47 s, decoding 0.02 s per hairstyle;
peak MLX memory 2.8 GiB, peak RSS 1.3 GiB.

## P2 results

Parity against PyTorch (CPU, fp32; `tools/dump_p2_fixtures.py` → `Fixtures/p2`), max abs: chain DiT forward 1.0e-5,
50-step sample 1.1e-5, decoded chains 6e-17 (chain and joint counts equal), card VAE encoder latents 9.8e-6,
head-frame card tokens 1.5e-5.

Table 4 on the 58 test hairstyles with chains, mean of seeds 0–2, the reference's noise, scored by the reference's
evaluation (`tools/eval_p2.py`); Swift on the GPU with default TF32:

| | CD | Rec. | Prec. | Motion | Fit |
|---|---|---|---|---|---|
| Artists' cards — paper | 0.279 | 0.625 | 0.585 | 0.702 | 0.953 |
| Artists' cards — PyTorch on this Mac, same seeds | 0.294 | 0.596 | 0.573 | 0.697 | 0.950 |
| Artists' cards — Swift-MLX | 0.292 | 0.597 | 0.576 | 0.696 | 0.951 |
| Point-cloud cards — PyTorch on this Mac | 0.322 | 0.538 | 0.551 | — | 0.957 |
| Point-cloud cards — Swift-MLX (on the Swift cards) | 0.321 | 0.541 | 0.550 | — | 0.957 |

(Chains CD varies 0.270–0.297 between seeds; seeds 0–2 sit above the paper's mean on PyTorch too.)
Chains: 0.03 s per hairstyle for sampling (58 in one batch) and 0.03 s for the head-frame card tokens; peak MLX
memory 2.0 GiB.

## P3 results

The rig (`HairRig`: card mesh, UVs, card → chain assignment, vertex binding) against `hairdeck.vrm.prep` on HairDeck's
avatar body, for test row 1 (165 cards, 11 chains; `tools/dump_p3_fixtures.py` → `Fixtures/p3`): 21,048 vertices equal
to 2.2e-16, 51,300 faces equal, UVs to 1.8e-15, chain and card indices equal, weights to 2.0e-14. One vertex lies exactly
on a joint's bisector and picks the other of its two segments (the same binding: both at α = 0.5).

The hair item (`HairItem.glb`, 1.8 MB for this hairstyle): `inspect-vrm.mjs` reports no problems (11 springs, 55 spring
joints, 45 skin joints). In three.js 0.162 with `@pixiv/three-vrm-springbone` 3.5.5 (`tools/check_hair_item.mjs`) it loads
as a skinned mesh (weights sum to 1 within 3e-8) and under a 50° head turn the chain ends lag by up to 0.29 m and
are back at rest after 2.5 s. Rendered on the avatar in Blender (`tools/render_hair_item.py`):

![hair item on the avatar](docs/p3_hair_item_on_avatar.jpg)

**Body from a VRM** (`VRMBody`): the face mesh's POSITION bounds and the humanoid head bone, read from the file's
JSON, give the same placement as HairDeck's Blender-made fit of its avatar — head-frame points land within 1.0e-8 m,
the head joint within 3.6e-7 m. VRM 0.x and 1.0 files are both read; the item is written in the file's own axes, with
the head bone's rest rotation (Spec's is not identity).

**On Spec, through Item Studio's merge** (`tools/check_merged_vrm.mjs`, with `mergeVrm` extended to carry an item's
spring bones): Spec.vrm + the hair item → a VRM 1.0 file with 13 springs (2 of the body, 11 hair), every hair chain
intact under the head bone, the skinned hair within 2.4e-7 m of the item's vertices at rest, chain ends lagging up to
0.29 m under the 50° head turn and back at rest after 2.5 s. The inspector reports no problems.

Not in the item: the body's colliders (so HairDeck's rule that keeps the scalp part of a chain rigid inside the head
collider is not applied; Spec has none), and per-card colours (P4). The face mesh is found by name (`face`), which
holds for VRoid models.

## P4a results

Parity against PyTorch (CPU, fp32; `tools/dump_p4_fixtures.py` → `Fixtures/p4`, demo image ex01), max abs: DINOv3 RoPE
4.6e-7, patch embedding 2.6e-5, block 0 2.5e-8 relative, final tokens 3.9e-3 (they are rounded to bfloat16; one step);
image DiT forward 5.2e-6, 50-step sample 8.3e-6, 89 cards equal to 1.8e-7; colour context 3.1e-6, colour output
3.7e-6, Lab 8.0e-5, the 4-decimal sRGB values equal; chains from the image tokens 1.2e-6.

End to end from a 448 × 448 model input (`RunHairDeck image-hair`, GPU): DINOv3 0.33 s, image DiT 1.39 s, decoding
and colours 0.80 s, chains 1.11 s, rig and `.glb` 0.01 s; peak MLX memory 3.8 GiB. The PyTorch reference on this Mac
(MPS) takes 4.7 + 2.3 + 1.8 s for the same stages, after 21.1 s of preprocessing (SAM 3 13.3 s, BiRefNet 7.2 s).
The demo image and the generated hair with its per-card colours on Spec:

![image to hair on Spec](docs/p4_image_to_hair_on_spec.jpg)

Per-card colours travel as a small texture (one 8-pixel column per card, 18 rows), used as the base colour and the
MToon shade texture. The crop above comes from the reference's preprocessing, and the matte handed to the colour
model was "not white".

## P4b so far

`ModelInput` turns a picture, its person matte and a head box into the 448 px model input and the matte crop with
Pillow's own integer arithmetic (alpha compositing, premultiplied bicubic resize, nearest for the matte): on demo
images ex01 and ex02 all 602,112 values of the model input and every matte pixel equal the reference's
(`RunHairDeck gate4b`). The head box is the square around hair ∪ face with an 8 % margin.

The head box is an input by decision (2026-11-13): the app passes it, for example a rectangle the user drags; an
automatic source (SAM 3 or another) can be added later behind the same input. With the picture, the matte and the box,
`RunHairDeck image-hair --image … --matte … --box …` goes all the way to the hair item: on ex02 the model input equals
the reference's and the item (103 cards, 13 springs) passes the inspector; a second call in a warm process takes
DINOv3 0.06 s, image DiT 0.64 s, colours 0.04 s, chains 0.19 s.

Still from outside: the person matte (BiRefNet-matting in the reference; the fleet has `mlx-birefnet-swift`) and the
hair and face boxes (SAM 3 with text prompts in the reference). Reference timings on this Mac, 12 demo images, MPS:
SAM 3 0.8–0.9 s and the matte 0.2 s per image once warm (13.3 s and 7.2 s for the first image); the boxes of all 12
are in `Fixtures/p4b/*_box.json`.

## How much the two outside inputs matter

On the 12 demo images (`tools/input_sensitivity.py`; same noise, hair compared by the chamfer distance between card
control points, in face widths), against the reference's own matte and head box:

| change | median | largest |
|---|---|---|
| another seed, same inputs | 0.072 | 0.338 |
| the matte from Gaea's `image.matte` (BiRefNet) instead of the reference's | 0.040 | 0.209 |
| head box 15 % larger | 0.043 | 0.110 |
| head box 10 % smaller | 0.036 | 0.091 |
| head box moved 8 % of its side right / down | 0.033 / 0.037 | 0.135 / 0.133 |

Each change moves the hair less than drawing another seed does, on the median and in the worst case; per image the
only exception is one picture where the larger box (0.062) exceeds that picture's seed spread (0.043). The mattes
themselves agree with the reference's at IoU 0.89–0.98 on ten pictures and badly on two busy illustrations (0.20 and
0.58: BiRefNet alone keeps the background the reference's SAM 3-guided matte drops), and even there the hair changes
less than with another seed.

## P5: the engine package

`HairDeckKit` wraps the image path as an MLXEngine `ModelPackage` on the `characterPart` capability (mlx-engine-swift
0.78.0, contract 1.63.0, AB-A-0277): `CharacterPartRequest(image:, subjectBox: [x0, y0, x1, y1], body:, seed:,
springBones:, colors:)` → `CharacterPartResponse(mesh:, part: .hair, fittedToBody:)`. With a body the item is placed on
its head; without one it comes back in the model's own head frame. A picture with transparency is used as it is cut
out; an opaque one needs the host's matting (`HairDeckConfiguration.matting`, the Trellis2 pattern) and is refused with
the reason otherwise. The point-cloud path is not a surface of the package: a bare cloud has no head to hang chains on
or to place on a body.

- Offline conformance (`swift test`, 10 tests: contract version, surface, declaration, specialty, licences, configuration,
  requirements, weight sources and probe, `run()`'s refusals, cancellation first): green.
- Through `MLXServeEngine` (`RunHairDeckEngine`: register → prepare → run): the hair item equals the CLI's byte for byte
  for the same picture, matte, box, body and seed; run 0.95 s, peak MLX 3.76 GiB.
- First use on an empty model store: `prepare()` downloads `xocialize/hairdeck-mlx` (the image path's five files) and
  `dino.safetensors` of `xocialize/trellis2-mlx` — 2.6 GB in 76 s here — and the result is again the same bytes. The
  weights repo is private for now, so the download needs `HF_TOKEN`.
- Declared footprint (one float32 tier): 2.9 GB resident + 1.3 GB peak activation, from the in-process peak; the
  engine's memory harness has not been run on it yet.

## Use

The weights come from either a HairDeck workspace (`--workspace HAIRDECK_ROOT`, the layout of its
`scripts/setup/download_weights.py`) or the consolidated snapshot `xocialize/hairdeck-mlx` (`--snapshot DIR`, built by
`tools/build_weights_repo.py`: the released tensors unchanged under flat names plus one `config.json`); the image path
also takes `--dino FILE`, the `dino.safetensors` of `xocialize/trellis2-mlx`.

```bash
swift run -c release RunHairDeck gate --workspace HAIRDECK_ROOT --fixtures Fixtures/p1
swift run -c release RunHairDeck run --workspace HAIRDECK_ROOT --plys DIR --noise NOISE.npy --out RUN
swift run -c release RunHairDeck gate2 --workspace HAIRDECK_ROOT --fixtures Fixtures/p2
swift run -c release RunHairDeck chains --workspace HAIRDECK_ROOT --cards RUN --fits FITS.json --noise NOISE.npy --out CHAINS
swift run -c release RunHairDeck gate3 --fixtures Fixtures/p3
swift run -c release RunHairDeck hair-item --cards CARDS.json --chains CHAINS.json --vrm BODY.vrm --out HAIR.glb
swift run -c release RunHairDeck body --vrm BODY.vrm
swift run -c release RunHairDeck gate4 --workspace HAIRDECK_ROOT --fixtures Fixtures/p4
swift run -c release RunHairDeck image-hair --workspace HAIRDECK_ROOT --image IMAGE.png --matte MATTE.png --box X0,Y0,X1,Y1 --vrm BODY.vrm --out HAIR.glb
MLX_ENGINE_PATH=… swift test
MLX_ENGINE_PATH=… IMAGE=cutout.png BOX=x0,y0,x1,y1 BODY=body.vrm OUT=hair.glb MODELS_ROOT=… swift run -c release RunHairDeckEngine
```

`run` reads `DIR/{idx:03d}.ply` (10,000 points with normals, the `hair_reform.ply` format) and writes CHARM-format
cards to `RUN/JsonResults/{idx:03d}.json`. The noise is an input for now (`[rows, 192, 40]`), so results line up
with the reference sample for sample. `chains` needs each row's head fit (`FITS.json`: `[{"row", "k", "b"}]`,
p_head = k · p + b) and writes the chains in the head frame to `CHAINS/chains_head.json`; it also needs
`ckpt/chain_model`, `data/spring/chain_stats_head.json`, `data/token_stats_head.json`.

## Licences

Port code: MIT. HairDeck code and checkpoints: Apache-2.0. Dora-VAE 1.1 code and weights: Apache-2.0. DINOv3 weights:
DINOv3 License (attribution "Built with DINOv3" where the image path is shown); `DINOv3.swift` is lifted from
mlx-trellis2-swift (same author).
`NPYReader`, `SafetensorsHeader`, `Weights` and the base of `Ops` come from mlx-skintokens-swift (MIT, same author).
