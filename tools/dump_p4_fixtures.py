"""P4 parity fixtures from the PyTorch reference (HairDeck, CPU, fp32, no autocast): DINOv3 tokens of a model-input
crop, the image DiT, the colour model, and the chains made from image-model tokens.

  source /Volumes/Satechi/charm/HairDeck/hairdeck_env.sh
  HAIRDECK_DEVICE=cpu python tools/dump_p4_fixtures.py Fixtures/p4 --crop results/p4_ref_ex01/model_input.png
--crop: a 448 x 448 model input (the matte on white, cropped) as hairdeck.infer.image writes it. The matte of the
fixtures is "not white" (any channel below 250): a stand-in with the right shape for the colour model's patch
statistics, not the preprocessing's matte.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.environ['HAIRDECK_ROOT'])
from hairdeck import dev, paths  # noqa: E402
from hairdeck.cards.tokens import LatentDecoder, clean_tokens, sign_token_normals, tokens_to_cards  # noqa: E402
from hairdeck.ckpt import load as load_ckpt  # noqa: E402
from hairdeck.infer.image import CHAIN_DIT, COLOR_DIT, IMAGE_DIT, euler_cfg  # noqa: E402
from hairdeck.models.colormodel import lab_to_srgb, load_color_dit  # noqa: E402
from hairdeck.models.imgmodel import DinoEncoder, load_img_dit  # noqa: E402
from hairdeck.sampling.flow import fm_sample  # noqa: E402
from hairdeck.spring.chaindit import chain_stats, decode as chain_decode, load_chain_dit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--crop', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = str(dev.select(None))
    assert device == 'cpu', 'run with HAIRDECK_DEVICE=cpu'
    torch.set_grad_enabled(False)

    def save(name, x, dtype=np.float32):
        x = x.detach().cpu().float().numpy() if torch.is_tensor(x) else np.asarray(x)
        np.save(os.path.join(a.out, name + '.npy'), np.ascontiguousarray(x.astype(dtype)))
        print(f'{name:24s} {tuple(x.shape)} {dtype.__name__}', flush=True)

    crop = np.asarray(Image.open(a.crop).convert('RGB'))
    assert crop.shape == (448, 448, 3), crop.shape
    Image.fromarray(crop).save(os.path.join(a.out, 'model_input.png'))
    save('crop', crop, np.int32)
    model, tok, ra = load_img_dit(paths.ckpt(*IMAGE_DIT), device)
    enc = DinoEncoder(ra.dino, ra.res)
    img = torch.from_numpy(crop.copy()).permute(2, 0, 1)[None]
    # DINOv3, stage by stage
    net = enc.net
    x = (img.float() / 255 - enc.mean) / enc.std
    save('dino_pixels', x)
    hw = None
    xt, hw = net.prepare_tokens_with_masks(x)
    save('dino_embed', xt)
    rope = net.rope_embed(H=hw[0], W=hw[1])
    save('dino_rope_sin', rope[0])
    save('dino_rope_cos', rope[1])
    save('dino_block0', net.blocks[0](xt, rope))
    tokens = enc(img)  # bfloat16, as the pipeline passes it on
    save('dino_tokens', tokens)
    ctx = model.encode_cond(tokens)
    save('img_ctx', ctx)
    noise = torch.randn((1, ra.nmax, tok.dim), generator=torch.Generator().manual_seed(0))
    save('noise_seed0', noise)
    t = torch.tensor([0.7])
    save('img_v_cond', model(noise, t, ctx))
    save('img_v_null', model(noise, t, ctx, torch.ones(1, dtype=torch.bool)))
    xs = euler_cfg(model, ctx, noise, 50, 2.0)
    save('img_sample', xs)
    vae = LatentDecoder(paths.ckpt(ra.vae), device)
    cards = tokens_to_cards(tok, xs, vae, fix=True)[0]
    json.dump([dict(shape=int(sh), seq=np.asarray(s, np.float64).tolist()) for s, sh in cards], open(os.path.join(a.out, 'cards.json'), 'w'))

    # colour model
    col, ca = load_color_dit(paths.ckpt(*COLOR_DIT), tok.dim, device)
    matte = (crop < 250).any(-1)
    save('matte', matte, np.int32)
    cctx = col.encode_ctx(tokens, img.float(), torch.from_numpy(matte)[None])
    save('col_ctx', cctx)
    xc = sign_token_normals(tok, clean_tokens(tok, xs.float()))
    save('col_tokens', xc)
    save('col_raw', col.raw(xc, cctx))
    lab = col(xc, cctx)
    save('col_lab', lab)
    slots = torch.nonzero(xc[0, :, tok.groups['exist']][:, 0] > 0)[:, 0]
    srgb = np.round(lab_to_srgb(lab[0].float()[slots].numpy()), 4)
    save('col_srgb', srgb, np.float64)

    # chains from the image-model tokens
    chain, cra, cdim = load_chain_dit(paths.ckpt(*CHAIN_DIT), device)
    cc = chain.encode_cond(clean_tokens(tok, xs.float()))
    save('chain_clean_tokens', clean_tokens(tok, xs.float()))
    xch = fm_sample(chain, cc, 48, cdim, 50, 2.0, 1.0, torch.Generator().manual_seed(0), 'euler')
    save('chain_noise', torch.randn((1, 48, cdim), generator=torch.Generator().manual_seed(0)))
    save('chain_sample', xch)
    json.dump(chain_decode(xch[0].float().numpy(), chain_stats()), open(os.path.join(a.out, 'chains.json'), 'w'))

    ck, kk = load_ckpt(paths.ckpt(*IMAGE_DIT)), load_ckpt(paths.ckpt(*COLOR_DIT))
    extra = sorted(set(kk['ema']) - {'dit.' + k for k in ck['ema']})
    json.dump(dict(img_args=ck['args'], col_args=kk['args'], n_cards=len(cards), n_slots=int(len(slots)),
                   img_keys_not_in_pc=sorted(set(ck['ema']) - set(load_ckpt(paths.resolve('ckpt/pointcloud_model/model.safetensors'))['ema'])),
                   col_extra_keys={k: list(kk['ema'][k].shape) for k in extra},
                   col_mu=kk['ema']['mu'].tolist(), col_sd=kk['ema']['sd'].tolist(),
                   dino_norm_eps=float(net.norm.eps), dino_block_norm_eps=float(net.blocks[0].norm1.eps),
                   rope_periods=net.rope_embed.periods.tolist()), open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
