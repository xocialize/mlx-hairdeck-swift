"""P2 parity fixtures from the PyTorch reference (HairDeck, CPU, fp32, no autocast): the chain DiT and the card VAE
encoder with the head-frame re-tokenisation of generated cards.

  source /Volumes/Satechi/charm/HairDeck/hairdeck_env.sh
  HAIRDECK_DEVICE=cpu python tools/dump_p2_fixtures.py Fixtures/p2 --cards results/swift_pointcloud/seed0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ['HAIRDECK_ROOT'])
from hairdeck import dev, paths  # noqa: E402
from hairdeck.cards.vae import arc_u, load_vae, pad_seqs  # noqa: E402
from hairdeck.ckpt import load as load_ckpt  # noqa: E402
from hairdeck.eval.cards import load_cards  # noqa: E402
from hairdeck.sampling.flow import fm_sample  # noqa: E402
from hairdeck.spring.chaindit import Data, chain_stats, decode, load_chain_dit, make_tok  # noqa: E402
from hairdeck.testset.chains import CKPT, pointcloud_run_cards  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--cards', required=True, help='a point-cloud run (RUN/JsonResults) for the encoder fixtures')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = str(dev.select(None))
    assert device == 'cpu', 'run with HAIRDECK_DEVICE=cpu'
    torch.set_grad_enabled(False)

    def save(name, x, dtype=np.float32):
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
        np.save(os.path.join(a.out, name + '.npy'), np.ascontiguousarray(x.astype(dtype)))
        print(f'{name:28s} {tuple(x.shape)} {dtype.__name__}', flush=True)

    model, ra, dim = load_chain_dit(paths.resolve(CKPT), device)
    stats = chain_stats()
    tok = make_tok(ra['vae'], device)
    D = Data('test', tok, ra['vae'], stats, device)
    hs = [int(h) for h in D.items]
    ctok = D.cards(torch.from_numpy(D.items), perm=False)[:4]
    save('chain_card_tokens', ctok)
    ctx = model.encode_cond(ctok)
    save('chain_ctx', ctx)
    noise = torch.randn((4, D.nmax, D.dim), generator=torch.Generator().manual_seed(0))
    save('chain_noise', noise)
    t = torch.full((4,), 0.7)
    save('chain_v_cond', model(noise, t, ctx))
    save('chain_v_null', model(noise, t, ctx, torch.ones(4, dtype=torch.bool)))
    x = fm_sample(model, ctx, D.nmax, D.dim, 50, 2.0, 1.0, torch.Generator().manual_seed(0), 'euler')
    save('chain_sample', x)
    json.dump([decode(r, stats) for r in x.float().numpy()], open(os.path.join(a.out, 'chains.json'), 'w'))

    # the card VAE encoder on generated cards, and their head-frame card tokens
    vae = load_vae(paths.ckpt(ra['vae']), device)
    hz = np.load(paths.data('head_frame_test.npz'))
    rows = hs[:2]
    h = rows[0]
    cards = load_cards(f'{a.cards}/JsonResults/{h:03d}.json')[:192]
    X, lens = pad_seqs([s for s, _ in cards], vae.L)
    k = float(hz['k'][h])
    save('enc_X', X * k)
    save('enc_lens', lens, np.int32)
    save('enc_u', arc_u(X * k, lens))
    save('enc_latents', vae.encode_padded(X * k, lens))
    ptok, n_cards = pointcloud_run_cards(a.cards, rows, tok, vae)
    save('enc_card_tokens', ptok)
    for i in rows:
        os.makedirs(os.path.join(a.out, 'cards'), exist_ok=True)
        json.dump(json.load(open(f'{a.cards}/JsonResults/{i:03d}.json')), open(os.path.join(a.out, 'cards', f'{i:03d}.json'), 'w'))
    ck = load_ckpt(paths.resolve(CKPT))
    json.dump(dict(rows=rows, n_cards=n_cards, hs=hs, k=[float(hz['k'][i]) for i in rows], b=[hz['b'][i].astype(float).tolist() for i in rows],
                   chain_args=ck['args'], chain_dim=ck['dim'], nmax=int(D.nmax),
                   chain_keys={k_: list(v.shape) for k_, v in ck['ema'].items() if not k_.startswith('blocks.') or k_.startswith('blocks.0.')},
                   chain_stats=stats, token_stats_head=json.load(open(paths.data('token_stats_head.json'))),
                   latent_stats_head=json.load(open(paths.ckpt(ra['vae'], 'latent_stats_head.json')))),
              open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
