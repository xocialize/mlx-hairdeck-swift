"""P1 parity fixtures from the PyTorch reference (HairDeck, CPU, fp32, no autocast).

  source /Volumes/Satechi/charm/HairDeck/hairdeck_env.sh
  HAIRDECK_DEVICE=cpu python tools/dump_p1_fixtures.py Fixtures/p1 --ply Fixtures/pictures/girl_hair.ply
  (--row N instead: CHARM's test point cloud N, which is GPL-3.0 data and is not kept in this repo)

Writes .npy (float32 / int32, C order) for: the Dora encoder (points, FPS indices, intermediate and final tokens), one
set-DiT forward (conditional and null), the full Euler-50 CFG-2 sampling from seeded CPU noise, the card VAE decoder
and the decoded cards (cards.json), plus a key/shape/dtype listing of the three checkpoints and the stats the Swift
side reads (meta.json).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HD = os.environ['HAIRDECK_ROOT']
sys.path.insert(0, HD)
from hairdeck import dev, paths  # noqa: E402
from hairdeck.cards.tokens import LatentDecoder, tokens_to_cards  # noqa: E402
from hairdeck.encoders import dora  # noqa: E402
from hairdeck.encoders.common import condition_plys, condition_points  # noqa: E402
from hairdeck.models.pc_dit import load_pc_dit  # noqa: E402
from hairdeck.models.setdit import timestep_embedding  # noqa: E402
from hairdeck.sampling.flow import fm_sample  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--row', type=int, default=0)
ap.add_argument('--ply', default='', help='a condition point cloud (hair_reform.ply layout) instead of a CHARM test row')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
device = str(dev.select(None))
assert device == 'cpu', 'run with HAIRDECK_DEVICE=cpu'
torch.set_grad_enabled(False)


def save(name, x, dtype=np.float32):
    x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    np.save(os.path.join(a.out, name + '.npy'), np.ascontiguousarray(x.astype(dtype)))
    print(f'{name:28s} {tuple(x.shape)} {dtype.__name__}', flush=True)


# ------------------------------------------------------------------ Dora encoder
pts = torch.from_numpy(condition_points(a.ply or condition_plys('test')[a.row]))[None]  # [1, 10000, 6]
save('dora_points', pts)
m = dora.build_dora(device=device)
ic, is_ = m.query_indices(pts, pts, 256)
save('dora_fps_idx', ic, np.int32)
W, H = m.W, m.H
dc, ds = m._embed(pts, m.k_in), m._embed(pts, m.k_in_sharp)
save('dora_fourier', dora.fourier_xyz(pts[..., :3], 8, m.with_pi)[:, :64])
save('dora_embed_coarse_head', dc[:, :64])
q = torch.cat([dora.gather_rows(dc, ic), dora.gather_rows(ds, is_)], 1)
save('dora_query', q)
x = dora._cross_block(q, dc, W, m.k_ca, H, m.qk_norm) + dora._cross_block(q, ds, W, m.k_ca_sharp, H, m.qk_norm)
save('dora_after_cross', x)
x0 = dora._self_block(x, W, 'encoder.self_attn.resblocks.0', H, m.qk_norm)
save('dora_after_self0', x0)
tokens = dora.encode_dora(m, pts, 512, 'copy')
save('dora_tokens', tokens)
dora_meta = dict(qk_norm=m.qk_norm, with_pi=m.with_pi, heads=H, n_enc=m.n_enc)
dora_keys = {k: [list(v.shape), str(v.dtype)] for k, v in W.items() if k.startswith('encoder.')}
del m

# ------------------------------------------------------------------ set DiT
model, tok, ra = load_pc_dit(paths.resolve('ckpt/pointcloud_model/model.safetensors'), device)
cond = tokens.half().float()  # the released condition is stored as float16
save('dit_cond', cond)
ctx = model.encode_cond(cond)
save('dit_ctx', ctx)
g = torch.Generator().manual_seed(0)
noise = torch.randn((1, 192, tok.dim), generator=g)
save('noise_seed0', noise)
t = torch.tensor([0.7])
save('dit_temb', timestep_embedding(t, 256))
c = model.t_mlp(timestep_embedding(t, 256)) + model.pool_mlp(ctx.mean(1))
save('dit_c', c)
h = model.inp(noise)
save('dit_inp', h)
save('dit_block0', model.blocks[0](h, c, ctx))
save('dit_v_cond', model(noise, t, ctx))
save('dit_v_null', model(noise, t, ctx, torch.ones(1, dtype=torch.bool)))
g = torch.Generator().manual_seed(0)
x = fm_sample(model, ctx, 192, tok.dim, 50, 2.0, 1.0, g, 'euler')
save('sample_tokens', x)
g = torch.Generator().manual_seed(0)
save('sample_tokens_4step', fm_sample(model, ctx, 192, tok.dim, 4, 2.0, 1.0, g, 'euler'))

# ------------------------------------------------------------------ tokens -> cards
d = tok.decode(x.float())
save('tok_root', d['root'])
save('tok_z', d['z'])
save('tok_nrm', d['nrm'])
save('tok_shape', d['shape'], np.int32)
save('tok_exist', d['exist'], np.int32)
dec = LatentDecoder(paths.ckpt(ra.vae), device)
vae = dec.vae
js = torch.nonzero(d['exist'][0])[:, 0]
z = d['z'][0, js].float()
logits = vae.model.n_head(z)
save('vae_n_logits', logits)
ns = (logits.argmax(-1) + 1).clamp(min=3)
save('vae_ns', ns, np.int32)
ar = torch.arange(vae.L)
mask = ar[None] < ns[:, None]
u = (ar[None].float() / (ns[:, None].float() - 1).clamp(min=1)).clamp(max=1.0) * mask
save('vae_u', u)
save('vae_from_z', vae.model.from_z(z))
feat = vae.model.decode(z, u, mask)
save('vae_decode_raw', feat)
save('vae_decode', vae.norm.inv(feat.float()))
cards = tokens_to_cards(tok, x, dec)[0]
json.dump([dict(shape=int(sh), seq=np.asarray(s, np.float64).tolist()) for s, sh in cards], open(os.path.join(a.out, 'cards.json'), 'w'))
print('cards', len(cards), flush=True)

from hairdeck.ckpt import load as load_ckpt  # noqa: E402
ck = load_ckpt(paths.resolve('ckpt/pointcloud_model/model.safetensors'))
vk = load_ckpt(paths.ckpt(ra.vae, 'model.safetensors'))
json.dump(dict(
    row=a.row, dora=dora_meta, dora_encoder_keys=dora_keys,
    dit_args=ck['args'], dit_keys={k: [list(v.shape), str(v.dtype)] for k, v in ck['ema'].items()},
    vae_args=vk['args'], vae_keys={k: [list(v.shape), str(v.dtype)] for k, v in vk['model'].items()},
    token_stats=json.load(open(paths.data('token_stats.json'))),
    latent_stats=json.load(open(paths.ckpt(ra.vae, 'latent_stats.json'))),
    vae_stats=json.load(open(paths.ckpt(ra.vae, 'stats.json'))),
    n_cards=len(cards), torch=torch.__version__), open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
