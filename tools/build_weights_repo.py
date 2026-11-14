"""The consolidated weights snapshot of mlx-hairdeck-swift from a HairDeck workspace: the released files under flat
names, unchanged (float32), plus one config.json with everything else the port reads.

  source /Volumes/Satechi/charm/HairDeck/hairdeck_env.sh
  python tools/build_weights_repo.py OUT_DIR
Files: pointcloud_dit / image_dit / color_dit / chain_dit / card_vae .safetensors (the released checkpoints, byte for
byte), dora_encoder.safetensors (the `encoder.*` tensors of the converted Dora-VAE 1.1 file, the only ones the
condition reads), config.json (run args of every model, the token / latent / card / chain statistics, the canonical
head), SHA256SUMS, the licences and a model card. DINOv3 is not copied: the image path reads `dino.safetensors` of
xocialize/trellis2-mlx, the same file byte for byte (sha256 dcb2e451…8179).
"""
import hashlib
import json
import os
import shutil
import sys

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

sys.path.insert(0, os.environ['HAIRDECK_ROOT'])
from hairdeck import paths  # noqa: E402
from hairdeck.ckpt import load as load_ckpt  # noqa: E402

out = os.path.abspath(sys.argv[1])
os.makedirs(out, exist_ok=True)
MODELS = dict(pointcloud_dit='pointcloud_model', image_dit='image_model', color_dit='color_model', chain_dit='chain_model', card_vae='card_vae')
cfg = dict(format='mlx-hairdeck-swift weights 1', source='hyz317/HairDeck cb20f70 (Apache-2.0); Seed3D/Dora-VAE-1.1 @ 7187b01 (Apache-2.0)', models={})
for name, d in MODELS.items():
    src = paths.ckpt(d, 'model.safetensors')
    shutil.copyfile(src, f'{out}/{name}.safetensors')
    with safe_open(src, 'np') as f:
        meta = json.loads(f.metadata()['hairdeck'])
    cfg['models'][name] = {k: v for k, v in meta.items() if k != 'weights'}
enc = {k: v for k, v in load_file(paths.weights('dora_vae_1_1', 'dora_vae_1_1_shape_model.safetensors')).items() if k.startswith('encoder.')}
save_file(enc, f'{out}/dora_encoder.safetensors', metadata=dict(source='Seed3D/Dora-VAE-1.1@7187b01c78e1bd4a2ddf8301f145f5dc98ac2c22 dora_vae_1_1.ckpt, shape_model encoder tensors'))
J = lambda p: json.load(open(p))  # noqa: E731
hz = np.load(paths.data('head_frame_train.npz'))
cfg.update(
    token_stats=J(paths.data('token_stats.json')), token_stats_head=J(paths.data('token_stats_head.json')),
    latent_stats=J(paths.ckpt('card_vae', 'latent_stats.json')), latent_stats_head=J(paths.ckpt('card_vae', 'latent_stats_head.json')),
    card_stats=J(paths.ckpt('card_vae', 'stats.json')), chain_stats_head=J(paths.data('spring', 'chain_stats_head.json')),
    canonical_head=dict(W=float(hz['W']), O=hz['O'].astype(float).tolist()),
    dinov3=dict(repo='xocialize/trellis2-mlx', file='dino.safetensors', sha256='dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179',
                license='DINOv3 License', attribution='Built with DINOv3'))
json.dump(cfg, open(f'{out}/config.json', 'w'), indent=1)
shutil.copyfile(os.path.join(os.environ['HAIRDECK_ROOT'], 'LICENSE'), f'{out}/LICENSE')
with open(f'{out}/SHA256SUMS', 'w') as f:
    for n in sorted(os.listdir(out)):
        if n.endswith(('.safetensors', '.json')):
            h = hashlib.sha256(open(f'{out}/{n}', 'rb').read()).hexdigest()
            f.write(f'{h}  {n}\n')
            print(n, round(os.path.getsize(f'{out}/{n}') / 1e6, 1), 'MB', h[:12])
