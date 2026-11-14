"""How much the image path's result depends on the matte source and on the head box, against its seed-to-seed spread.

  python tools/input_sensitivity.py OPTIONS_DIR --workspace HAIRDECK_ROOT --body BODY.vrm [--examples ex01 ...]
OPTIONS_DIR: ref/<ex>_{rgb,person}.png and ref/<ex>_box.json (tools/dump_p4b_fixtures.py), gaea/<ex>_matte.png
(tools/gaea_options.py). Runs RunHairDeck image-hair (release build, MLX noise) per example:
  base      the reference's matte and box, seed 0
  seed1     the same inputs, seed 1                       (the spread a different seed gives)
  matte     another matte source (gaea/), reference box, seed 0
  box+15    the box 15 % larger about its centre          box-10  10 % smaller
  shift8    the box moved right by 8 % of its side        shift8d moved down by 8 %
and compares each with base: symmetric chamfer distance between the cards' control points (head frame, in face
widths: / 0.1276), and the card counts. Writes OPTIONS_DIR/sensitivity.json.
"""
import argparse
import json
import os
import subprocess

import numpy as np


def pts(d):
    c = json.load(open(f'{d}/cards_head.json'))
    return np.concatenate([np.asarray(x['seq'])[:, :3] for x in c]) if c else np.zeros((0, 3)), len(c)


def chamfer(a, b):
    if not len(a) or not len(b):
        return float('nan')
    d = np.linalg.norm(a[:, None] - b[None], axis=-1)
    return float((d.min(1).mean() + d.min(0).mean()) / 2 / 0.12759823777166274)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('--workspace', required=True)
    ap.add_argument('--body', required=True)
    ap.add_argument('--examples', nargs='+', default=[f'ex{i:02d}' for i in range(1, 13)])
    a = ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exe = os.path.join(here, '.build', 'release', 'RunHairDeck')
    res = {}
    for ex in a.examples:
        box = json.load(open(f'{a.dir}/ref/{ex}_box.json'))['box']
        cx, cy, side = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0]

        def sq(s=1.0, dx=0.0, dy=0.0):
            h = side * s / 2
            return [cx + dx * side - h, cy + dy * side - h, cx + dx * side + h, cy + dy * side + h]
        runs = dict(base=('ref', box, 0), seed1=('ref', box, 1), matte=('gaea', box, 0), **{'box+15': ('ref', sq(1.15), 0)},
                    **{'box-10': ('ref', sq(0.9), 0)}, shift8=('ref', sq(dx=0.08), 0), shift8d=('ref', sq(dy=0.08), 0))
        out = {}
        for name, (src, b, seed) in runs.items():
            d = f'{a.dir}/runs/{ex}/{name}'
            matte = f'{a.dir}/ref/{ex}_person.png' if src == 'ref' else f'{a.dir}/gaea/{ex}_matte.png'
            if not os.path.exists(f'{d}/cards_head.json'):
                os.makedirs(d, exist_ok=True)
                subprocess.run([exe, 'image-hair', '--workspace', a.workspace, '--image', f'{a.dir}/ref/{ex}_rgb.png', '--matte', matte,
                                '--box', ','.join(str(v) for v in b), '--vrm', a.body, '--out', f'{d}/hair.glb', '--save', d, '--seed', str(seed)],
                               check=True, capture_output=True)
            out[name] = pts(d)
        base = out['base']
        res[ex] = {k: dict(chamfer=round(chamfer(base[0], v[0]), 4), cards=v[1]) for k, v in out.items()}
        print(ex, 'cards', base[1], ' '.join(f"{k} {res[ex][k]['chamfer']:.3f} ({res[ex][k]['cards']})" for k in runs if k != 'base'), flush=True)
    json.dump(res, open(f'{a.dir}/sensitivity.json', 'w'), indent=1)
    keys = [k for k in next(iter(res.values())) if k != 'base']
    print('median  ' + ' '.join(f"{k} {np.nanmedian([r[k]['chamfer'] for r in res.values()]):.3f}" for k in keys))
    print('max     ' + ' '.join(f"{k} {np.nanmax([r[k]['chamfer'] for r in res.values()]):.3f}" for k in keys))


if __name__ == '__main__':
    main()
