"""Table 1 of the paper for cards written by the Swift port, with the reference's own evaluation.

  source /Volumes/Satechi/charm/HairDeck/hairdeck_env.sh
  python tools/eval_p1.py noise OUT [--seeds 0 1 2]     OUT/noise_seed<k>.npy: the reference's noise of each seed
                                                        ([100, 192, 40], CPU generator, as hairdeck.dev.randn on MPS)
  python tools/eval_p1.py eval OUT [--seeds 0 1 2]      after RunHairDeck run ... --out OUT/seed<k>: card meshes
                                                        (hairdeck.cards.mesh), the CHARM-protocol scan and metrics
                                                        (hairdeck.testset.pointcloud.report), OUT/eval.json and the table
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ['HAIRDECK_ROOT'])
from hairdeck.cards.mesh import cards_to_mesh  # noqa: E402
from hairdeck.eval.cards import load_cards  # noqa: E402
from hairdeck.testset import pointcloud  # noqa: E402
from hairdeck.testset.common import seed_dir, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('noise', 'eval'))
    ap.add_argument('out')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    if a.mode == 'noise':
        for s in a.seeds:
            x = torch.randn((100, 192, 40), generator=torch.Generator().manual_seed(s))
            np.save(f'{out}/noise_seed{s}.npy', x.numpy())
        return
    for s in a.seeds:
        run = seed_dir(out, s)
        for i in range(100):
            cards = load_cards(f'{run}/JsonResults/{i:03d}.json')
            m = cards_to_mesh(cards)
            if m.geometry:
                m.export(f'{run}/JsonResults/{i:03d}.glb')
            else:
                open(f'{run}/JsonResults/{i:03d}.glb', 'wb').close()
        pointcloud.report(run, dict(ckpt='swift: ' + pointcloud.CKPT, seed=s, steps=50, cfg=2.0), a.workers)
    summarize(out, 'pointcloud', list(a.seeds), pointcloud.TABLES)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('fork')  # the evaluation pools read the parent's globals
    main()
