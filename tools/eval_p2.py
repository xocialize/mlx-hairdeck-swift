"""Table 4 of the paper for chains made by the Swift port, with the reference's own evaluation.

  source /Volumes/Satechi/charm/HairDeck/hairdeck_env.sh
  python tools/eval_p2.py prep OUT          OUT/noise_seed<k>.npy ([58, 48, 27], the reference's CPU noise),
                                            OUT/card_tokens_gt.npy (the artists' cards, head frame), OUT/rows.json,
                                            OUT/fits.json (the head fit of every test row with chains)
  python tools/eval_p2.py eval OUT [--on PC_OUT]
      after RunHairDeck chains ... --out OUT/seed<k>: chains back to the dataset frame with the characters' own
      chains beside them (RUN/chains.json), then hairdeck.testset.chains.report; --on: the point-cloud output
      folder whose cards the chains were made for (default: the artists' cards)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ['HAIRDECK_ROOT'])
from hairdeck import paths  # noqa: E402
from hairdeck.eval.cards import load_cards  # noqa: E402
from hairdeck.spring.chaindit import NCARD, Data, chain_stats, chains_head_to_dataset, decode, load_chain_dit, make_tok  # noqa: E402
from hairdeck.testset import chains as T  # noqa: E402
from hairdeck.testset.common import seed_dir, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('prep', 'eval'))
    ap.add_argument('out')
    ap.add_argument('--on', default='gt')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    model, ra, dim = load_chain_dit(paths.resolve(T.CKPT), 'cpu')
    stats = chain_stats()
    D = Data('test', make_tok(ra['vae'], 'cpu'), ra['vae'], stats, 'cpu')
    hs = [int(h) for h in D.items]
    hz = np.load(paths.data('head_frame_test.npz'))
    if a.mode == 'prep':
        for s in a.seeds:
            np.save(f'{out}/noise_seed{s}.npy', torch.randn((len(hs), D.nmax, D.dim), generator=torch.Generator().manual_seed(s)).numpy())
        np.save(f'{out}/card_tokens_gt.npy', D.cards(torch.from_numpy(D.items), perm=False).numpy())
        json.dump(hs, open(f'{out}/rows.json', 'w'))
        json.dump([dict(row=h, k=float(hz['k'][h]), b=hz['b'][h].astype(float).tolist()) for h in hs], open(f'{out}/fits.json', 'w'))
        return
    on = a.on if a.on == 'gt' else os.path.abspath(a.on)
    gt = D.chains(torch.tensor(hs), perm=False).numpy()
    for s in a.seeds:
        run = seed_dir(out, s)
        cards = on if on == 'gt' else seed_dir(on, s)
        pred = json.load(open(f'{run}/chains_head.json'))
        res = {}
        for h, xg in zip(hs, gt):
            r = dict(pred=pred[str(h)], gt=decode(xg, stats))
            if cards != 'gt':
                r['n_cards'] = len(load_cards(f'{cards}/JsonResults/{h:03d}.json')[:NCARD])
            chains_head_to_dataset(r['pred'] + r['gt'], float(hz['k'][h]), hz['b'][h].astype(np.float64))
            res[h] = r
        json.dump(res, open(f'{run}/chains.json', 'w'))
        T.report(run, dict(ckpt='swift: ' + T.CKPT, seed=s, cards=cards, steps=50, cfg=2.0), a.workers)
    summarize(out, 'chains', list(a.seeds), ((f'Table 4 (Swift chains) on {on}', T.COLUMNS),))


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('fork')
    main()
