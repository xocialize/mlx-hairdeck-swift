"""P3 parity fixtures from the Python reference: hairdeck.vrm.prep (card mesh, UVs, skinning with chains) on
HairDeck's avatar body for one test hairstyle, from cards and chains made by the Swift port.

  source /Volumes/Satechi/charm/HairDeck/hairdeck_env.sh
  python tools/dump_p3_fixtures.py Fixtures/p3 --cards results/swift_pointcloud/seed0 \
      --chains results/swift_chains_on_pointcloud/seed0 --fits results/swift_chains/fits.json --row 1
Inputs written for the Swift side: cards_head.json, chains_head.json (head frame), body.json (the avatar body's fit, k,
b and the head bone). Outputs of prep: V (glTF axes, metres), F, UV, vcard, vchain, vseg, vw, vseg2, valpha,
chain_ptr, chain_J (glTF axes), stiffness, and info.json.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.environ['HAIRDECK_ROOT'])
from hairdeck.vrm import body as B  # noqa: E402
from hairdeck.vrm.prep import card_list, head_to_body, prep  # noqa: E402


def glb_json(path):
    with open(path, 'rb') as f:
        f.read(12)
        n, _ = struct.unpack('<II', f.read(8))
        return json.loads(f.read(n))


def gltf(p):
    """Blender world (Z up, -Y front) -> glTF axes (Y up, +Z front)."""
    p = np.asarray(p, np.float64)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--cards', required=True)
    ap.add_argument('--chains', required=True)
    ap.add_argument('--fits', required=True)
    ap.add_argument('--row', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fit = next(f for f in json.load(open(a.fits)) if f['row'] == a.row)
    k, b = fit['k'], np.array(fit['b'])
    cards = json.load(open(f'{a.cards}/JsonResults/{a.row:03d}.json'))
    for c in cards:  # the test row's dataset frame -> head frame
        q = np.asarray(c['seq'], np.float64)
        q[:, :3] = k * q[:, :3] + b
        q[:, 3:5] *= k
        c['seq'] = q.tolist()
    chains = json.load(open(f'{a.chains}/chains_head.json'))[str(a.row)]
    json.dump(cards, open(f'{a.out}/cards_head.json', 'w'))
    json.dump(chains, open(f'{a.out}/chains_head.json', 'w'))

    body = B.body()
    vj = glb_json(body['vrm'])
    hum = vj['extensions']['VRM']['humanoid']['humanBones']
    head = next(h['node'] for h in hum if h['bone'] == 'head')
    json.dump(dict(k=body['k'], b=body['b'], fit={x: body['fit'][x] for x in ('s', 't', 'sx', 'sz', 'w', 'h', 'o', 'head_joint')},
                   head_bone=vj['nodes'][head]['name'], vrm_version='0'), open(f'{a.out}/body.json', 'w'), indent=1)

    cb, pred = head_to_body(cards, chains, body['k'], np.asarray(body['b'], np.float64))
    out, info = prep(body['h'], body['key'], card_list(cards=[{'shape': c['shape'], 'seq': c['seq']} for c in cb]), pred, body['fit'])

    def save(name, x, dtype):
        np.save(f'{a.out}/{name}.npy', np.ascontiguousarray(np.asarray(x).astype(dtype)))
        print(f'{name:12s} {np.asarray(x).shape} {dtype.__name__}')
    save('V', gltf(out['V']), np.float64)
    save('F', out['F'], np.int32)
    save('UV', out['UV'], np.float64)
    for n in ('vcard', 'vchain', 'vseg', 'vseg2', 'chain_ptr'):
        save(n, out[n], np.int32)
    for n in ('vw', 'valpha', 'stiffness'):
        save(n, out[n], np.float64)
    save('chain_J', gltf(out['chain_J']), np.float64)
    save('head_joint', gltf(out['head_joint']), np.float64)
    json.dump(info, open(f'{a.out}/info.json', 'w'), indent=1)
    print(json.dumps(info))

    # where the body's own hair sits in the raw VRM file against the dataset frame of data/avatar/hair.glb
    import trimesh
    sc = trimesh.load(os.path.join(body['dir'], 'hair.glb'))
    Vh = np.concatenate([g.vertices for g in sc.dump()])
    f = body['fit']
    w = (np.stack([Vh.min(0), Vh.max(0)]) - np.array(f['t'])) / f['s']
    lo, hi = [], []
    for m in vj['meshes']:
        if 'hair' in m['name'].lower():
            for p in m['primitives']:
                acc = vj['accessors'][p['attributes']['POSITION']]
                lo.append(acc['min'])
                hi.append(acc['max'])
    print('hair bbox, (dataset - t) / s :', np.round(w, 4).tolist())
    print('hair bbox, raw VRM 0.x file  :', np.round([np.min(lo, 0), np.max(hi, 0)], 4).tolist())


if __name__ == '__main__':
    main()
