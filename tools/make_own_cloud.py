"""A hairstyle point cloud of the repo's own: the hair generated from Fixtures/pictures/girl.png (head-frame cards,
`RunHairDeck image-hair --save DIR`) put on HairDeck's avatar body, meshed and scan-sampled as the reference samples its
condition clouds — 10,000 points with normals in the `hair_reform.ply` layout, in that body's dataset frame.

  source /Volumes/Satechi/charm/HairDeck/hairdeck_env.sh
  python tools/make_own_cloud.py DIR/cards_head.json OUT.ply
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ['HAIRDECK_ROOT'])
from hairdeck.cards.mesh import cards_to_mesh  # noqa: E402
from hairdeck.sampling import scan  # noqa: E402
from hairdeck.vrm import body as B  # noqa: E402
from hairdeck.vrm.prep import card_list, head_to_body  # noqa: E402

cards = json.load(open(sys.argv[1]))
body = B.body()
cb, _ = head_to_body(cards, [], body['k'], np.asarray(body['b'], np.float64))
mesh = cards_to_mesh(card_list([{'shape': c['shape'], 'seq': c['seq']} for c in cb])).dump(concatenate=True)
scanner = scan.make_scanner(400, threads=1) if scan.BACKEND == 'raycast' else scan.make_scanner(400)
p, n = scan.fast_scan_sample(mesh, 10000, 0, scanner=scanner)
scan.write_ply(sys.argv[2], p, n)
print(len(p), 'points', np.round(p.min(0), 3).tolist(), np.round(p.max(0), 3).tolist())
