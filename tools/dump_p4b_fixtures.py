"""P4b fixtures: the reference preprocessing of demo images (person matte, hair and face masks, crop box) and the
model input and matte crop made from them (hairdeck.infer.image.Pipeline.prep / matte_crop).

  source /Volumes/Satechi/charm/HairDeck/hairdeck_env.sh
  python tools/dump_p4b_fixtures.py Fixtures/p4b [--examples ex01 ex02 ...] [--full ex01]
Per example: <ex>_box.json (box, status, mask boxes, timings). For the --full examples also <ex>_rgb.png (the pixels
the preprocessing saw), <ex>_person.png (the matte, 8-bit), <ex>_input.png (the 448 px model input) and
<ex>_matte.png (the matte crop the colour model reads, white = character).
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.environ['HAIRDECK_ROOT'])
from hairdeck import paths  # noqa: E402
from hairdeck.data.crop import apply_crop, composite, mask_box  # noqa: E402
from hairdeck.data.illus import crop_white  # noqa: E402
from hairdeck.infer.image import load_image  # noqa: E402
from hairdeck.prep.preprocess import Preprocessor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--examples', nargs='+', default=[f'ex{i:02d}' for i in range(1, 13)])
    ap.add_argument('--full', nargs='+', default=['ex01', 'ex02'])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pre = Preprocessor('best')
    for ex in a.examples:
        f = paths.data('demo_examples', f'{ex}.png')
        rgb, alpha, note, src = load_image(f, want_path=True)
        p = pre(rgb, alpha, src)
        info = dict(example=ex, size=[int(rgb.shape[1]), int(rgb.shape[0])], status=p['status'], box=list(p['box']) if p['box'] is not None else None,
                    hair_box=mask_box(p['hair']), face_box=mask_box(p['face']), person_box=mask_box(p['person'] > 127),
                    input_alpha=alpha is not None, time=p['info'].get('matting', {}).get('time'))
        json.dump(info, open(f'{a.out}/{ex}_box.json', 'w'))
        print(json.dumps(info), flush=True)
        if ex in a.full and p['box'] is not None:
            am = p['person'].copy()
            if am.max() <= 127:
                am[:] = 255
            full = np.asarray(composite(Image.fromarray(np.concatenate([p['rgb'], am[..., None]], -1), 'RGBA')))
            Image.fromarray(p['rgb']).save(f'{a.out}/{ex}_rgb.png')
            Image.fromarray(p['person']).save(f'{a.out}/{ex}_person.png')
            Image.fromarray(crop_white(full, tuple(p['box']), 448)).save(f'{a.out}/{ex}_input.png')
            m = np.asarray(apply_crop(Image.fromarray(p['person']), tuple(p['box']), 448, resample=Image.Resampling.NEAREST)) > 127
            Image.fromarray((m * 255).astype(np.uint8)).save(f'{a.out}/{ex}_matte.png')


if __name__ == '__main__':
    main()
