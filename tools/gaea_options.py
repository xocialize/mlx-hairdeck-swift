"""Candidate sources of the image path's two outside inputs, asked of a running Gaea (MES v0 on this Mac):
  matte   image.matte (BiRefNet, the Forge pack)            -> OUT/gaea/<ex>_matte.png
  box     image.describe (Mage-VL) asked for the head box   -> OUT/gaea/<ex>_vlm.json (the answer and the parsed box)

  python tools/gaea_options.py IMAGES_DIR OUT [--mes http://127.0.0.1:17630/mes/v0] [--examples ex01 ...]
IMAGES_DIR: <ex>_rgb.png as tools/dump_p4b_fixtures.py writes them (the pixels the reference preprocessing saw).
Only the standard library; nothing leaves this machine.
"""
import argparse
import json
import os
import re
import struct
import time
import urllib.error
import urllib.request
import zlib

QUESTION = ('This image is {w} pixels wide and {h} pixels high. Find the head of the main character, including all of '
            'its hair down to the hair tips. Answer with only the bounding box of that region as four integers in '
            'pixels: x_min, y_min, x_max, y_max.')


def call(url, data=None, ctype=None):
    req = urllib.request.Request(url, data=data, headers={'MES-Version': '0.1', **({'Content-Type': ctype} if ctype else {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def png_size(b):
    return struct.unpack('>II', b[16:24])


def job(mes, body, tries=90):
    j = json.loads(call(f'{mes}/jobs', json.dumps(body).encode(), 'application/json'))
    jid = j['id']
    for _ in range(tries):
        j = json.loads(call(f'{mes}/jobs/{jid}?wait=10'))
        if j['state'] in ('holding', 'purged'):
            return j
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images')
    ap.add_argument('out')
    ap.add_argument('--mes', default='http://127.0.0.1:17630/mes/v0')
    ap.add_argument('--examples', nargs='+', default=[f'ex{i:02d}' for i in range(1, 13)])
    a = ap.parse_args()
    os.makedirs(f'{a.out}/gaea', exist_ok=True)
    for ex in a.examples:
        png = open(f'{a.images}/{ex}_rgb.png', 'rb').read()
        w, h = png_size(png)
        up = json.loads(call(f'{a.mes}/uploads', png, 'image/png'))['upload_id']
        t = time.time()
        j = job(a.mes, dict(operation='image.matte', model='birefnet-matte', inputs=dict(image=dict(upload=up))))
        ok = j.get('outcome') == 'succeeded'
        if ok:
            o = next(x for x in j['result']['outputs'] if x['name'] == 'mask')
            open(f'{a.out}/gaea/{ex}_matte.png', 'wb').write(call(a.mes.rsplit('/mes/', 1)[0] + o['href']))
        tm = time.time() - t
        up = json.loads(call(f'{a.mes}/uploads', png, 'image/png'))['upload_id']
        t = time.time()
        try:
            j2 = job(a.mes, dict(operation='image.describe', model='mage-vl', inputs=dict(image=dict(upload=up), question=QUESTION.format(w=w, h=h))))
        except urllib.error.HTTPError as e:  # Gaea 1.4.0: image.describe takes no question from a client
            j2 = dict(outcome='refused', error=e.read().decode()[:300])
        text = None
        if j2.get('outcome') == 'succeeded':
            r = j2['result']
            text = (r.get('values') or {}).get('text') or next((v.get('text') for v in r.get('outputs', []) if v.get('text')), None) or json.dumps(r)[:600]
        nums = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', text or '')]
        box = nums[:4] if len(nums) >= 4 else None
        rec = dict(example=ex, size=[w, h], matte_ok=ok, matte_s=round(tm, 2), vlm_s=round(time.time() - t, 2), vlm_outcome=j2.get('outcome'),
                   vlm_error=str(j2.get('error'))[:200] if j2.get('error') else None, answer=text, box=box)
        json.dump(rec, open(f'{a.out}/gaea/{ex}_vlm.json', 'w'), indent=1)
        print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
