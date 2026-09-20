"""Cut more 4K backgrounds out of the Danish orthophoto strips.

Each strip in raesga02's denmark_ortho.zip is 16701x4175 at 0.2395 m/px, about 4 km of
ground. Only four 3840x2160 tiles per site were taken from them; at the challenge's
0.21 m/px a strip holds about ten side by side, and more with a little overlap.

    python training/cut_denmark_strips.py --zip <path>/denmark_ortho.zip --per-site 12

Writes backgrounds/denmark/dk_<site>_<n>.jpg, skipping names that already exist, so the
four originals stay as they are. Run training/ground_masks.py afterwards for the masks.
"""

import argparse
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 3840, 2160
TARGET_MPP = 0.21   # the challenge imagery, as in fetch_backgrounds.py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', required=True, help='denmark_ortho.zip')
    parser.add_argument('--out', default=str(ROOT / 'backgrounds' / 'denmark'))
    parser.add_argument('--per-site', type=int, default=12)
    parser.add_argument('--start', type=int, default=4, help='first index to write (originals are 0-3)')
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(args.zip) as z:
        sites = sorted({name.split('/')[1] for name in z.namelist() if name.endswith('strip.png')})
        for site in sites:
            meta = json.loads(z.read(f'denmark/{site}/strip.json'))
            scale = meta['target_gsd_m'] / TARGET_MPP          # 0.2395 -> 0.21, so > 1
            src_w, src_h = int(WIDTH / scale), int(HEIGHT / scale)
            data = np.frombuffer(z.read(f'denmark/{site}/strip.png'), np.uint8)
            strip = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if strip is None:
                print(f'{site}: could not decode')
                continue
            h, w = strip.shape[:2]
            cols = max(1, round(w / src_w))
            rows = max(1, h // src_h)
            # Spread the requested number of tiles over the strip, overlapping as needed.
            n = args.per_site
            per_row = max(1, -(-n // rows))
            spots = []
            for r in range(rows):
                y = min(r * src_h, max(0, h - src_h))
                for c in range(per_row):
                    x = 0 if per_row == 1 else round(c * (w - src_w) / (per_row - 1))
                    spots.append((max(0, x), y))
            for i, (x, y) in enumerate(spots[:n]):
                name = f'dk_{site}_{args.start + i}.jpg'
                path = out / name
                if path.exists():
                    continue
                crop = strip[y:y + src_h, x:x + src_w]
                if crop.shape[0] < src_h * 0.9 or crop.shape[1] < src_w * 0.9:
                    continue
                tile = cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(str(path), tile, [cv2.IMWRITE_JPEG_QUALITY, 92])
                written += 1
            print(f'{site}: strip {w}x{h}, {cols} across x {rows} down, wrote up to {n}')
            del strip
    print(f'{written} new tiles in {out}')


if __name__ == '__main__':
    main()
