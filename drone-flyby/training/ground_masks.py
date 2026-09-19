"""Open-ground masks for photo backgrounds (orthophotos have no height to tell roofs from ground).

    python training/ground_masks.py backgrounds/denmark backgrounds/naip

Writes <name>_ground.png (quarter resolution, 255 = open ground) next to each .jpg, which
synth_dataset.py then pastes objects on only, as render_city.py's depth masks do for the mesh
renders. Open ground here is plain natural ground, where the supplied scenes put their objects:
grass, fields, sand and soil (green to tan, lit, little texture). Roofs, roads, water, shadows
and tree crowns (dark or busy) are left out.
"""

import sys
from pathlib import Path

import cv2
import numpy as np


def ground_mask(image: np.ndarray) -> np.ndarray:
    small = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    natural = (h >= 10) & (h <= 85) & (s >= 22)          # tan/sand through green, not grey or blue
    lit = v >= 95                                         # not shadow, not dark tree crowns
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(grey, (9, 9))
    texture = np.sqrt(np.maximum(cv2.blur(grey * grey, (9, 9)) - mean * mean, 0))
    plain = texture < 14                                  # fields and lots, not trees, roofs or rubble
    mask = (natural & lit & plain).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return mask * 255


def main():
    for folder in sys.argv[1:]:
        paths = [p for p in sorted(Path(folder).glob('*.jpg'))]
        shares = []
        for p in paths:
            mask = ground_mask(cv2.imread(str(p)))
            cv2.imwrite(str(p.with_name(p.stem + '_ground.png')), mask)
            shares.append((mask > 0).mean())
        print(f'{folder}: {len(paths)} masks, open ground {np.mean(shares):.0%} on average '
              f'(min {min(shares):.0%}, max {max(shares):.0%})')


if __name__ == '__main__':
    main()
