"""Cut every ground-truth object out of the 4K frames as an RGBA sprite.

GrabCut is seeded with the ground-truth box on a 4x upscaled crop (it struggles
on objects that are only ~20 px). Writes:

    sprites/<class>/f<frame>_<n>.png     RGBA sprite at source resolution
    sprites/index.json                   bbox, truncation flag, mask fill per sprite
    sprites/review/<class>.png           review sheet: crop | mask outline | cut-out
    sprites/review/_overview.png         one instance per class

    python training/extract_sprites.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH, OBJECT_CLASSES  # noqa: E402
from utils import frame_numbers, load_annotations, load_frame  # noqa: E402

OUT = ROOT / 'sprites'
PAD = 8          # context pixels around the box for GrabCut
UPSCALE = 4
EDGE_MARGIN = 8  # a box this close to the frame edge is probably cut off
SUSPECT_FILL = 0.35  # flag fill ratios this far (relative) from the class median
PANEL = 128      # review panel size


def extract(frame_img, bbox):
    x1, y1, x2, y2 = bbox
    cx1, cy1 = max(x1 - PAD, 0), max(y1 - PAD, 0)
    cx2, cy2 = min(x2 + PAD, IMAGE_WIDTH), min(y2 + PAD, IMAGE_HEIGHT)
    crop = frame_img[cy1:cy2, cx1:cx2]
    big = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)

    mask = np.zeros(big.shape[:2], np.uint8)
    rect = ((x1 - cx1) * UPSCALE, (y1 - cy1) * UPSCALE, (x2 - x1) * UPSCALE, (y2 - y1) * UPSCALE)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(big, mask, rect, bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        fg = np.zeros(big.shape[:2], np.uint8)
        fg[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = 255

    fg = cv2.resize(fg, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA)
    fg = np.where(fg >= 128, 255, 0).astype(np.uint8)
    # Only the box interior can be object.
    inside = np.zeros_like(fg)
    inside[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1] = 255
    fg &= inside

    sprite = np.dstack([crop, fg])[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1]
    return crop, fg, sprite


def fit(img, size=PANEL):
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((size, size, 3), 40, np.uint8)
    oy, ox = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    canvas[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img
    return canvas


def review_cell(crop, fg, sprite, label):
    outline = crop.copy()
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    big_outline = cv2.resize(outline, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_NEAREST)
    cv2.drawContours(big_outline, [c * UPSCALE for c in contours], -1, (255, 0, 255), 1)

    # Cut-out on a checkerboard so both dark and light objects are visible.
    h, w = sprite.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    checker = np.where(((yy // 3 + xx // 3) % 2)[..., None] == 0, 90, 160).astype(np.uint8).repeat(3, axis=2)
    alpha = sprite[..., 3:4] / 255.0
    cutout = (sprite[..., :3] * alpha + checker * (1 - alpha)).astype(np.uint8)

    cell = np.hstack([fit(crop), fit(big_outline), fit(cutout)])
    header = np.full((16, cell.shape[1], 3), 20, np.uint8)
    cv2.putText(header, label, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    return np.vstack([header, cell])


def sheet(cells, per_row=4):
    blank = np.zeros_like(cells[0])
    while len(cells) % per_row:
        cells.append(blank)
    rows = [np.hstack(cells[i:i + per_row]) for i in range(0, len(cells), per_row)]
    return np.vstack(rows)


def main():
    (OUT / 'review').mkdir(parents=True, exist_ok=True)
    manual_path = OUT / 'manual.json'
    manual = json.loads(manual_path.read_text()) if manual_path.exists() else {}
    index = []
    results = []
    for frame_no in frame_numbers():
        img = load_frame(frame_no)
        for n, ann in enumerate(load_annotations(frame_no)):
            cls = ann['object_id']
            x1, y1, x2, y2 = ann['bbox']
            truncated = x1 <= EDGE_MARGIN or y1 <= EDGE_MARGIN or x2 >= IMAGE_WIDTH - EDGE_MARGIN or y2 >= IMAGE_HEIGHT - EDGE_MARGIN
            crop, fg, sprite = extract(img, ann['bbox'])
            name = f'f{frame_no:03d}_{n}.png'
            (OUT / cls).mkdir(exist_ok=True)
            if manual.get(f'{cls}/{name}', {}).get('status') == 'manual':
                # Hand-drawn in the review tool: keep it, show it in the sheets.
                sprite = cv2.imread(str(OUT / cls / name), cv2.IMREAD_UNCHANGED)
                fg[:] = 0
                bx, by = ann['bbox'][0] - max(ann['bbox'][0] - PAD, 0), ann['bbox'][1] - max(ann['bbox'][1] - PAD, 0)
                fg[by:by + sprite.shape[0], bx:bx + sprite.shape[1]] = sprite[..., 3]
            else:
                cv2.imwrite(str(OUT / cls / name), sprite)
            fill = float((sprite[..., 3] > 0).mean())
            index.append({'class': cls, 'file': f'{cls}/{name}', 'frame': frame_no, 'bbox': ann['bbox'],
                          'truncated': truncated, 'fill': round(fill, 3)})
            results.append((index[-1], crop, fg, sprite))

    medians = {c: float(np.median([r[0]['fill'] for r in results if r[0]['class'] == c and not r[0]['truncated']] or [0]))
               for c in {r[0]['class'] for r in results}}
    cells = defaultdict(list)
    best = {}
    for entry, crop, fg, sprite in results:
        cls, med = entry['class'], medians[entry['class']]
        entry['suspect'] = bool(med and abs(entry['fill'] - med) / med > SUSPECT_FILL)
        x1, y1, x2, y2 = entry['bbox']
        label = f"f{entry['frame']} {x2 - x1}x{y2 - y1} fill {entry['fill']:.2f}"
        label += ' CUT' if entry['truncated'] else ''
        label += ' SUSPECT' if entry['suspect'] else ''
        cells[cls].append(review_cell(crop, fg, sprite, label))
        area = (x2 - x1) * (y2 - y1)
        if not entry['truncated'] and not entry['suspect'] and area > best.get(cls, (0, None))[0]:
            best[cls] = (area, review_cell(crop, fg, sprite, f"{cls} f{entry['frame']}"))

    for cls, cs in cells.items():
        cv2.imwrite(str(OUT / 'review' / f'{cls}.png'), sheet(cs))
    overview = [best[c][1] for c in OBJECT_CLASSES if c in best]
    cv2.imwrite(str(OUT / 'review' / '_overview.png'), sheet(overview))
    (OUT / 'index.json').write_text(json.dumps(index, indent=1))
    print(f'{len(index)} sprites, {sum(i["truncated"] for i in index)} truncated, {sum(i["suspect"] for i in index)} suspect -> {OUT}')


if __name__ == '__main__':
    main()
