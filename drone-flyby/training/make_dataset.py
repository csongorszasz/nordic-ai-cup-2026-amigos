"""Cut YOLO training crops out of the 4K reference frames.

Crops are made exactly the way the evaluator makes views: a source region at
resolution level 0/1/2, downsampled to 960x540 with INTER_AREA. Labels are the
ground-truth boxes clipped to that region.

    python training/make_dataset.py                    # -> datasets/helsinki_yolo
    python training/make_dataset.py --val-frames 6 13 21
    python training/make_dataset.py --all-val --out datasets/helsinki_real   # a real test set for synthetic training
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))  # dtos.py / utils.py live in src/

from dtos import OBJECT_CLASSES, SOURCE_REGION_SIZES, TRANSMITTED_VIEW_SIZE  # noqa: E402
from utils import (  # noqa: E402
    center_bounds_for_level,
    frame_numbers,
    load_annotations,
    load_frame,
    source_region_for_view,
)

CLASS_INDEX = {name: i for i, name in enumerate(OBJECT_CLASSES)}
VIEW_W, VIEW_H = TRANSMITTED_VIEW_SIZE

# A box cut by the crop edge is kept if at least this much of it is visible.
MIN_VISIBLE_FRACTION = 0.4
# Drop labels smaller than this in the transmitted 960x540 image.
MIN_LABEL_PIXELS = 2.0


def crop_view(frame, level, cx, cy):
    x1, y1, x2, y2 = source_region_for_view(level, cx, cy)
    view = frame[y1:y2, x1:x2]
    if level < 2:  # level 2 is already 960x540
        view = cv2.resize(view, TRANSMITTED_VIEW_SIZE, interpolation=cv2.INTER_AREA)
    return view, (x1, y1, x2, y2)


def labels_for_region(annotations, region):
    rx1, ry1, rx2, ry2 = region
    rw, rh = rx2 - rx1, ry2 - ry1
    scale_x, scale_y = VIEW_W / rw, VIEW_H / rh
    lines = []
    for ann in annotations:
        bx1, by1, bx2, by2 = ann['bbox']
        area = max(0, bx2 - bx1) * max(0, by2 - by1)
        cx1, cy1 = max(bx1, rx1), max(by1, ry1)
        cx2, cy2 = min(bx2, rx2), min(by2, ry2)
        if cx2 <= cx1 or cy2 <= cy1 or area == 0:
            continue
        if (cx2 - cx1) * (cy2 - cy1) / area < MIN_VISIBLE_FRACTION:
            continue
        w_px, h_px = (cx2 - cx1) * scale_x, (cy2 - cy1) * scale_y
        if w_px < MIN_LABEL_PIXELS or h_px < MIN_LABEL_PIXELS:
            continue
        xc = ((cx1 + cx2) / 2 - rx1) / rw
        yc = ((cy1 + cy2) / 2 - ry1) / rh
        lines.append(
            f"{CLASS_INDEX[ann['object_id']]} {xc:.6f} {yc:.6f} {(cx2 - cx1) / rw:.6f} {(cy2 - cy1) / rh:.6f}"
        )
    return lines


def view_centres(level, annotations, n_random, n_per_object, rng):
    if level == 0:
        cx, cy = 1920, 1080
        return [(cx, cy)]
    min_x, max_x, min_y, max_y = center_bounds_for_level(level)
    rw, rh = SOURCE_REGION_SIZES[level]
    centres = [(rng.randint(min_x, max_x), rng.randint(min_y, max_y)) for _ in range(n_random)]
    # Views that contain each object at a random position, so every class is seen zoomed in.
    for ann in annotations:
        bx1, by1, bx2, by2 = ann['bbox']
        ox, oy = (bx1 + bx2) / 2, (by1 + by2) / 2
        for _ in range(n_per_object):
            cx = int(ox + rng.uniform(-0.4, 0.4) * rw)
            cy = int(oy + rng.uniform(-0.4, 0.4) * rh)
            centres.append((min(max(cx, min_x), max_x), min(max(cy, min_y), max_y)))
    return centres


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(ROOT / 'datasets' / 'helsinki_yolo'))
    parser.add_argument('--val-frames', type=int, nargs='*', default=[6, 13, 21])
    parser.add_argument('--all-val', action='store_true',
                        help='every frame to val: the real scene as the test set of a synthetic-only model')
    parser.add_argument('--l1-random', type=int, default=12)
    parser.add_argument('--l1-per-object', type=int, default=1)
    parser.add_argument('--l2-random', type=int, default=20)
    parser.add_argument('--l2-per-object', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for split in ('train', 'val'):
        (out / 'images' / split).mkdir(parents=True)
        (out / 'labels' / split).mkdir(parents=True)

    counts = {'train': 0, 'val': 0}
    for frame_no in frame_numbers():
        split = 'val' if args.all_val or frame_no in args.val_frames else 'train'
        frame = load_frame(frame_no)
        annotations = load_annotations(frame_no)
        plan = [
            (0, view_centres(0, annotations, 0, 0, rng)),
            (1, view_centres(1, annotations, args.l1_random, args.l1_per_object, rng)),
            (2, view_centres(2, annotations, args.l2_random, args.l2_per_object, rng)),
        ]
        for level, centres in plan:
            for i, (cx, cy) in enumerate(centres):
                view, region = crop_view(frame, level, cx, cy)
                lines = labels_for_region(annotations, region)
                stem = f'f{frame_no:03d}_L{level}_{i:03d}_{cx}_{cy}'
                cv2.imwrite(str(out / 'images' / split / f'{stem}.jpg'), view, [cv2.IMWRITE_JPEG_QUALITY, 95])
                (out / 'labels' / split / f'{stem}.txt').write_text('\n'.join(lines))
                counts[split] += 1

    names = '\n'.join(f'  {i}: {name}' for i, name in enumerate(OBJECT_CLASSES))
    (out / 'data.yaml').write_text(f'path: {out}\ntrain: images/train\nval: images/val\nnames:\n{names}\n')
    print(f"wrote {counts['train']} train / {counts['val']} val views to {out}")


if __name__ == '__main__':
    main()
