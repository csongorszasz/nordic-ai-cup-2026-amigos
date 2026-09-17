"""Rebuild full 3840x2160 source frames of a recorded sequence from many runs.

Every validation attempt replays the same flight, so views recorded in different
runs can be combined:

1. Registration: consecutive Level-0 views of a full-view run give the exact ground
   shift per frame (phase correlation), integrated into offset[f], so that a ground
   point g appears at source position g + offset[f] in frame f.
2. Each frame F starts from its (upscaled) Level-0 view, then Level-1 and Level-2
   views from all runs are pasted where they fall in F, closest-in-time on top.

    python training/reconstruct_frames.py --l0-run recordings/<l0 run> \
        --runs recordings/<run a> recordings/<run b> ... --out recordings/validation_4k

Writes frame_XXXX.jpg (4K), and coverage.json with the share of each frame that
came from Level 2 / Level 1 / only Level 0.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

W, H = 3840, 2160
UPSCALE = {0: 4, 1: 2, 2: 1}


def load_views(run: Path):
    return [dict(json.loads(l), run=run) for l in open(run / 'views.jsonl')]


def register(l0_views):
    """offset[frame] in source px, relative to the first Level-0 frame."""
    l0_views = sorted(l0_views, key=lambda v: v['frame'])
    offsets = {l0_views[0]['frame']: np.zeros(2)}
    prev = None
    for v in l0_views:
        img = cv2.cvtColor(cv2.imread(str(v['run'] / v['file'])), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            (dx, dy), response = cv2.phaseCorrelate(prev[1], img)
            offsets[v['frame']] = offsets[prev[0]['frame']] + np.array([dx, dy]) * UPSCALE[0]
        prev = (v, img)
    frames = np.array(sorted(offsets))
    table = np.array([offsets[f] for f in frames])
    return frames, table


def offset_at(frames, table, f):
    return np.array([np.interp(f, frames, table[:, 0]), np.interp(f, frames, table[:, 1])])


def paste(canvas, level_map, img, x, y, level):
    """Paste img with its top-left at (x, y) in canvas coords, clipped to the frame."""
    h, w = img.shape[:2]
    x0, y0 = max(int(round(x)), 0), max(int(round(y)), 0)
    x1, y1 = min(int(round(x)) + w, W), min(int(round(y)) + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    sx, sy = x0 - int(round(x)), y0 - int(round(y))
    canvas[y0:y1, x0:x1] = img[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
    level_map[y0:y1, x0:x1] = level


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--l0-run', required=True)
    parser.add_argument('--runs', nargs='+', required=True, help='runs whose Level-1/2 views are pasted')
    parser.add_argument('--out', default='recordings/validation_4k')
    parser.add_argument('--frames', type=int, nargs='*', help='only these frames')
    parser.add_argument('--max-age', type=int, default=6, help='only use views within this many frames of F')
    args = parser.parse_args()

    l0_run = Path(args.l0_run)
    l0_views = [v for v in load_views(l0_run) if v['level'] == 0]
    frames, table = register(l0_views)
    print(f'registered {len(frames)} Level-0 frames; total ground shift {table[-1].round(1)} px')

    zoomed = [v for r in args.runs for v in load_views(Path(r)) if v['level'] in (1, 2)]
    print(f'{len(zoomed)} zoomed views from {len(args.runs)} runs')
    cache = {}

    def image(v):
        key = (str(v['run']), v['file'])
        if key not in cache:
            img = cv2.imread(str(v['run'] / v['file']))
            if v['level'] == 1:
                img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            cache[key] = img
        return cache[key]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Base views: the registration run first, then Level-0 views of other runs for frames it skipped.
    l0_by_frame = {}
    for v in [v for r in args.runs for v in load_views(Path(r)) if v['level'] == 0] + l0_views:
        l0_by_frame[v['frame']] = v
    targets = args.frames or list(range(min(l0_by_frame), max(l0_by_frame) + 1))
    coverage = {}
    for F in targets:
        oF = offset_at(frames, table, F)
        canvas = np.zeros((H, W, 3), np.uint8)
        level_map = np.full((H, W), -1, np.int8)

        # Base: nearest Level-0 view, shifted to frame F.
        nearest = min(l0_by_frame, key=lambda f: abs(f - F))
        base_view = l0_by_frame[nearest]
        base = cv2.resize(cv2.imread(str(base_view['run'] / base_view['file'])), (W, H), interpolation=cv2.INTER_CUBIC)
        shift = oF - offset_at(frames, table, nearest)
        paste(canvas, level_map, base, shift[0], shift[1], 0)

        # Zoomed views, Level 1 first, then Level 2; within a level the closest in time last (on top).
        candidates = [v for v in zoomed if abs(v['frame'] - F) <= args.max_age]
        candidates.sort(key=lambda v: (v['level'], -abs(v['frame'] - F)))
        for v in candidates:
            of = offset_at(frames, table, v['frame'])
            x1, y1 = v['source_region_xyxy'][:2]
            paste(canvas, level_map, image(v), x1 - of[0] + oF[0], y1 - of[1] + oF[1], v['level'])

        cv2.imwrite(str(out / f'frame_{F:04d}.jpg'), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        total = level_map.size
        coverage[F] = {k: round(float((level_map == lvl).sum()) / total, 3) for k, lvl in (('L2', 2), ('L1', 1), ('L0_only', 0), ('empty', -1))}
        if len(cache) > 600:
            cache.clear()
    (out / 'coverage.json').write_text(json.dumps(coverage, indent=1))
    mean = {k: round(float(np.mean([c[k] for c in coverage.values()])), 3) for k in ('L2', 'L1', 'L0_only', 'empty')}
    print(f'{len(coverage)} frames -> {out}; mean coverage {mean}')


if __name__ == '__main__':
    main()
