"""Stitch a recorded sequence's zoomed views into one ground mosaic.

The ground drifts through the frame at a roughly constant speed, so a view at
frame f is pasted at (region - f * drift) in ground coordinates. The drift is
estimated by matching overlapping consecutive views, then smoothed.

    python training/stitch_recording.py recordings/<sequence_id>
    python training/stitch_recording.py recordings/<sequence_id> --levels 1 2 --scale 0.5

Writes <sequence_dir>/mosaic.jpg and mosaic_views.json (where each view landed).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def estimate_pair_drift(a_img, a_region, b_img, b_region, gap, prior):
    """Ground drift (px/frame) between two views of the same level, or None."""
    sa = (a_region[2] - a_region[0]) / a_img.shape[1]
    # Where B's centre patch should appear in A if the ground moved prior * gap.
    ph, pw = 160, 240
    h, w = b_img.shape[:2]
    # B pixel -> A pixel offset under the prior; put the patch where it lands mid-overlap.
    off_x = (b_region[0] - a_region[0] - prior[0] * gap) / sa
    off_y = (b_region[1] - a_region[1] - prior[1] * gap) / sa
    lo_x, hi_x = max(0, -off_x), min(w, w - off_x)
    lo_y, hi_y = max(0, -off_y), min(h, h - off_y)
    if hi_x - lo_x < pw + 2 * 90 or hi_y - lo_y < ph + 2 * 90:
        return None
    bx = int((lo_x + hi_x) / 2 - pw / 2)
    by = int((lo_y + hi_y) / 2 - ph / 2)
    patch = b_img[by:by + ph, bx:bx + pw]
    src_x = b_region[0] + (bx + pw / 2) * sa - prior[0] * gap
    src_y = b_region[1] + (by + ph / 2) * sa - prior[1] * gap
    ax, ay = (src_x - a_region[0]) / sa, (src_y - a_region[1]) / sa
    search = 90
    x0, y0 = int(ax - pw / 2 - search), int(ay - ph / 2 - search)
    x1, y1 = int(ax + pw / 2 + search), int(ay + ph / 2 + search)
    if x0 < 0 or y0 < 0 or x1 > a_img.shape[1] or y1 > a_img.shape[0]:
        return None
    res = cv2.matchTemplate(a_img[y0:y1, x0:x1], patch, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    if score < 0.6:
        return None
    found_x = x0 + loc[0] + pw / 2
    found_y = y0 + loc[1] + ph / 2
    # B's patch sits at source (b_region + offset) at frame b; in A it was at (a_region + found).
    dx = (b_region[0] + (bx + pw / 2) * sa - (a_region[0] + found_x * sa)) / gap
    dy = (b_region[1] + (by + ph / 2) * sa - (a_region[1] + found_y * sa)) / gap
    return dx, dy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sequence_dir')
    parser.add_argument('--levels', type=int, nargs='*', default=[2])
    parser.add_argument('--scale', type=float, default=0.5, help='mosaic scale relative to source pixels')
    parser.add_argument('--drift', type=float, nargs=2, default=None, help='force drift dx dy (px/frame)')
    args = parser.parse_args()

    d = Path(args.sequence_dir)
    views = [json.loads(l) for l in open(d / 'views.jsonl')]
    views = sorted((v for v in views if v['level'] in args.levels), key=lambda v: v['frame'])
    if not views:
        raise SystemExit('no views at those levels')
    grey = lambda v: cv2.cvtColor(cv2.imread(str(d / v['file'])), cv2.COLOR_BGR2GRAY)

    if args.drift:
        drift = np.array(args.drift)
    else:
        prior, samples = (0.0, 62.0), []
        for a, b in zip(views, views[1:]):
            gap = b['frame'] - a['frame']
            if a['level'] != b['level'] or not 1 <= gap <= 3:
                continue
            est = estimate_pair_drift(grey(a), a['source_region_xyxy'], grey(b), b['source_region_xyxy'], gap, prior)
            if est:
                samples.append(est)
        drift = np.median(np.array(samples), axis=0) if samples else np.array(prior)
        print(f'drift estimate from {len(samples)} view pairs: dx {drift[0]:.1f}, dy {drift[1]:.1f} px/frame')

    placed = []
    for v in views:
        x1, y1, x2, y2 = v['source_region_xyxy']
        gx, gy = x1 - v['frame'] * drift[0], y1 - v['frame'] * drift[1]
        placed.append((v, gx, gy, x2 - x1, y2 - y1))
    min_x = min(p[1] for p in placed); min_y = min(p[2] for p in placed)
    max_x = max(p[1] + p[3] for p in placed); max_y = max(p[2] + p[4] for p in placed)
    s = args.scale
    W, H = int((max_x - min_x) * s) + 1, int((max_y - min_y) * s) + 1
    mosaic = np.zeros((H, W, 3), np.uint8)
    meta = []
    # Later frames on top; for a downward drift that keeps the freshest view of each patch.
    for v, gx, gy, w, h in placed:
        img = cv2.imread(str(d / v['file']))
        tw, th = max(1, int(w * s)), max(1, int(h * s))
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        ox, oy = int((gx - min_x) * s), int((gy - min_y) * s)
        mosaic[oy:oy + th, ox:ox + tw] = img[:max(0, min(th, H - oy)), :max(0, min(tw, W - ox))]
        meta.append({'file': v['file'], 'frame': v['frame'], 'mosaic_xy': [ox, oy], 'size': [tw, th]})
    # Newest frames end up at the top: flip so the scroll reads in flight order top->bottom.
    cv2.imwrite(str(d / 'mosaic.jpg'), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 90])
    (d / 'mosaic_views.json').write_text(json.dumps({'drift': list(map(float, drift)), 'scale': s, 'views': meta}, indent=1))
    print(f'mosaic {W}x{H} from {len(views)} views -> {d / "mosaic.jpg"}')


if __name__ == '__main__':
    main()
