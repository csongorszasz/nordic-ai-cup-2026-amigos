"""Rebuild full 3840x2160 source frames of a recorded sequence from many runs.

Every validation attempt replays the same flight, so views recorded in different
runs can be combined:

1. Registration: consecutive Level-0 views (from any run) are matched with SIFT, which
   gives a homography per frame step: where a ground point of frame f appears in frame
   f+1. A homography rather than a shift, because the camera is pitched: the ground near
   the bottom of the frame moves faster and grows as it comes closer (up to 6% in the
   six frames a view is reused over). A single shift per frame misplaced views by up to
   95 px; the homographies place them within a pixel. Frames with no Level-0 view get an
   even share of the step across the gap (fractional matrix power).
2. Each frame F starts from its Level-0 view (upscaled; the nearest one, warped, when F
   has none). Then Level-1 and Level-2 views from all runs within --max-age frames are
   warped into F, closest-in-time on top, so views recorded at F itself end up on top
   and exact.

    python training/reconstruct_frames.py --runs recordings/<run a> recordings/<run b> ... \
        --out recordings/validation_4k

Writes frame_XXXX.jpg (4K), coverage.json with the share of each frame that came from
Level 2 / Level 1 / only Level 0, and homographies.json (the per-step homographies, 4K px).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.linalg import fractional_matrix_power

W, H = 3840, 2160
UPSCALE = {0: 4, 1: 2, 2: 1}
MIN_INLIERS = 100  # a step matched on fewer SIFT inliers than this is reported as weak


def load_views(run: Path):
    return [dict(json.loads(l), run=run) for l in open(run / 'views.jsonl')]


def register(l0_by_frame: dict) -> dict:
    """{f: 3x3 homography taking frame f's 4K pixels to frame f+1's}."""
    sift, matcher = cv2.SIFT_create(6000), cv2.BFMatcher()
    scale = np.diag([UPSCALE[0], UPSCALE[0], 1.0])  # Level-0 pixels -> 4K pixels
    features = {}
    for f, v in l0_by_frame.items():
        gray = cv2.cvtColor(cv2.imread(str(v['run'] / v['file'])), cv2.COLOR_BGR2GRAY)
        features[f] = sift.detectAndCompute(gray, None)
    frames = sorted(l0_by_frame)
    steps = {}
    for a, b in zip(frames, frames[1:]):
        (ka, da), (kb, db) = features[a], features[b]
        matches = [m for m, n in matcher.knnMatch(da, db, k=2) if m.distance < 0.7 * n.distance]
        src = np.float32([ka[m.queryIdx].pt for m in matches])
        dst = np.float32([kb[m.trainIdx].pt for m in matches])
        homography, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 1.0)
        if inliers.sum() < MIN_INLIERS:
            print(f'  weak registration {a}->{b}: {int(inliers.sum())} inliers')
        full = scale @ homography @ np.linalg.inv(scale)
        per_step = np.real(fractional_matrix_power(full, 1 / (b - a))) if b - a > 1 else full
        for f in range(a, b):
            steps[f] = per_step / per_step[2, 2]
    return steps


def between(steps: dict, f: int, F: int) -> np.ndarray:
    """Homography taking frame f's 4K pixels to frame F's."""
    M = np.eye(3)
    for g in range(f, F):
        M = steps[g] @ M
    for g in range(F, f):
        M = np.linalg.inv(steps[g]) @ M
    return M / M[2, 2]


def paste_warped(canvas, level_map, img, M, level):
    """Warp img by M (its pixels -> canvas pixels) and paste it, clipped to the frame."""
    h, w = img.shape[:2]
    corners = cv2.perspectiveTransform(np.float32([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]]), M)[:, 0]
    x0, y0 = np.floor(corners.min(axis=0)).astype(int)
    x1, y1 = np.ceil(corners.max(axis=0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 <= x0 or y1 <= y0:
        return
    shifted = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64) @ M
    size = (x1 - x0, y1 - y0)
    # Edge pixels repeat outside the view, so interpolation at its border never blends in black.
    warped = cv2.warpPerspective(img, shifted, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), shifted, size, flags=cv2.INTER_NEAREST)
    mask = mask > 0
    canvas[y0:y1, x0:x1][mask] = warped[mask]
    level_map[y0:y1, x0:x1][mask] = level


def placement(v) -> np.ndarray:
    """The view's pixels (after its upscale to source resolution) -> its own frame's 4K pixels."""
    x1, y1 = v['source_region_xyxy'][:2]
    return np.array([[1, 0, x1], [0, 1, y1], [0, 0, 1]], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True, help='recorded runs; their Level-0 views register the flight')
    parser.add_argument('--l0-run', help='(older interface) one more run, used like the others')
    parser.add_argument('--out', default='recordings/validation_4k')
    parser.add_argument('--frames', type=int, nargs='*', help='only these frames')
    parser.add_argument('--max-age', type=int, default=6, help='only use views within this many frames of F')
    args = parser.parse_args()

    runs = [Path(r) for r in ([args.l0_run] if args.l0_run else []) + args.runs]
    views = [v for r in runs for v in load_views(r)]
    l0_by_frame = {}
    for v in views:
        if v['level'] == 0:
            l0_by_frame.setdefault(v['frame'], v)
    steps = register(l0_by_frame)
    print(f'registered {len(steps)} frame steps from {len(l0_by_frame)} Level-0 views')

    zoomed = [v for v in views if v['level'] in (1, 2)]
    print(f'{len(zoomed)} zoomed views from {len(runs)} runs')
    cache = {}

    def image(v):
        key = (str(v['run']), v['file'])
        if key not in cache:
            img = cv2.imread(str(v['run'] / v['file']))
            if v['level'] > 0 and UPSCALE[v['level']] > 1:
                img = cv2.resize(img, None, fx=UPSCALE[v['level']], fy=UPSCALE[v['level']], interpolation=cv2.INTER_CUBIC)
            cache[key] = img
        return cache[key]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    first, last = min(l0_by_frame), max(l0_by_frame)
    targets = args.frames or list(range(first, last + 1))
    coverage = json.loads((out / 'coverage.json').read_text()) if args.frames and (out / 'coverage.json').exists() else {}
    for F in targets:
        canvas = np.zeros((H, W, 3), np.uint8)
        level_map = np.full((H, W), -1, np.int8)

        # Base: the Level-0 view of F, or the nearest one warped into F.
        nearest = min(l0_by_frame, key=lambda f: abs(f - F))
        base_view = l0_by_frame[nearest]
        base = cv2.resize(cv2.imread(str(base_view['run'] / base_view['file'])), (W, H), interpolation=cv2.INTER_CUBIC)
        if nearest == F:
            canvas[:], level_map[:] = base, 0
        else:
            paste_warped(canvas, level_map, base, between(steps, nearest, F), 0)

        # Zoomed views, Level 1 first, then Level 2; within a level the closest in time last (on top).
        candidates = [v for v in zoomed if abs(v['frame'] - F) <= args.max_age]
        candidates.sort(key=lambda v: (v['level'], -abs(v['frame'] - F)))
        for v in candidates:
            paste_warped(canvas, level_map, image(v), between(steps, v['frame'], F) @ placement(v), v['level'])

        cv2.imwrite(str(out / f'frame_{F:04d}.jpg'), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        total = level_map.size
        coverage[str(F)] = {k: round(float((level_map == lvl).sum()) / total, 3)
                            for k, lvl in (('L2', 2), ('L1', 1), ('L0_only', 0), ('empty', -1))}
        if len(cache) > 600:
            cache.clear()
        if F % 25 == 0:
            print(f'  frame {F}')
    (out / 'coverage.json').write_text(json.dumps(coverage, indent=1))
    (out / 'homographies.json').write_text(json.dumps({f: m.round(8).tolist() for f, m in sorted(steps.items())}))
    mean = {k: round(float(np.mean([c[k] for c in coverage.values()])), 3) for k in ('L2', 'L1', 'L0_only', 'empty')}
    print(f'{len(targets)} frames -> {out}; mean coverage {mean}')


if __name__ == '__main__':
    main()
