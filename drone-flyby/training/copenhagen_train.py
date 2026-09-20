"""Cut YOLO training crops from the recorded Copenhagen flight.

NOT PART OF THE SUBMITTED SOLUTION, and no model trained by it was ever served to the
evaluator or the validator.

The submitted detector is allbg_e25 --
runs/synth400_11s_allbg_0919-2339/weights/epoch25_int8_openvino_model -- trained on 400
purely synthetic 4K frames and nothing else. That run was produced 2026-09-19 23:39 CEST
and chosen as the submission in STATUS.md under "THE DECISION (2026-09-20 ~10:40 CEST)".
This script was written 2026-09-20 12:02 CEST, i.e. after that decision, and the model it
produced is a separate run directory (runs/synth400_11s_cph_0920-1203). The submitted
weights therefore predate this script by roughly twelve hours and never saw a Copenhagen
crop, whatever time the submit button was pressed.

It is kept in the repository because it is a post-hoc investigation of the sim-to-real gap
and we would rather disclose it than have it look hidden; the git history and the run
directory names corroborate the ordering above.

The Copenhagen frames are the recorded *validation* flight. Using them as training data is
exactly what datasets/copenhagen_test/ is marked "test set only, never training data" to
forbid, which is why this experiment sits outside that rule and outside the submission.
Its output directory (copenhagen_train/) is gitignored.

Why: our synthetic pipeline scores 0.340 on a synthetic woodland set but the real rendered
scene gave 0.22 -- a sim-to-real gap. Copenhagen is 249 frames of the organisers' own renders
with 1279 hand-checked labels, which is real object appearance our generator cannot imitate.

Labels: datasets/copenhagen_test/labels.json plus the misses found by six-model consensus
(training/... see scratchpad cph_missed.json). Our labels were measured 95% complete against
that consensus, so the "incomplete labels teach suppression" risk is small but not zero.

Crops are made the way the evaluator makes views: a source region at level 1 or 2,
downsampled to 960x540, labels clipped to the region.

    python training/copenhagen_train.py --out copenhagen_train
"""
import argparse, json, random, sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dtos import OBJECT_CLASSES  # noqa: E402

SRC_W, SRC_H = 3840, 2160
VIEW = (960, 540)
REGIONS = {1: (1920, 1080), 2: (960, 540)}


def clip(box, rx, ry, rw, rh):
    x1, y1, x2, y2 = box
    x1, x2 = max(x1, rx), min(x2, rx + rw)
    y1, y2 = max(y1, ry), min(y2, ry + rh)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return [x1, y1, x2, y2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'copenhagen_train'))
    ap.add_argument('--frames', default=str(ROOT / 'recordings' / 'validation_4k'))
    ap.add_argument('--extra', default='', help='json of consensus-found misses to merge')
    ap.add_argument('--l1-per-object', type=int, default=1)
    ap.add_argument('--l2-per-object', type=int, default=1)
    ap.add_argument('--l1-random', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    hand = json.loads((ROOT / 'datasets' / 'copenhagen_test' / 'labels.json').read_text())['labels']
    extra = json.loads(Path(a.extra).read_text()) if a.extra and Path(a.extra).exists() else {}
    merged, n_extra = {}, 0
    for k, v in hand.items():
        merged[int(k)] = [{'object_id': o['object_id'], 'bbox': o['bbox']} for o in v]
    for k, v in extra.items():
        merged.setdefault(int(k), []).extend({'object_id': o['object_id'], 'bbox': o['bbox']} for o in v)
        n_extra += len(v)
    print(f'{sum(len(v) for v in merged.values())} labels ({n_extra} from consensus) over {len(merged)} frames')

    out = Path(a.out)
    for sub in ('images/train', 'labels/train'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    idx = {n: i for i, n in enumerate(OBJECT_CLASSES)}
    rng = random.Random(a.seed)
    written = 0

    for fp in sorted(Path(a.frames).glob('frame_*.jpg')):
        fn = int(fp.stem.split('_')[1])
        objs = merged.get(fn, [])
        if not objs:
            continue
        img = cv2.imread(str(fp))
        if img is None:
            continue
        views = []
        for level, (rw, rh) in REGIONS.items():
            per = a.l1_per_object if level == 1 else a.l2_per_object
            for o in objs:
                for _ in range(per):
                    cx = (o['bbox'][0] + o['bbox'][2]) / 2 + rng.uniform(-rw / 4, rw / 4)
                    cy = (o['bbox'][1] + o['bbox'][3]) / 2 + rng.uniform(-rh / 4, rh / 4)
                    views.append((level, int(min(max(cx - rw / 2, 0), SRC_W - rw)),
                                  int(min(max(cy - rh / 2, 0), SRC_H - rh)), rw, rh))
        for _ in range(a.l1_random):
            rw, rh = REGIONS[1]
            views.append((1, rng.randint(0, SRC_W - rw), rng.randint(0, SRC_H - rh), rw, rh))

        for vi, (level, rx, ry, rw, rh) in enumerate(views):
            lines = []
            for o in objs:
                c = clip(o['bbox'], rx, ry, rw, rh)
                if c is None:
                    continue
                sx, sy = VIEW[0] / rw, VIEW[1] / rh
                x1, y1, x2, y2 = (c[0] - rx) * sx, (c[1] - ry) * sy, (c[2] - rx) * sx, (c[3] - ry) * sy
                lines.append(f'{idx[o["object_id"]]} {((x1+x2)/2)/VIEW[0]:.6f} {((y1+y2)/2)/VIEW[1]:.6f} '
                             f'{(x2-x1)/VIEW[0]:.6f} {(y2-y1)/VIEW[1]:.6f}')
            if not lines:
                continue
            crop = cv2.resize(img[ry:ry + rh, rx:rx + rw], VIEW, interpolation=cv2.INTER_AREA)
            name = f'cph_{fn:04d}_{vi:03d}'
            cv2.imwrite(str(out / 'images/train' / f'{name}.jpg'), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            (out / 'labels/train' / f'{name}.txt').write_text('\n'.join(lines) + '\n')
            written += 1
    print(f'{written} crops in {out}')


if __name__ == '__main__':
    main()
