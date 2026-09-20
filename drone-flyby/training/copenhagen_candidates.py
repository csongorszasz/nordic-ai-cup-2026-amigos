"""Detections on the recorded Copenhagen flight, linked into tracks, for review by hand.

The recorded validation frames (recordings/validation_4k) have no labels. This runs a
detector over every frame, links its detections across frames into tracks (the same
object stays in view for ~30 frames; the ground motion between frames is known from
reconstruct_frames.py's homographies.json), and writes one candidate per track. The
review page (/review on the sprite review server) shows them to be accepted, rejected
or given another class; accepted tracks become per-frame labels.

TEST SET ONLY. Nothing here may ever be used for training: it is the only honest
measure we have of how the detector does on unseen imagery.

    python training/copenhagen_candidates.py --model runs/<run>/weights/best.pt

Writes datasets/copenhagen_test/candidates.json. Decisions are kept apart, in
datasets/copenhagen_test/decisions.json, keyed by track id. Before regenerating, copy the
reviewed candidates.json to the next candidates_r<N>.json (kept in git): a regenerated track
that is the same object as a decided one takes its id, so only new objects need a review.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dtos import OBJECT_CLASSES  # noqa: E402
from reconstruct_frames import between  # noqa: E402

FRAMES = ROOT / 'recordings' / 'validation_4k'
OUT = ROOT / 'datasets' / 'copenhagen_test'
W, H = 3840, 2160
# Level-2 tiles (native resolution) overlapping by a sixth, so objects on a seam are whole in one.
TILE, STRIDE = (960, 540), (800, 450)
LINK_IOU = 0.2      # a detection continues a track when it overlaps the track's predicted box this much
MAX_MISSES = 4      # frames a track may go undetected before it ends
NMS_IOU = 0.5


def tiles():
    xs = list(range(0, W - TILE[0], STRIDE[0])) + [W - TILE[0]]
    ys = list(range(0, H - TILE[1], STRIDE[1])) + [H - TILE[1]]
    return [(x, y, x + TILE[0], y + TILE[1]) for y in ys for x in xs]


def detect(model, frame, conf):
    """(class, conf, box) over the frame: Level-2 tiles, plus Level-1 views for the big objects."""
    views = [(frame[y1:y2, x1:x2], (x1, y1), 1.0) for x1, y1, x2, y2 in tiles()]
    for x1 in (0, 1920):
        for y1 in (0, 1080):
            crop = frame[y1:y1 + 1080, x1:x1 + 1920]
            views.append((cv2.resize(crop, (960, 540), interpolation=cv2.INTER_AREA), (x1, y1), 2.0))
    found = []
    for k in range(0, len(views), 16):
        batch = views[k:k + 16]
        for (img, (ox, oy), scale), result in zip(batch, model.predict([v[0] for v in batch], imgsz=960, conf=conf,
                                                                         verbose=False, half=True)):
            for (x1, y1, x2, y2), c, s in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                              result.boxes.conf.tolist()):
                found.append((OBJECT_CLASSES[int(c)], float(s),
                              np.array([ox + x1 * scale, oy + y1 * scale, ox + x2 * scale, oy + y2 * scale])))
    return nms(found)


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def nms(found):
    """Class-agnostic: overlapping tiles see the same object, sometimes as two classes."""
    kept = []
    for item in sorted(found, key=lambda d: -d[1]):
        if all(iou(item[2], k[2]) < NMS_IOU for k in kept):
            kept.append(item)
    return kept


def move(box, M):
    """The box's corners through homography M, as the enclosing box."""
    x1, y1, x2, y2 = box
    pts = cv2.perspectiveTransform(np.float32([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]]), M)[:, 0]
    return np.array([*pts.min(axis=0), *pts.max(axis=0)])


CARRY_IOU = 0.5    # a new track is an already-decided one if their boxes overlap this much (median over shared frames)


def carry_over(candidates) -> int:
    """Give a new track the id of an already-decided track of an earlier round (candidates_r*.json)
    that is the same object, so the decision applies to it and it is not reviewed again."""
    decisions_path = OUT / 'decisions.json'
    if not decisions_path.exists():
        return 0
    decided = set(json.loads(decisions_path.read_text()))
    old = {}
    for path in sorted(OUT.glob('candidates_r*.json')):
        for c in json.loads(path.read_text())['candidates']:
            if c['id'] in decided:
                old.setdefault(c['id'], dict(zip(c['frames'], c['boxes'])))
    carried = 0
    for c in candidates:
        best, best_overlap = None, CARRY_IOU
        for oid, boxes in old.items():
            shared = [iou(box, boxes[f]) for f, box in zip(c['frames'], c['boxes']) if f in boxes]
            if len(shared) >= 2 and np.median(shared) >= best_overlap:
                best, best_overlap = oid, float(np.median(shared))
        if best:
            c['id'] = best
            carried += 1
    return carried


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--conf', type=float, default=0.1, help='detections below this are not linked at all')
    parser.add_argument('--min-score', type=float, default=0.3,
                        help='keep a track if its best confidence reaches this, or if it was seen in --min-frames frames')
    parser.add_argument('--min-frames', type=int, default=3,
                        help='frames that keep a track below --min-score (low, persistent junk on the map passed at 3)')
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    steps = {int(k): np.array(v) for k, v in json.loads((FRAMES / 'homographies.json').read_text()).items()}
    frames = sorted(int(p.stem.split('_')[-1]) for p in FRAMES.glob('frame_*.jpg'))

    tracks, active = [], []
    for F in frames:
        found = detect(model, cv2.imread(str(FRAMES / f'frame_{F:04d}.jpg')), args.conf)
        # Predict where each active track is now, then match greedily by overlap.
        predicted = [move(t['boxes'][-1], between(steps, t['frames'][-1], F)) for t in active]
        pairs = sorted(((iou(p, d[2]), i, j) for i, p in enumerate(predicted) for j, d in enumerate(found)), reverse=True)
        used_t, used_d = set(), set()
        for overlap, i, j in pairs:
            if overlap < LINK_IOU or i in used_t or j in used_d:
                continue
            used_t.add(i)
            used_d.add(j)
            cls, conf, box = found[j]
            active[i]['frames'].append(F)
            active[i]['boxes'].append(box)
            active[i]['classes'].append(cls)
            active[i]['confs'].append(conf)
        for j, (cls, conf, box) in enumerate(found):
            if j not in used_d:
                active.append({'frames': [F], 'boxes': [box], 'classes': [cls], 'confs': [conf]})
        still = []
        for t in active:
            gone = move(t['boxes'][-1], between(steps, t['frames'][-1], F))
            off_frame = gone[1] > H or gone[3] < 0 or gone[0] > W or gone[2] < 0
            (tracks if F - t['frames'][-1] > MAX_MISSES or off_frame else still).append(t)
        active = still
        print(f'frame {F}: {len(found)} detections, {len(active)} active tracks', flush=True)
    tracks += active

    candidates = []
    for t in tracks:
        best = int(np.argmax(t['confs']))
        votes = {}
        for cls, conf in zip(t['classes'], t['confs']):
            votes[cls] = votes.get(cls, 0.0) + conf
        if max(t['confs']) < args.min_score and len(t['frames']) < args.min_frames:
            continue
        frame, box = t['frames'][best], t['boxes'][best]
        candidates.append({
            # Stable enough to key decisions on: where and when the track is at its best.
            'id': f'f{frame:03d}_{int((box[0] + box[2]) / 2):04d}_{int((box[1] + box[3]) / 2):04d}',
            'class': max(votes, key=votes.get), 'votes': {k: round(v, 2) for k, v in votes.items()},
            'score': round(max(t['confs']), 3), 'best_frame': frame, 'best_box': [round(v) for v in box],
            'frames': t['frames'], 'boxes': [[round(v) for v in b] for b in t['boxes']],
            'confs': [round(c, 3) for c in t['confs']], 'classes': t['classes'],
        })
    candidates.sort(key=lambda c: -c['score'])
    carried = carry_over(candidates)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'candidates.json').write_text(json.dumps({
        'model': str(Path(args.model).resolve()), 'conf': args.conf, 'min_score': args.min_score,
        'created': datetime.now().isoformat(timespec='seconds'), 'candidates': candidates,
    }))
    per_class = {}
    for c in candidates:
        per_class[c['class']] = per_class.get(c['class'], 0) + 1
    print(f'{len(candidates)} candidate tracks (of {len(tracks)}) -> {OUT / "candidates.json"}; '
          f'{carried} are tracks already decided in an earlier round (their decision applies)')
    print('  by class: ' + ', '.join(f'{k} {v}' for k, v in sorted(per_class.items(), key=lambda kv: -kv[1])))


if __name__ == '__main__':
    main()
