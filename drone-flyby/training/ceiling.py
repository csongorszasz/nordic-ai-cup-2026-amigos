"""Upper bounds on the Copenhagen flight: what the detector reaches when every frame is seen
whole at Level 1 (4 quadrants, 2x shrunk), at Level 2 (all native tiles), or both. No memory
and no camera limits, so a camera policy can only approach these.

    python training/ceiling.py runs/<run>/weights/last.pt [--small 1600]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from copenhagen_candidates import FRAMES, nms, tiles  # noqa: E402
from local_evaluator import score  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--small', type=int, default=0, help='second pass for the small classes on L1 views (DRONE_SMALL_IMGSZ)')
    args = parser.parse_args()
    from ultralytics import YOLO
    m = YOLO(args.model)
    small = [k for k, v in m.names.items() if v in ('ta-ta', 'small_launcher', 'medium_launcher')]
    truth = {int(k): v for k, v in json.loads((ROOT / 'datasets/copenhagen_test/labels.json').read_text())['labels'].items()}

    def run(img, off, s):
        rs = [m.predict(img, imgsz=960, conf=0.05, verbose=False)[0]]
        if args.small and s > 1:
            rs[0] = rs[0][[int(c) not in small for c in rs[0].boxes.cls]] if len(rs[0].boxes) else rs[0]
            rs.append(m.predict(img, imgsz=args.small, conf=0.05, classes=small, verbose=False)[0])
        return [(m.names[int(c)], p, [off[0] + b[0] * s, off[1] + b[1] * s, off[0] + b[2] * s, off[1] + b[3] * s])
                for r in rs for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]

    preds = {'L1': {}, 'L2': {}, 'L1+L2': {}}
    for f in sorted(truth):
        frame = cv2.imread(str(FRAMES / f'frame_{f:04d}.jpg'))
        l1 = [d for x in (0, 1920) for y in (0, 1080)
              for d in run(cv2.resize(frame[y:y + 1080, x:x + 1920], (960, 540), interpolation=cv2.INTER_AREA), (x, y), 2.0)]
        l2 = [d for x1, y1, x2, y2 in tiles() for d in run(frame[y1:y2, x1:x2], (x1, y1), 1.0)]
        for key, found in (('L1', l1), ('L2', l2), ('L1+L2', l1 + l2)):
            preds[key][f] = [{'object_id': c, 'bbox': b, 'confidence': p} for c, p, b in nms(found)]
    for key, pr in preds.items():
        mean, per = score('validation_4k', pr, truth)
        halves = [score('validation_4k', {f: v for f, v in pr.items() if keep(f)}, {f: v for f, v in truth.items() if keep(f)})[0]
                  for keep in (lambda f: f <= 125, lambda f: f > 125)]
        print(f'{key:6} all {mean:.3f} tune {halves[0]:.3f} check {halves[1]:.3f}   '
              + ', '.join(f'{c} {v:.2f}' for c, v in sorted(per.items(), key=lambda kv: kv[1])), flush=True)


if __name__ == '__main__':
    main()
