"""Replay a live run's exact camera path offline: the same frames, in the same views.

    DRONE_MODEL=... DRONE_SMALL_IMGSZ=1280 ... python training/replay_live_views.py recordings/<live run>

The live server saves every view it is sent (solution.record_view: views.jsonl). Here the
recorded Copenhagen frames are cut to those views and passed to solution.predict in order, so the
answers depend only on the camera path, not on the network. Scored against our labels: compare with
training/score_live_log.py on the same run's response log, and with run_policy.py's own path.
"""

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dtos import DroneFlybyPredictRequestDto  # noqa: E402
from local_evaluator import Camera, build_request, render_view, score  # noqa: E402
import solution  # noqa: E402


def main():
    run = Path(sys.argv[1])
    views = [json.loads(line) for line in open(run / 'views.jsonl')]
    truth = {int(k): v for k, v in json.loads((ROOT / 'datasets' / 'copenhagen_test' / 'labels.json').read_text())['labels'].items()}
    predictions = {}
    for v in views:
        x1, y1, x2, y2 = v['source_region_xyxy']
        camera = Camera(resolution_level=v['level'], center_x=(x1 + x2) // 2, center_y=(y1 + y2) // 2)
        image = cv2.imread(str(ROOT / 'recordings' / 'validation_4k' / f"frame_{v['frame']:04d}.jpg"))
        payload = build_request(v['frame'], v['frame_index'], camera, render_view(image, camera), None)
        response = solution.predict(DroneFlybyPredictRequestDto.model_validate(payload))
        predictions[v['frame']] = [{'object_id': a.object_id, 'confidence': a.confidence,
                                    'bbox': [round(c * s) for c, s in zip(a.bbox, (3840, 2160, 3840, 2160))]}
                                   for a in response.annotations]
    mean, per_class = score('validation_4k', predictions, truth)
    print(f'{run.name}: {len(views)} views, COCO mAP@0.50 against our labels: {mean:.3f}')
    print('  ' + ', '.join(f'{k} {v:.2f}' for k, v in sorted(per_class.items(), key=lambda kv: kv[1])))


if __name__ == '__main__':
    main()
