"""Score what the server answered in a live validation against our Copenhagen labels.

    python training/score_live_log.py logs/live/responses.jsonl

The log comes from the server run with DRONE_LOG_RESPONSES=<file> (src/api.py). If our labels
score the live answers about as well as the offline replay (training/run_policy.py), the gap
to the website's score is in the ground truth (objects or classes our labels lack); if they
score them as low as the website, the gap is in serving (skipped frames, timing, camera lag).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from local_evaluator import score  # noqa: E402


def main():
    lines = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
    sequence = lines[-1]['sequence_id']   # the last run in the file
    lines = [line for line in lines if line['sequence_id'] == sequence]
    predictions = {line['frame']: [{'object_id': a['object_id'], 'confidence': a['confidence'],
                                    'bbox': [round(v * s) for v, s in zip(a['bbox'], (3840, 2160, 3840, 2160))]}
                                   for a in line['annotations']] for line in lines}
    truth = {int(k): v for k, v in json.loads((ROOT / 'datasets' / 'copenhagen_test' / 'labels.json').read_text())['labels'].items()}
    ms = sorted(line['ms'] for line in lines)
    print(f'{sequence}: {len(lines)} frames answered (frames {min(predictions)}-{max(predictions)}), '
          f'median {ms[len(ms) // 2]} ms, max {ms[-1]} ms')
    mean, per_class = score('validation_4k', predictions, truth)
    print(f'COCO mAP@0.50 against our labels: {mean:.3f}')
    print('  ' + ', '.join(f'{k} {v:.2f}' for k, v in sorted(per_class.items(), key=lambda kv: kv[1])))


if __name__ == '__main__':
    main()
