"""Score one explicitly identified live sequence against the same local labels.

These are server-generated responses, not proof of evaluator acceptance. Missing
responses retain their ground truth. Local labels may omit objects/classes or
use different box/visibility conventions from the private competition labels.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dtos import IMAGE_HEIGHT, IMAGE_WIDTH, OBJECT_CLASSES, DroneFlybyPredictionDto, RequestedViewDto  # noqa: E402
from local_evaluator import score  # noqa: E402
from utils import annotations_to_predictions  # noqa: E402


def read_records(path: Path) -> list[dict]:
    records = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f'{path}:{number}: invalid JSON') from error
        if not isinstance(record, dict) or not isinstance(record.get('sequence_id'), str) or not record['sequence_id']:
            raise ValueError(f'{path}:{number}: missing sequence identity')
        if type(record.get('frame')) is not int or record['frame'] < 0:
            raise ValueError(f'{path}:{number}: invalid source frame')
        if not isinstance(record.get('annotations'), list):
            raise ValueError(f'{path}:{number}: missing annotations')
        records.append(record)
    if not records:
        raise ValueError(f'No response records in {path}')
    return records


def list_sequences(records: list[dict]) -> list[dict]:
    groups = {}
    for record in records:
        groups.setdefault(record['sequence_id'], []).append(record)
    return [
        {'sequence_id': sequence, 'records': len(rows),
         'unique_frames': len({row['frame'] for row in rows}),
         'first_frame': min(row['frame'] for row in rows),
         'last_frame': max(row['frame'] for row in rows)}
        for sequence, rows in groups.items()
    ]


def select_sequence(records: list[dict], sequence: str | None = None) -> list[dict]:
    available = list_sequences(records)
    if sequence is None:
        if len(available) != 1:
            raise ValueError(f'Multiple sequences; select --sequence explicitly: {json.dumps(available)}')
        sequence = available[0]['sequence_id']
    selected = [record for record in records if record['sequence_id'] == sequence]
    if not selected:
        raise ValueError(f'Sequence {sequence!r} is not in the log')
    return selected


def logged_predictions(records: list[dict]) -> tuple[dict, int]:
    predictions, identities = {}, {}
    duplicates = 0
    for row in records:
        if row.get('response_validated') is False:
            raise ValueError(f"Frame {row['frame']} was logged as an invalid response")
        width, height = row.get('original_width', IMAGE_WIDTH), row.get('original_height', IMAGE_HEIGHT)
        if type(width) is not int or type(height) is not int:
            raise ValueError('Recorded source dimensions must be integers')
        annotations = [DroneFlybyPredictionDto.model_validate(value) for value in row['annotations']]
        requested = row.get('requested_view')
        if requested is not None:
            RequestedViewDto.model_validate(requested)
        current = annotations_to_predictions(annotations, width, height)
        frame = row['frame']
        identity = (row.get('request_id'), row.get('frame_index'), row.get('level'),
                    row.get('region'), requested)
        if frame in predictions:
            if predictions[frame] != current or identities[frame] != identity:
                raise ValueError(f'Conflicting responses for frame {frame}; evaluator acceptance is unknown')
            duplicates += 1
        else:
            predictions[frame], identities[frame] = current, identity
    return predictions, duplicates


def score_records(records: list[dict], truth: dict[int, list[dict]]) -> dict:
    if len({row['sequence_id'] for row in records}) != 1:
        raise ValueError('Select exactly one sequence before scoring')
    predictions, duplicates = logged_predictions(records)
    mean, per_class = score('validation_4k', predictions, truth)
    latencies = []
    for row in records:
        value = row.get('predict_ms', row.get('ms'))
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid recorded server latency')
            latencies.append(float(value))
    unlabelled_classes = set(OBJECT_CLASSES) - set(per_class)
    unscored_counts = Counter(
        annotation['object_id'] for frame in truth for annotation in predictions.get(frame, [])
        if annotation['object_id'] in unlabelled_classes
    )
    report = {
        'sequence_id': records[0]['sequence_id'], 'records': len(records),
        'unique_reported_frames': len(predictions), 'identical_duplicate_records': duplicates,
        'labelled_frames': len(truth),
        'missing_prediction_frames': sorted(set(truth) - set(predictions)),
        'unscored_prediction_frames': sorted(set(predictions) - set(truth)),
        'map50': mean, 'ap50': per_class, 'evaluated_classes': list(per_class),
        'classes_absent_from_labels': sorted(unlabelled_classes),
        'predictions_in_classes_without_labels': dict(unscored_counts),
        'legacy_validation_status_unknown': sum(row.get('response_validated') is not True for row in records),
        'evaluator_acceptance_known': False,
        'score_scope': 'Same-label comparison, not the private competition score or a proof of its class set',
        'prediction_precision': 'full; no box or confidence rounding',
        'latency_scope': 'Recorded server/predict time, not client round-trip time',
        'median_recorded_ms': statistics.median(latencies) if latencies else None,
        'max_recorded_ms': max(latencies) if latencies else None,
    }
    for name, keep in (('tune', lambda frame: frame <= 125), ('check', lambda frame: frame > 125)):
        part = {frame: annotations for frame, annotations in truth.items() if keep(frame)}
        if any(part.values()):
            part_mean, part_classes = score('validation_4k', predictions, part)
            report[name] = {'map50': part_mean, 'labelled_frames': len(part),
                            'evaluated_classes': list(part_classes), 'ap50': part_classes}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--sequence', help='Required when the file contains multiple sequences, including Verify calls.')
    parser.add_argument('--list-sequences', action='store_true')
    parser.add_argument('--labels', type=Path, default=ROOT / 'datasets' / 'copenhagen_test' / 'labels.json')
    parser.add_argument('--output-json', type=Path)
    args = parser.parse_args()
    records = read_records(args.log)
    if args.list_sequences:
        print(json.dumps(list_sequences(records), indent=2))
        return 0
    selected = select_sequence(records, args.sequence)
    source = json.loads(args.labels.read_text(encoding='utf-8'))['labels']
    truth = {int(frame): annotations for frame, annotations in source.items()}
    if len(truth) != len(source):
        raise ValueError('Label frame identifiers collide after integer conversion')
    report = score_records(selected, truth)
    print(f"{report['sequence_id']}: {report['unique_reported_frames']}/{report['labelled_frames']} labelled frames reported; "
          f"{len(report['missing_prediction_frames'])} missing, {report['identical_duplicate_records']} identical duplicates")
    print(f"COCO mAP@0.50 against these labels: {report['map50']:.6f} ({len(report['evaluated_classes'])} classes)")
    print('  ' + ', '.join(f'{name} {value:.3f}' for name, value in sorted(report['ap50'].items(), key=lambda item: item[1])))
    print('Classes absent from these labels: ' + ', '.join(report['classes_absent_from_labels']))
    print('Evaluator acceptance is not known from a server response log.')
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
