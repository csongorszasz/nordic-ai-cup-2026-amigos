import json

import pytest

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH
from training.score_live_log import logged_predictions, read_records, score_records, select_sequence


def row(frame=1, sequence='validation', confidence=0.9):
    return {
        'sequence_id': sequence, 'frame': frame, 'frame_index': frame - 1,
        'level': 0, 'region': [0, 0, IMAGE_WIDTH, IMAGE_HEIGHT], 'ms': 12.5,
        'annotations': [{
            'object_id': 'tank',
            'bbox': [10 / IMAGE_WIDTH, 10 / IMAGE_HEIGHT, 20 / IMAGE_WIDTH, 20 / IMAGE_HEIGHT],
            'confidence': confidence,
        }],
        'requested_view': None,
    }


def test_later_verify_call_does_not_silently_replace_validation_sequence():
    rows = [row(1), row(2), row(1, 'verify')]
    with pytest.raises(ValueError, match='--sequence'):
        select_sequence(rows)
    assert select_sequence(rows, 'validation') == rows[:2]
    with pytest.raises(ValueError, match='not in the log'):
        select_sequence(rows, 'missing')


def test_missing_frames_retain_the_complete_label_denominator():
    truth = {frame: [{'object_id': 'tank', 'bbox': [10, 10, 20, 20]}] for frame in (1, 2)}
    report = score_records([row(1)], truth)
    assert report['labelled_frames'] == 2
    assert report['missing_prediction_frames'] == [2]
    assert 0 < report['map50'] < 0.6
    assert report['evaluator_acceptance_known'] is False


def test_identical_duplicates_are_visible_but_conflicting_answers_are_rejected():
    first = row()
    second = {**first, 'at': 42, 'ms': 20}
    predictions, duplicates = logged_predictions([first, second])
    assert len(predictions) == 1 and duplicates == 1
    with pytest.raises(ValueError, match='Conflicting'):
        logged_predictions([first, row(confidence=0.8)])


def test_missing_label_classes_are_explicit_not_assumed_absent_in_reality():
    record = row()
    record['annotations'].append({
        'object_id': 'spacecraft', 'bbox': [0.3, 0.3, 0.4, 0.4], 'confidence': 0.99,
    })
    report = score_records([record], {1: [{'object_id': 'tank', 'bbox': [10, 10, 20, 20]}]})
    assert report['map50'] == pytest.approx(1)
    assert report['evaluated_classes'] == ['tank']
    assert 'spacecraft' in report['classes_absent_from_labels']
    assert report['predictions_in_classes_without_labels'] == {'spacecraft': 1}


def test_empty_and_malformed_logs_fail_explicitly(tmp_path):
    path = tmp_path / 'responses.jsonl'
    path.write_text('')
    with pytest.raises(ValueError, match='No response records'):
        read_records(path)
    path.write_text(json.dumps(row()) + '\nnot json\n')
    with pytest.raises(ValueError, match=':2: invalid JSON'):
        read_records(path)


def test_logged_invalid_response_is_not_scored_as_success():
    with pytest.raises(ValueError, match='invalid response'):
        logged_predictions([{**row(), 'response_validated': False}])
