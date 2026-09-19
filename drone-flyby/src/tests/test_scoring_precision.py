import pytest

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH, DroneFlybyPredictionDto
from local_evaluator import score
from utils import annotations_to_predictions


def annotation(box, confidence):
    return DroneFlybyPredictionDto(
        object_id='tank',
        bbox=[value / size for value, size in zip(box, (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT))],
        confidence=confidence,
    )


def test_scoring_conversion_preserves_confidence_order_that_rounding_erases():
    truth = {1: [{'object_id': 'tank', 'bbox': [10, 10, 20, 20]}]}
    predictions = annotations_to_predictions([
        annotation([40, 40, 50, 50], 0.5001),
        annotation([10, 10, 20, 20], 0.5002),
    ])
    full = score('fixture', {1: predictions}, truth)[0]
    rounded = score('fixture', {1: [{**p, 'confidence': round(p['confidence'], 3)} for p in predictions]}, truth)[0]
    assert full == pytest.approx(1)
    assert rounded == pytest.approx(0.5)
    assert predictions[1]['confidence'] == 0.5002


def test_box_rounding_can_inflate_an_iou50_miss_to_a_perfect_match():
    truth = {1: [{'object_id': 'tank', 'bbox': [10, 10, 12, 12]}]}
    predictions = annotations_to_predictions([annotation([10.49, 10.49, 12.49, 12.49], 0.9)])
    full = score('fixture', {1: predictions}, truth)[0]
    rounded = score('fixture', {1: [{**p, 'bbox': [round(v) for v in p['bbox']]} for p in predictions]}, truth)[0]
    assert full == 0
    assert rounded == pytest.approx(1)
    assert predictions[0]['bbox'][0] == pytest.approx(10.49)


def test_scoring_conversion_rejects_invalid_source_dimensions():
    with pytest.raises(ValueError, match='positive'):
        annotations_to_predictions([], original_width=0)
