from core.interfaces import DetectionResult
from offline.zoom_diagnostics import fully_contained, match_observations


def detection(name="tank"):
    return DetectionResult(name, (0.1, 0.1, 0.2, 0.2), 0.8, 0, (10, 10, 30, 30))


def test_matching_distinguishes_classification_from_localization():
    annotations = [{"object_id": "tank", "bbox": [10, 10, 30, 30]}]
    detections = [detection("jammer")]
    assert match_observations(annotations, detections) == {}
    assert match_observations(annotations, detections, require_class=False) == {0: 0.8}


def test_one_detection_cannot_credit_two_ground_truth_objects():
    annotations = [{"object_id": "tank", "bbox": [10, 10, 30, 30]}] * 2
    assert len(match_observations(annotations, [detection()])) == 1


def test_border_truncated_boxes_do_not_enter_full_visibility_diagnostic():
    annotations = [
        {"object_id": "tank", "bbox": [10, 10, 30, 30]},
        {"object_id": "jammer", "bbox": [0, 10, 20, 30]},
        {"object_id": "hangar", "bbox": [90, 10, 120, 30]},
    ]
    assert fully_contained(annotations, (0, 0, 100, 100)) == annotations[:1]
