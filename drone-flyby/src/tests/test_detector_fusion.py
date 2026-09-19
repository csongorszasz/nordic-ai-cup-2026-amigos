import pytest
import numpy as np

from config import DroneFlybyConfig
from core.detector import create_detector, fuse_detection_pairs
from core.interfaces import DetectionResult
from utils import pairwise_iou, source_bbox_to_global


def detection(box=(100, 100, 140, 140), confidence=0.8, name="tank"):
    return DetectionResult(name, source_bbox_to_global(box), confidence, 0, box)


def test_pair_fusion_weights_boxes_and_averages_model_scores():
    first = detection(confidence=0.9)
    second = detection((102, 100, 142, 140), confidence=0.6)
    result = fuse_detection_pairs([first], [second])
    assert len(result) == 1
    assert result[0].source_pixel_bbox == pytest.approx((100.8, 100, 140.8, 140))
    assert result[0].confidence == pytest.approx(0.75)
    assert first.confidence == 0.9


def test_different_classes_are_not_fused():
    result = fuse_detection_pairs([detection()], [detection(name="jammer")])
    assert {item.class_name for item in result} == {"tank", "jammer"}
    assert all(item.confidence == 0.4 for item in result)


def test_one_model_cannot_count_twice_as_cross_model_agreement():
    result = fuse_detection_pairs(
        [detection(confidence=0.9), detection((101, 100, 141, 140), confidence=0.7)],
        [detection(confidence=0.8)],
    )
    assert len(result) == 2
    assert sorted(item.confidence for item in result) == pytest.approx([0.35, 0.85])


def test_single_model_and_empty_outputs_are_explicit():
    assert fuse_detection_pairs([], []) == []
    result = fuse_detection_pairs([detection()], [])
    assert result[0].confidence == 0.4
    assert result[0].source_pixel_bbox == (100, 100, 140, 140)


def test_proposal_limit_and_zero_confidence_geometry():
    result = fuse_detection_pairs([detection(confidence=0)], [detection(confidence=0)], max_proposals=1)
    assert len(result) == 1
    assert result[0].confidence == 0
    assert result[0].source_pixel_bbox == (100, 100, 140, 140)


def test_pair_factory_requires_explicit_auxiliary_weights():
    with pytest.raises(ValueError, match="AUX_WEIGHTS"):
        create_detector(DroneFlybyConfig(DETECTOR_TYPE="yolo_pair"))


def test_shared_iou_supports_integer_pixels_and_empty_sets():
    assert pairwise_iou(np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10]])).item() == 1.0
    assert pairwise_iou(np.empty((0, 4)), np.empty((3, 4))).shape == (0, 3)


def test_self_fusion_preserves_boxes_classes_and_scores():
    predictions = [
        detection((100, 100, 140, 140), 0.9),
        detection((108, 100, 148, 140), 0.7),
        detection((500, 500, 550, 560), 0.001, name="jammer"),
    ]
    result = fuse_detection_pairs(predictions, predictions)
    assert len(result) == len(predictions)
    for fused, original in zip(result, predictions):
        assert fused.class_name == original.class_name
        assert fused.confidence == pytest.approx(original.confidence)
        assert fused.source_pixel_bbox == pytest.approx(original.source_pixel_bbox)
        assert fused.bbox_global == pytest.approx(original.bbox_global)
