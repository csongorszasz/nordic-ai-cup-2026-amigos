"""Ground-truth-to-detection conversion used by the oracle memory gate."""

import pytest

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH
from offline.oracle_memory_benchmark import _oracle_detections
from utils import load_annotations


def test_oracle_detections_mirror_ground_truth():
    annotations = load_annotations(0, "helsinki")
    detections = _oracle_detections(0, "helsinki")

    assert len(detections) == len(annotations)
    detection = detections[0]
    annotation = annotations[0]
    x1, y1, x2, y2 = (float(c) for c in annotation["bbox"])

    assert detection.class_name == annotation["object_id"]
    assert detection.confidence == 0.99
    assert detection.source_pixel_bbox == (x1, y1, x2, y2)
    assert detection.bbox_global == pytest.approx(
        (x1 / IMAGE_WIDTH, y1 / IMAGE_HEIGHT, x2 / IMAGE_WIDTH, y2 / IMAGE_HEIGHT)
    )
