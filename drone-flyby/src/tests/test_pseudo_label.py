"""Pseudo-label generation writes numeric YOLO class IDs."""

import cv2
import numpy as np

from core.interfaces import DetectionResult
from dtos import OBJECT_CLASSES
from offline.pseudo_label import pseudo_label_directory


class _StubDetector:
    def __init__(self, detections):
        self._detections = detections

    def warmup(self):
        pass

    def detect(self, image, zoom_level, source_region_xyxy):
        return self._detections


def _image(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    cv2.imwrite(str(images / "frame_000000.png"), np.zeros((540, 960, 3), dtype=np.uint8))
    return images


def _detection(class_name):
    return DetectionResult(
        class_name=class_name,
        bbox_global=(0.1, 0.1, 0.2, 0.2),
        confidence=0.9,
        zoom_level=0,
        source_pixel_bbox=(96.0, 54.0, 192.0, 108.0),
    )


def test_writes_numeric_class_ids(tmp_path):
    labels = tmp_path / "labels"
    counts = pseudo_label_directory(
        _image(tmp_path), labels, detector=_StubDetector([_detection("tank")])
    )

    token = (labels / "frame_000000.txt").read_text().strip().split()
    assert int(token[0]) == OBJECT_CLASSES.index("tank")
    assert counts["tank"] == 1


def test_drops_unknown_class(tmp_path):
    labels = tmp_path / "labels"
    counts = pseudo_label_directory(
        _image(tmp_path), labels, detector=_StubDetector([_detection("car")])
    )

    assert (labels / "frame_000000.txt").read_text().strip() == ""
    assert counts == {}
