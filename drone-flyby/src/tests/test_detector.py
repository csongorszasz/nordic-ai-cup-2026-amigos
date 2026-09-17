"""Tests for detector backends using the supplied Helsinki scene."""

import pytest
import numpy as np

from config import DroneFlybyConfig
from core.detector import (
    DummyCannyDetector,
    TemplateBankDetector,
    YoloDetector,
    _normalize_class_name,
    create_detector,
)
from dtos import OBJECT_CLASSES
from utils import load_frame


SOURCE_REGION = (0, 0, 3840, 2160)


def _assert_valid_detections(detections):
    for det in detections:
        x1, y1, x2, y2 = det.bbox_global
        sx1, sy1, sx2, sy2 = det.source_pixel_bbox
        assert 0.0 <= x1 < x2 <= 1.0
        assert 0.0 <= y1 < y2 <= 1.0
        assert det.confidence >= 0.0
        assert sx1 < sx2
        assert sy1 < sy2


def test_dummy_canny_detector_runs_on_real_helsinki_frame():
    detector = DummyCannyDetector()
    detector.warmup()

    image = load_frame(0, scene="helsinki")
    detections = detector.detect(image, zoom_level=0, source_region_xyxy=SOURCE_REGION)

    assert detections
    _assert_valid_detections(detections)


@pytest.mark.parametrize("frame_index", [0, 1, 24])
def test_template_bank_detector_finds_real_helsinki_objects(frame_index: int):
    detector = TemplateBankDetector(scene="helsinki", max_templates_per_class=1, max_proposals=20)
    detector.warmup()

    image = load_frame(frame_index, scene="helsinki")
    detections = detector.detect(image, zoom_level=0, source_region_xyxy=SOURCE_REGION)

    assert detections
    _assert_valid_detections(detections)

    classes = {det.class_name for det in detections}
    assert classes & {"jammer", "helicopter", "tank", "small_launcher"}


def test_detector_factory_falls_back_safely_for_yolo_backends():
    detector = create_detector(DroneFlybyConfig(DETECTOR_TYPE="yolo_standard"))
    assert hasattr(detector, "detect")
    assert hasattr(detector, "warmup")
    assert type(detector).__name__ in {"YoloDetector", "TemplateBankDetector"}


class _FakeScalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _FakeBox:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = _FakeScalar(cls_id)
        self.conf = _FakeScalar(conf)
        self.xyxy = np.array([xyxy], dtype=float)


class _FakeResult:
    def __init__(self, boxes, orig_shape=(540, 960)):
        self.boxes = boxes
        self.orig_shape = orig_shape


class _FakeYOLO:
    names = {0: "airplane", 1: "helicopter"}

    def __init__(self, weights_path):
        self.weights_path = weights_path

    def predict(self, image_bgr, conf, device, verbose):
        _ = (image_bgr, conf, device, verbose)
        return [
            _FakeResult(
                boxes=[
                    _FakeBox(0, 0.91, [40.0, 50.0, 120.0, 135.0]),
                    _FakeBox(1, 0.77, [200.0, 180.0, 260.0, 245.0]),
                ]
            )
        ]


def test_yolo_detector_parses_and_normalizes_labels(monkeypatch):
    import core.detector as detector_module

    monkeypatch.setattr(detector_module, "_ULTRALYTICS_AVAILABLE", True)
    monkeypatch.setattr(detector_module, "YOLO", _FakeYOLO)

    detector = YoloDetector(weights_path="fake.pt", device="cpu")
    image = load_frame(0, scene="helsinki")
    detections = detector.detect(image, zoom_level=0, source_region_xyxy=SOURCE_REGION)

    assert {det.class_name for det in detections} == {"jet_plane", "helicopter"}
    _assert_valid_detections(detections)
    assert detections[0].confidence >= detections[1].confidence


def test_yolo_factory_prefers_yolo_when_backend_is_available(monkeypatch):
    import core.detector as detector_module

    monkeypatch.setattr(detector_module, "_ULTRALYTICS_AVAILABLE", True)
    monkeypatch.setattr(detector_module, "YOLO", _FakeYOLO)

    detector = create_detector(DroneFlybyConfig(DETECTOR_TYPE="yolo_standard"))
    assert isinstance(detector, YoloDetector)


def test_normalize_class_name_preserves_challenge_labels():
    assert _normalize_class_name("ta-ta") == "ta-ta"
    assert _normalize_class_name("ta-ta") in OBJECT_CLASSES
    assert _normalize_class_name("airplane") == "jet_plane"
    assert _normalize_class_name("car") == "small_plane"


class _FakeYOLOWithUnknown(_FakeYOLO):
    names = {0: "ta-ta", 1: "not_a_real_class"}

    def predict(self, image_bgr, conf, device, verbose):
        _ = (image_bgr, conf, device, verbose)
        return [
            _FakeResult(
                boxes=[
                    _FakeBox(0, 0.91, [40.0, 50.0, 120.0, 135.0]),
                    _FakeBox(1, 0.80, [200.0, 180.0, 260.0, 245.0]),
                ]
            )
        ]


def test_yolo_detector_keeps_ta_ta_and_drops_unknown_class(monkeypatch):
    import core.detector as detector_module

    monkeypatch.setattr(detector_module, "_ULTRALYTICS_AVAILABLE", True)
    monkeypatch.setattr(detector_module, "YOLO", _FakeYOLOWithUnknown)

    detector = YoloDetector(weights_path="fake.pt", device="cpu")
    image = load_frame(0, scene="helsinki")
    detections = detector.detect(image, zoom_level=0, source_region_xyxy=SOURCE_REGION)

    assert {det.class_name for det in detections} == {"ta-ta"}
    assert all(det.class_name in OBJECT_CLASSES for det in detections)
    _assert_valid_detections(detections)
