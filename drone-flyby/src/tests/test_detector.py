"""Tests for detector backends using the supplied Helsinki scene.

The wire protocol always transmits a 960x540 image, whatever the resolution
level. Tests here therefore feed 960x540 views; feeding a 4K frame and calling
it level 0 passes while exercising a geometry the evaluator never sends.
"""

from pathlib import Path

import pytest
import numpy as np

from config import DroneFlybyConfig
from core.detector import (
    DummyCannyDetector,
    TemplateBankDetector,
    YoloDetector,
    _normalize_class_name,
    create_detector,
    load_calibration,
)
from dtos import OBJECT_CLASSES
from utils import load_frame

WIRE_SIZE = (960, 540)
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


def _wire_sized_view() -> np.ndarray:
    """A 960x540 BGR image, the only size the evaluator ever transmits."""
    image = np.zeros((WIRE_SIZE[1], WIRE_SIZE[0], 3), dtype=np.uint8)
    image[100:150, 200:250] = 255
    return image


def test_dummy_canny_detector_runs_on_wire_sized_view():
    detector = DummyCannyDetector()
    detector.warmup()

    detections = detector.detect(_wire_sized_view(), zoom_level=0, source_region_xyxy=SOURCE_REGION)

    assert detections
    _assert_valid_detections(detections)


def test_template_bank_is_debug_only_and_runs_on_wire_sized_view():
    # The template bank is a debug backend, not a production detector. It is
    # built from 4K crops, so it is expected to find nothing at the 960x540 L0
    # scale; this test only guarantees it does not raise.
    detector = TemplateBankDetector(scene="helsinki", max_templates_per_class=1, max_proposals=5)
    detections = detector.detect(_wire_sized_view(), zoom_level=0, source_region_xyxy=SOURCE_REGION)
    assert isinstance(detections, list)


def test_detector_factory_fails_when_weights_are_missing(tmp_path):
    config = DroneFlybyConfig(
        DETECTOR_TYPE="yolo_standard",
        YOLO_WEIGHTS_PATH=tmp_path / "does_not_exist.pt",
    )
    with pytest.raises(FileNotFoundError):
        create_detector(config)


def test_detector_factory_rejects_unknown_type():
    with pytest.raises(ValueError):
        create_detector(DroneFlybyConfig(DETECTOR_TYPE="not_a_detector"))


def test_load_calibration_reads_known_classes_and_ignores_unknown(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text('{"jet_plane": 0.5, "not_a_class": 0.9}')
    assert load_calibration(path) == {"jet_plane": 0.5}


def test_load_calibration_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_calibration(tmp_path / "missing.json")


def test_load_calibration_rejects_out_of_range(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text('{"jet_plane": 2.0}')
    with pytest.raises(ValueError):
        load_calibration(path)


def test_calibration_threshold_filters_per_class(monkeypatch, tmp_path):
    import core.detector as detector_module

    monkeypatch.setattr(detector_module, "_ULTRALYTICS_AVAILABLE", True)
    monkeypatch.setattr(detector_module, "YOLO", _FakeYOLO)

    # jet_plane is predicted at 0.91; a 0.95 per-class threshold must drop it,
    # while helicopter (0.77, no calibration) survives.
    detector = YoloDetector(
        weights_path=str(_fake_weights(tmp_path)),
        device="cpu",
        calibration={"jet_plane": 0.95},
    )
    detections = detector.detect(_wire_sized_view(), zoom_level=0, source_region_xyxy=SOURCE_REGION)
    assert {det.class_name for det in detections} == {"helicopter"}


def test_debug_detector_ignores_calibration_path(tmp_path):
    config = DroneFlybyConfig(
        DETECTOR_TYPE="dummy",
        CALIBRATION_PATH=tmp_path / "missing.json",
    )
    assert type(create_detector(config)).__name__ == "DummyCannyDetector"


def test_yolo_detector_fails_on_missing_calibration(tmp_path):
    config = DroneFlybyConfig(
        DETECTOR_TYPE="yolo_standard",
        YOLO_WEIGHTS_PATH=tmp_path / "fake.pt",
        CALIBRATION_PATH=tmp_path / "missing.json",
    )
    with pytest.raises(FileNotFoundError):
        create_detector(config)


def test_tensorrt_factory_requires_an_engine_path():
    config = DroneFlybyConfig(DETECTOR_TYPE="tensorrt", TRT_ENGINE_PATH=None)
    with pytest.raises(ValueError):
        create_detector(config)


def test_tensorrt_factory_fails_on_missing_engine(tmp_path):
    config = DroneFlybyConfig(
        DETECTOR_TYPE="tensorrt",
        TRT_ENGINE_PATH=tmp_path / "missing.engine",
    )
    with pytest.raises(FileNotFoundError):
        create_detector(config)


def test_tensorrt_invalid_engine_does_not_fall_back(monkeypatch, tmp_path):
    """An engine that fails to load must stop startup, not change the model."""
    import core.detector as detector_module

    class _BrokenYOLO:
        def __init__(self, weights_path):
            raise RuntimeError(f"invalid engine: {weights_path}")

    engine = tmp_path / "broken.engine"
    engine.write_bytes(b"not-a-serialized-engine")

    monkeypatch.setattr(detector_module, "_ULTRALYTICS_AVAILABLE", True)
    monkeypatch.setattr(detector_module, "YOLO", _BrokenYOLO)

    config = DroneFlybyConfig(DETECTOR_TYPE="tensorrt", TRT_ENGINE_PATH=engine)
    with pytest.raises(RuntimeError):
        create_detector(config)


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
    """A fake Ultralytics model exposing the challenge's 16-class map."""

    names = {index: name for index, name in enumerate(OBJECT_CLASSES)}

    def __init__(self, weights_path):
        self.weights_path = weights_path

    def predict(self, image_bgr, conf, device, verbose):
        _ = (image_bgr, conf, device, verbose)
        # Class 2 = jet_plane, class 1 = helicopter.
        return [
            _FakeResult(
                boxes=[
                    _FakeBox(2, 0.91, [40.0, 50.0, 120.0, 135.0]),
                    _FakeBox(1, 0.77, [200.0, 180.0, 260.0, 245.0]),
                ]
            )
        ]


def _fake_weights(tmp_path: Path) -> Path:
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"not-a-real-checkpoint")
    return weights


def test_yolo_detector_parses_and_normalizes_labels(monkeypatch, tmp_path):
    import core.detector as detector_module

    monkeypatch.setattr(detector_module, "_ULTRALYTICS_AVAILABLE", True)
    monkeypatch.setattr(detector_module, "YOLO", _FakeYOLO)

    detector = YoloDetector(weights_path=str(_fake_weights(tmp_path)), device="cpu")
    detections = detector.detect(_wire_sized_view(), zoom_level=0, source_region_xyxy=SOURCE_REGION)

    assert {det.class_name for det in detections} == {"jet_plane", "helicopter"}
    _assert_valid_detections(detections)
    assert detections[0].confidence >= detections[1].confidence


def test_yolo_detector_rejects_class_map_mismatch(monkeypatch, tmp_path):
    import core.detector as detector_module

    class _FakeYOLOMismatch(_FakeYOLO):
        names = {0: "hangar", 1: "helicopter"}

    monkeypatch.setattr(detector_module, "_ULTRALYTICS_AVAILABLE", True)
    monkeypatch.setattr(detector_module, "YOLO", _FakeYOLOMismatch)

    with pytest.raises(ValueError):
        YoloDetector(weights_path=str(_fake_weights(tmp_path)), device="cpu")


def test_yolo_factory_prefers_yolo_when_backend_is_available(monkeypatch, tmp_path):
    import core.detector as detector_module

    monkeypatch.setattr(detector_module, "_ULTRALYTICS_AVAILABLE", True)
    monkeypatch.setattr(detector_module, "YOLO", _FakeYOLO)

    config = DroneFlybyConfig(
        DETECTOR_TYPE="yolo_standard",
        YOLO_WEIGHTS_PATH=_fake_weights(tmp_path),
    )
    detector = create_detector(config)
    assert isinstance(detector, YoloDetector)


def test_normalize_class_name_does_not_alias_generic_names():
    assert _normalize_class_name("ta-ta") == "ta-ta"
    assert _normalize_class_name("ta-ta") in OBJECT_CLASSES
    # Generic detector names must not be remapped onto challenge classes.
    assert _normalize_class_name("car") == "car"
    assert _normalize_class_name("car") not in OBJECT_CLASSES


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


def test_yolo_detector_drops_unknown_class(monkeypatch, tmp_path):
    import core.detector as detector_module

    monkeypatch.setattr(detector_module, "_ULTRALYTICS_AVAILABLE", True)
    # The fake exposes 2 classes but is only used to exercise name filtering,
    # so allow the class-map check to be bypassed for this test.
    monkeypatch.setattr(detector_module, "YOLO", _FakeYOLOWithUnknown)

    # Build the detector with the real 16-class map but a fake model whose names
    # contain an unknown class; the mismatch check would reject it, so validate
    # filtering by calling the parser through a partially constructed instance.
    detector = object.__new__(YoloDetector)
    detector.model = _FakeYOLOWithUnknown("fake.pt")
    detector.device = "cpu"
    detector.conf_thresholds = {0: 0.10, 1: 0.15, 2: 0.20}
    detector.calibration = {}
    detections = detector.detect(_wire_sized_view(), zoom_level=0, source_region_xyxy=SOURCE_REGION)

    assert {det.class_name for det in detections} == {"ta-ta"}
    assert all(det.class_name in OBJECT_CLASSES for det in detections)
    _assert_valid_detections(detections)
