"""AP math used by the per-class calibrator."""

from offline.calibrate import average_precision
import pytest
from types import SimpleNamespace

from config import DroneFlybyConfig
from core.detector import YoloDetector
import offline.calibrate as calibration_module

BOX_A = (0.0, 0.0, 10.0, 10.0)
BOX_B = (100.0, 100.0, 110.0, 110.0)


def test_perfect_prediction_scores_one():
    assert average_precision([(0, BOX_A, 0.9)], {0: [BOX_A]}) == pytest.approx(1.0)


def test_no_predictions_scores_zero():
    assert average_precision([], {0: [BOX_A]}) == 0.0


def test_missing_one_of_two_ground_truths_halves_ap():
    ap = average_precision([(0, BOX_A, 0.9)], {0: [BOX_A, BOX_B]})
    assert ap == pytest.approx(51 / 101)


def test_non_overlapping_prediction_scores_zero():
    assert average_precision([(0, (500.0, 500.0, 510.0, 510.0), 0.9)], {0: [BOX_A]}) == 0.0


def test_calibration_replays_each_candidate_with_the_intended_config(monkeypatch, tmp_path):
    class FakeDetector(YoloDetector):
        def __init__(self):
            self.calibration = {}

        def warmup(self):
            pass

    detector = FakeDetector()
    config = DroneFlybyConfig(YOLO_WEIGHTS_PATH=tmp_path / "chosen.pt", DEVICE="cpu", POLICY_TYPE="hold")
    def create(selected):
        assert selected.YOLO_WEIGHTS_PATH == config.YOLO_WEIGHTS_PATH
        return detector
    observed = []
    def replay(selected, scene, **kwargs):
        assert selected.POLICY_TYPE == "hold"
        assert kwargs["detector"] is detector
        observed.append(dict(detector.calibration))
        return SimpleNamespace(map50=0.8 if detector.calibration.get("tank") == 0.05 else 0.6)
    monkeypatch.setattr(calibration_module, "create_detector", create)
    monkeypatch.setattr(calibration_module, "run_simulation", replay)
    monkeypatch.setattr(calibration_module, "scene_directory", lambda scene: tmp_path)
    monkeypatch.setattr(calibration_module, "frame_numbers", lambda scene: [0])
    monkeypatch.setattr(calibration_module, "load_frame", lambda frame, scene: None)
    monkeypatch.setattr(calibration_module, "load_annotations", lambda frame, scene: [{"object_id": "tank", "bbox": BOX_A}])
    result = calibration_module.calibrate("scene", min_instances=1, thresholds=[0.05, 0.9], config=config)
    assert result == {"tank": 0.05}
    assert observed == [{}, {"tank": 0.05}, {"tank": 0.9}]


def test_calibration_rejects_evaluation_recordings(monkeypatch, tmp_path):
    monkeypatch.setattr(calibration_module, "scene_directory", lambda scene: tmp_path / "recorded_validation_data")
    with pytest.raises(ValueError, match="Evaluation recording"):
        calibration_module.calibrate("scene")
