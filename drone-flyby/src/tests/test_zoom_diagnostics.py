from core.interfaces import DetectionResult
from offline.zoom_diagnostics import fully_contained, match_observations
import offline.zoom_diagnostics as diagnostics
import numpy as np


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


def test_zoom_report_uses_measured_tensor_shape_not_requested_size(tmp_path, monkeypatch):
    class Detector:
        last_input_shape = (1, 3, 544, 960)
        last_input_dtype = "torch.float16"
        def warmup(self):
            pass
        def detect(self, *args):
            return [detection()]
    monkeypatch.setattr(diagnostics, "create_detector", lambda config: Detector())
    monkeypatch.setattr(diagnostics, "frame_numbers", lambda scene: [7])
    monkeypatch.setattr(diagnostics, "load_frame", lambda *args: np.zeros((100, 100, 3), np.uint8))
    monkeypatch.setattr(diagnostics, "load_annotations",
                        lambda *args: [{"object_id": "tank", "bbox": [10, 10, 30, 30]}])
    monkeypatch.setattr(diagnostics, "grid_centers", lambda level: [(50, 50)])
    monkeypatch.setattr(diagnostics, "render_view",
                        lambda *args: (np.zeros((540, 960, 3), np.uint8), (0, 0, 100, 100)))
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"fixture")
    report = diagnostics.diagnose("fixture", weights, tmp_path / "report.json", imgsz=1600, frame_count=1)
    assert report["config"]["INFERENCE_IMAGE_SIZE"] == 1600
    for level in report["levels"].values():
        assert level["actual_inference_shape"] == (1, 3, 544, 960)
        assert level["actual_inference_dtype"] == "torch.float16"
        assert level["per_class"]["tank"]["recall"] == 1
