"""Pipeline session-order and deadline-budget behaviour."""

import time

from config import DroneFlybyConfig
from core import build_pipeline
from core.interfaces import DetectionResult
from tests.test_scaffolding import create_synthetic_request
from utils import validate_response


def _detection():
    return DetectionResult(
        class_name="tank",
        bbox_global=(0.1, 0.1, 0.2, 0.2),
        confidence=0.9,
        zoom_level=0,
        source_pixel_bbox=(384.0, 216.0, 768.0, 432.0),
    )


def test_stale_frame_zero_does_not_reset_state():
    config = DroneFlybyConfig(DEBUG=True, DETECTOR_TYPE="dummy", TRACKER_TYPE="world_map")
    pipeline = build_pipeline(config)
    pipeline.warmup()

    for frame_index in (0, 1, 2):
        pipeline.handle_request(create_synthetic_request("stale0_seq", frame_index))
    last_index = pipeline.last_frame_index

    # A late duplicate of frame 0 must be dropped, not treated as a new session.
    response = pipeline.handle_request(create_synthetic_request("stale0_seq", 0))

    validate_response(response)
    assert response.requested_view is None
    assert pipeline.last_frame_index == last_index
    assert pipeline.active_sequence_id == "stale0_seq"


def test_budget_exhausted_holds_camera_but_keeps_detections(monkeypatch):
    config = DroneFlybyConfig(
        DEBUG=True,
        DETECTOR_TYPE="dummy",
        POLICY_TYPE="active_coverage",
    )
    pipeline = build_pipeline(config)
    pipeline.warmup()

    def slow_detect(**_kwargs):
        time.sleep(0.06)
        return [_detection()]

    monkeypatch.setattr(pipeline.detector, "detect", slow_detect)

    request = create_synthetic_request("deadline_seq", 0).model_copy(
        update={"response_timeout_ms": 50}
    )
    response = pipeline.handle_request(request)

    validate_response(response)
    # Detections still count; only the camera move is withheld.
    assert len(response.annotations) >= 1
    assert response.requested_view is None
