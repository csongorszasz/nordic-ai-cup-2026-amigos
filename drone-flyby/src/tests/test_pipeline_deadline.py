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


def test_stale_mid_session_frame_is_discarded():
    config = DroneFlybyConfig(DEBUG=True, DETECTOR_TYPE="dummy", TRACKER_TYPE="world_map")
    pipeline = build_pipeline(config)
    pipeline.warmup()

    for frame_index in (0, 1, 2):
        pipeline.handle_request(create_synthetic_request("stale_seq", frame_index))
    last_index = pipeline.last_frame_index

    # A late out-of-order frame inside the session is dropped, and the
    # session state is untouched.
    response = pipeline.handle_request(create_synthetic_request("stale_seq", 1))

    validate_response(response)
    assert response.requested_view is None
    assert pipeline.last_frame_index == last_index
    assert pipeline.active_sequence_id == "stale_seq"


def test_frame_zero_restarts_the_session():
    """A rerun that reuses the sequence id must reset, not be discarded.

    The local evaluator always replays sequence id 'local': without this,
    every frame of the second run sat below the previous run's last
    frame_index, was served from stale memory and never moved the camera.
    """
    config = DroneFlybyConfig(DEBUG=True, DETECTOR_TYPE="dummy", TRACKER_TYPE="world_map")
    pipeline = build_pipeline(config)
    pipeline.warmup()

    for frame_index in (0, 1, 2):
        pipeline.handle_request(create_synthetic_request("rerun_seq", frame_index))
    assert pipeline.last_frame_index == 2

    # A restarted attempt: same sequence id, frame_index 0 again.
    response = pipeline.handle_request(create_synthetic_request("rerun_seq", 0))

    validate_response(response)
    # Fully processed (fresh session): the camera policy planned a move and
    # the session cursor restarted at 0. The stale path would have returned
    # requested_view=None and left last_frame_index at 2.
    assert response.requested_view is not None
    assert response.requested_view.resolution_level == 1
    assert pipeline.last_frame_index == 0
    assert pipeline.active_sequence_id == "rerun_seq"


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
