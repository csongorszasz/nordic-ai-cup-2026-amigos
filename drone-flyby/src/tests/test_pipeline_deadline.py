"""Pipeline session-order and deadline-budget behaviour."""

import time
import pytest

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
    # Late observations can still update future memory, but an actual timed-out
    # HTTP response is not credited by the evaluator.
    assert len(response.annotations) >= 1
    assert response.requested_view is None


def test_detector_failure_advances_memory_to_the_requested_frame(monkeypatch):
    pipeline = build_pipeline(DroneFlybyConfig(DETECTOR_TYPE="dummy", TRACKER_TYPE="world_map"))
    monkeypatch.setattr(pipeline.detector, "detect", lambda **kwargs: [_detection()])
    pipeline.handle_request(create_synthetic_request("failure", 0))
    def fail(**kwargs):
        raise RuntimeError("inference unavailable")
    monkeypatch.setattr(pipeline.detector, "detect", fail)
    response = pipeline.handle_request(create_synthetic_request("failure", 2))
    assert response.annotations[0].bbox[1] * 2160 == pytest.approx(216 + 2 * 58)
    assert pipeline.tracker.last_frame_index == 2
    assert pipeline.tracker.tracks[0].existence > 0.7


def test_exhausted_budget_does_not_start_inference(monkeypatch):
    pipeline = build_pipeline(DroneFlybyConfig(DETECTOR_TYPE="dummy", TRACKER_TYPE="world_map"))
    pipeline._detector_estimate_ms = 100
    calls = []
    monkeypatch.setattr(pipeline.detector, "detect", lambda **kwargs: calls.append(kwargs))
    request = create_synthetic_request("budget", 0).model_copy(update={"response_timeout_ms": 50})
    response = pipeline.handle_request(request)
    assert not calls
    assert response.annotations == []


def test_duplicate_nonzero_request_does_not_add_track_hits(monkeypatch):
    pipeline = build_pipeline(DroneFlybyConfig(DETECTOR_TYPE="dummy", TRACKER_TYPE="world_map"))
    monkeypatch.setattr(pipeline.detector, "detect", lambda **kwargs: [_detection()])
    request = create_synthetic_request("duplicate", 1)
    first = pipeline.handle_request(request)
    second = pipeline.handle_request(request)
    assert first == second
    assert pipeline.tracker.tracks[0].hits == 1


def test_lock_wait_is_bounded_by_request_deadline():
    pipeline = build_pipeline(DroneFlybyConfig(DETECTOR_TYPE="dummy"))
    class BusyLock:
        def acquire(self, timeout):
            assert timeout == 0.05
            return False
    pipeline._lock = BusyLock()
    request = create_synthetic_request("locked", 0).model_copy(update={"response_timeout_ms": 50})
    response = pipeline.handle_request(request)
    assert response.annotations == []
    assert pipeline.active_sequence_id is None


def test_camera_failure_retains_weak_fresh_detections(monkeypatch):
    pipeline = build_pipeline(DroneFlybyConfig(DETECTOR_TYPE="dummy", TRACKER_TYPE="world_map"))
    detection = _detection()
    detection.confidence = 0.001
    monkeypatch.setattr(pipeline.detector, "detect", lambda **kwargs: [detection])
    def fail(*args):
        raise RuntimeError("planner unavailable")
    monkeypatch.setattr(pipeline.camera_policy, "decide_next_view", fail)
    response = pipeline.handle_request(create_synthetic_request("camera_failure", 0))
    assert len(response.annotations) == 1
    assert response.requested_view is None


def test_transient_latency_spike_does_not_disable_inference_forever(monkeypatch):
    pipeline = build_pipeline(DroneFlybyConfig(DETECTOR_TYPE="dummy", TRACKER_TYPE="world_map"))
    calls = []
    def detect(**kwargs):
        calls.append(1)
        return [_detection()]
    monkeypatch.setattr(pipeline.detector, "detect", detect)
    pipeline.handle_request(create_synthetic_request("recovery", 0))
    pipeline._detector_estimate_ms = 5000
    for frame in range(1, 10):
        response = pipeline.handle_request(create_synthetic_request("recovery", frame))
        validate_response(response)
    assert len(calls) >= 2
    assert pipeline._detector_estimate_ms < 3333
