"""Unit tests for the modular scaffolding and pipeline orchestrator."""

import numpy as np
import pytest

from config import DroneFlybyConfig
from core import build_pipeline
from core.camera_policy import CameraConstraintGuard
from core.detector import DummyCannyDetector
from core.tracker import PassthroughTracker, compute_iou
from dtos import (
    CameraConstraintsDto,
    CameraLevelBoundsDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyViewDto,
    RequestedViewDto,
)
from utils import encode_image, validate_response


def create_synthetic_request(
    sequence_id: str = "test_seq_01",
    frame_index: int = 0,
    resolution_level: int = 0,
    center_x: int = 1920,
    center_y: int = 1080,
) -> DroneFlybyPredictRequestDto:
    """Helper to build a valid synthetic request for unit testing."""
    # Synthetic 960x540 image with random noise + small rectangle
    synthetic_img = np.zeros((540, 960, 3), dtype=np.uint8)
    synthetic_img[100:150, 200:250] = 255  # Box for Canny detector to catch
    encoded = encode_image(synthetic_img)

    view = DroneFlybyViewDto(
        resolution_level=resolution_level,
        center_x=center_x,
        center_y=center_y,
        view_id=f"{sequence_id}:{frame_index}:{resolution_level}:{center_x}:{center_y}",
        image=encoded,
        image_media_type="image/png",
        width=960,
        height=540,
        source_region_xyxy=[0, 0, 3840, 2160] if resolution_level == 0 else [960, 540, 2880, 1620],
    )

    bounds = [
        CameraLevelBoundsDto(
            resolution_level=0,
            width=3840,
            height=2160,
            minimum_center_x=1920,
            maximum_center_x=1920,
            minimum_center_y=1080,
            maximum_center_y=1080,
        ),
        CameraLevelBoundsDto(
            resolution_level=1,
            width=1920,
            height=1080,
            minimum_center_x=960,
            maximum_center_x=2880,
            minimum_center_y=540,
            maximum_center_y=1620,
        ),
        CameraLevelBoundsDto(
            resolution_level=2,
            width=960,
            height=540,
            minimum_center_x=480,
            maximum_center_x=3360,
            minimum_center_y=270,
            maximum_center_y=1890,
        ),
    ]

    constraints = CameraConstraintsDto(
        maximum_center_delta=2203.0 if resolution_level == 0 else 1102.0,
        allowed_resolution_levels=[0, 1] if resolution_level == 0 else [0, 1, 2],
        center_bounds=bounds,
        full_view_reset_exempt_from_delta=True,
    )

    return DroneFlybyPredictRequestDto(
        sequence_id=sequence_id,
        frame=frame_index,
        frame_index=frame_index,
        request_id=f"{sequence_id}:{frame_index}",
        frame_interval_ms=333,
        response_timeout_ms=3333,
        original_width=3840,
        original_height=2160,
        view=view,
        camera_constraints=constraints,
    )


def test_iou_calculation():
    box1 = (0.1, 0.1, 0.3, 0.3)
    box2 = (0.1, 0.1, 0.3, 0.3)
    assert compute_iou(box1, box2) == pytest.approx(1.0)

    box3 = (0.5, 0.5, 0.7, 0.7)
    assert compute_iou(box1, box3) == pytest.approx(0.0)


def test_camera_constraint_guard():
    req = create_synthetic_request(resolution_level=0)
    
    # Legal move from L0 to L1 center
    target = RequestedViewDto(resolution_level=1, center_x=1920, center_y=1080)
    clamped = CameraConstraintGuard.clamp_and_validate(req, target)
    assert clamped is not None
    assert clamped.resolution_level == 1
    assert clamped.center_x == 1920
    assert clamped.center_y == 1080
    assert isinstance(clamped.center_x, int)

    # Illegal jump L0 -> L2
    illegal_target = RequestedViewDto(resolution_level=2, center_x=1920, center_y=1080)
    assert CameraConstraintGuard.clamp_and_validate(req, illegal_target) is None


def test_pipeline_scaffolding_end_to_end():
    # Plumbing test: use the debug detector so it does not depend on GPU weights.
    config = DroneFlybyConfig(DEBUG=True, DETECTOR_TYPE="dummy")
    pipeline = build_pipeline(config)
    pipeline.warmup()

    # Process frame 0
    req_0 = create_synthetic_request(sequence_id="session_alpha", frame_index=0)
    resp_0 = pipeline.handle_request(req_0)

    # Validate response schema & constraints
    validate_response(resp_0)
    assert resp_0.frame == 0
    assert resp_0.request_id == req_0.request_id
    assert pipeline.active_sequence_id == "session_alpha"

    # Process frame 1
    req_1 = create_synthetic_request(sequence_id="session_alpha", frame_index=1, resolution_level=1)
    resp_1 = pipeline.handle_request(req_1)
    validate_response(resp_1)
    assert resp_1.frame == 1

    # Verify session reset when sequence_id changes
    req_new_seq = create_synthetic_request(sequence_id="session_beta", frame_index=0)
    _ = pipeline.handle_request(req_new_seq)
    assert pipeline.active_sequence_id == "session_beta"

