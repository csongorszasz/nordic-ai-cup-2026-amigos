"""Tests for the belief-map value-of-information camera policy."""

from dataclasses import dataclass

import numpy as np
import pytest

from config import DroneFlybyConfig
from core.camera_policy import BeliefVoIPolicy, create_camera_policy
from core.interfaces import TrackBelief, TrackerSummary
from dtos import (
    ALLOWED_RESOLUTION_LEVELS,
    FULL_FRAME_CENTER,
    MAXIMUM_CENTER_DELTA_PIXELS,
    TRANSMITTED_VIEW_SIZE,
    CameraConstraintsDto,
    CameraLevelBoundsDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyViewDto,
)
from utils import (
    center_bounds_for_level,
    describe_camera_rejection,
    encode_image,
    source_region_for_view,
)


SOURCE_WIDTH = 3840
SOURCE_HEIGHT = 2160


@dataclass
class SimulatedCamera:
    resolution_level: int = 0
    center_x: int = FULL_FRAME_CENTER[0]
    center_y: int = FULL_FRAME_CENTER[1]

    def apply(self, view) -> None:
        rejection = describe_camera_rejection(
            self.resolution_level,
            (self.center_x, self.center_y),
            view.resolution_level,
            (view.center_x, view.center_y),
        )
        assert rejection is None, rejection
        self.resolution_level = view.resolution_level
        self.center_x = view.center_x
        self.center_y = view.center_y


def _build_constraints(current_level: int) -> CameraConstraintsDto:
    allowed = list(ALLOWED_RESOLUTION_LEVELS[current_level])
    bounds = []
    for level in allowed:
        minimum_x, maximum_x, minimum_y, maximum_y = center_bounds_for_level(level)
        bounds.append(
            CameraLevelBoundsDto(
                resolution_level=level,
                width=TRANSMITTED_VIEW_SIZE[0],
                height=TRANSMITTED_VIEW_SIZE[1],
                minimum_center_x=minimum_x,
                maximum_center_x=maximum_x,
                minimum_center_y=minimum_y,
                maximum_center_y=maximum_y,
            )
        )
    return CameraConstraintsDto(
        maximum_center_delta=MAXIMUM_CENTER_DELTA_PIXELS[current_level],
        allowed_resolution_levels=allowed,
        center_bounds=bounds,
        full_view_reset_exempt_from_delta=True,
    )


def _build_request(
    camera: SimulatedCamera,
    frame_index: int,
    sequence_id: str = "belief_seq",
) -> DroneFlybyPredictRequestDto:
    image = np.zeros((TRANSMITTED_VIEW_SIZE[1], TRANSMITTED_VIEW_SIZE[0], 3), dtype=np.uint8)
    encoded = encode_image(image)
    source_region_xyxy = source_region_for_view(
        camera.resolution_level, camera.center_x, camera.center_y
    )
    return DroneFlybyPredictRequestDto(
        sequence_id=sequence_id,
        frame=frame_index,
        frame_index=frame_index,
        request_id=f"{sequence_id}:{frame_index}",
        frame_interval_ms=333,
        response_timeout_ms=3333,
        original_width=SOURCE_WIDTH,
        original_height=SOURCE_HEIGHT,
        view=DroneFlybyViewDto(
            resolution_level=camera.resolution_level,
            center_x=camera.center_x,
            center_y=camera.center_y,
            view_id=f"{sequence_id}:{frame_index}:{camera.resolution_level}:{camera.center_x}:{camera.center_y}",
            image=encoded,
            image_media_type="image/png",
            width=TRANSMITTED_VIEW_SIZE[0],
            height=TRANSMITTED_VIEW_SIZE[1],
            source_region_xyxy=list(source_region_xyxy),
        ),
        camera_constraints=_build_constraints(camera.resolution_level),
    )


def _summary(*tracks: TrackBelief) -> TrackerSummary:
    return TrackerSummary(
        num_active_tracks=len(tracks),
        unscanned_clusters=[(int(t.center_x), int(t.center_y)) for t in tracks],
        current_shift_estimate=(0.0, 58.0),
        track_beliefs=list(tracks),
    )


def _track(cx, cy, existence=0.9, confidence=0.2, best_zoom=0) -> TrackBelief:
    return TrackBelief(
        class_name="tank",
        center_x=float(cx),
        center_y=float(cy),
        existence=existence,
        confidence=confidence,
        best_zoom=best_zoom,
        position_std=0.0,
    )


def test_factory_returns_belief_policy():
    policy = create_camera_policy(DroneFlybyConfig(POLICY_TYPE="belief_voi"))
    assert isinstance(policy, BeliefVoIPolicy)


def test_enters_l1_from_l0_and_stays_legal():
    policy = BeliefVoIPolicy()
    policy.reset("entry_seq")
    camera = SimulatedCamera()

    request = _build_request(camera, 0, "entry_seq")
    next_view = policy.decide_next_view(request, _summary())

    assert next_view is not None
    assert next_view.resolution_level == 1
    # Frame 0's full view was marked even though the camera is only about to
    # move to L1.
    assert policy.coverage_fraction("entry_seq", 0) > 0.0

    camera.apply(next_view)
    policy.decide_next_view(_build_request(camera, 1, "entry_seq"), _summary())
    assert policy.coverage_fraction("entry_seq", 1) > 0.0


def test_exploration_never_idles_while_ground_is_uncovered():
    policy = BeliefVoIPolicy()
    policy.reset("coverage_seq")
    camera = SimulatedCamera()

    for frame_index in range(40):
        request = _build_request(camera, frame_index, "coverage_seq")
        next_view = policy.decide_next_view(request, _summary())
        if policy.coverage_fraction("coverage_seq", 1) < 1.0:
            assert next_view is not None, "policy idled with unobserved ground"
        if next_view is None:
            break
        camera.apply(next_view)

    assert policy.coverage_fraction("coverage_seq", 1) > 0.5


def test_coverage_is_monotonic():
    policy = BeliefVoIPolicy()
    policy.reset("monotonic_seq")
    camera = SimulatedCamera()
    previous = 0.0

    for frame_index in range(15):
        request = _build_request(camera, frame_index, "monotonic_seq")
        next_view = policy.decide_next_view(request, _summary())
        current = policy.coverage_fraction("monotonic_seq", 1)
        assert current >= previous
        previous = current
        if next_view is not None:
            camera.apply(next_view)


def test_zooms_l2_on_an_in_view_verification_target():
    policy = BeliefVoIPolicy(l2_min_interval=1)
    policy.reset("verify_seq")
    camera = SimulatedCamera(resolution_level=1, center_x=1920, center_y=1080)
    request = _build_request(camera, 10, "verify_seq")

    next_view = policy.decide_next_view(request, _summary(_track(1920, 1080)))

    assert next_view is not None
    assert next_view.resolution_level == 2
    camera.apply(next_view)


def test_steps_down_from_l2_to_l1():
    policy = BeliefVoIPolicy()
    policy.reset("stepdown_seq")
    camera = SimulatedCamera(resolution_level=2, center_x=2000, center_y=900)
    request = _build_request(camera, 11, "stepdown_seq")

    next_view = policy.decide_next_view(request, _summary())

    assert next_view is not None
    assert next_view.resolution_level == 1
    camera.apply(next_view)


def test_l2_is_rate_limited():
    policy = BeliefVoIPolicy(l2_min_interval=5)
    policy.reset("rate_seq")
    camera = SimulatedCamera(resolution_level=1, center_x=1920, center_y=1080)
    tracks = _summary(_track(1920, 1080))

    first = policy.decide_next_view(
        _build_request(camera, 10, "rate_seq"), tracks
    )
    assert first.resolution_level == 2
    camera.apply(first)

    # Step down from L2 is forced.
    step_down = policy.decide_next_view(
        _build_request(camera, 11, "rate_seq"), tracks
    )
    assert step_down.resolution_level == 1
    camera.apply(step_down)

    # Still inside the interval, so the next L1 decision may not zoom again.
    limited = policy.decide_next_view(
        _build_request(camera, 12, "rate_seq"), tracks
    )
    assert limited.resolution_level == 1


def test_tracks_outside_the_view_do_not_trigger_l2():
    policy = BeliefVoIPolicy(l2_min_interval=1)
    policy.reset("outside_seq")
    camera = SimulatedCamera(resolution_level=1, center_x=960, center_y=540)
    request = _build_request(camera, 10, "outside_seq")

    # Track at the far corner is not inside the current L1 view.
    next_view = policy.decide_next_view(request, _summary(_track(3300, 1800)))

    assert next_view is not None
    assert next_view.resolution_level == 1


def test_reset_clears_coverage():
    policy = BeliefVoIPolicy()
    policy.reset("reset_seq")
    camera = SimulatedCamera()
    policy.decide_next_view(_build_request(camera, 0, "reset_seq"), _summary())
    assert policy.coverage_fraction("reset_seq", 0) > 0.0

    policy.reset("reset_seq")
    assert policy.coverage_fraction("reset_seq", 0) == 0.0
    assert policy.coverage_fraction("reset_seq", 1) == 0.0
