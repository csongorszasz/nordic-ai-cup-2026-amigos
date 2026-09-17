"""Tests for the survey-and-zoom camera policy."""

from dataclasses import dataclass

import cv2
import numpy as np

from config import DroneFlybyConfig
from core.camera_policy import SurveyAndZoomPolicy, create_camera_policy
from core.interfaces import TrackerSummary
from core.interfaces import DetectionResult
from core.tracker import WorldMapTracker
from dtos import (
    ALLOWED_RESOLUTION_LEVELS,
    FULL_FRAME_CENTER,
    MAXIMUM_CENTER_DELTA_PIXELS,
    TRANSMITTED_VIEW_SIZE,
    CameraConstraintsDto,
    CameraLevelBoundsDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyViewDto,
    RequestedViewDto,
)
from utils import (
    describe_camera_rejection,
    encode_image,
    center_bounds_for_level,
    load_annotations,
    load_frame,
    source_region_for_view,
)


SOURCE_WIDTH = 3840
SOURCE_HEIGHT = 2160


@dataclass
class SimulatedCamera:
    resolution_level: int = 0
    center_x: int = FULL_FRAME_CENTER[0]
    center_y: int = FULL_FRAME_CENTER[1]

    def apply(self, requested_view: RequestedViewDto) -> None:
        rejection = describe_camera_rejection(
            self.resolution_level,
            (self.center_x, self.center_y),
            requested_view.resolution_level,
            (requested_view.center_x, requested_view.center_y),
        )
        assert rejection is None
        self.resolution_level = requested_view.resolution_level
        self.center_x = requested_view.center_x
        self.center_y = requested_view.center_y


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
    sequence_id: str = "survey_zoom_seq",
) -> DroneFlybyPredictRequestDto:
    image = np.zeros((TRANSMITTED_VIEW_SIZE[1], TRANSMITTED_VIEW_SIZE[0], 3), dtype=np.uint8)
    encoded = encode_image(image)
    source_region_xyxy = source_region_for_view(camera.resolution_level, camera.center_x, camera.center_y)

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


def _real_detections(frame_index: int):
    annotations = load_annotations(frame_index, scene="helsinki")
    return annotations


def _real_tracker_summary(frame_indices=(0, 1)) -> TrackerSummary:
    tracker = WorldMapTracker(min_hits_to_confirm=2, default_shift=(0.0, 58.0))
    tracker.reset("real_policy_summary")

    for frame_index in frame_indices:
        annotations = load_annotations(frame_index, scene="helsinki")
        tracker.update(
            detections=[
                DetectionResult(
                    class_name=annotation["object_id"],
                    bbox_global=(
                        annotation["bbox"][0] / SOURCE_WIDTH,
                        annotation["bbox"][1] / SOURCE_HEIGHT,
                        annotation["bbox"][2] / SOURCE_WIDTH,
                        annotation["bbox"][3] / SOURCE_HEIGHT,
                    ),
                    confidence=0.99,
                    zoom_level=0,
                    source_pixel_bbox=(
                        float(annotation["bbox"][0]),
                        float(annotation["bbox"][1]),
                        float(annotation["bbox"][2]),
                        float(annotation["bbox"][3]),
                    ),
                )
                for annotation in annotations
            ],
            zoom_level=0,
            source_region_xyxy=(0, 0, SOURCE_WIDTH, SOURCE_HEIGHT),
            frame_index=frame_index,
            l0_image_gray=cv2.cvtColor(load_frame(frame_index, scene="helsinki"), cv2.COLOR_BGR2GRAY),
        )

    return tracker.get_summary()


def _simple_summary(*clusters: tuple[int, int]) -> TrackerSummary:
    return TrackerSummary(
        num_active_tracks=len(clusters),
        unscanned_clusters=list(clusters),
        current_shift_estimate=(0.0, 58.0),
    )


def _assert_legal_move(camera: SimulatedCamera, requested_view: RequestedViewDto) -> None:
    rejection = describe_camera_rejection(
        camera.resolution_level,
        (camera.center_x, camera.center_y),
        requested_view.resolution_level,
        (requested_view.center_x, requested_view.center_y),
    )
    assert rejection is None
    camera.apply(requested_view)


def test_factory_returns_survey_zoom_policy():
    policy = create_camera_policy(DroneFlybyConfig(POLICY_TYPE="survey_zoom"))
    assert isinstance(policy, SurveyAndZoomPolicy)


def test_survey_zoom_uses_real_helsinki_summary():
    policy = SurveyAndZoomPolicy(survey_interval_frames=10)
    policy.reset("real_policy_summary")

    camera = SimulatedCamera()
    request = _build_request(camera, frame_index=2, sequence_id="real_policy_summary")
    summary = _real_tracker_summary()

    next_view = policy.decide_next_view(request, summary)

    assert next_view is not None
    assert next_view.resolution_level == 1
    _assert_legal_move(camera, next_view)


def test_survey_zoom_progresses_through_two_clusters():
    policy = SurveyAndZoomPolicy(survey_interval_frames=10)
    policy.reset("cluster_progression")

    camera = SimulatedCamera()
    summaries = {
        0: _simple_summary((900, 600), (2500, 1400)),
        1: _simple_summary((900, 600), (2500, 1400)),
        2: _simple_summary((900, 600), (2500, 1400)),
        3: _simple_summary((2500, 1400)),
        4: _simple_summary((2500, 1400)),
        5: _simple_summary((2500, 1400)),
        6: _simple_summary((2500, 1400)),
        7: _simple_summary((2500, 1400)),
    }

    requested_levels = []
    for frame_index in range(8):
        request = _build_request(camera, frame_index, sequence_id="cluster_progression")
        next_view = policy.decide_next_view(request, summaries[frame_index])
        if next_view is None:
            continue
        requested_levels.append(next_view.resolution_level)
        _assert_legal_move(camera, next_view)

    assert requested_levels == [1, 2, 1, 0, 1, 2, 1, 0]


def test_survey_zoom_periodic_survey_resets_from_l2():
    policy = SurveyAndZoomPolicy(survey_interval_frames=10)
    policy.reset("periodic_survey")

    camera = SimulatedCamera(resolution_level=2, center_x=2500, center_y=1400)
    request = _build_request(camera, frame_index=10, sequence_id="periodic_survey")
    summary = _simple_summary()

    next_view = policy.decide_next_view(request, summary)

    assert next_view is not None
    assert next_view.resolution_level == 1
    _assert_legal_move(camera, next_view)

    request = _build_request(camera, frame_index=11, sequence_id="periodic_survey")
    next_view = policy.decide_next_view(request, summary)

    assert next_view is not None
    assert next_view.resolution_level == 0
    assert (next_view.center_x, next_view.center_y) == FULL_FRAME_CENTER
    _assert_legal_move(camera, next_view)


def test_survey_zoom_trajectory_remains_legal_for_25_frames():
    policy = SurveyAndZoomPolicy(survey_interval_frames=10)
    policy.reset("trajectory_25")

    camera = SimulatedCamera()
    frame_summaries = {}
    for frame_index in range(25):
        if frame_index in {0, 1, 2}:
            frame_summaries[frame_index] = _simple_summary((900, 600), (2500, 1400))
        elif frame_index in {3, 4, 5}:
            frame_summaries[frame_index] = _simple_summary((2500, 1400))
        elif frame_index in {8, 9, 10}:
            frame_summaries[frame_index] = _simple_summary((3000, 700))
        elif frame_index in {15, 16, 17}:
            frame_summaries[frame_index] = _simple_summary((1300, 1700))
        else:
            frame_summaries[frame_index] = _simple_summary()

    requested_views = []
    for frame_index in range(25):
        request = _build_request(camera, frame_index, sequence_id="trajectory_25")
        next_view = policy.decide_next_view(request, frame_summaries[frame_index])
        if next_view is None:
            continue
        requested_views.append(next_view)
        _assert_legal_move(camera, next_view)

    assert requested_views, "policy should emit at least one movement"
    assert any(view.resolution_level == 0 for view in requested_views)
    assert any(view.resolution_level == 1 for view in requested_views)
    assert any(view.resolution_level == 2 for view in requested_views)
