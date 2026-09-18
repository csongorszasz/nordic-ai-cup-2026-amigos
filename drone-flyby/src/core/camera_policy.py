"""Camera policy and active vision implementations for drone-flyby."""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
from config import DroneFlybyConfig
from core.interfaces import BaseCameraPolicy, TrackerSummary
from dtos import (
    DroneFlybyPredictRequestDto,
    RequestedViewDto,
    MAXIMUM_CENTER_DELTA_PIXELS,
)

logger = logging.getLogger(__name__)


@dataclass
class _SurveyZoomState:
    """Sequence-local state for the survey-and-zoom policy."""

    pending_targets: Deque[Tuple[int, int]] = field(default_factory=deque)
    active_target: Optional[Tuple[int, int]] = None
    completed_targets: List[Tuple[int, int]] = field(default_factory=list)
    pending_l0_reset: bool = False
    last_survey_frame_index: int = 0


class CameraConstraintGuard:
    """Validates and enforces camera movement constraints to prevent command rejections."""

    @staticmethod
    def clamp_and_validate(
        request: DroneFlybyPredictRequestDto,
        requested_view: Optional[RequestedViewDto],
    ) -> Optional[RequestedViewDto]:
        if requested_view is None:
            return None

        constraints = request.camera_constraints
        current = request.view
        target_level = requested_view.resolution_level

        # 1. Validate zoom level transition (must be in allowed_resolution_levels)
        if target_level not in constraints.allowed_resolution_levels:
            logger.warning(
                "Requested level %d not in allowed levels %s",
                target_level,
                constraints.allowed_resolution_levels,
            )
            return None

        # 2. Validate center bounds
        bounds = constraints.bounds_for_level(target_level)
        if bounds is None:
            logger.warning("No bounds defined for requested level %d", target_level)
            return None

        # Clamp into bounds and enforce strict Python integer
        clamped_x = int(min(max(requested_view.center_x, bounds.minimum_center_x), bounds.maximum_center_x))
        clamped_y = int(min(max(requested_view.center_y, bounds.minimum_center_y), bounds.maximum_center_y))

        # 3. Validate delta constraint (unless resetting to Level 0 full frame)
        is_l0_reset = target_level == 0 and clamped_x == 1920 and clamped_y == 1080
        if not is_l0_reset and not constraints.full_view_reset_exempt_from_delta:
            max_delta = constraints.maximum_center_delta or MAXIMUM_CENTER_DELTA_PIXELS.get(
                current.resolution_level, 551.0
            )
            dx = clamped_x - current.center_x
            dy = clamped_y - current.center_y
            dist = (dx**2 + dy**2) ** 0.5
            if dist > max_delta:
                # Scale back to max allowable distance
                scale = (max_delta * 0.98) / dist
                clamped_x = int(current.center_x + dx * scale)
                clamped_y = int(current.center_y + dy * scale)
                # Re-clamp
                clamped_x = int(min(max(clamped_x, bounds.minimum_center_x), bounds.maximum_center_x))
                clamped_y = int(min(max(clamped_y, bounds.minimum_center_y), bounds.maximum_center_y))

        return RequestedViewDto(
            resolution_level=int(target_level),
            center_x=int(clamped_x),
            center_y=int(clamped_y),
        )


class HoldCameraPolicy(BaseCameraPolicy):
    """Never move the camera.

    Used as the fixed-L0 / fixed-view baseline: with no camera motion the
    detector is measured on its own, and any active policy must beat this.
    """

    def reset(self, sequence_id: str) -> None:
        logger.info("HoldCameraPolicy reset for sequence '%s'", sequence_id)

    def decide_next_view(
        self,
        request: DroneFlybyPredictRequestDto,
        tracker_summary: TrackerSummary,
    ) -> Optional[RequestedViewDto]:
        return None


class SweepCameraPolicy(BaseCameraPolicy):
    """Horizontal sweep policy at the deepest accessible zoom level."""

    def __init__(self):
        self._sweep_direction: Dict[str, int] = {}

    def reset(self, sequence_id: str) -> None:
        """Reset sweep direction for a new sequence."""
        self._sweep_direction[sequence_id] = 1
        logger.info("Camera policy reset for sequence '%s'", sequence_id)

    def decide_next_view(
        self,
        request: DroneFlybyPredictRequestDto,
        tracker_summary: TrackerSummary,
    ) -> Optional[RequestedViewDto]:
        constraints = request.camera_constraints
        current = request.view
        allowed = [lvl for lvl in constraints.allowed_resolution_levels if lvl > 0]
        if not allowed:
            return None

        # Zoom in one step at a time (e.g. L0 -> L1, L1 -> L2)
        target_level = min(max(allowed), current.resolution_level + 1)
        bounds = constraints.bounds_for_level(target_level)
        if bounds is None:
            return None

        # Starting from Level 0 full view, center the crop
        if current.resolution_level == 0:
            center_x = (bounds.minimum_center_x + bounds.maximum_center_x) // 2
            center_y = (bounds.minimum_center_y + bounds.maximum_center_y) // 2
            raw_view = RequestedViewDto(
                resolution_level=target_level,
                center_x=int(center_x),
                center_y=int(center_y),
            )
            return CameraConstraintGuard.clamp_and_validate(request, raw_view)

        direction = self._sweep_direction.setdefault(request.sequence_id, 1)
        limit = constraints.maximum_center_delta or MAXIMUM_CENTER_DELTA_PIXELS.get(
            current.resolution_level, 551.0
        )
        step = int(limit * 0.9)

        center_x = current.center_x + direction * step
        if center_x > bounds.maximum_center_x or center_x < bounds.minimum_center_x:
            # Turn around at bounds edge
            direction = -direction
            self._sweep_direction[request.sequence_id] = direction
            center_x = current.center_x + direction * step

        center_y = current.center_y

        raw_view = RequestedViewDto(
            resolution_level=target_level,
            center_x=int(center_x),
            center_y=int(center_y),
        )

        return CameraConstraintGuard.clamp_and_validate(request, raw_view)


class SurveyAndZoomPolicy(BaseCameraPolicy):
    """Explore the frame at Level 0, then zoom in on tracker-derived clusters."""

    def __init__(self, survey_interval_frames: int = 10, cluster_merge_radius: float = 128.0):
        self.survey_interval_frames = survey_interval_frames
        self.cluster_merge_radius = cluster_merge_radius
        self._states: Dict[str, _SurveyZoomState] = {}

    def reset(self, sequence_id: str) -> None:
        """Reset the target queue and timing for a new sequence."""
        self._states[sequence_id] = _SurveyZoomState()
        logger.info(
            "SurveyAndZoomPolicy reset for sequence '%s'", sequence_id
        )

    def _state_for(self, sequence_id: str) -> _SurveyZoomState:
        return self._states.setdefault(sequence_id, _SurveyZoomState())

    def _same_cluster(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx * dx + dy * dy) ** 0.5 <= self.cluster_merge_radius

    def _enqueue_new_clusters(
        self,
        state: _SurveyZoomState,
        tracker_summary: TrackerSummary,
    ) -> None:
        sorted_clusters = sorted(tracker_summary.unscanned_clusters, key=lambda item: (item[1], item[0]))
        for cluster in sorted_clusters:
            if state.active_target is not None and self._same_cluster(cluster, state.active_target):
                continue
            if any(self._same_cluster(cluster, queued) for queued in state.pending_targets):
                continue
            if any(self._same_cluster(cluster, done) for done in state.completed_targets):
                continue
            state.pending_targets.append((int(cluster[0]), int(cluster[1])))

    def _build_requested_view(
        self,
        request: DroneFlybyPredictRequestDto,
        resolution_level: int,
        center: Tuple[int, int],
    ) -> Optional[RequestedViewDto]:
        raw_view = RequestedViewDto(
            resolution_level=int(resolution_level),
            center_x=int(center[0]),
            center_y=int(center[1]),
        )
        return CameraConstraintGuard.clamp_and_validate(request, raw_view)

    def decide_next_view(
        self,
        request: DroneFlybyPredictRequestDto,
        tracker_summary: TrackerSummary,
    ) -> Optional[RequestedViewDto]:
        state = self._state_for(request.sequence_id)
        self._enqueue_new_clusters(state, tracker_summary)

        current = request.view
        survey_due = (
            request.frame_index > 0
            and (request.frame_index - state.last_survey_frame_index) >= self.survey_interval_frames
        )

        # Once we have stepped down from L2 to L1, use the free reset to go back to L0.
        if state.pending_l0_reset:
            if current.resolution_level == 1:
                state.pending_l0_reset = False
                state.last_survey_frame_index = request.frame_index
                return self._build_requested_view(request, 0, (1920, 1080))
            if current.resolution_level == 2:
                return self._build_requested_view(request, 1, (current.center_x, current.center_y))
            state.pending_l0_reset = False

        # Periodic surveys are implemented as a legal two-step descent from L2 -> L1 -> L0.
        if survey_due and current.resolution_level == 1:
            state.last_survey_frame_index = request.frame_index
            return self._build_requested_view(request, 0, (1920, 1080))
        if survey_due and current.resolution_level == 2:
            state.pending_l0_reset = True
            state.last_survey_frame_index = request.frame_index
            return self._build_requested_view(request, 1, (current.center_x, current.center_y))

        # Keep zooming toward the active target when one exists.
        if state.active_target is not None:
            if current.resolution_level == 0:
                return self._build_requested_view(request, 1, state.active_target)
            if current.resolution_level == 1:
                return self._build_requested_view(request, 2, state.active_target)

            if current.resolution_level == 2:
                state.completed_targets.append(state.active_target)
                state.active_target = None
                state.pending_l0_reset = True
                return self._build_requested_view(request, 1, (current.center_x, current.center_y))

        # No active target, but there are queued clusters to inspect.
        if state.pending_targets:
            state.active_target = state.pending_targets.popleft()
            if current.resolution_level == 0:
                return self._build_requested_view(request, 1, state.active_target)
            if current.resolution_level == 1:
                return self._build_requested_view(request, 2, state.active_target)

            # From L2 we step down to L1, then use the free reset on the next frame.
            state.pending_l0_reset = True
            return self._build_requested_view(request, 1, (current.center_x, current.center_y))

        # When there is nothing left to inspect, continue periodic Level 0 surveys.
        if survey_due and current.resolution_level == 2:
            state.pending_l0_reset = True
            state.last_survey_frame_index = request.frame_index
            return self._build_requested_view(request, 1, (current.center_x, current.center_y))
        if survey_due and current.resolution_level == 1:
            state.last_survey_frame_index = request.frame_index
            return self._build_requested_view(request, 0, (1920, 1080))

        return None


def create_camera_policy(config: DroneFlybyConfig) -> BaseCameraPolicy:
    """Factory function for camera policies."""
    if config.POLICY_TYPE == "hold":
        return HoldCameraPolicy()
    elif config.POLICY_TYPE == "sweep":
        return SweepCameraPolicy()
    elif config.POLICY_TYPE == "survey_zoom":
        return SurveyAndZoomPolicy(survey_interval_frames=config.SURVEY_INTERVAL_FRAMES)
    elif config.POLICY_TYPE == "belief_map":
        raise NotImplementedError(
            "Camera policy 'belief_map' will be implemented in subsequent phases."
        )
    else:
        logger.warning("Unknown policy type '%s', falling back to SweepCameraPolicy", config.POLICY_TYPE)
        return SweepCameraPolicy()

