"""Camera policy and active vision implementations for drone-flyby."""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple
from config import DroneFlybyConfig
from core.belief import BeliefField
from core.interfaces import BaseCameraPolicy, TrackerSummary
from dtos import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    DroneFlybyPredictRequestDto,
    RequestedViewDto,
    MAXIMUM_CENTER_DELTA_PIXELS,
    SOURCE_REGION_SIZES,
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

        def clamp_into_bounds(point: Tuple[float, float]) -> Tuple[int, int]:
            return (
                int(min(max(point[0], bounds.minimum_center_x), bounds.maximum_center_x)),
                int(min(max(point[1], bounds.minimum_center_y), bounds.maximum_center_y)),
            )

        # Clamp into bounds and enforce strict Python integer
        clamped_x, clamped_y = clamp_into_bounds(
            (requested_view.center_x, requested_view.center_y)
        )

        # 3. Validate delta constraint. Only a level-0 full-frame reset is
        #    exempt; the old check also skipped it whenever
        #    full_view_reset_exempt_from_delta was true, which is always, so a
        #    3304 px L2 move passed a 551 px limit.
        is_l0_reset = target_level == 0 and clamped_x == 1920 and clamped_y == 1080
        if not is_l0_reset:
            max_delta = constraints.maximum_center_delta or MAXIMUM_CENTER_DELTA_PIXELS.get(
                current.resolution_level, 551.0
            )
            current_center = (float(current.center_x), float(current.center_y))

            def distance(point: Tuple[float, float]) -> float:
                return (
                    (point[0] - current_center[0]) ** 2
                    + (point[1] - current_center[1]) ** 2
                ) ** 0.5

            if distance((clamped_x, clamped_y)) > max_delta:
                # Scale the move back toward the current centre, then re-clamp.
                # Re-clamping can push the point back outside the delta disk
                # when the current centre lies outside the target level's
                # bounds (a common L2 -> L1 step near the frame edge), so fall
                # back to the nearest point inside the bounds, which is always
                # within the limit for legal level transitions.
                dx = clamped_x - current_center[0]
                dy = clamped_y - current_center[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > 0:
                    scale = (max_delta * 0.98) / dist
                    scaled = clamp_into_bounds(
                        (current_center[0] + dx * scale, current_center[1] + dy * scale)
                    )
                else:
                    scaled = (clamped_x, clamped_y)

                if distance(scaled) <= max_delta:
                    clamped_x, clamped_y = scaled
                else:
                    entry = clamp_into_bounds(current_center)
                    if distance(entry) > max_delta:
                        # No legal move toward this level exists from here.
                        logger.warning(
                            "No legal %d move from %s; holding",
                            target_level,
                            current_center,
                        )
                        return None
                    clamped_x, clamped_y = entry

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


def _level_cells(level: int) -> List[Tuple[int, int]]:
    """Legal centres covering the frame at one level, in a serpentine order.

    Spacing is half the region, i.e. 50% overlap, and consecutive cells are at
    most one diagonal apart, so each move stays inside the level's delta limit.
    """
    width, height = SOURCE_REGION_SIZES[level]
    min_x, max_x = width // 2, IMAGE_WIDTH - width // 2
    min_y, max_y = height // 2, IMAGE_HEIGHT - height // 2
    xs = list(range(min_x, max_x + 1, max(1, width // 2)))
    if xs[-1] != max_x:
        xs.append(max_x)
    ys = list(range(min_y, max_y + 1, max(1, height // 2)))
    if ys[-1] != max_y:
        ys.append(max_y)

    order: List[Tuple[int, int]] = []
    for row_index, y in enumerate(ys):
        row_xs = xs if row_index % 2 == 0 else list(reversed(xs))
        order.extend((x, y) for x in row_xs)
    return order


@dataclass
class _CoverageState:
    next_cell: int = 0
    covered: Set[int] = field(default_factory=set)
    last_l2_frame: int = -1000


class ActiveCoveragePolicy(BaseCameraPolicy):
    """L1-first active vision with an exploration quota.

    The frame is covered by a deterministic overlapping L1 grid independent of
    any detections, so a frame with no detections can never freeze the camera at
    L0. L2 is spent only on a tracker candidate that lies inside the current L1
    view, rate-limited by ``l2_interval_frames``.
    """

    def __init__(self, l2_interval_frames: int = 4, allow_l2: bool = True):
        self.l2_interval_frames = l2_interval_frames
        self.allow_l2 = allow_l2
        self.cells = _level_cells(1)
        self._states: Dict[str, _CoverageState] = {}

    def reset(self, sequence_id: str) -> None:
        self._states[sequence_id] = _CoverageState()
        logger.info("ActiveCoveragePolicy reset for sequence '%s'", sequence_id)

    def _state_for(self, sequence_id: str) -> _CoverageState:
        return self._states.setdefault(sequence_id, _CoverageState())

    def _next_cell(self, state: _CoverageState) -> Tuple[int, int]:
        index = state.next_cell % len(self.cells)
        state.covered.add(index)
        state.next_cell += 1
        return self.cells[index]

    @staticmethod
    def _region(level: int, center: Tuple[int, int]) -> Tuple[int, int, int, int]:
        width, height = SOURCE_REGION_SIZES[level]
        return (
            center[0] - width // 2,
            center[1] - height // 2,
            center[0] + width // 2,
            center[1] + height // 2,
        )

    def _inside_current_view(self, request: DroneFlybyPredictRequestDto, level: int, center: Tuple[int, int]) -> bool:
        region = self._region(level, (request.view.center_x, request.view.center_y))
        return region[0] <= center[0] <= region[2] and region[1] <= center[1] <= region[3]

    def _clamp(self, request: DroneFlybyPredictRequestDto, level: int, center: Tuple[int, int]) -> Tuple[int, int]:
        bounds = request.camera_constraints.bounds_for_level(level)
        if bounds is None:
            return center
        return (
            int(min(max(center[0], bounds.minimum_center_x), bounds.maximum_center_x)),
            int(min(max(center[1], bounds.minimum_center_y), bounds.maximum_center_y)),
        )

    def _build(self, request, level, center):
        center = self._clamp(request, level, center)
        raw_view = RequestedViewDto(
            resolution_level=int(level),
            center_x=int(center[0]),
            center_y=int(center[1]),
        )
        return CameraConstraintGuard.clamp_and_validate(request, raw_view)

    def _step_towards(self, request: DroneFlybyPredictRequestDto, target: Tuple[int, int]) -> Tuple[int, int]:
        """Move toward a target by at most the current level's delta limit."""
        current = request.view
        limit = request.camera_constraints.maximum_center_delta or MAXIMUM_CENTER_DELTA_PIXELS.get(
            current.resolution_level, 551.0
        )
        dx = target[0] - current.center_x
        dy = target[1] - current.center_y
        distance = (dx * dx + dy * dy) ** 0.5
        if distance <= limit * 0.95 or distance == 0.0:
            return target
        scale = (limit * 0.95) / distance
        return (int(current.center_x + dx * scale), int(current.center_y + dy * scale))

    def decide_next_view(
        self,
        request: DroneFlybyPredictRequestDto,
        tracker_summary: TrackerSummary,
    ) -> Optional[RequestedViewDto]:
        state = self._state_for(request.sequence_id)
        current = request.view
        candidate = tuple(tracker_summary.unscanned_clusters[0]) if tracker_summary.unscanned_clusters else None

        # Leaving L2 always steps down to L1 first.
        if current.resolution_level == 2:
            return self._build(request, 1, candidate or self._next_cell(state))

        # L1 is the search workhorse: move to a verification candidate, or to
        # the next exploration cell when there is nothing to verify.
        if current.resolution_level == 1:
            can_zoom = (
                self.allow_l2
                and candidate is not None
                and self._inside_current_view(request, 1, candidate)
            )
            if can_zoom and (request.frame_index - state.last_l2_frame) >= self.l2_interval_frames:
                state.last_l2_frame = request.frame_index
                return self._build(request, 2, candidate)
            target = candidate if candidate is not None else self._next_cell(state)
            return self._build(request, 1, self._step_towards(request, target))

        # From L0, enter L1 toward the first candidate or coverage cell.
        target = candidate if candidate is not None else self._next_cell(state)
        return self._build(request, 1, target)


@dataclass
class _BeliefPolicyState:
    """Sequence-local state for the belief-map value-of-information policy."""

    field: BeliefField
    last_l2_frame: int = -1000


class BeliefVoIPolicy(BaseCameraPolicy):
    """Belief-map camera planner that maximises expected information gain.

    Level 1 is the search workhorse. Unobserved L1 cells always compete as
    exploration candidates, so the camera can never freeze on a frame with no
    detections; confirmed tracks with unresolved class or position compete as
    verification candidates. Level 2 is spent only on a verification candidate
    that lies inside the current L1 view, is due by the rate limit, and is
    worth more than the best available L1 move.
    """

    def __init__(
        self,
        cell_size: int = 120,
        explore_weight: float = 1.0,
        verify_weight: float = 1.2,
        travel_weight: float = 0.35,
        l2_min_interval: int = 4,
        min_track_existence: float = 0.20,
        zoom_bias: float = 1.0,
    ):
        self.cell_size = cell_size
        self.explore_weight = explore_weight
        self.verify_weight = verify_weight
        self.travel_weight = travel_weight
        self.l2_min_interval = max(1, int(l2_min_interval))
        self.min_track_existence = min_track_existence
        self.zoom_bias = zoom_bias
        self._states: Dict[str, _BeliefPolicyState] = {}

    # -- state ---------------------------------------------------------- #

    def _make_field(self) -> BeliefField:
        return BeliefField(
            cell_size=self.cell_size,
            explore_weight=self.explore_weight,
            verify_weight=self.verify_weight,
            travel_weight=self.travel_weight,
            min_track_existence=self.min_track_existence,
        )

    def reset(self, sequence_id: str) -> None:
        self._states[sequence_id] = _BeliefPolicyState(field=self._make_field())
        logger.info("BeliefVoIPolicy reset for sequence '%s'", sequence_id)

    def _state_for(self, sequence_id: str) -> _BeliefPolicyState:
        return self._states.setdefault(
            sequence_id, _BeliefPolicyState(field=self._make_field())
        )

    def coverage_fraction(self, sequence_id: str, level: int) -> float:
        """Exposed for tests and offline analysis."""
        state = self._states.get(sequence_id)
        return 0.0 if state is None else state.field.coverage_fraction(level)

    # -- planning helpers ---------------------------------------------- #

    @staticmethod
    def _bounds(
        request: DroneFlybyPredictRequestDto, level: int
    ) -> Optional[Tuple[int, int, int, int]]:
        bounds = request.camera_constraints.bounds_for_level(level)
        if bounds is None:
            return None
        return (
            bounds.minimum_center_x,
            bounds.maximum_center_x,
            bounds.minimum_center_y,
            bounds.maximum_center_y,
        )

    @staticmethod
    def _inside_view(
        current_level: int,
        current_center: Tuple[int, int],
        target_center: Tuple[int, int],
    ) -> bool:
        width, height = SOURCE_REGION_SIZES[current_level]
        half_width, half_height = width // 2, height // 2
        return (
            current_center[0] - half_width
            <= target_center[0]
            <= current_center[0] + half_width
            and current_center[1] - half_height
            <= target_center[1]
            <= current_center[1] + half_height
        )

    def _build(
        self,
        request: DroneFlybyPredictRequestDto,
        level: int,
        center: Tuple[int, int],
    ) -> Optional[RequestedViewDto]:
        raw_view = RequestedViewDto(
            resolution_level=int(level),
            center_x=int(center[0]),
            center_y=int(center[1]),
        )
        return CameraConstraintGuard.clamp_and_validate(request, raw_view)

    # -- decision ------------------------------------------------------- #

    def decide_next_view(
        self,
        request: DroneFlybyPredictRequestDto,
        tracker_summary: TrackerSummary,
    ) -> Optional[RequestedViewDto]:
        state = self._state_for(request.sequence_id)
        belief_field = state.field
        current = request.view
        constraints = request.camera_constraints
        current_level = int(current.resolution_level)
        current_center = (int(current.center_x), int(current.center_y))
        tracks = list(tracker_summary.track_beliefs)
        max_delta = constraints.maximum_center_delta or MAXIMUM_CENTER_DELTA_PIXELS.get(
            current_level, 551.0
        )

        # The current view is now knowledge, whatever we decide next.
        belief_field.mark_observed(
            current.source_region_xyxy, current_level, request.frame_index
        )

        # Level 2 cannot stay: the next move is always the step down to L1.
        if current_level == 2:
            return self._plan_l1_move(
                request, belief_field, tracks, current_center, max_delta
            )

        # From the full view, enter L1 toward the best candidate.
        if current_level == 0:
            l1_bounds = self._bounds(request, 1)
            if l1_bounds is None:
                return None
            candidate = belief_field.best_candidate(
                1, current_center, tracks, max_delta, l1_bounds
            )
            if candidate is None:
                return None
            return self._build(request, 1, (candidate.center_x, candidate.center_y))

        # At L1, consider spending a frame on an in-view verification target.
        if (
            2 in constraints.allowed_resolution_levels
            and (request.frame_index - state.last_l2_frame) >= self.l2_min_interval
        ):
            in_view_tracks = [
                track
                for track in tracks
                if self._inside_view(
                    current_level,
                    current_center,
                    (int(round(track.center_x)), int(round(track.center_y))),
                )
            ]
            l2_bounds = self._bounds(request, 2)
            verify = belief_field.best_candidate(
                2,
                current_center,
                in_view_tracks,
                max_delta,
                l2_bounds,
                include_exploration=False,
            )
            if verify is not None:
                l1_bounds = self._bounds(request, 1)
                l1_move = belief_field.best_candidate(
                    1, current_center, tracks, max_delta, l1_bounds
                )
                l1_value = l1_move.value if l1_move is not None else 0.0
                if verify.value >= self.zoom_bias * l1_value:
                    state.last_l2_frame = request.frame_index
                    return self._build(request, 2, (verify.center_x, verify.center_y))

        return self._plan_l1_move(
            request, belief_field, tracks, current_center, max_delta
        )

    def _plan_l1_move(
        self,
        request: DroneFlybyPredictRequestDto,
        belief_field: BeliefField,
        tracks: List,
        current_center: Tuple[int, int],
        max_delta: float,
    ) -> Optional[RequestedViewDto]:
        l1_bounds = self._bounds(request, 1)
        if l1_bounds is None:
            return None
        candidate = belief_field.best_candidate(
            1, current_center, tracks, max_delta, l1_bounds
        )
        if candidate is None:
            return None
        return self._build(request, 1, (candidate.center_x, candidate.center_y))


def create_camera_policy(config: DroneFlybyConfig) -> BaseCameraPolicy:
    """Factory function for camera policies."""
    if config.POLICY_TYPE == "hold":
        return HoldCameraPolicy()
    elif config.POLICY_TYPE == "sweep":
        return SweepCameraPolicy()
    elif config.POLICY_TYPE == "survey_zoom":
        return SurveyAndZoomPolicy(survey_interval_frames=config.SURVEY_INTERVAL_FRAMES)
    elif config.POLICY_TYPE == "deterministic_l1":
        # Coverage-only baseline: legal overlapping L1 sweep, never zooms.
        return ActiveCoveragePolicy(l2_interval_frames=1, allow_l2=False)
    elif config.POLICY_TYPE == "belief_voi":
        return BeliefVoIPolicy(
            cell_size=config.BELIEF_CELL_SIZE,
            explore_weight=config.BELIEF_EXPLORE_WEIGHT,
            verify_weight=config.BELIEF_VERIFY_WEIGHT,
            travel_weight=config.BELIEF_TRAVEL_WEIGHT,
            l2_min_interval=config.BELIEF_L2_MIN_INTERVAL,
            min_track_existence=config.MIN_EXISTENCE,
        )
    elif config.POLICY_TYPE in ("active_coverage", "belief_map"):
        return ActiveCoveragePolicy(l2_interval_frames=max(1, config.SURVEY_INTERVAL_FRAMES))
    else:
        logger.warning("Unknown policy type '%s', falling back to HoldCameraPolicy", config.POLICY_TYPE)
        return HoldCameraPolicy()

