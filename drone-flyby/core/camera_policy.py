"""Camera policy and active vision implementations for drone-flyby."""

import logging
from typing import Dict, Optional
from config import DroneFlybyConfig
from core.interfaces import BaseCameraPolicy, TrackerSummary
from dtos import (
    DroneFlybyPredictRequestDto,
    RequestedViewDto,
    MAXIMUM_CENTER_DELTA_PIXELS,
)

logger = logging.getLogger(__name__)


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


def create_camera_policy(config: DroneFlybyConfig) -> BaseCameraPolicy:
    """Factory function for camera policies."""
    if config.POLICY_TYPE == "sweep":
        return SweepCameraPolicy()
    elif config.POLICY_TYPE in ("survey_zoom", "belief_map"):
        raise NotImplementedError(
            f"Camera policy '{config.POLICY_TYPE}' will be implemented in subsequent phases."
        )
    else:
        logger.warning("Unknown policy type '%s', falling back to SweepCameraPolicy", config.POLICY_TYPE)
        return SweepCameraPolicy()

