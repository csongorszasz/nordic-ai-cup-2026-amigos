"""Abstract base classes and shared data contracts for drone-flyby subsystems."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from dtos import (
    DroneFlybyPredictionDto,
    DroneFlybyPredictRequestDto,
    RequestedViewDto,
)


@dataclass(slots=True)
class DetectionResult:
    """Internal detection representation before tracker registration.
    
    Coordinates are normalized to the full 4K source frame [0, 1].
    """
    class_name: str
    bbox_global: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in [0, 1]
    confidence: float
    zoom_level: int
    source_pixel_bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in 3840x2160 pixels


@dataclass(slots=True)
class TrackBelief:
    """One track's belief state, exported for information-gain camera planning.

    Only the fields a planner needs are exposed: where the object is believed
    to be, how sure we are it exists, how sure we are of its class, and how
    deeply it has been observed. Keeping the planner decoupled from the full
    ``TrackedObject`` lets the camera policy stay independent of tracker
    internals.
    """
    class_name: str
    center_x: float
    center_y: float
    existence: float
    confidence: float
    best_zoom: int
    position_std: float


@dataclass(slots=True)
class TrackerSummary:
    """High-level spatial memory summary supplied to camera policy decisions."""
    num_active_tracks: int
    unscanned_clusters: List[Tuple[int, int]]  # Candidate target centers (cx, cy)
    current_shift_estimate: Tuple[float, float]  # (dx, dy) ego-motion shift per frame
    # Richer per-track beliefs, when the tracker can provide them. Policies that
    # only need cluster centres may ignore this; belief-map planners use it to
    # compute value of information. Defaults to empty so simple trackers stay
    # source-compatible.
    track_beliefs: List[TrackBelief] = field(default_factory=list)


class BaseDetector(ABC):
    """Abstract interface for detection implementations."""

    @abstractmethod
    def warmup(self) -> None:
        """Run dummy inferences to pre-compile CUDA/TensorRT kernels."""
        pass

    @abstractmethod
    def detect(
        self,
        image_bgr: np.ndarray,
        zoom_level: int,
        source_region_xyxy: Tuple[int, int, int, int],
    ) -> List[DetectionResult]:
        """Detect objects in the 960x540 crop and return global 4K boxes."""
        pass


class BaseTracker(ABC):
    """Abstract interface for spatial memory and tracking."""

    @abstractmethod
    def reset(self, sequence_id: str) -> None:
        """Clear all active tracks and reset state for a new flight sequence."""
        pass

    @abstractmethod
    def update(
        self,
        detections: List[DetectionResult],
        zoom_level: int,
        source_region_xyxy: Tuple[int, int, int, int],
        frame_index: int,
        l0_image_gray: Optional[np.ndarray] = None,
    ) -> List[DroneFlybyPredictionDto]:
        """Update world map, apply ego-motion compensation, and produce frame annotations."""
        pass

    @abstractmethod
    def predict_only(self) -> List[DroneFlybyPredictionDto]:
        """Return the current belief without incorporating new observations.

        Used on the deadline path: when a detector cannot finish in time, the
        memory still owes the evaluator its best full-frame answer.
        """
        pass

    @abstractmethod
    def get_summary(self) -> TrackerSummary:
        """Export world state summary for camera steering decisions."""
        pass


class BaseCameraPolicy(ABC):
    """Abstract interface for camera steering policies."""

    @abstractmethod
    def reset(self, sequence_id: str) -> None:
        """Reset internal exploration maps/heuristics for a new sequence."""
        pass

    @abstractmethod
    def decide_next_view(
        self,
        request: DroneFlybyPredictRequestDto,
        tracker_summary: TrackerSummary,
    ) -> Optional[RequestedViewDto]:
        """Determine the next camera view adhering to constraints."""
        pass

