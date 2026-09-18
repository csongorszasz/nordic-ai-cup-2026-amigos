"""Abstract base classes and shared data contracts for drone-flyby subsystems."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
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
class TrackerSummary:
    """High-level spatial memory summary supplied to camera policy decisions."""
    num_active_tracks: int
    unscanned_clusters: List[Tuple[int, int]]  # Candidate target centers (cx, cy)
    current_shift_estimate: Tuple[float, float]  # (dx, dy) ego-motion shift per frame


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

