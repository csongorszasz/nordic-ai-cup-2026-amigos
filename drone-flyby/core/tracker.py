"""Spatial memory and tracking implementations for drone-flyby."""

from collections import defaultdict
import logging
from typing import List, Optional, Tuple
import numpy as np

from config import DroneFlybyConfig
from core.interfaces import BaseTracker, DetectionResult, TrackerSummary
from dtos import DroneFlybyPredictionDto

logger = logging.getLogger(__name__)


def compute_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Compute Intersection-over-Union between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0.0 else 0.0


def apply_class_aware_nms(
    predictions: List[DroneFlybyPredictionDto],
    iou_threshold: float = 0.45,
    max_total: int = 500,
) -> List[DroneFlybyPredictionDto]:
    """Class-aware Non-Maximum Suppression to prevent duplicate penalties."""
    by_class = defaultdict(list)
    for pred in predictions:
        by_class[pred.object_id].append(pred)

    selected: List[DroneFlybyPredictionDto] = []
    for object_id, class_preds in by_class.items():
        # Sort descending by confidence
        class_preds.sort(key=lambda p: p.confidence, reverse=True)
        kept_boxes: List[Tuple[float, float, float, float]] = []

        for p in class_preds:
            if not any(compute_iou(p.bbox, kb) > iou_threshold for kb in kept_boxes):
                selected.append(p)
                kept_boxes.append(p.bbox)

    # Sort final predictions by confidence descending and cap at max_total
    selected.sort(key=lambda p: p.confidence, reverse=True)
    return selected[:max_total]


class PassthroughTracker(BaseTracker):
    """Scaffold tracker that validates and formats current-frame detections.
    
    Serves as the baseline implementation before enabling full persistent WorldMap.
    """

    def __init__(self, nms_threshold: float = 0.45):
        self.nms_threshold = nms_threshold
        self.current_sequence_id: Optional[str] = None
        self.last_frame_index: int = -1
        self.total_detections_seen: int = 0

    def reset(self, sequence_id: str) -> None:
        """Reset internal session state when a new flight starts."""
        self.current_sequence_id = sequence_id
        self.last_frame_index = -1
        self.total_detections_seen = 0
        logger.info("Tracker session reset for sequence '%s'", sequence_id)

    def update(
        self,
        detections: List[DetectionResult],
        zoom_level: int,
        source_region_xyxy: Tuple[int, int, int, int],
        frame_index: int,
        l0_image_gray: Optional[np.ndarray] = None,
    ) -> List[DroneFlybyPredictionDto]:
        """Convert raw detections to prediction DTOs and apply local NMS."""
        self.last_frame_index = frame_index
        self.total_detections_seen += len(detections)

        raw_predictions = [
            DroneFlybyPredictionDto(
                object_id=d.class_name,
                bbox=d.bbox_global,
                confidence=d.confidence,
            )
            for d in detections
        ]

        # Apply strict NMS to protect against duplicate penalties
        return apply_class_aware_nms(raw_predictions, iou_threshold=self.nms_threshold)

    def get_summary(self) -> TrackerSummary:
        """Provide status summary for camera policy decision making."""
        return TrackerSummary(
            num_active_tracks=self.total_detections_seen,
            unscanned_clusters=[],
            current_shift_estimate=(0.0, 58.0),  # Default geometric downward shift
        )


def create_tracker(config: DroneFlybyConfig) -> BaseTracker:
    """Factory function for instantiating tracker backends."""
    if config.TRACKER_TYPE == "passthrough":
        return PassthroughTracker(nms_threshold=config.GLOBAL_NMS_IOU_THRESHOLD)
    elif config.TRACKER_TYPE == "world_map":
        raise NotImplementedError("Persistent WorldMap tracker will be implemented in subsequent phases.")
    else:
        logger.warning("Unknown tracker type '%s', falling back to PassthroughTracker", config.TRACKER_TYPE)
        return PassthroughTracker(nms_threshold=config.GLOBAL_NMS_IOU_THRESHOLD)

