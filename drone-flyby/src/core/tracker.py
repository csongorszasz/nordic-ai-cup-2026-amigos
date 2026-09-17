"""Spatial memory and tracking implementations for drone-flyby."""

from collections import defaultdict
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from config import DroneFlybyConfig
from core.interfaces import BaseTracker, DetectionResult, TrackerSummary
from dtos import DroneFlybyPredictionDto
from utils import clip_bbox_to_frame, source_bbox_to_global

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


@dataclass
class TrackedObject:
    """Persistent representation of a ground object in 4K source coordinates."""
    track_id: int
    class_name: str
    bbox_4k: Tuple[float, float, float, float]  # [x1, y1, x2, y2] in 3840x2160 source pixels
    confidence: float
    best_zoom: int
    hits: int = 1
    frames_since_seen: int = 0
    age: int = 1


class WorldMapTracker(BaseTracker):
    """Persistent spatial memory with ego-motion compensation and cross-zoom refinement."""

    def __init__(
        self,
        iou_match_threshold: float = 0.30,
        min_hits_to_confirm: int = 2,
        confidence_decay_rate: float = 0.98,
        out_of_view_max_age: int = 60,
        nms_threshold: float = 0.45,
        default_shift: Tuple[float, float] = (0.0, 58.0),
    ):
        self.iou_match_threshold = iou_match_threshold
        self.min_hits_to_confirm = min_hits_to_confirm
        self.confidence_decay_rate = confidence_decay_rate
        self.out_of_view_max_age = out_of_view_max_age
        self.nms_threshold = nms_threshold
        self.default_shift = default_shift

        self.tracks: Dict[int, TrackedObject] = {}
        self.next_track_id: int = 0
        self.current_sequence_id: Optional[str] = None
        self.last_frame_index: int = -1
        self.prev_l0_gray: Optional[np.ndarray] = None
        self.current_shift: Tuple[float, float] = default_shift

    def reset(self, sequence_id: str) -> None:
        """Reset the world map when transitioning to a new sequence."""
        self.tracks.clear()
        self.next_track_id = 0
        self.current_sequence_id = sequence_id
        self.last_frame_index = -1
        self.prev_l0_gray = None
        self.current_shift = self.default_shift
        logger.info("WorldMapTracker session reset for sequence '%s'", sequence_id)

    def _estimate_ego_motion(self, l0_image_gray: Optional[np.ndarray]) -> Tuple[float, float]:
        """Estimate frame-to-frame pixel shift using phase correlation on L0 views."""
        if l0_image_gray is None:
            return self.current_shift

        if self.prev_l0_gray is not None:
            try:
                prev = np.ascontiguousarray(self.prev_l0_gray, dtype=np.float32)
                curr = np.ascontiguousarray(l0_image_gray, dtype=np.float32)
                phase_correlate: Any = cv2.phaseCorrelate
                shift, response = phase_correlate(prev, curr)
                # Phase correlation returns (dx, dy) in L0 pixels. L0 is downsampled 4x from 4K.
                shift_4k = (shift[0] * 4.0, shift[1] * 4.0)

                # Sanity check: drone flies forward, so objects drift downward (+y).
                # Expected downward drift ~30-90px, small lateral drift -20 to +20px.
                if response >= 0.20 and 20.0 <= shift_4k[1] <= 110.0 and abs(shift_4k[0]) <= 30.0:
                    # Exponential smoothing
                    self.current_shift = (
                        0.7 * shift_4k[0] + 0.3 * self.current_shift[0],
                        0.7 * shift_4k[1] + 0.3 * self.current_shift[1],
                    )
            except Exception as exc:
                logger.warning("Phase correlation failed: %s", exc)

        self.prev_l0_gray = l0_image_gray
        return self.current_shift

    def _shift_tracks_for_motion(self, frame_gap: int) -> None:
        """Advance track coordinates to compensate for drone forward movement."""
        dx = self.current_shift[0] * frame_gap
        dy = self.current_shift[1] * frame_gap

        for track in self.tracks.values():
            x1, y1, x2, y2 = track.bbox_4k
            track.bbox_4k = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            track.age += frame_gap

    def _is_in_view(
        self, bbox_4k: Tuple[float, float, float, float], source_region_xyxy: Tuple[int, int, int, int]
    ) -> bool:
        """Determine if a 4K box is inside the current camera view crop."""
        cx = (bbox_4k[0] + bbox_4k[2]) / 2.0
        cy = (bbox_4k[1] + bbox_4k[3]) / 2.0
        sx1, sy1, sx2, sy2 = source_region_xyxy
        return sx1 <= cx <= sx2 and sy1 <= cy <= sy2

    def update(
        self,
        detections: List[DetectionResult],
        zoom_level: int,
        source_region_xyxy: Tuple[int, int, int, int],
        frame_index: int,
        l0_image_gray: Optional[np.ndarray] = None,
    ) -> List[DroneFlybyPredictionDto]:
        """Update tracks, correlate with new detections, and return full-frame annotations."""
        # 1. Estimate ego-motion
        self._estimate_ego_motion(l0_image_gray)

        # 2. Shift tracks forward for elapsed frame interval
        frame_gap = max(1, frame_index - self.last_frame_index) if self.last_frame_index >= 0 else 1
        self.last_frame_index = frame_index
        self._shift_tracks_for_motion(frame_gap)

        # 3. Data Association (Hungarian Algorithm) in 4K coordinate space
        track_ids = list(self.tracks.keys())
        num_tracks = len(track_ids)
        num_dets = len(detections)

        matched_track_ids = set()
        matched_det_indices = set()

        if num_tracks > 0 and num_dets > 0:
            iou_matrix = np.zeros((num_dets, num_tracks), dtype=np.float32)
            for i, det in enumerate(detections):
                for j, tid in enumerate(track_ids):
                    iou_matrix[i, j] = compute_iou(det.source_pixel_bbox, self.tracks[tid].bbox_4k)

            cost_matrix = 1.0 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_match_threshold:
                    matched_det_indices.add(r)
                    tid = track_ids[c]
                    matched_track_ids.add(tid)

                    track = self.tracks[tid]
                    det = detections[r]

                    track.hits += 1
                    track.frames_since_seen = 0

                    # Prefer higher resolution observations
                    if zoom_level >= track.best_zoom:
                        track.bbox_4k = det.source_pixel_bbox
                        track.class_name = det.class_name
                        track.confidence = max(track.confidence, det.confidence)
                        track.best_zoom = zoom_level
                    else:
                        track.confidence = max(track.confidence, det.confidence * 0.95)

        # 4. Handle unmatched existing tracks
        for tid in track_ids:
            if tid not in matched_track_ids:
                track = self.tracks[tid]
                # Unmatched tracks age regardless of whether they are currently in view.
                # This keeps out-of-view objects alive briefly while still decaying confidence.
                track.frames_since_seen += frame_gap

        # 5. Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_det_indices:
                self.tracks[self.next_track_id] = TrackedObject(
                    track_id=self.next_track_id,
                    class_name=det.class_name,
                    bbox_4k=det.source_pixel_bbox,
                    confidence=det.confidence,
                    best_zoom=zoom_level,
                    hits=1,
                    frames_since_seen=0,
                    age=1,
                )
                self.next_track_id += 1

        # 6. Prune dead or drifted tracks
        dead_tids = []
        for tid, track in self.tracks.items():
            x1, y1, x2, y2 = track.bbox_4k

            # Drifted past the bottom edge (or completely outside 4K frame)
            if y1 > 2160 or y2 < 0 or x1 > 3840 or x2 < 0:
                dead_tids.append(tid)
            # Ephemeral false positive candidate pruning
            elif track.hits < self.min_hits_to_confirm and track.age > 4:
                dead_tids.append(tid)
            # Stale track pruning
            elif (track.age - track.hits) > self.out_of_view_max_age:
                dead_tids.append(tid)

        for tid in dead_tids:
            del self.tracks[tid]

        # 7. Generate output annotations for full 4K frame
        output_predictions: List[DroneFlybyPredictionDto] = []
        for track in self.tracks.values():
            is_confirmed = track.hits >= self.min_hits_to_confirm
            if not is_confirmed:
                continue

            # Decay confidence slowly based on frames since last seen in-view
            effective_conf = track.confidence * (self.confidence_decay_rate ** track.frames_since_seen)
            if effective_conf < 0.05:
                continue

            global_bbox = source_bbox_to_global(track.bbox_4k, 3840, 2160)
            clipped_bbox = clip_bbox_to_frame(global_bbox)
            if clipped_bbox is None:
                continue

            output_predictions.append(
                DroneFlybyPredictionDto(
                    object_id=track.class_name,
                    bbox=clipped_bbox,
                    confidence=float(effective_conf),
                )
            )

        # 8. Apply class-aware NMS to prevent duplicate penalties
        return apply_class_aware_nms(output_predictions, iou_threshold=self.nms_threshold)

    def get_summary(self) -> TrackerSummary:
        """Provide world state summary for camera steering decisions."""
        unscanned_clusters = []
        for track in self.tracks.values():
            # Flag objects that haven't been inspected at high zoom (best_zoom < 2)
            if track.best_zoom < 2 and track.hits >= 1:
                cx = int((track.bbox_4k[0] + track.bbox_4k[2]) / 2.0)
                cy = int((track.bbox_4k[1] + track.bbox_4k[3]) / 2.0)
                unscanned_clusters.append((cx, cy))

        return TrackerSummary(
            num_active_tracks=len(self.tracks),
            unscanned_clusters=unscanned_clusters,
            current_shift_estimate=self.current_shift,
        )


class PassthroughTracker(BaseTracker):
    """Simple non-persistent tracker for testing and baseline comparison."""

    def __init__(self, nms_threshold: float = 0.45):
        self.nms_threshold = nms_threshold
        self.current_sequence_id: Optional[str] = None
        self.last_frame_index: int = -1
        self.total_detections_seen: int = 0

    def reset(self, sequence_id: str) -> None:
        self.current_sequence_id = sequence_id
        self.last_frame_index = -1
        self.total_detections_seen = 0

    def update(
        self,
        detections: List[DetectionResult],
        zoom_level: int,
        source_region_xyxy: Tuple[int, int, int, int],
        frame_index: int,
        l0_image_gray: Optional[np.ndarray] = None,
    ) -> List[DroneFlybyPredictionDto]:
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
        return apply_class_aware_nms(raw_predictions, iou_threshold=self.nms_threshold)

    def get_summary(self) -> TrackerSummary:
        return TrackerSummary(
            num_active_tracks=self.total_detections_seen,
            unscanned_clusters=[],
            current_shift_estimate=(0.0, 58.0),
        )


def create_tracker(config: DroneFlybyConfig) -> BaseTracker:
    """Factory function for instantiating tracker backends."""
    if config.TRACKER_TYPE == "world_map":
        return WorldMapTracker(
            iou_match_threshold=config.IOU_MATCH_THRESHOLD,
            min_hits_to_confirm=config.MIN_HITS_TO_CONFIRM,
            confidence_decay_rate=config.CONFIDENCE_DECAY_RATE,
            out_of_view_max_age=config.OUT_OF_VIEW_MAX_AGE_FRAMES,
            nms_threshold=config.GLOBAL_NMS_IOU_THRESHOLD,
        )
    elif config.TRACKER_TYPE == "passthrough":
        return PassthroughTracker(nms_threshold=config.GLOBAL_NMS_IOU_THRESHOLD)
    else:
        logger.warning("Unknown tracker type '%s', defaulting to WorldMapTracker", config.TRACKER_TYPE)
        return WorldMapTracker()
