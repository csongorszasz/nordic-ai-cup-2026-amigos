"""Spatial memory and tracking implementations for drone-flyby.

The world map keeps every object in cumulative source-frame coordinates and
treats the camera as a moving sensor. Three things it deliberately does that the
old image-relative tracker did not:

* motion is predicted with a frame-gap-normalised shift (phase correlation
  measures displacement between two L0 frames, which is not a per-frame rate
  when frames are skipped);
* association is class-aware and combines IoU with a motion-gated centre
  distance, so a fast small object is still matched after it outruns its own
  box;
* a track that is inside the current crop but not detected receives strong
  negative evidence, while an out-of-view track is only gently decayed.
"""

from collections import defaultdict
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from config import DroneFlybyConfig
from core.interfaces import BaseTracker, DetectionResult, TrackerSummary
from dtos import DroneFlybyPredictionDto, IMAGE_HEIGHT, IMAGE_WIDTH
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


def _box_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _box_diagonal(box: Tuple[float, float, float, float]) -> float:
    width = box[2] - box[0]
    height = box[3] - box[1]
    return (width * width + height * height) ** 0.5


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
    # Per-class evidence, used to keep the best class instead of letting a
    # stale high-confidence observation overwrite a better one.
    class_scores: Dict[str, float] = field(default_factory=dict)
    # Probability the object still exists in the scene.
    existence: float = 0.5
    # Approximate positional uncertainty in source pixels (grows while unseen).
    position_std: float = 0.0
    last_seen_frame: int = -1
    last_zoom: int = -1
    velocity: Tuple[float, float] = (0.0, 0.0)


class WorldMapTracker(BaseTracker):
    """Persistent spatial memory with ego-motion prediction and class-aware association."""

    def __init__(
        self,
        iou_match_threshold: float = 0.30,
        min_hits_to_confirm: int = 2,
        confidence_decay_rate: float = 0.98,
        out_of_view_max_age: int = 60,
        nms_threshold: float = 0.45,
        default_shift: Tuple[float, float] = (0.0, 58.0),
        # A track is confirmed after min_hits, or immediately on one detection
        # this confident. 0.85 dropped classes that are only ever seen for one
        # or two frames (e.g. mine_roller); 0.30 keeps them and still filtered
        # the local harness for false positives.
        single_hit_confirm_confidence: float = 0.30,
        min_existence: float = 0.20,
        in_view_miss_decay: float = 0.55,
        out_of_view_miss_decay: float = 0.97,
    ):
        self.iou_match_threshold = iou_match_threshold
        self.min_hits_to_confirm = min_hits_to_confirm
        self.confidence_decay_rate = confidence_decay_rate
        self.out_of_view_max_age = out_of_view_max_age
        self.nms_threshold = nms_threshold
        self.default_shift = default_shift
        self.single_hit_confirm_confidence = single_hit_confirm_confidence
        self.min_existence = min_existence
        self.in_view_miss_decay = in_view_miss_decay
        self.out_of_view_miss_decay = out_of_view_miss_decay

        self.tracks: Dict[int, TrackedObject] = {}
        self.next_track_id: int = 0
        self.current_sequence_id: Optional[str] = None
        self.last_frame_index: int = -1
        self.last_zoom: int = 0
        self.prev_l0_gray: Optional[np.ndarray] = None
        self.prev_l0_frame_index: int = -1
        self.current_shift: Tuple[float, float] = default_shift

    def reset(self, sequence_id: str) -> None:
        """Reset the world map when transitioning to a new sequence."""
        self.tracks.clear()
        self.next_track_id = 0
        self.current_sequence_id = sequence_id
        self.last_frame_index = -1
        self.last_zoom = 0
        self.prev_l0_gray = None
        self.prev_l0_frame_index = -1
        self.current_shift = self.default_shift
        logger.info("WorldMapTracker session reset for sequence '%s'", sequence_id)

    # ------------------------------------------------------------------ #
    # Ego-motion
    # ------------------------------------------------------------------ #

    def _estimate_ego_motion(
        self, l0_image_gray: Optional[np.ndarray], frame_index: int
    ) -> Tuple[float, float]:
        """Estimate a per-frame 4K shift from phase correlation between L0 views.

        Phase correlation reports the total displacement between the two L0
        frames it is given. Skipped frames make that a multi-frame displacement,
        so it is divided by the frame gap before use, otherwise a four-frame gap
        looks like a 240 px/frame jump and is rejected by the sanity gate.
        """
        if l0_image_gray is None:
            return self.current_shift

        gray = np.ascontiguousarray(l0_image_gray, dtype=np.float32)

        if self.prev_l0_gray is not None and self.prev_l0_frame_index >= 0:
            gap = max(1, frame_index - self.prev_l0_frame_index)
            try:
                shift, response = cv2.phaseCorrelate(self.prev_l0_gray, gray)
                # The transmitted L0 image is 960 px wide; tests may pass the 4K
                # frame directly, so derive the downsample factor from the input.
                scale = IMAGE_WIDTH / float(gray.shape[1])
                per_frame = (shift[0] * scale / gap, shift[1] * scale / gap)

                # Sanity check: the drone flies forward, so the scene drifts down.
                if response >= 0.15 and 10.0 <= per_frame[1] <= 130.0 and abs(per_frame[0]) <= 40.0:
                    self.current_shift = (
                        0.7 * per_frame[0] + 0.3 * self.current_shift[0],
                        0.7 * per_frame[1] + 0.3 * self.current_shift[1],
                    )
            except Exception as exc:
                logger.warning("Phase correlation failed: %s", exc)

        self.prev_l0_gray = gray
        self.prev_l0_frame_index = frame_index
        return self.current_shift

    def _predict_tracks(self, frame_gap: int) -> None:
        """Advance track coordinates to compensate for drone forward movement."""
        dx = self.current_shift[0] * frame_gap
        dy = self.current_shift[1] * frame_gap

        for track in self.tracks.values():
            x1, y1, x2, y2 = track.bbox_4k
            track.bbox_4k = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            track.age += frame_gap
            # Uncertainty grows while the object is not observed.
            track.position_std += 0.5 * frame_gap * (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def _is_in_view(
        bbox_4k: Tuple[float, float, float, float], source_region_xyxy: Tuple[int, int, int, int]
    ) -> bool:
        """Determine if a 4K box is inside the current camera view crop."""
        cx, cy = _box_center(bbox_4k)
        sx1, sy1, sx2, sy2 = source_region_xyxy
        return sx1 <= cx <= sx2 and sy1 <= cy <= sy2

    # ------------------------------------------------------------------ #
    # Association
    # ------------------------------------------------------------------ #

    def _associate(
        self,
        detections: List[DetectionResult],
        track_ids: List[int],
    ) -> Tuple[List[Tuple[int, int]], set, set]:
        """Match detections to tracks, class-aware and motion-gated.

        Returns matched (detection_index, track_id) pairs plus the matched index
        sets. IoU alone drops fast small objects; a same-class centre-distance
        fallback keeps them associated without letting different classes collide.
        """
        num_dets = len(detections)
        num_tracks = len(track_ids)
        cost = np.ones((num_dets, num_tracks), dtype=np.float32)
        gate = np.zeros((num_dets, num_tracks), dtype=bool)

        for i, det in enumerate(detections):
            det_box = det.source_pixel_bbox
            for j, tid in enumerate(track_ids):
                track = self.tracks[tid]
                iou = compute_iou(det_box, track.bbox_4k)
                det_cx, det_cy = _box_center(det_box)
                tr_cx, tr_cy = _box_center(track.bbox_4k)
                distance = ((det_cx - tr_cx) ** 2 + (det_cy - tr_cy) ** 2) ** 0.5
                diagonal = max(_box_diagonal(track.bbox_4k), 1.0)
                class_match = det.class_name == track.class_name

                accepted = iou >= self.iou_match_threshold or (
                    class_match and distance <= max(diagonal, 150.0)
                )
                if not accepted:
                    continue

                gate[i, j] = True
                # Prefer overlap, then proximity.
                centre_bonus = 0.25 * max(0.0, 1.0 - distance / max(diagonal, 1.0))
                cost[i, j] = 1.0 - iou - centre_bonus

        matches: List[Tuple[int, int]] = []
        matched_dets: set = set()
        matched_tracks: set = set()
        if num_dets and num_tracks:
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if gate[r, c]:
                    matches.append((int(r), track_ids[c]))
                    matched_dets.add(int(r))
                    matched_tracks.add(track_ids[c])
        return matches, matched_dets, matched_tracks

    def _apply_detection(self, track: TrackedObject, det: DetectionResult, zoom_level: int) -> None:
        """Fuse one detection into a track."""
        track.hits += 1
        track.frames_since_seen = 0
        track.last_seen_frame = self.last_frame_index

        # Class posterior: accumulate evidence, decay competing classes.
        for class_name in list(track.class_scores):
            track.class_scores[class_name] *= 0.9
        track.class_scores[det.class_name] = track.class_scores.get(det.class_name, 0.0) + det.confidence
        track.class_name = max(track.class_scores, key=track.class_scores.get)

        track.existence = min(1.0, track.existence + 0.5 + 0.5 * det.confidence)

        if zoom_level >= track.best_zoom:
            previous_center = _box_center(track.bbox_4k)
            track.bbox_4k = det.source_pixel_bbox
            track.confidence = max(track.confidence, det.confidence)
            track.best_zoom = zoom_level
            track.last_zoom = zoom_level
            new_center = _box_center(track.bbox_4k)
            track.velocity = (new_center[0] - previous_center[0], new_center[1] - previous_center[1])
            track.position_std = max(0.0, track.position_std * 0.5)
        else:
            track.confidence = max(track.confidence, det.confidence * 0.95)

    def update(
        self,
        detections: List[DetectionResult],
        zoom_level: int,
        source_region_xyxy: Tuple[int, int, int, int],
        frame_index: int,
        l0_image_gray: Optional[np.ndarray] = None,
    ) -> List[DroneFlybyPredictionDto]:
        """Update tracks, correlate with new detections, and return full-frame annotations."""
        self._estimate_ego_motion(l0_image_gray, frame_index)

        frame_gap = max(1, frame_index - self.last_frame_index) if self.last_frame_index >= 0 else 1
        self.last_frame_index = frame_index
        self.last_zoom = zoom_level
        self._predict_tracks(frame_gap)

        track_ids = list(self.tracks.keys())
        matches, matched_dets, matched_tracks = self._associate(detections, track_ids)

        for det_index, track_id in matches:
            self._apply_detection(self.tracks[track_id], detections[det_index], zoom_level)

        # Negative evidence: an in-view miss is informative, an out-of-view one
        # is not. Collapsing them is why the old map kept ghost tracks alive.
        for track_id in track_ids:
            if track_id in matched_tracks:
                continue
            track = self.tracks[track_id]
            track.frames_since_seen += frame_gap
            if self._is_in_view(track.bbox_4k, source_region_xyxy):
                track.existence *= self.in_view_miss_decay
                track.confidence *= self.confidence_decay_rate ** frame_gap
            else:
                track.existence *= self.out_of_view_miss_decay

        # New tracks for unmatched detections.
        for det_index, det in enumerate(detections):
            if det_index in matched_dets:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = TrackedObject(
                track_id=track_id,
                class_name=det.class_name,
                bbox_4k=det.source_pixel_bbox,
                confidence=det.confidence,
                best_zoom=zoom_level,
                hits=1,
                frames_since_seen=0,
                age=1,
                class_scores={det.class_name: det.confidence},
                existence=min(0.9, 0.4 + 0.5 * det.confidence),
                position_std=0.0,
                last_seen_frame=frame_index,
                last_zoom=zoom_level,
            )

        self._prune()
        return self._emit()

    def predict_only(self) -> List[DroneFlybyPredictionDto]:
        """Return the current full-frame belief without new observations."""
        return self._emit()

    def _prune(self) -> None:
        dead_tids: List[int] = []
        for tid, track in self.tracks.items():
            x1, y1, x2, y2 = track.bbox_4k
            if y1 > IMAGE_HEIGHT or y2 < 0 or x1 > IMAGE_WIDTH or x2 < 0:
                dead_tids.append(tid)
            elif track.existence < self.min_existence * 0.5:
                dead_tids.append(tid)
            elif track.hits < self.min_hits_to_confirm and track.age > 4:
                dead_tids.append(tid)
            elif (track.age - track.hits) > self.out_of_view_max_age:
                dead_tids.append(tid)

        for tid in dead_tids:
            del self.tracks[tid]

    def _is_confirmed(self, track: TrackedObject) -> bool:
        return (
            track.hits >= self.min_hits_to_confirm
            or track.confidence >= self.single_hit_confirm_confidence
        )

    def _emit(self) -> List[DroneFlybyPredictionDto]:
        output_predictions: List[DroneFlybyPredictionDto] = []
        for track in self.tracks.values():
            if not self._is_confirmed(track):
                continue
            if track.existence < self.min_existence:
                continue

            effective_conf = track.confidence * (self.confidence_decay_rate ** track.frames_since_seen)
            if effective_conf < 0.05:
                continue

            global_bbox = source_bbox_to_global(track.bbox_4k, IMAGE_WIDTH, IMAGE_HEIGHT)
            clipped_bbox = clip_bbox_to_frame(global_bbox)
            if clipped_bbox is None:
                continue

            output_predictions.append(
                DroneFlybyPredictionDto(
                    object_id=track.class_name,
                    bbox=clipped_bbox,
                    confidence=float(min(1.0, effective_conf)),
                )
            )

        return apply_class_aware_nms(output_predictions, iou_threshold=self.nms_threshold)

    def get_summary(self) -> TrackerSummary:
        """Provide world state summary for camera steering decisions.

        ``unscanned_clusters`` is ordered by expected verification value: tracks
        never seen at high zoom first, then the most positionally uncertain.
        """
        candidates: List[Tuple[float, int, int]] = []
        for track in self.tracks.values():
            # Only confirmed, still-existing tracks are worth spending a camera
            # move on; unconfirmed detections are often single-frame noise.
            if not self._is_confirmed(track):
                continue
            if track.existence < self.min_existence:
                continue
            if track.best_zoom >= 2:
                continue
            cx, cy = _box_center(track.bbox_4k)
            # Higher score = more valuable to inspect.
            zoom_deficit = (2 - track.best_zoom) / 2.0
            uncertainty = 1.0 - min(1.0, track.confidence)
            score = zoom_deficit + uncertainty + 0.5 * min(1.0, track.position_std / 100.0)
            candidates.append((score, int(cx), int(cy)))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return TrackerSummary(
            num_active_tracks=len(self.tracks),
            unscanned_clusters=[(x, y) for _, x, y in candidates],
            current_shift_estimate=self.current_shift,
        )


class PassthroughTracker(BaseTracker):
    """Simple non-persistent tracker for testing and baseline comparison."""

    def __init__(self, nms_threshold: float = 0.45):
        self.nms_threshold = nms_threshold
        self.current_sequence_id: Optional[str] = None
        self.last_frame_index: int = -1
        self.total_detections_seen: int = 0
        self._last_predictions: List[DroneFlybyPredictionDto] = []

    def reset(self, sequence_id: str) -> None:
        self.current_sequence_id = sequence_id
        self.last_frame_index = -1
        self.total_detections_seen = 0
        self._last_predictions = []

    def update(
        self,
        detections: List[DetectionResult],
        zoom_level: int,
        source_region_xyxy: Tuple[int, int, int, int],
        frame_index: int,
        l0_image_gray: Optional[np.ndarray] = None,
    ) -> List[DroneFlybyPredictionDto]:
        _ = (zoom_level, source_region_xyxy, l0_image_gray)
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
        self._last_predictions = apply_class_aware_nms(raw_predictions, iou_threshold=self.nms_threshold)
        return self._last_predictions

    def predict_only(self) -> List[DroneFlybyPredictionDto]:
        return list(self._last_predictions)

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
            single_hit_confirm_confidence=config.SINGLE_HIT_CONFIRM_CONFIDENCE,
            confidence_decay_rate=config.CONFIDENCE_DECAY_RATE,
            out_of_view_max_age=config.OUT_OF_VIEW_MAX_AGE_FRAMES,
            nms_threshold=config.GLOBAL_NMS_IOU_THRESHOLD,
        )
    elif config.TRACKER_TYPE == "passthrough":
        return PassthroughTracker(nms_threshold=config.GLOBAL_NMS_IOU_THRESHOLD)
    else:
        logger.warning("Unknown tracker type '%s', defaulting to WorldMapTracker", config.TRACKER_TYPE)
        return WorldMapTracker()
