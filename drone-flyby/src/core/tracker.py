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
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from config import DroneFlybyConfig
from core.ego_motion import EgoMotionEstimator, EgoMotionResult
from core.interfaces import BaseTracker, DetectionResult, TrackerSummary, TrackBelief
from dtos import DroneFlybyPredictionDto, IMAGE_HEIGHT, IMAGE_WIDTH
from utils import clip_bbox_to_frame, source_bbox_to_global, pairwise_iou

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


def class_confidences(detections: List[DetectionResult]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for detection in detections:
        result[detection.class_name] = max(result.get(detection.class_name, 0.0), detection.confidence)
    return result


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
        boxes = np.asarray([prediction.bbox for prediction in class_preds], dtype=np.float64)
        remaining = np.arange(len(class_preds))
        while len(remaining):
            index = remaining[0]
            selected.append(class_preds[index])
            candidates = remaining[1:]
            overlaps = pairwise_iou(boxes[index:index + 1], boxes[candidates])[0]
            remaining = candidates[overlaps <= iou_threshold]

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
        ego_motion_method: str = "phase_correlation",
        ego_motion_min_response: float = 0.15,
        min_detectable_pixels: float = 16.0,
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
        self.min_detectable_pixels = min_detectable_pixels
        self.ego_motion = EgoMotionEstimator(
            method=ego_motion_method, min_response=ego_motion_min_response
        )

        self.tracks: Dict[int, TrackedObject] = {}
        self.next_track_id: int = 0
        self.current_sequence_id: Optional[str] = None
        self.last_frame_index: int = -1
        self.last_zoom: int = 0
        self.prev_l0_gray: Optional[np.ndarray] = None
        self.prev_l0_frame_index: int = -1
        self.prev_region: Optional[Tuple[int, int, int, int]] = None
        self.prev_pixel_scale: Optional[float] = None
        self.current_shift: Tuple[float, float] = default_shift
        self.fresh_class_confidences: Dict[str, float] = {}

    def reset(self, sequence_id: str) -> None:
        """Reset the world map when transitioning to a new sequence."""
        self.tracks.clear()
        self.next_track_id = 0
        self.current_sequence_id = sequence_id
        self.last_frame_index = -1
        self.last_zoom = 0
        self.prev_l0_gray = None
        self.prev_l0_frame_index = -1
        self.prev_region = None
        self.prev_pixel_scale = None
        self.current_shift = self.default_shift
        self.fresh_class_confidences.clear()
        logger.info("WorldMapTracker session reset for sequence '%s'", sequence_id)

    # ------------------------------------------------------------------ #
    # Ego-motion
    # ------------------------------------------------------------------ #

    def _estimate_ego_motion(
        self,
        l0_image_gray: Optional[np.ndarray],
        frame_index: int,
        source_region_xyxy: Tuple[int, int, int, int],
    ) -> Tuple[float, float]:
        """Estimate a per-frame source-pixel shift from consecutive views.

        Both views are resampled over common source-space terrain, compensating
        for pan and zoom. The residual corrects the motion prior. Frame gaps
        convert displacement into a rate rather than a single huge jump.
        """
        if l0_image_gray is None:
            return self.current_shift

        gray = np.ascontiguousarray(l0_image_gray, dtype=np.float32)
        source_width = max(1, int(source_region_xyxy[2]) - int(source_region_xyxy[0]))
        pixel_scale = source_width / float(gray.shape[1])

        gap = max(1, frame_index - self.prev_l0_frame_index) if self.prev_l0_frame_index >= 0 else 1

        if self.prev_l0_gray is not None and self.prev_l0_frame_index >= 0 and self.prev_region is not None:
            result: EgoMotionResult = self.ego_motion.estimate_views(
                previous_gray=self.prev_l0_gray,
                current_gray=gray,
                previous_region=self.prev_region,
                current_region=source_region_xyxy,
                gap=gap,
                prior=self.current_shift,
            )
            if result.accepted:
                self.current_shift = (
                    0.7 * result.dx + 0.3 * self.current_shift[0],
                    0.7 * result.dy + 0.3 * self.current_shift[1],
                )

        self.prev_l0_gray = gray
        self.prev_l0_frame_index = frame_index
        self.prev_region = source_region_xyxy
        self.prev_pixel_scale = pixel_scale
        return self.current_shift

    def _predict_tracks(self, frame_gap: int) -> None:
        """Advance track coordinates to compensate for drone forward movement."""
        dx = self.current_shift[0] * frame_gap
        dy = self.current_shift[1] * frame_gap

        for track in self.tracks.values():
            x1, y1, x2, y2 = track.bbox_4k
            track.bbox_4k = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            track.age += frame_gap
            track.frames_since_seen += frame_gap
            process_std = 2.0 + 0.1 * np.hypot(*self.current_shift)
            track.position_std = float(np.sqrt(track.position_std ** 2 + process_std ** 2 * frame_gap))

    @staticmethod
    def _is_in_view(
        bbox_4k: Tuple[float, float, float, float], source_region_xyxy: Tuple[int, int, int, int]
    ) -> bool:
        """Determine if a 4K box is inside the current camera view crop."""
        cx, cy = _box_center(bbox_4k)
        sx1, sy1, sx2, sy2 = source_region_xyxy
        return sx1 <= cx <= sx2 and sy1 <= cy <= sy2

    def _is_observable(self, bbox, source_region) -> bool:
        x1, y1, x2, y2 = bbox
        sx1, sy1, sx2, sy2 = source_region
        fully_inside = sx1 <= x1 < x2 <= sx2 and sy1 <= y1 < y2 <= sy2
        apparent_size = max(x2 - x1, y2 - y1) * 960.0 / (sx2 - sx1)
        return fully_inside and apparent_size >= self.min_detectable_pixels

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
        if not num_dets or not num_tracks:
            return [], set(), set()

        det_boxes = np.asarray([det.source_pixel_bbox for det in detections], dtype=np.float64)
        track_boxes = np.asarray([self.tracks[tid].bbox_4k for tid in track_ids], dtype=np.float64)
        iou = pairwise_iou(det_boxes, track_boxes)
        track_sizes = track_boxes[:, 2:] - track_boxes[:, :2]

        det_centers = (det_boxes[:, :2] + det_boxes[:, 2:]) / 2.0
        track_centers = (track_boxes[:, :2] + track_boxes[:, 2:]) / 2.0
        delta = det_centers[:, None, :] - track_centers[None, :, :]
        distance = np.sqrt((delta * delta).sum(axis=2))
        diagonal = np.maximum(np.sqrt((track_sizes * track_sizes).sum(axis=1)), 1.0)
        class_match = (
            np.asarray([det.class_name for det in detections], dtype=object)[:, None]
            == np.asarray([self.tracks[tid].class_name for tid in track_ids], dtype=object)[None, :]
        )
        gate = (iou >= self.iou_match_threshold) | (class_match & (distance <= np.maximum(diagonal, 150.0)))
        centre_bonus = 0.25 * np.maximum(0.0, 1.0 - distance / diagonal)
        cost = np.full((num_dets, num_tracks + num_dets), 1e6, dtype=np.float32)
        cost[:, num_tracks:] = 1.5
        cost[:, :num_tracks] = np.where(
            gate, 1.0 - iou - centre_bonus + np.where(class_match, 0.0, 0.2), 1e6,
        )

        matches: List[Tuple[int, int]] = []
        matched_dets: set = set()
        matched_tracks: set = set()
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if c < num_tracks and gate[r, c]:
                matches.append((int(r), track_ids[c]))
                matched_dets.add(int(r))
                matched_tracks.add(track_ids[c])
        return matches, matched_dets, matched_tracks

    @staticmethod
    def _truncated_edges(box, region):
        margin = (region[2] - region[0]) / 960.0
        return (
            region[0] > 0 and box[0] <= region[0] + margin,
            region[1] > 0 and box[1] <= region[1] + margin,
            region[2] < IMAGE_WIDTH and box[2] >= region[2] - margin,
            region[3] < IMAGE_HEIGHT and box[3] >= region[3] - margin,
        )

    def _apply_detection(self, track: TrackedObject, det: DetectionResult, zoom_level: int, region) -> None:
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

        previous_center = _box_center(track.bbox_4k)
        left, top, right, bottom = self._truncated_edges(det.source_pixel_bbox, region)
        truncated = left or top or right or bottom
        improved_zoom = zoom_level > track.best_zoom and not truncated
        if zoom_level >= track.best_zoom and not truncated:
            track.bbox_4k = det.source_pixel_bbox
            track.best_zoom = zoom_level
        else:
            cx, cy = _box_center(det.source_pixel_bbox)
            width = track.bbox_4k[2] - track.bbox_4k[0]
            height = track.bbox_4k[3] - track.bbox_4k[1]
            if left:
                cx = previous_center[0] if right else det.source_pixel_bbox[2] - width / 2
            elif right:
                cx = det.source_pixel_bbox[0] + width / 2
            if top:
                cy = previous_center[1] if bottom else det.source_pixel_bbox[3] - height / 2
            elif bottom:
                cy = det.source_pixel_bbox[1] + height / 2
            track.bbox_4k = (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
        track.confidence = det.confidence if improved_zoom else 0.6 * track.confidence + 0.4 * det.confidence
        track.last_zoom = zoom_level
        new_center = _box_center(track.bbox_4k)
        track.velocity = (new_center[0] - previous_center[0], new_center[1] - previous_center[1])
        track.position_std = (4.0 if truncated else 2.0) * 2 ** (2 - zoom_level)

    def _advance_to(self, frame_index: int) -> int:
        if frame_index < self.last_frame_index:
            raise ValueError("Cannot advance spatial memory backwards")
        gap = frame_index - self.last_frame_index if self.last_frame_index >= 0 else 0
        self._predict_tracks(gap)
        self.last_frame_index = frame_index
        return gap

    def update(
        self,
        detections: List[DetectionResult],
        zoom_level: int,
        source_region_xyxy: Tuple[int, int, int, int],
        frame_index: int,
        l0_image_gray: Optional[np.ndarray] = None,
    ) -> List[DroneFlybyPredictionDto]:
        """Update tracks, correlate with new detections, and return full-frame annotations."""
        if frame_index < self.last_frame_index:
            raise ValueError("Cannot advance spatial memory backwards")
        self._estimate_ego_motion(l0_image_gray, frame_index, source_region_xyxy)

        frame_gap = self._advance_to(frame_index)
        self.fresh_class_confidences = class_confidences(detections)
        self.last_zoom = zoom_level

        track_ids = list(self.tracks.keys())
        matches, matched_dets, matched_tracks = self._associate(detections, track_ids)

        for det_index, track_id in matches:
            self._apply_detection(self.tracks[track_id], detections[det_index], zoom_level, source_region_xyxy)

        # Negative evidence: an in-view miss is informative, an out-of-view one
        # is not. Collapsing them is why the old map kept ghost tracks alive.
        for track_id in track_ids:
            if track_id in matched_tracks:
                continue
            track = self.tracks[track_id]
            if self._is_observable(track.bbox_4k, source_region_xyxy):
                track.existence *= self.in_view_miss_decay
            else:
                track.existence *= self.out_of_view_miss_decay ** frame_gap

        # New tracks for unmatched detections.
        for det_index, det in enumerate(detections):
            if det_index in matched_dets:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            truncated = any(self._truncated_edges(det.source_pixel_bbox, source_region_xyxy))
            self.tracks[track_id] = TrackedObject(
                track_id=track_id,
                class_name=det.class_name,
                bbox_4k=det.source_pixel_bbox,
                confidence=det.confidence,
                best_zoom=max(0, zoom_level - 1) if truncated else zoom_level,
                hits=1,
                frames_since_seen=0,
                age=1,
                class_scores={det.class_name: det.confidence},
                existence=min(0.9, 0.4 + 0.5 * det.confidence),
                position_std=16.0 if truncated else 0.0,
                last_seen_frame=frame_index,
                last_zoom=zoom_level,
            )

        self._prune()
        return self._emit(detections)

    def predict_only(self, frame_index: Optional[int] = None) -> List[DroneFlybyPredictionDto]:
        """Advance memory without negative evidence from an unavailable detector."""
        if frame_index is not None:
            gap = self._advance_to(frame_index)
            if gap:
                self.fresh_class_confidences.clear()
            for track in self.tracks.values():
                track.existence *= self.out_of_view_miss_decay ** gap
            self._prune()
        return self._emit()

    def _prune(self) -> None:
        dead_tids: List[int] = []
        for tid, track in self.tracks.items():
            x1, y1, x2, y2 = track.bbox_4k
            if y1 > IMAGE_HEIGHT or y2 < 0 or x1 > IMAGE_WIDTH or x2 < 0:
                dead_tids.append(tid)
            elif track.existence < self.min_existence * 0.5:
                dead_tids.append(tid)
            elif not self._is_confirmed(track) and track.age > 4:
                # Only *unconfirmed* tracks die of old age. A track confirmed by
                # a single confident hit must survive while the camera looks
                # elsewhere, otherwise every L0-only object is lost the moment
                # the loop starts zooming.
                dead_tids.append(tid)
            elif track.frames_since_seen > self.out_of_view_max_age:
                dead_tids.append(tid)

        for tid in dead_tids:
            del self.tracks[tid]

    def _is_confirmed(self, track: TrackedObject) -> bool:
        return (
            track.hits >= self.min_hits_to_confirm
            or track.confidence >= self.single_hit_confirm_confidence
        )

    def _emit(self, current_detections: Optional[List[DetectionResult]] = None) -> List[DroneFlybyPredictionDto]:
        output_predictions: List[DroneFlybyPredictionDto] = []
        # Confidence is a ranking score in this protocol. Reserve the upper
        # interval for fresh detections so speculative memory cannot outrank or
        # suppress the detector's current-frame evidence.
        for detection in current_detections or []:
            bbox = clip_bbox_to_frame(source_bbox_to_global(detection.source_pixel_bbox))
            if bbox is not None:
                output_predictions.append(DroneFlybyPredictionDto(
                    object_id=detection.class_name, bbox=bbox,
                    confidence=0.5 + 0.5 * detection.confidence,
                ))
        fresh_count = len(output_predictions)
        for track in self.tracks.values():
            if not self._is_confirmed(track):
                continue
            if track.existence < self.min_existence:
                continue

            effective_conf = track.confidence * (self.confidence_decay_rate ** track.frames_since_seen)
            if effective_conf <= 0:
                continue

            global_bbox = source_bbox_to_global(track.bbox_4k, IMAGE_WIDTH, IMAGE_HEIGHT)
            clipped_bbox = clip_bbox_to_frame(global_bbox)
            if clipped_bbox is None:
                continue

            output_predictions.append(
                DroneFlybyPredictionDto(
                    object_id=track.class_name,
                    bbox=clipped_bbox,
                    confidence=float(0.49 * min(1.0, effective_conf)),
                )
            )

        selected = apply_class_aware_nms(output_predictions, iou_threshold=self.nms_threshold)
        self.last_emission_counts = {
            "fresh_before_nms": fresh_count,
            "memory_before_nms": len(output_predictions) - fresh_count,
            "after_nms": len(selected),
            "output_at_cap": len(selected) == 500,
        }
        return selected

    def get_summary(self) -> TrackerSummary:
        """Provide world state summary for camera steering decisions.

        ``unscanned_clusters`` is ordered by expected verification value: tracks
        never seen at high zoom first, then the most positionally uncertain.
        ``track_beliefs`` carries the same tracks with the per-track evidence a
        belief-map planner needs (existence, confidence, best zoom, spread).
        """
        candidates: List[Tuple[float, int, int]] = []
        beliefs: List[TrackBelief] = []
        for track in self.tracks.values():
            # Only confirmed, still-existing tracks are worth spending a camera
            # move on; unconfirmed detections are often single-frame noise.
            if not self._is_confirmed(track):
                continue
            if track.existence < self.min_existence:
                continue
            cx, cy = _box_center(track.bbox_4k)
            beliefs.append(
                TrackBelief(
                    class_name=track.class_name,
                    center_x=cx,
                    center_y=cy,
                    existence=track.existence,
                    confidence=track.confidence,
                    best_zoom=track.best_zoom,
                    position_std=track.position_std,
                )
            )
            if track.best_zoom >= 2:
                continue
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
            track_beliefs=beliefs,
            fresh_class_confidences=dict(self.fresh_class_confidences),
        )


class PassthroughTracker(BaseTracker):
    """Simple non-persistent tracker for testing and baseline comparison."""

    def __init__(self, nms_threshold: float = 0.45):
        self.nms_threshold = nms_threshold
        self.current_sequence_id: Optional[str] = None
        self.last_frame_index: int = -1
        self.total_detections_seen: int = 0
        self._last_predictions: List[DroneFlybyPredictionDto] = []
        self.fresh_class_confidences: Dict[str, float] = {}

    def reset(self, sequence_id: str) -> None:
        self.current_sequence_id = sequence_id
        self.last_frame_index = -1
        self.total_detections_seen = 0
        self._last_predictions = []
        self.fresh_class_confidences.clear()

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
        self.fresh_class_confidences = class_confidences(detections)

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

    def predict_only(self, frame_index: Optional[int] = None) -> List[DroneFlybyPredictionDto]:
        if frame_index is not None and frame_index < self.last_frame_index:
            raise ValueError("Cannot advance passthrough state backwards")
        if frame_index is not None and frame_index != self.last_frame_index:
            self.last_frame_index = frame_index
            self._last_predictions = []
            self.fresh_class_confidences.clear()
            return []
        return list(self._last_predictions)

    def get_summary(self) -> TrackerSummary:
        return TrackerSummary(
            num_active_tracks=len(self._last_predictions),
            unscanned_clusters=[],
            current_shift_estimate=(0.0, 58.0),
            fresh_class_confidences=dict(self.fresh_class_confidences),
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
            min_existence=config.MIN_EXISTENCE,
            nms_threshold=config.GLOBAL_NMS_IOU_THRESHOLD,
            ego_motion_method=config.EGO_MOTION_METHOD,
            ego_motion_min_response=config.EGO_MOTION_MIN_RESPONSE,
            min_detectable_pixels=config.TRACK_MIN_DETECTABLE_PIXELS,
        )
    elif config.TRACKER_TYPE == "passthrough":
        return PassthroughTracker(nms_threshold=config.GLOBAL_NMS_IOU_THRESHOLD)
    else:
        raise ValueError(f"Unknown tracker type: {config.TRACKER_TYPE!r}")
