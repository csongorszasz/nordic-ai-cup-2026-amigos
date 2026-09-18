"""Pipeline orchestrator coordinating detector, tracker, and camera policy."""

import logging
import threading
import time
from typing import Optional
import cv2

from config import DroneFlybyConfig
from core.interfaces import BaseCameraPolicy, BaseDetector, BaseTracker
from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from utils import decode_view

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Coordinates detection, spatial memory, and camera steering across requests."""

    def __init__(
        self,
        detector: BaseDetector,
        tracker: BaseTracker,
        camera_policy: BaseCameraPolicy,
        config: DroneFlybyConfig,
    ):
        self.detector = detector
        self.tracker = tracker
        self.camera_policy = camera_policy
        self.config = config

        self.active_sequence_id: Optional[str] = None
        self.last_frame_index: int = -1
        # The evaluator is sequential, but the server is multithreaded: state
        # mutations must be serialised so two frames cannot interleave.
        self._lock = threading.Lock()

    def warmup(self) -> None:
        """Warm up subsystems prior to receiving live requests."""
        logger.info("Warming up pipeline orchestrator...")
        self.detector.warmup()
        logger.info("Pipeline orchestrator warmed up successfully.")

    def _predict_only_response(self, request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
        """Best-effort answer from memory alone, holding the camera."""
        try:
            annotations = self.tracker.predict_only()
        except Exception:
            logger.exception("Tracker predict_only failed on frame %d", request.frame)
            annotations = []
        return DroneFlybyPredictResponseDto(
            request_id=request.request_id,
            frame=request.frame,
            annotations=annotations,
            requested_view=None,
        )

    def handle_request(
        self,
        request: DroneFlybyPredictRequestDto,
    ) -> DroneFlybyPredictResponseDto:
        """Process one incoming frame request end-to-end with crash protection."""
        with self._lock:
            t_start = time.perf_counter()

            same_sequence = request.sequence_id == self.active_sequence_id

            # 1. Discard stale/out-of-order frames before touching any state. A
            #    late duplicate of frame 0 must not reset the session either, so
            #    this runs before the frame_index == 0 boundary check below.
            if same_sequence and request.frame_index < self.last_frame_index:
                logger.warning(
                    "Discarding stale frame_index %d (< %d) for sequence %s",
                    request.frame_index,
                    self.last_frame_index,
                    request.sequence_id,
                )
                return self._predict_only_response(request)

            # 2. Manage flight session boundaries. frame_index 0 marks the first
            #    frame of a new sequence; the evaluator uses a unique sequence_id
            #    per attempt, so a same-id restart is treated as stale above.
            if not same_sequence or request.frame_index == 0:
                self._reset_session(request.sequence_id)

            self.last_frame_index = request.frame_index
            budget_ms = request.response_timeout_ms or 3333

            try:
                # 3. Decode view image
                image_bgr = decode_view(request.view)

                # 4. Object Detection stage. If it fails, do not feed an empty
                #    detection set to the memory: that would be read as "nothing
                #    is in this crop" and wrongly erode real tracks.
                t_det_start = time.perf_counter()
                try:
                    raw_detections = self.detector.detect(
                        image_bgr=image_bgr,
                        zoom_level=request.view.resolution_level,
                        source_region_xyxy=request.view.source_region_xyxy,
                    )
                except Exception:
                    logger.exception(
                        "Detector failed on frame %d; serving memory only", request.frame
                    )
                    return self._predict_only_response(request)
                t_det_ms = (time.perf_counter() - t_det_start) * 1000

                # 5. Spatial Memory & Tracking stage. The grayscale view feeds
                #    ego-motion estimation; with the belief-map policy the camera
                #    is rarely at L0, so it must be supplied at every level. The
                #    tracker rejects comparisons across level changes.
                t_trk_start = time.perf_counter()
                view_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                annotations = self.tracker.update(
                    detections=raw_detections,
                    zoom_level=request.view.resolution_level,
                    source_region_xyxy=request.view.source_region_xyxy,
                    frame_index=request.frame_index,
                    l0_image_gray=view_gray,
                )
                t_trk_ms = (time.perf_counter() - t_trk_start) * 1000

                # 6. Camera Steering Policy stage. If detection has already spent
                #    the request budget, hold the camera rather than risk a
                #    timeout; the frame's detections still count.
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                t_cam_start = time.perf_counter()
                next_view = None
                if elapsed_ms < budget_ms * 0.90:
                    tracker_summary = self.tracker.get_summary()
                    next_view = self.camera_policy.decide_next_view(
                        request=request,
                        tracker_summary=tracker_summary,
                    )
                else:
                    logger.warning(
                        "frame %d exceeded %.0f%% of the %dms budget before camera planning; holding",
                        request.frame,
                        90,
                        budget_ms,
                    )
                t_cam_ms = (time.perf_counter() - t_cam_start) * 1000

                t_total_ms = (time.perf_counter() - t_start) * 1000
                if self.config.DEBUG or request.frame_index % 10 == 0:
                    logger.info(
                        "frame %d (seq: %s) -> %d dets | det: %.1fms, trk: %.1fms, cam: %.1fms, total: %.1fms",
                        request.frame,
                        request.sequence_id[:8] if request.sequence_id else "unknown",
                        len(annotations),
                        t_det_ms,
                        t_trk_ms,
                        t_cam_ms,
                        t_total_ms,
                    )

                return DroneFlybyPredictResponseDto(
                    request_id=request.request_id,
                    frame=request.frame,
                    annotations=annotations,
                    requested_view=next_view,
                )

            except Exception as exc:
                # Critical: never allow an uncaught exception to return HTTP 500,
                # as that forfeits all detections for the frame. Serve whatever
                # memory already knows instead.
                logger.exception("Catastrophic error processing frame %d: %s", request.frame, exc)
                return self._predict_only_response(request)

    def _reset_session(self, new_sequence_id: str) -> None:
        """Reset state across all stateful subsystems when a new flight sequence starts."""
        logger.info(
            "Starting new session for sequence_id='%s' (previous='%s')",
            new_sequence_id,
            self.active_sequence_id,
        )
        self.active_sequence_id = new_sequence_id
        self.tracker.reset(new_sequence_id)
        self.camera_policy.reset(new_sequence_id)

