"""Pipeline orchestrator coordinating detector, tracker, and camera policy."""

import logging
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

    def warmup(self) -> None:
        """Warm up subsystems prior to receiving live requests."""
        logger.info("Warming up pipeline orchestrator...")
        self.detector.warmup()
        logger.info("Pipeline orchestrator warmed up successfully.")

    def handle_request(
        self,
        request: DroneFlybyPredictRequestDto,
    ) -> DroneFlybyPredictResponseDto:
        """Process one incoming frame request end-to-end with crash protection."""
        t_start = time.perf_counter()

        # 1. Manage flight session boundaries
        if request.sequence_id != self.active_sequence_id or request.frame_index == 0:
            self._reset_session(request.sequence_id)

        self.last_frame_index = request.frame_index

        try:
            # 2. Decode view image
            image_bgr = decode_view(request.view)

            # 3. Object Detection stage
            t_det_start = time.perf_counter()
            raw_detections = self.detector.detect(
                image_bgr=image_bgr,
                zoom_level=request.view.resolution_level,
                source_region_xyxy=request.view.source_region_xyxy,
            )
            t_det_ms = (time.perf_counter() - t_det_start) * 1000

            # 4. Spatial Memory & Tracking stage
            t_trk_start = time.perf_counter()
            l0_gray = (
                cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                if request.view.resolution_level == 0
                else None
            )
            annotations = self.tracker.update(
                detections=raw_detections,
                zoom_level=request.view.resolution_level,
                source_region_xyxy=request.view.source_region_xyxy,
                frame_index=request.frame_index,
                l0_image_gray=l0_gray,
            )
            t_trk_ms = (time.perf_counter() - t_trk_start) * 1000

            # 5. Camera Steering Policy stage
            t_cam_start = time.perf_counter()
            tracker_summary = self.tracker.get_summary()
            next_view = self.camera_policy.decide_next_view(
                request=request,
                tracker_summary=tracker_summary,
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
            # as that forfeits all detections for the frame.
            logger.exception("Catastrophic error processing frame %d: %s", request.frame, exc)
            return DroneFlybyPredictResponseDto(
                request_id=request.request_id,
                frame=request.frame,
                annotations=[],
                requested_view=None,
            )

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

