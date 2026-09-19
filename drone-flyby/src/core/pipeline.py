"""Coordinate causal detection, spatial memory and camera commands."""

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
    def __init__(
        self, detector: BaseDetector, tracker: BaseTracker,
        camera_policy: BaseCameraPolicy, config: DroneFlybyConfig,
    ):
        self.detector = detector
        self.tracker = tracker
        self.camera_policy = camera_policy
        self.config = config
        self.active_sequence_id: Optional[str] = None
        self.last_frame_index = -1
        self._lock = threading.Lock()
        self._detector_estimate_ms = 0.0
        self._healthy_detector_ms = 0.0
        self._budget_fallbacks = 0
        self._last_response: Optional[DroneFlybyPredictResponseDto] = None
        self.last_timings = {}

    def warmup(self) -> None:
        self.detector.warmup()
        started = time.perf_counter()
        self.detector.warmup()
        self._detector_estimate_ms = (time.perf_counter() - started) * 1000
        self._healthy_detector_ms = self._detector_estimate_ms
        logger.info("Pipeline warmed up; steady detector estimate %.1f ms", self._detector_estimate_ms)

    @staticmethod
    def _empty_response(request):
        return DroneFlybyPredictResponseDto(
            request_id=request.request_id, frame=request.frame, annotations=[], requested_view=None,
        )

    def _predict_only_response(self, request):
        try:
            annotations = self.tracker.predict_only(frame_index=request.frame_index)
        except Exception:
            logger.exception("Memory fallback failed on frame %d", request.frame)
            return self._empty_response(request)
        return DroneFlybyPredictResponseDto(
            request_id=request.request_id, frame=request.frame,
            annotations=annotations, requested_view=None,
        )

    def handle_request(self, request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
        started = time.perf_counter()
        budget_ms = request.response_timeout_ms or 3333
        if not self._lock.acquire(timeout=max(0.0, budget_ms / 1000)):
            logger.error("Frame %d exhausted its deadline waiting for pipeline state", request.frame)
            return self._empty_response(request)
        try:
            return self._process_locked(request, started, budget_ms)
        finally:
            self._lock.release()

    def _process_locked(self, request, started, budget_ms):
        same_sequence = request.sequence_id == self.active_sequence_id
        if same_sequence and request.frame_index > 0 and request.frame_index <= self.last_frame_index:
            if (request.frame_index == self.last_frame_index and self._last_response is not None
                    and self._last_response.request_id == request.request_id):
                return self._last_response.model_copy(deep=True)
            logger.warning("Discarding stale/duplicate frame_index %d", request.frame_index)
            return self._empty_response(request)

        if not same_sequence or request.frame_index == 0:
            self._reset_session(request.sequence_id)
        self.last_frame_index = request.frame_index
        try:
            elapsed_ms = (time.perf_counter() - started) * 1000
            if budget_ms - elapsed_ms <= 5:
                logger.warning("Frame %d has insufficient remaining budget; advancing memory only", request.frame)
                response = self._predict_only_response(request)
            else:
                response = self._observe(request, started, budget_ms)
        except Exception:
            logger.exception("Frame %d failed; advancing memory without negative evidence", request.frame)
            response = self._predict_only_response(request)
        self._last_response = response.model_copy(deep=True)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if elapsed_ms >= budget_ms:
            logger.error("Frame %d exceeded its hard deadline: %.1f >= %d ms",
                         request.frame, elapsed_ms, budget_ms)
        return response

    def _observe(self, request, started, budget_ms):
        image = decode_view(request.view)
        remaining_ms = budget_ms - (time.perf_counter() - started) * 1000
        if remaining_ms <= self._detector_estimate_ms * 1.2 + 5:
            self._budget_fallbacks += 1
            # A one-off accelerator stall must not disable detection forever.
            # Retry only after bounded backoff and when the known healthy cost fits.
            if self._budget_fallbacks < 8 or remaining_ms <= self._healthy_detector_ms * 1.2 + 5:
                logger.warning("Frame %d has insufficient inference budget; advancing memory", request.frame)
                return self._predict_only_response(request)
            logger.warning("Frame %d is probing detector recovery after budget backoff", request.frame)
        self._budget_fallbacks = 0
        detection_started = time.perf_counter()
        try:
            detections = self.detector.detect(
                image_bgr=image, zoom_level=request.view.resolution_level,
                source_region_xyxy=request.view.source_region_xyxy,
            )
        except Exception:
            logger.exception("Detector unavailable on frame %d; advancing memory", request.frame)
            return self._predict_only_response(request)
        detector_ms = (time.perf_counter() - detection_started) * 1000
        self._detector_estimate_ms = detector_ms
        if detector_ms + 5 < remaining_ms:
            self._healthy_detector_ms = detector_ms

        tracking_started = time.perf_counter()
        annotations = self.tracker.update(
            detections=detections, zoom_level=request.view.resolution_level,
            source_region_xyxy=request.view.source_region_xyxy,
            frame_index=request.frame_index, l0_image_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        )
        tracker_ms = (time.perf_counter() - tracking_started) * 1000
        camera_started = time.perf_counter()
        next_view = None
        if (camera_started - started) * 1000 < budget_ms * 0.9:
            try:
                next_view = self.camera_policy.decide_next_view(request, self.tracker.get_summary())
            except Exception:
                logger.exception("Camera planning failed on frame %d; retaining current detections", request.frame)
        else:
            logger.warning("Frame %d has no camera-planning budget; holding", request.frame)
        self.last_timings = {
            "frame": request.frame, "detector_ms": detector_ms, "tracker_ms": tracker_ms,
            "camera_ms": (time.perf_counter() - camera_started) * 1000,
            "total_ms": (time.perf_counter() - started) * 1000,
        }
        if self.config.DEBUG or request.frame_index % 10 == 0:
            logger.info("frame %d -> %d annotations | timings=%s",
                        request.frame, len(annotations), self.last_timings)
        return DroneFlybyPredictResponseDto(
            request_id=request.request_id, frame=request.frame,
            annotations=annotations, requested_view=next_view,
        )

    def _reset_session(self, new_sequence_id: str) -> None:
        logger.info("Starting sequence %s", new_sequence_id)
        self.active_sequence_id = new_sequence_id
        self.last_frame_index = -1
        self._last_response = None
        self.tracker.reset(new_sequence_id)
        self.camera_policy.reset(new_sequence_id)
