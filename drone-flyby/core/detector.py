"""Detector implementations and factory for drone-flyby."""

import logging
from typing import List, Tuple
import cv2
import numpy as np

from config import DroneFlybyConfig
from core.interfaces import BaseDetector, DetectionResult
from utils import clip_bbox_to_frame, view_bbox_to_global, view_bbox_to_source

logger = logging.getLogger(__name__)


class DummyCannyDetector(BaseDetector):
    """Fast OpenCV Canny contour detector baseline.
    
    Serves as the plumbing verification baseline and mock detector for testing.
    """

    def __init__(self, placeholder_class: str = "jammer", max_proposals: int = 20):
        self.placeholder_class = placeholder_class
        self.max_proposals = max_proposals

    def warmup(self) -> None:
        """Warm up by running edge detection on a blank canvas."""
        dummy = np.zeros((540, 960, 3), dtype=np.uint8)
        _ = self.detect(dummy, zoom_level=0, source_region_xyxy=(0, 0, 3840, 2160))
        logger.info("DummyCannyDetector warmed up.")

    def detect(
        self,
        image_bgr: np.ndarray,
        zoom_level: int,
        source_region_xyxy: Tuple[int, int, int, int],
    ) -> List[DetectionResult]:
        """Detect edge contour proposals and project them to global 4K coordinates."""
        results: List[DetectionResult] = []
        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 200)
            dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            proposals = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                longest = max(w, h)
                if longest < 8 or longest > 320:
                    continue
                area = cv2.contourArea(contour)
                compactness = area / float(w * h) if w * h > 0 else 0.0
                proposals.append((compactness, x, y, w, h))

            # Prefer denser contours
            proposals.sort(reverse=True, key=lambda item: item[0])

            h_view, w_view = image_bgr.shape[:2]
            for rank, (score, x, y, w, h) in enumerate(proposals[: self.max_proposals]):
                # Normalized bbox relative to 960x540 view
                view_bbox = (
                    float(x) / w_view,
                    float(y) / h_view,
                    float(x + w) / w_view,
                    float(y + h) / h_view,
                )

                # Transform to global normalized 4K source coordinates [0, 1]
                global_bbox = view_bbox_to_global(view_bbox, source_region_xyxy)
                clipped_global = clip_bbox_to_frame(global_bbox)
                if clipped_global is None:
                    continue

                # 4K pixel bbox for spatial memory matching
                source_pixel_bbox = view_bbox_to_source(view_bbox, source_region_xyxy)

                confidence = max(0.05, min(0.30, 0.30 - 0.01 * rank))
                results.append(
                    DetectionResult(
                        class_name=self.placeholder_class,
                        bbox_global=clipped_global,
                        confidence=confidence,
                        zoom_level=zoom_level,
                        source_pixel_bbox=source_pixel_bbox,
                    )
                )
        except Exception as exc:
            logger.error("Error during dummy detection: %s", exc)

        return results


def create_detector(config: DroneFlybyConfig) -> BaseDetector:
    """Factory function for instantiating detectors."""
    if config.DETECTOR_TYPE == "dummy":
        return DummyCannyDetector()
    elif config.DETECTOR_TYPE in ("yolo_standard", "yolo_sahi", "tensorrt"):
        raise NotImplementedError(
            f"Detector backend '{config.DETECTOR_TYPE}' will be implemented in subsequent phases."
        )
    else:
        logger.warning(
            "Unknown detector type '%s', falling back to DummyCannyDetector",
            config.DETECTOR_TYPE,
        )
        return DummyCannyDetector()

