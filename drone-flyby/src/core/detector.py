"""Detector implementations and factory for drone-flyby."""

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from config import DroneFlybyConfig
from core.interfaces import BaseDetector, DetectionResult
from dtos import OBJECT_CLASSES
from utils import (
    clip_bbox_to_frame,
    load_annotations,
    load_frame,
    scene_directory,
    source_bbox_to_global,
    view_bbox_to_global,
    view_bbox_to_source,
)

logger = logging.getLogger(__name__)

try:  # Optional backend; the detector still works without it.
    from ultralytics import YOLO  # type: ignore

    _ULTRALYTICS_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when the package exists.
    YOLO = None
    _ULTRALYTICS_AVAILABLE = False


def _compute_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass(slots=True)
class _TemplateExample:
    """One representative crop for a class in the supplied Helsinki scene."""

    class_name: str
    template_gray: np.ndarray
    source_bbox: Tuple[int, int, int, int]
    source_frame: int


def _normalize_class_name(raw_name: str) -> str:
    """Map detector-style class names into the challenge label format."""
    name = raw_name.strip()
    if name in OBJECT_CLASSES:
        return name
    normalized = name.replace(" ", "_").replace("-", "_")
    aliases = {
        "airplane": "jet_plane",
        "plane": "jet_plane",
        "aeroplane": "jet_plane",
        "car": "small_plane",
        "truck": "large_launcher",
        "bus": "hangar",
    }
    return aliases.get(normalized, normalized)


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


class TemplateBankDetector(BaseDetector):
    """Template-matching detector seeded from the supplied Helsinki frames.

    This is an initial, dependency-light phase-3 detector that is materially
    better than the fixed-label Canny baseline on the provided scene while
    still being fully deterministic and testable without a deep learning stack.
    """

    def __init__(
        self,
        scene: str = "helsinki",
        max_templates_per_class: int = 1,
        max_proposals: int = 40,
        score_threshold: float = 0.82,
    ):
        self.scene = scene
        self.max_templates_per_class = max_templates_per_class
        self.max_proposals = max_proposals
        self.score_threshold = score_threshold
        self.templates: Dict[str, List[_TemplateExample]] = self._build_template_bank()

    def _build_template_bank(self) -> Dict[str, List[_TemplateExample]]:
        templates: Dict[str, List[_TemplateExample]] = defaultdict(list)
        scene_dir = scene_directory(self.scene)

        # Use all supplied frames to pick a high-quality representative crop per class.
        for frame_index in range(25):
            try:
                frame = load_frame(frame_index, scene=self.scene)
                annotations = load_annotations(frame_index, scene=self.scene)
            except Exception:
                continue

            for annotation in annotations:
                class_name = str(annotation["object_id"])
                x1, y1, x2, y2 = (int(round(float(v))) for v in annotation["bbox"])
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue

                crop = frame[y1:y2, x1:x2]
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                templates[class_name].append(
                    _TemplateExample(
                        class_name=class_name,
                        template_gray=crop_gray,
                        source_bbox=(x1, y1, x2, y2),
                        source_frame=frame_index,
                    )
                )

        # Keep the sharpest/largest templates per class.
        bank: Dict[str, List[_TemplateExample]] = {}
        for class_name, examples in templates.items():
            examples.sort(
                key=lambda item: (item.template_gray.shape[0] * item.template_gray.shape[1], -item.source_frame),
                reverse=True,
            )
            bank[class_name] = examples[: self.max_templates_per_class]

        logger.info("TemplateBankDetector built with %d classes from %s", len(bank), scene_dir)
        return bank

    def warmup(self) -> None:
        """No heavy compilation; ensure the template bank exists."""
        if not self.templates:
            logger.warning("TemplateBankDetector template bank is empty.")

    def _scales_for_zoom(self, zoom_level: int) -> Sequence[float]:
        if zoom_level <= 0:
            return (0.85, 1.0, 1.15)
        if zoom_level == 1:
            return (0.9, 1.0, 1.2)
        return (1.0, 1.15, 1.3)

    def _detect_single_template(
        self,
        image_gray: np.ndarray,
        template: np.ndarray,
        class_name: str,
        zoom_level: int,
        source_region_xyxy: Tuple[int, int, int, int],
    ) -> Optional[DetectionResult]:
        best_score = -1.0
        best_bbox: Optional[Tuple[float, float, float, float]] = None

        for scale in self._scales_for_zoom(zoom_level):
            resized = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            tpl_h, tpl_w = resized.shape[:2]
            img_h, img_w = image_gray.shape[:2]
            if tpl_h < 3 or tpl_w < 3 or tpl_h > img_h or tpl_w > img_w:
                continue

            response = cv2.matchTemplate(image_gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(response)
            if max_val > best_score:
                x1 = float(max_loc[0])
                y1 = float(max_loc[1])
                x2 = float(max_loc[0] + tpl_w)
                y2 = float(max_loc[1] + tpl_h)
                best_score = float(max_val)
                best_bbox = (x1, y1, x2, y2)

        if best_bbox is None or best_score < self.score_threshold:
            return None

        img_h, img_w = image_gray.shape[:2]
        view_bbox = (
            best_bbox[0] / float(img_w),
            best_bbox[1] / float(img_h),
            best_bbox[2] / float(img_w),
            best_bbox[3] / float(img_h),
        )
        global_bbox = view_bbox_to_global(view_bbox, source_region_xyxy)
        clipped_global = clip_bbox_to_frame(global_bbox)
        if clipped_global is None:
            return None

        source_pixel_bbox = view_bbox_to_source(view_bbox, source_region_xyxy)
        return DetectionResult(
            class_name=class_name,
            bbox_global=clipped_global,
            confidence=min(0.99, max(0.0, best_score)),
            zoom_level=zoom_level,
            source_pixel_bbox=source_pixel_bbox,
        )

    def detect(
        self,
        image_bgr: np.ndarray,
        zoom_level: int,
        source_region_xyxy: Tuple[int, int, int, int],
    ) -> List[DetectionResult]:
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        results: List[DetectionResult] = []

        for class_name, examples in self.templates.items():
            for example in examples:
                detection = self._detect_single_template(
                    image_gray=image_gray,
                    template=example.template_gray,
                    class_name=class_name,
                    zoom_level=zoom_level,
                    source_region_xyxy=source_region_xyxy,
                )
                if detection is not None:
                    results.append(detection)

        # Keep only the best instance per class and suppress near-duplicates.
        results.sort(key=lambda det: det.confidence, reverse=True)
        selected: List[DetectionResult] = []
        by_class: Dict[str, List[DetectionResult]] = defaultdict(list)
        for det in results:
            by_class[det.class_name].append(det)

        for class_name, class_results in by_class.items():
            for det in class_results:
                if any(_compute_iou(det.source_pixel_bbox, kept.source_pixel_bbox) > 0.45 for kept in selected if kept.class_name == class_name):
                    continue
                selected.append(det)

        selected.sort(key=lambda det: det.confidence, reverse=True)
        return selected[: self.max_proposals]


class YoloDetector(BaseDetector):
    """Optional Ultralytics-based detector scaffold for future fine-tuned weights."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        use_sahi: bool = False,
        confidence_threshold_l0: float = 0.10,
        confidence_threshold_l1: float = 0.15,
        confidence_threshold_l2: float = 0.20,
    ):
        if not _ULTRALYTICS_AVAILABLE:
            raise RuntimeError("ultralytics is not installed; YOLO backend is unavailable")

        yolo_cls = YOLO
        assert yolo_cls is not None
        self.model = yolo_cls(weights_path or "yolo11s.pt")
        self.device = device
        self.use_sahi = use_sahi
        self.conf_thresholds = {
            0: confidence_threshold_l0,
            1: confidence_threshold_l1,
            2: confidence_threshold_l2,
        }

    def warmup(self) -> None:
        dummy = np.zeros((540, 960, 3), dtype=np.uint8)
        _ = self.detect(dummy, zoom_level=0, source_region_xyxy=(0, 0, 3840, 2160))

    def _parse_result(
        self,
        result,
        zoom_level: int,
        source_region_xyxy: Tuple[int, int, int, int],
    ) -> List[DetectionResult]:
        detections: List[DetectionResult] = []
        names = getattr(self.model, "names", {})
        image_h, image_w = result.orig_shape[:2]

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls)
            raw_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            class_name = _normalize_class_name(str(raw_name))
            if class_name not in OBJECT_CLASSES:
                continue
            conf = float(box.conf.item() if hasattr(box.conf, "item") else box.conf)
            xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy, "__getitem__") else list(box.xyxy)
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            view_bbox = (x1 / image_w, y1 / image_h, x2 / image_w, y2 / image_h)
            global_bbox = view_bbox_to_global(view_bbox, source_region_xyxy)
            clipped_global = clip_bbox_to_frame(global_bbox)
            if clipped_global is None:
                continue
            detections.append(
                DetectionResult(
                    class_name=class_name,
                    bbox_global=clipped_global,
                    confidence=conf,
                    zoom_level=zoom_level,
                    source_pixel_bbox=view_bbox_to_source(view_bbox, source_region_xyxy),
                )
            )
        return detections

    def detect(
        self,
        image_bgr: np.ndarray,
        zoom_level: int,
        source_region_xyxy: Tuple[int, int, int, int],
    ) -> List[DetectionResult]:
        conf = self.conf_thresholds.get(zoom_level, 0.15)
        if self.use_sahi and zoom_level == 0:
            # SAHI can be plugged in later; for now the scaffold still works with standard inference.
            pass

        results = self.model.predict(image_bgr, conf=conf, device=self.device, verbose=False)
        if not results:
            return []
        return self._parse_result(results[0], zoom_level, source_region_xyxy)


def create_detector(config: DroneFlybyConfig) -> BaseDetector:
    """Factory function for instantiating detectors."""
    if config.DETECTOR_TYPE == "dummy":
        return DummyCannyDetector()
    elif config.DETECTOR_TYPE == "template_bank":
        return TemplateBankDetector()
    elif config.DETECTOR_TYPE in ("yolo_standard", "yolo_sahi", "tensorrt"):
        if _ULTRALYTICS_AVAILABLE:
            return YoloDetector(
                weights_path=str(config.YOLO_WEIGHTS_PATH) if config.YOLO_WEIGHTS_PATH else None,
                device=config.DEVICE,
                use_sahi=config.DETECTOR_TYPE == "yolo_sahi",
                confidence_threshold_l0=config.CONFIDENCE_THRESHOLD_L0,
                confidence_threshold_l1=config.CONFIDENCE_THRESHOLD_L1,
                confidence_threshold_l2=config.CONFIDENCE_THRESHOLD_L2,
            )

        logger.warning(
            "ultralytics is unavailable; falling back to TemplateBankDetector for '%s'",
            config.DETECTOR_TYPE,
        )
        return TemplateBankDetector()
    else:
        logger.warning(
            "Unknown detector type '%s', falling back to DummyCannyDetector",
            config.DETECTOR_TYPE,
        )
        return DummyCannyDetector()

