"""Detector implementations and factory for drone-flyby."""

import json
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


def load_calibration(path: Path) -> Dict[str, float]:
    """Load a per-class confidence calibration file.

    The file maps class name to a minimum confidence; classes not listed keep
    the per-zoom thresholds. Raises if the file is missing or malformed, so a
    misconfigured calibration stops startup instead of being ignored.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"Calibration file not found at '{path}'")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Calibration file '{path}' must contain a JSON object")
    calibration: Dict[str, float] = {}
    for name, value in data.items():
        if name not in OBJECT_CLASSES:
            logger.warning("Ignoring calibration for unknown class '%s'", name)
            continue
        threshold = float(value)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Calibration threshold for '{name}' must be in [0, 1], got {threshold}")
        calibration[name] = threshold
    return calibration


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
    """Return a trimmed detector class name.

    There is deliberately no generic-to-challenge aliasing here. Mapping names
    such as ``car -> small_plane`` or ``truck -> large_launcher`` manufactures
    confident semantic false positives, so unknown names are returned as-is and
    dropped by the caller when they are not in :data:`OBJECT_CLASSES`.
    """
    return raw_name.strip()


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
    """Ultralytics detector backed by task-finetuned weights.

    Construction is deliberately fail-fast: a missing weight file, an
    unavailable backend or a class map that does not match the challenge stops
    startup instead of quietly falling back to a zero-scoring detector.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold_l0: float = 0.10,
        confidence_threshold_l1: float = 0.15,
        confidence_threshold_l2: float = 0.20,
        expected_classes: Sequence[str] = OBJECT_CLASSES,
        calibration: Optional[Dict[str, float]] = None,
    ):
        if not _ULTRALYTICS_AVAILABLE:
            raise RuntimeError("ultralytics is not installed; the YOLO backend is unavailable")

        if not weights_path:
            raise ValueError(
                "A weights path is required for the YOLO backend. Set "
                "DRONE_FLYBY_YOLO_WEIGHTS_PATH or DRONE_FLYBY_TRT_ENGINE_PATH."
            )

        weights_file = Path(weights_path)
        if not weights_file.is_file():
            raise FileNotFoundError(
                f"Detector weights not found at '{weights_file}'. Refusing to start "
                f"with an unvalidated fallback model."
            )

        yolo_cls = YOLO
        assert yolo_cls is not None
        self.model = yolo_cls(str(weights_file))
        self.device = device
        self.conf_thresholds = {
            0: confidence_threshold_l0,
            1: confidence_threshold_l1,
            2: confidence_threshold_l2,
        }
        self.calibration = dict(calibration) if calibration else {}
        self._validate_class_map(list(expected_classes))

    def _validate_class_map(self, expected_classes: List[str]) -> None:
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            ordered = [names[key] for key in sorted(names)]
        elif names is not None:
            ordered = list(names)
        else:
            raise ValueError("The YOLO model exposes no class names; cannot verify the label map.")

        if ordered != expected_classes:
            raise ValueError(
                "The detector's class map does not match the challenge classes. "
                f"Expected {expected_classes}, got {ordered}."
            )

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
            class_threshold = self.calibration.get(class_name)
            if class_threshold is not None and conf < class_threshold:
                continue
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
        results = self.model.predict(image_bgr, conf=conf, device=self.device, verbose=False)
        if not results:
            return []
        return self._parse_result(results[0], zoom_level, source_region_xyxy)


def _load_calibration(config: DroneFlybyConfig) -> Optional[Dict[str, float]]:
    """Load the per-class calibration, if configured.

    Only the YOLO/TensorRT backends consume it, so it is loaded lazily inside
    those branches rather than for every backend. Otherwise a stray
    CALIBRATION_PATH would make the debug detectors fail for no reason.
    """
    if not config.CALIBRATION_PATH:
        return None
    return load_calibration(config.CALIBRATION_PATH)


def create_detector(config: DroneFlybyConfig) -> BaseDetector:
    """Factory function for instantiating detectors.

    The production backends fail startup on misconfiguration. The debug-only
    ``dummy`` and ``template_bank`` backends remain available but are expected
    to score zero on the wire protocol and must be selected explicitly.
    """
    if config.DETECTOR_TYPE == "yolo_standard":
        return YoloDetector(
            weights_path=str(config.YOLO_WEIGHTS_PATH) if config.YOLO_WEIGHTS_PATH else None,
            device=config.DEVICE,
            confidence_threshold_l0=config.CONFIDENCE_THRESHOLD_L0,
            confidence_threshold_l1=config.CONFIDENCE_THRESHOLD_L1,
            confidence_threshold_l2=config.CONFIDENCE_THRESHOLD_L2,
            calibration=_load_calibration(config),
        )
    elif config.DETECTOR_TYPE == "tensorrt":
        return YoloDetector(
            weights_path=str(config.TRT_ENGINE_PATH) if config.TRT_ENGINE_PATH else None,
            device=config.DEVICE,
            confidence_threshold_l0=config.CONFIDENCE_THRESHOLD_L0,
            confidence_threshold_l1=config.CONFIDENCE_THRESHOLD_L1,
            confidence_threshold_l2=config.CONFIDENCE_THRESHOLD_L2,
            calibration=_load_calibration(config),
        )
    elif config.DETECTOR_TYPE == "template_bank":
        logger.warning(
            "DETECTOR_TYPE='template_bank' is a debug backend and will score near "
            "zero on the 960x540 wire protocol."
        )
        return TemplateBankDetector()
    elif config.DETECTOR_TYPE == "dummy":
        logger.warning(
            "DETECTOR_TYPE='dummy' is a plumbing backend and will score zero."
        )
        return DummyCannyDetector()
    elif config.DETECTOR_TYPE == "yolo_sahi":
        raise NotImplementedError(
            "DETECTOR_TYPE='yolo_sahi' is not implemented. Use 'yolo_standard' with "
            "exact-zoom training, or tile the 960x540 view yourself."
        )
    else:
        raise ValueError(
            f"Unknown DETECTOR_TYPE '{config.DETECTOR_TYPE}'. Valid values: "
            f"yolo_standard, tensorrt, template_bank, dummy."
        )

