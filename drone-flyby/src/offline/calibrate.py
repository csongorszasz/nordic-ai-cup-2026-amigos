"""Per-class confidence calibration.

Sweeps a confidence threshold per class and keeps the value that maximises that
class's AP@0.50 on a supplied scene, then writes a JSON file the server can load
via ``DRONE_FLYBY_CALIBRATION_PATH``.

    python src/offline/calibrate.py --scene helsinki --output calibration.json

A single global threshold trades rare classes against common ones. Classes with
too few ground-truth instances to calibrate reliably keep the default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import DroneFlybyConfig  # noqa: E402
from core.detector import create_detector  # noqa: E402
from dtos import IMAGE_HEIGHT, IMAGE_WIDTH, OBJECT_CLASSES  # noqa: E402
from local_evaluator import frame_numbers, load_annotations, load_frame  # noqa: E402
from utils import DEFAULT_SCENE  # noqa: E402

REGION = (0, 0, IMAGE_WIDTH, IMAGE_HEIGHT)
Box = Tuple[float, float, float, float]


def _iou(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def average_precision(
    predictions: List[Tuple[int, Box, float]],
    ground_truth: Dict[int, List[Box]],
    iou_threshold: float = 0.5,
) -> float:
    """All-point-interpolated AP for one class at one IoU threshold."""
    total_positives = sum(len(boxes) for boxes in ground_truth.values())
    if total_positives == 0:
        return 0.0

    matched = {frame: [False] * len(boxes) for frame, boxes in ground_truth.items()}
    true_positive: List[int] = []
    false_positive: List[int] = []

    for frame, box, _confidence in sorted(predictions, key=lambda item: -item[2]):
        best_iou, best_index = 0.0, -1
        for index, gt_box in enumerate(ground_truth.get(frame, [])):
            if matched.get(frame, [])[index]:
                continue
            iou = _iou(box, gt_box)
            if iou > best_iou:
                best_iou, best_index = iou, index
        if best_index >= 0 and best_iou >= iou_threshold:
            matched[frame][best_index] = True
            true_positive.append(1)
            false_positive.append(0)
        else:
            true_positive.append(0)
            false_positive.append(1)

    if not true_positive:
        return 0.0

    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)
    recall = cumulative_tp / total_positives
    precision = cumulative_tp / (cumulative_tp + cumulative_fp)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    ramp = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[ramp + 1] - mrec[ramp]) * mpre[ramp + 1]))


def collect(scene: str) -> Tuple[Dict[str, List], Dict[str, Dict[int, List[Box]]]]:
    config = DroneFlybyConfig(DETECTOR_TYPE="yolo_standard")
    detector = create_detector(config)
    detector.warmup()

    predictions: Dict[str, List[Tuple[int, Box, float]]] = {name: [] for name in OBJECT_CLASSES}
    ground_truth: Dict[str, Dict[int, List[Box]]] = {name: {} for name in OBJECT_CLASSES}

    for frame in frame_numbers(scene):
        image = load_frame(frame, scene)
        l0 = cv2.resize(image, (IMAGE_WIDTH // 4, IMAGE_HEIGHT // 4), interpolation=cv2.INTER_AREA)
        for detection in detector.detect(l0, zoom_level=0, source_region_xyxy=REGION):
            predictions[detection.class_name].append(
                (frame, detection.source_pixel_bbox, detection.confidence)
            )
        for annotation in load_annotations(frame, scene):
            ground_truth[annotation["object_id"]].setdefault(frame, []).append(
                tuple(float(c) for c in annotation["bbox"])
            )
    return predictions, ground_truth


def calibrate(
    scene: str,
    min_instances: int = 5,
    default_threshold: float = 0.25,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    predictions, ground_truth = collect(scene)
    calibration: Dict[str, float] = {}

    print(f"{'class':16s} {'n':>3s} {'default':>8s} {'best_t':>7s} {'best_ap':>8s}")
    for name in OBJECT_CLASSES:
        boxes = ground_truth[name]
        instances = sum(len(value) for value in boxes.values())
        if instances < min_instances:
            calibration[name] = default_threshold
            print(f"{name:16s} {instances:3d} {default_threshold:8.2f} {'-':>7s} {'-':>8s}")
            continue

        best_ap, best_threshold = -1.0, default_threshold
        for threshold in thresholds:
            filtered = [p for p in predictions[name] if p[2] >= threshold]
            ap = average_precision(filtered, boxes)
            if ap > best_ap:
                best_ap, best_threshold = ap, float(threshold)
        calibration[name] = round(best_threshold, 3)
        print(f"{name:16s} {instances:3d} {default_threshold:8.2f} {best_threshold:7.2f} {best_ap:8.3f}")

    return calibration


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-class confidence calibration.")
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, default=None, help="Where to write the calibration JSON.")
    parser.add_argument("--min-instances", type=int, default=5)
    parser.add_argument("--default-threshold", type=float, default=0.25)
    arguments = parser.parse_args()

    calibration = calibrate(
        arguments.scene,
        min_instances=arguments.min_instances,
        default_threshold=arguments.default_threshold,
    )
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(calibration, indent=2))
        print(f"\nWrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
