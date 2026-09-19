"""Tune class proposal thresholds by replaying their temporal consequences.

Use training-development scenes only. The objective is the common full-frame
COCO AP50 scorer, not a separate AP approximation or isolated detection count.
Any selected calibration still needs an independent realtime comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import DroneFlybyConfig
from core.detector import YoloDetector, create_detector
from dtos import OBJECT_CLASSES
from local_evaluator import score_ground_truth
from offline.camera_simulator import run_simulation
from offline.dataset_provenance import assert_training_source
from utils import DEFAULT_SCENE, frame_numbers, load_annotations, load_frame, scene_directory

Box = Tuple[float, float, float, float]


def average_precision(
    predictions: List[Tuple[int, Box, float]],
    ground_truth: Dict[int, List[Box]],
    iou_threshold: float = 0.5,
) -> float:
    """Single-class adapter to exactly the same COCO AP50 implementation."""
    if iou_threshold != 0.5:
        raise ValueError("The competition scorer uses IoU 0.50")
    if not any(ground_truth.values()):
        return 0.0
    truth = {
        frame: [{"object_id": "tank", "bbox": box} for box in boxes]
        for frame, boxes in ground_truth.items()
    }
    detections: Dict[int, List[dict]] = {}
    for frame, box, confidence in predictions:
        truth.setdefault(frame, [])
        detections.setdefault(frame, []).append({
            "object_id": "tank", "bbox": box, "confidence": confidence,
        })
    return score_ground_truth(truth, detections)[0]


def calibrate(
    scene: str,
    min_instances: int = 5,
    default_threshold: Optional[float] = None,
    thresholds: Optional[Sequence[float]] = None,
    config: Optional[DroneFlybyConfig] = None,
) -> Dict[str, float]:
    assert_training_source(scene_directory(scene))
    config = replace(config or DroneFlybyConfig.from_env())
    if config.CALIBRATION_PATH is not None:
        assert_training_source(config.CALIBRATION_PATH)
    candidates = list(thresholds) if thresholds is not None else [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
    if any(not 0 <= threshold <= 1 for threshold in candidates):
        raise ValueError("Calibration thresholds must be in [0, 1]")
    detector = create_detector(config)
    if not isinstance(detector, YoloDetector):
        raise ValueError("Calibration requires the production YOLO/TensorRT detector")
    if default_threshold is not None:
        if not 0 <= default_threshold <= 1:
            raise ValueError("Default threshold must be in [0, 1]")
        detector.calibration = {name: default_threshold for name in OBJECT_CLASSES}
    detector.warmup()
    frames = frame_numbers(scene)
    images = {frame: load_frame(frame, scene) for frame in frames}
    truth = {frame: load_annotations(frame, scene) for frame in frames}

    def evaluate() -> float:
        return run_simulation(
            config, scene, frames=frames, detector=detector,
            frame_provider=images.__getitem__, annotation_provider=truth.__getitem__,
            score_fn=lambda _scene, predictions: score_ground_truth(truth, predictions),
        ).map50

    best_score = evaluate()
    print(f"Starting closed-loop development AP50: {best_score:.6f}")
    for name in OBJECT_CLASSES:
        observed_frames = sum(any(a["object_id"] == name for a in annotations) for annotations in truth.values())
        if observed_frames < min_instances:
            print(f"{name}: insufficient labeled frames ({observed_frames}); unchanged")
            continue
        incumbent = dict(detector.calibration)
        best_calibration = incumbent
        for threshold in candidates:
            detector.calibration = {**incumbent, name: float(threshold)}
            candidate_score = evaluate()
            if candidate_score > best_score + 1e-8:
                best_score = candidate_score
                best_calibration = dict(detector.calibration)
        detector.calibration = best_calibration
        print(f"{name}: threshold={best_calibration.get(name, 'per-zoom default')} "
              f"closed-loop AP50={best_score:.6f}", flush=True)
    return dict(detector.calibration)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--min-instances", type=int, default=5, help="Minimum distinct labeled source frames.")
    parser.add_argument("--default-threshold", type=float)
    parser.add_argument("--thresholds", nargs="+", type=float)
    arguments = parser.parse_args()
    config = DroneFlybyConfig.from_env()
    if arguments.weights is not None:
        config.YOLO_WEIGHTS_PATH = arguments.weights
    if arguments.device is not None:
        config.DEVICE = arguments.device
    if arguments.imgsz is not None:
        config.INFERENCE_IMAGE_SIZE = arguments.imgsz
        config.INFERENCE_IMAGE_SIZES = None
    result = calibrate(
        arguments.scene, arguments.min_instances, arguments.default_threshold,
        arguments.thresholds, config,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
