"""Measure conditional detector recall at real camera zooms, not competition AP."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import DroneFlybyConfig
from core.detector import create_detector
from core.tracker import compute_iou
from dtos import OBJECT_CLASSES
from offline.build_exact_view_dataset import grid_centers, render_view
from offline.experiment_runner import checkpoint_hash, write_json
from utils import frame_numbers, load_annotations, load_frame


def match_observations(annotations, detections, require_class=True):
    """One-to-one, confidence-ordered matches at the competition's IoU."""
    matched = {}
    for detection in sorted(detections, key=lambda item: -item.confidence):
        eligible = [
            (compute_iou(annotation["bbox"], detection.source_pixel_bbox), index)
            for index, annotation in enumerate(annotations)
            if index not in matched and (not require_class or annotation["object_id"] == detection.class_name)
        ]
        if not eligible:
            continue
        overlap, index = max(eligible)
        if overlap >= 0.5:
            matched[index] = detection.confidence
    return matched


def fully_contained(annotations, region):
    return [
        annotation for annotation in annotations
        if region[0] < annotation["bbox"][0] < annotation["bbox"][2] < region[2]
        and region[1] < annotation["bbox"][1] < annotation["bbox"][3] < region[3]
    ]


def diagnose(scene, weights, output, imgsz=1600, frame_count=8, confidence=0.001):
    if output.exists():
        raise FileExistsError(f"Diagnostic output already exists: {output}")
    if frame_count < 1 or not 0 <= confidence <= 1:
        raise ValueError("Use a positive sample count and confidence in [0, 1]")
    config = DroneFlybyConfig.from_env()
    config.DETECTOR_TYPE = "yolo_standard"
    config.YOLO_WEIGHTS_PATH = weights
    config.INFERENCE_IMAGE_SIZE = imgsz
    config.INFERENCE_IMAGE_SIZES = None
    config.CALIBRATION_PATH = None
    config.CONFIDENCE_THRESHOLD_L0 = confidence
    config.CONFIDENCE_THRESHOLD_L1 = confidence
    config.CONFIDENCE_THRESHOLD_L2 = confidence
    detector = create_detector(config)
    detector.warmup()
    available = frame_numbers(scene)
    if not available:
        raise ValueError(f"Scene contains no frames: {scene}")
    indices = np.linspace(0, len(available) - 1, min(frame_count, len(available)), dtype=int)
    frames = [available[index] for index in indices]
    images = {frame: load_frame(frame, scene) for frame in frames}
    truth = {frame: load_annotations(frame, scene) for frame in frames}
    report = {
        "scope": "Crop recall on fully-contained objects; not full-frame AP, policy value, or independent instances.",
        "scene": scene, "frames": frames, "config": asdict(config),
        "checkpoint_sha256": checkpoint_hash(weights), "levels": {},
    }
    for level in (0, 1, 2):
        counts = {
            name: {"eligible": 0, "detected": 0, "localized_any_class": 0, "short_side_pixels": []}
            for name in OBJECT_CLASSES
        }
        proposals, views = 0, 0
        started = time.monotonic()
        for frame in frames:
            for cx, cy in grid_centers(level):
                view, region = render_view(images[frame], cx, cy, level)
                annotations = fully_contained(truth[frame], region)
                detections = detector.detect(view, level, region)
                correct = match_observations(annotations, detections)
                localized = match_observations(annotations, detections, require_class=False)
                proposals += len(detections)
                views += 1
                for index, annotation in enumerate(annotations):
                    row = counts[annotation["object_id"]]
                    row["eligible"] += 1
                    row["detected"] += index in correct
                    row["localized_any_class"] += index in localized
                    x1, y1, x2, y2 = annotation["bbox"]
                    row["short_side_pixels"].append(min(x2 - x1, y2 - y1) * 960 / (region[2] - region[0]))
        for row in counts.values():
            row["recall"] = row["detected"] / row["eligible"] if row["eligible"] else None
            row["localization_recall"] = row["localized_any_class"] / row["eligible"] if row["eligible"] else None
            sizes = row.pop("short_side_pixels")
            row["median_short_side_pixels"] = float(np.median(sizes)) if sizes else None
        report["levels"][level] = {
            "views": views, "proposals": proposals, "proposals_per_view": proposals / views,
            "elapsed_seconds": time.monotonic() - started, "per_class": counts,
        }
        print(f"L{level}: {views} views, {proposals} proposals", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="helsinki")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1600)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.001)
    arguments = parser.parse_args()
    diagnose(arguments.scene, arguments.weights, arguments.output, arguments.imgsz,
             arguments.frames, arguments.confidence)


if __name__ == "__main__":
    main()
