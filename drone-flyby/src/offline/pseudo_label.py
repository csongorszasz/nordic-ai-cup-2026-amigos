"""Offline pseudo-label generation for recorded validation frames.

Turns a directory of recorded images into YOLO-format label files. The default
detector is the production YOLO backend, built from the normal configuration,
so pseudo-labels come from the same model the server serves. A detector can be
injected directly for testing.

The template-bank backend is deliberately not the default: it is built from 4K
crops and finds nothing on the 960x540 views the evaluator transmits, which
silently produced empty labels.

    python src/offline/pseudo_label.py --images-dir recorded/.../images --labels-dir out/labels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2

from config import DroneFlybyConfig
from core.detector import create_detector
from core.interfaces import BaseDetector
from dtos import OBJECT_CLASSES
from offline.dataset_provenance import assert_training_source

CLASS_INDEX = {name: index for index, name in enumerate(OBJECT_CLASSES)}


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def _iter_images(images_dir: Path) -> Iterable[Path]:
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _xyxy_to_yolo(bbox: Tuple[float, float, float, float], width: int, height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    cx = x1 + box_w / 2.0
    cy = y1 + box_h / 2.0
    return cx / width, cy / height, box_w / width, box_h / height


def pseudo_label_directory(
    images_dir: Path,
    labels_dir: Path,
    detector: Optional[BaseDetector] = None,
    config: Optional[DroneFlybyConfig] = None,
) -> Dict[str, int]:
    """Generate YOLO labels for every image in a directory."""
    assert_training_source(images_dir)
    assert_training_source(labels_dir)
    if detector is None:
        detector = create_detector(config or DroneFlybyConfig())
    detector.warmup()
    labels_dir.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {}
    for image_path in _iter_images(images_dir):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        detections = detector.detect(image, zoom_level=0, source_region_xyxy=(0, 0, image.shape[1], image.shape[0]))
        label_path = labels_dir / f"{image_path.stem}.txt"
        with open(label_path, "w", encoding="utf-8") as handle:
            for det in detections:
                if det.class_name not in CLASS_INDEX:
                    continue
                x1, y1, x2, y2 = det.source_pixel_bbox
                cx, cy, w, h = _xyxy_to_yolo((x1, y1, x2, y2), image.shape[1], image.shape[0])
                if w <= 0 or h <= 0:
                    continue
                # YOLO label files require numeric class IDs, not names.
                handle.write(f"{CLASS_INDEX[det.class_name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                counts[det.class_name] = counts.get(det.class_name, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate pseudo-labels for recorded validation images.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory containing images.")
    parser.add_argument("--labels-dir", type=Path, required=True, help="Output directory for YOLO labels.")
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional summary JSON output.")
    parser.add_argument("--detector-type", default="yolo_standard", help="Detector backend to pseudo-label with.")
    parser.add_argument("--weights", type=Path, default=None, help="Override the detector weights path.")
    arguments = parser.parse_args()

    config = DroneFlybyConfig(
        DETECTOR_TYPE=arguments.detector_type,
        YOLO_WEIGHTS_PATH=arguments.weights if arguments.weights else DroneFlybyConfig.YOLO_WEIGHTS_PATH,
    )
    counts = pseudo_label_directory(arguments.images_dir, arguments.labels_dir, config=config)
    if arguments.summary_json is not None:
        arguments.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with open(arguments.summary_json, "w", encoding="utf-8") as handle:
            json.dump({"counts": counts}, handle, indent=2)

    total = sum(counts.values())
    print(f"Wrote {total} pseudo-labels to {arguments.labels_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
