"""Oracle-visible memory gate.

Feed the ground truth in as perfect detections at fixed Level 0 and score what
the memory emits. This isolates the tracker from the detector: if a sound memory
cannot reach ~1.0 mAP@0.50 on oracle observations, integrating a real detector
and a camera policy on top of it cannot help.

    python src/offline/oracle_memory_benchmark.py --tracker world_map --target 0.95

Exits non-zero when the score is below the target, so it can be used as a gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import DroneFlybyConfig  # noqa: E402
from core.interfaces import DetectionResult  # noqa: E402
from core.tracker import create_tracker  # noqa: E402
from dtos import IMAGE_HEIGHT, IMAGE_WIDTH  # noqa: E402
from local_evaluator import frame_numbers, load_annotations, load_frame, score  # noqa: E402
from utils import DEFAULT_SCENE  # noqa: E402

REGION = (0, 0, IMAGE_WIDTH, IMAGE_HEIGHT)


def _oracle_detections(frame: int, scene: str):
    detections = []
    for annotation in load_annotations(frame, scene):
        x1, y1, x2, y2 = (float(c) for c in annotation["bbox"])
        detections.append(
            DetectionResult(
                class_name=annotation["object_id"],
                bbox_global=(x1 / IMAGE_WIDTH, y1 / IMAGE_HEIGHT, x2 / IMAGE_WIDTH, y2 / IMAGE_HEIGHT),
                confidence=0.99,
                zoom_level=0,
                source_pixel_bbox=(x1, y1, x2, y2),
            )
        )
    return detections


def run(scene: str, tracker_type: str) -> float:
    config = DroneFlybyConfig(DETECTOR_TYPE="yolo_standard", TRACKER_TYPE=tracker_type)
    tracker = create_tracker(config)
    tracker.reset("oracle_memory")

    predictions = {}
    for frame_index, frame in enumerate(frame_numbers(scene)):
        image = load_frame(frame, scene)
        gray = cv2.cvtColor(
            cv2.resize(image, (IMAGE_WIDTH // 4, IMAGE_HEIGHT // 4), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        emitted = tracker.update(_oracle_detections(frame, scene), 0, REGION, frame_index, gray)
        predictions[frame] = [
            {
                "object_id": p.object_id,
                "bbox": (
                    p.bbox[0] * IMAGE_WIDTH,
                    p.bbox[1] * IMAGE_HEIGHT,
                    p.bbox[2] * IMAGE_WIDTH,
                    p.bbox[3] * IMAGE_HEIGHT,
                ),
                "confidence": float(p.confidence),
            }
            for p in emitted
        ]

    mAP, per_class = score(scene, predictions)
    print(f"Oracle-visible memory: tracker={tracker_type} scene={scene}")
    print("AP@0.50 by class")
    for name, value in sorted(per_class.items(), key=lambda item: item[1]):
        print(f"  {name:16s} {value:.3f}")
    print(f"COCO mAP@0.50: {mAP:.4f}")
    return mAP


def main() -> int:
    parser = argparse.ArgumentParser(description="Oracle-visible memory gate.")
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--tracker", default="world_map", choices=["world_map", "passthrough"])
    parser.add_argument("--target", type=float, default=0.95)
    arguments = parser.parse_args()

    mAP = run(arguments.scene, arguments.tracker)
    if mAP >= arguments.target:
        print(f"PASS (>= {arguments.target:.2f})")
        return 0
    print(f"FAIL (< {arguments.target:.2f})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
