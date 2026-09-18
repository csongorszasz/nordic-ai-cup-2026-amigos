"""Offline closed-loop simulator for camera policies.

This is the tool the architecture research calls for: it renders evaluator-exact
views from the supplied 4K frames and replays the full loop - camera, detector,
world map, emission - without touching the competition endpoint. It lets a
policy be compared against another offline, and it can generate trajectories for
later imitation learning.

The detector is deliberately replaced by a *view-limited oracle*: it reports a
ground-truth object only when that object is actually inside the current view.
That is the honest upper bound of what a perfect detector could see, so the
score measures the architecture (did we look at and remember everything?), not
the detector.

    python src/offline/camera_simulator.py --policy belief_voi \
        --tracker world_map --frames 60
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import DroneFlybyConfig  # noqa: E402
from core.camera_policy import create_camera_policy  # noqa: E402
from core.interfaces import DetectionResult  # noqa: E402
from core.tracker import create_tracker  # noqa: E402
from dtos import (  # noqa: E402
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    DroneFlybyPredictRequestDto,
)
from local_evaluator import (  # noqa: E402
    SEQUENCE_ID,
    Camera,
    CameraRejection,
    build_request,
    frame_numbers,
    load_annotations,
    load_frame,
    render_view,
    score,
)


# The requests carry the local evaluator's sequence id, so policy and tracker
# state must be keyed to the same value.
SIMULATION_SEQUENCE_ID = SEQUENCE_ID


FrameProvider = Callable[[int], np.ndarray]
AnnotationProvider = Callable[[int], List[Dict]]


@dataclass
class SimulatorReport:
    """What one offline policy replay produced."""

    policy: str
    tracker: str
    frames: int
    illegal_moves: int
    coverage_l1: float
    coverage_l2: float
    instances_seen_l1: int
    instances_seen_l2: int
    total_instances: int
    map50: float
    per_class: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"policy={self.policy} tracker={self.tracker} frames={self.frames}\n"
            f"  illegal camera moves : {self.illegal_moves}\n"
            f"  L1 coverage          : {self.coverage_l1:.1%}\n"
            f"  L2 coverage          : {self.coverage_l2:.1%}\n"
            f"  instances seen L1    : {self.instances_seen_l1}/{self.total_instances}\n"
            f"  instances seen L2    : {self.instances_seen_l2}/{self.total_instances}\n"
            f"  COCO mAP@0.50        : {self.map50:.4f}"
        )


def view_oracle_detections(
    annotations: Sequence[Dict],
    source_region: Sequence[int],
    level: int,
    min_view_pixels: int = 0,
) -> List[DetectionResult]:
    """Ground-truth objects the current view can plausibly resolve.

    An object is reported when its centre lies inside the view and, if
    ``min_view_pixels`` is set, when its largest apparent dimension in the
    transmitted 960x540 image is at least that many pixels. The size gate is
    what makes the simulation honest: at Level 0 a 50-px object is under 13 px
    and a real detector cannot see it, so a policy that holds at Level 0 must
    not be credited with detecting it.

    Boxes are returned in both global-normalized and source-pixel coordinates,
    the shape the tracker's association step expects.
    """
    x1_region, y1_region, x2_region, y2_region = (float(v) for v in source_region)
    region_width = max(1.0, x2_region - x1_region)
    view_scale = 960.0 / region_width

    detections: List[DetectionResult] = []
    for annotation in annotations:
        x1, y1, x2, y2 = (float(c) for c in annotation["bbox"])
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        if not (x1_region <= center_x <= x2_region and y1_region <= center_y <= y2_region):
            continue
        if min_view_pixels > 0:
            apparent = max(x2 - x1, y2 - y1) * view_scale
            if apparent < min_view_pixels:
                continue
        detections.append(
            DetectionResult(
                class_name=annotation["object_id"],
                bbox_global=(
                    x1 / IMAGE_WIDTH,
                    y1 / IMAGE_HEIGHT,
                    x2 / IMAGE_WIDTH,
                    y2 / IMAGE_HEIGHT,
                ),
                confidence=0.99,
                zoom_level=level,
                source_pixel_bbox=(x1, y1, x2, y2),
            )
        )
    return detections


def _instance_key(annotation: Dict) -> Tuple[str, int, int]:
    x1, y1, _, _ = (float(c) for c in annotation["bbox"])
    return annotation["object_id"], int(round(x1 / 40.0)), int(round(y1 / 40.0))


def _view_gray(frame_image: np.ndarray, source_region: Sequence[int]) -> np.ndarray:
    x1, y1, x2, y2 = (int(v) for v in source_region)
    view = frame_image[y1:y2, x1:x2]
    view = cv2.resize(view, (960, 540), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)


def run_simulation(
    config: DroneFlybyConfig,
    scene: str,
    frames: Optional[Sequence[int]] = None,
    frame_provider: Optional[FrameProvider] = None,
    annotation_provider: Optional[AnnotationProvider] = None,
    score_fn: Optional[Callable[[str, Dict[int, List[Dict]]], Tuple[float, Dict[str, float]]]] = None,
    min_view_pixels: int = 0,
) -> SimulatorReport:
    """Replay one policy through a scene and score the whole loop."""
    frame_provider = frame_provider or (lambda frame: load_frame(frame, scene))
    annotation_provider = annotation_provider or (
        lambda frame: load_annotations(frame, scene)
    )
    score_fn = score_fn or score

    tracker = create_tracker(config)
    policy = create_camera_policy(config)
    tracker.reset(SIMULATION_SEQUENCE_ID)
    policy.reset(SIMULATION_SEQUENCE_ID)
    camera = Camera()

    frame_list = list(frames) if frames is not None else frame_numbers(scene)
    predictions: Dict[int, List[Dict]] = {}
    seen_l1: set = set()
    seen_l2: set = set()
    illegal_moves = 0
    feedback = None

    for frame_index, frame in enumerate(frame_list):
        frame_image = frame_provider(frame)
        annotations = annotation_provider(frame)
        source_region = camera.source_region

        for annotation in annotations:
            if camera.resolution_level >= 1:
                seen_l1.add(_instance_key(annotation))
            if camera.resolution_level >= 2:
                seen_l2.add(_instance_key(annotation))

        detections = view_oracle_detections(
            annotations, source_region, camera.resolution_level, min_view_pixels
        )
        gray = _view_gray(frame_image, source_region)
        emitted = tracker.update(
            detections=detections,
            zoom_level=camera.resolution_level,
            source_region_xyxy=source_region,
            frame_index=frame_index,
            l0_image_gray=gray,
        )
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

        encoded = render_view(frame_image, camera)
        request = DroneFlybyPredictRequestDto(
            **build_request(frame, frame_index, camera, encoded, feedback)
        )
        next_view = policy.decide_next_view(request, tracker.get_summary())
        feedback = None
        if next_view is not None:
            try:
                camera.apply(
                    next_view.resolution_level,
                    next_view.center_x,
                    next_view.center_y,
                )
            except CameraRejection as rejection:
                illegal_moves += 1
                feedback = {
                    "frame": frame,
                    "requested_view": next_view.model_dump(),
                    "reason": str(rejection),
                }

    map50, per_class = score_fn(scene, predictions)
    coverage_l1 = _policy_coverage(policy, 1)
    coverage_l2 = _policy_coverage(policy, 2)
    total_instances = len({_instance_key(a) for f in frame_list for a in annotation_provider(f)})

    return SimulatorReport(
        policy=config.POLICY_TYPE,
        tracker=config.TRACKER_TYPE,
        frames=len(frame_list),
        illegal_moves=illegal_moves,
        coverage_l1=coverage_l1,
        coverage_l2=coverage_l2,
        instances_seen_l1=len(seen_l1),
        instances_seen_l2=len(seen_l2),
        total_instances=total_instances,
        map50=map50,
        per_class=per_class,
    )


def _policy_coverage(policy, level: int) -> float:
    getter = getattr(policy, "coverage_fraction", None)
    if getter is None:
        return 0.0
    return float(getter(SIMULATION_SEQUENCE_ID, level))


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline camera-policy simulator.")
    parser.add_argument("--scene", default="helsinki")
    parser.add_argument("--policy", default="belief_voi")
    parser.add_argument("--tracker", default="world_map")
    parser.add_argument("--ego-motion", default="phase_correlation")
    parser.add_argument("--frames", type=int, default=None, help="Limit the replay length.")
    parser.add_argument(
        "--min-view-pixels",
        type=int,
        default=0,
        help="Detectability gate: an object is only 'seen' if its largest "
        "apparent dimension in the transmitted image is at least this many "
        "pixels. Use ~16 to model a real detector; 0 credits even L0.",
    )
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    arguments = parser.parse_args()

    config = DroneFlybyConfig(
        POLICY_TYPE=arguments.policy,
        TRACKER_TYPE=arguments.tracker,
        EGO_MOTION_METHOD=arguments.ego_motion,
    )
    frame_list = frame_numbers(arguments.scene)
    if arguments.frames is not None:
        frame_list = frame_list[: arguments.frames]

    report = run_simulation(
        config,
        arguments.scene,
        frames=frame_list,
        min_view_pixels=arguments.min_view_pixels,
    )
    if arguments.json:
        import json

        print(json.dumps(asdict(report), indent=2))
    else:
        print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
