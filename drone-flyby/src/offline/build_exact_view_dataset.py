"""Build a YOLO dataset from evaluator-exact resolution-level views.

The evaluator never sends a 4K frame. Every request carries a 960x540 PNG of
whatever the camera is pointed at, produced by cropping the source region and
downsampling with ``INTER_AREA``. Training on full-resolution frames resized
generically therefore optimises a geometry the wire protocol never uses.

This script renders the views the evaluator would render for a grid of legal
camera positions at each resolution level, converts the source-frame ground
truth into view-normalised YOLO labels, and writes a train/val split.

    python src/offline/build_exact_view_dataset.py \
        --scene helsinki \
        --output-dir training_artifacts/exact_views

The supplied Helsinki scene is a single sequence: every split shares its
objects, so validation here is a harness, not a generalization estimate. Use
recorded validation passes for the latter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dtos import (  # noqa: E402
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    OBJECT_CLASSES,
    SOURCE_REGION_SIZES,
    TRANSMITTED_VIEW_SIZE,
)
from utils import (  # noqa: E402
    DEFAULT_SCENE,
    frame_numbers,
    load_annotations,
    load_frame,
    source_bbox_to_view,
    scene_directory,
)
from offline.dataset_provenance import assert_training_source, mark_training_dataset, split_source_frames

# A label is kept only if at least this fraction of the object's area survives
# the crop. Slivers produce noisy boxes that hurt more than they help.
MIN_VISIBLE_FRACTION = 0.30

CLASS_INDEX = {name: index for index, name in enumerate(OBJECT_CLASSES)}


def _positions(minimum: int, maximum: int, step: int) -> List[int]:
    """Return inclusive grid positions from minimum to maximum."""
    if maximum <= minimum:
        return [minimum]
    positions = list(range(minimum, maximum + 1, step))
    if positions[-1] != maximum:
        positions.append(maximum)
    return positions


def grid_centers(level: int, step_fraction: float = 0.5) -> List[Tuple[int, int]]:
    """Return legal camera centers covering the full frame at one level.

    ``step_fraction`` is the spacing as a fraction of the region size, so 0.5
    gives 50% overlap between neighbouring views.
    """
    width, height = SOURCE_REGION_SIZES[level]
    min_x, max_x = width // 2, IMAGE_WIDTH - width // 2
    min_y, max_y = height // 2, IMAGE_HEIGHT - height // 2
    step_x = max(1, int(width * step_fraction))
    step_y = max(1, int(height * step_fraction))
    xs = _positions(min_x, max_x, step_x) if level > 0 else [IMAGE_WIDTH // 2]
    ys = _positions(min_y, max_y, step_y) if level > 0 else [IMAGE_HEIGHT // 2]
    return [(x, y) for y in ys for x in xs]


def render_view(frame_image, center_x: int, center_y: int, level: int):
    """Crop and downsample a view exactly as ``local_evaluator.render_view``."""
    width, height = SOURCE_REGION_SIZES[level]
    x1 = center_x - width // 2
    y1 = center_y - height // 2
    source_region = (x1, y1, x1 + width, y1 + height)
    view = frame_image[y1:y1 + height, x1:x1 + width]
    if (view.shape[1], view.shape[0]) != TRANSMITTED_VIEW_SIZE:
        view = cv2.resize(view, TRANSMITTED_VIEW_SIZE, interpolation=cv2.INTER_AREA)
    return view, source_region


def _annotation_to_yolo(annotation: Dict, source_region: Sequence[int]) -> Optional[Tuple[int, float, float, float, float]]:
    """Convert one source-frame annotation into a view-normalised YOLO row."""
    object_id = annotation["object_id"]
    if object_id not in CLASS_INDEX:
        return None

    x1, y1, x2, y2 = source_bbox_to_view(annotation["bbox"], source_region)
    clip_x1, clip_y1 = max(0.0, x1), max(0.0, y1)
    clip_x2, clip_y2 = min(1.0, x2), min(1.0, y2)
    if clip_x2 <= clip_x1 or clip_y2 <= clip_y1:
        return None

    full_area = (x2 - x1) * (y2 - y1)
    visible_area = (clip_x2 - clip_x1) * (clip_y2 - clip_y1)
    if full_area <= 0.0 or visible_area / full_area < MIN_VISIBLE_FRACTION:
        return None

    width = clip_x2 - clip_x1
    height = clip_y2 - clip_y1
    center_x = (clip_x1 + clip_x2) / 2.0
    center_y = (clip_y1 + clip_y2) / 2.0
    return (CLASS_INDEX[object_id], center_x, center_y, width, height)


def _iter_samples(
    scene: str,
    levels: Sequence[int],
    grids: Dict[int, List[Tuple[int, int]]],
) -> Iterable[Tuple[int, int, int, int, List[str], Optional[int]]]:
    """Yield (level, frame, center_x, center_y, label_lines, image) lazily."""
    for frame in frame_numbers(scene):
        image = load_frame(frame, scene)
        annotations = load_annotations(frame, scene)
        for level in levels:
            for center_x, center_y in grids[level]:
                view, source_region = render_view(image, center_x, center_y, level)
                labels: List[str] = []
                for annotation in annotations:
                    row = _annotation_to_yolo(annotation, source_region)
                    if row is None:
                        continue
                    class_index, cx, cy, width, height = row
                    labels.append(f"{class_index} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")
                # Encode the rendered view now; the caller decides to keep it.
                encoded, buffer = cv2.imencode(".png", view, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                if not encoded:
                    raise RuntimeError(f"Could not encode frame {frame} level {level}")
                yield level, frame, center_x, center_y, labels, buffer


def build_exact_view_dataset(
    scene: str,
    output_dir: Path,
    levels: Sequence[int] = (0, 1, 2),
    val_fraction: float = 0.15,
    neg_stride: int = 5,
    step_fraction: float = 0.5,
) -> Path:
    """Render exact views and write a YOLO dataset. Returns the data.yaml path."""
    assert_training_source(scene_directory(scene))
    source_splits = split_source_frames(frame_numbers(scene), val_fraction)
    dataset_dir = output_dir / "drone_flyby_exact"
    mark_training_dataset(dataset_dir)
    dirs = {
        split: {
            "images": dataset_dir / "images" / split,
            "labels": dataset_dir / "labels" / split,
        }
        for split in ("train", "val")
    }
    for split_dirs in dirs.values():
        for path in split_dirs.values():
            path.mkdir(parents=True, exist_ok=True)

    grids = {level: grid_centers(level, step_fraction) for level in levels}

    manifest: List[Dict] = []
    counts = {"train": 0, "val": 0, "empty": 0}
    sample_index = 0

    for level, frame, center_x, center_y, labels, buffer in _iter_samples(scene, levels, grids):
        is_empty = not labels
        if is_empty:
            counts["empty"] += 1
            if neg_stride > 0 and sample_index % neg_stride != 0:
                sample_index += 1
                continue

        split = source_splits[frame]
        stem = f"L{level}_f{frame:06d}_x{center_x}_y{center_y}"
        image_path = dirs[split]["images"] / f"{stem}.png"
        label_path = dirs[split]["labels"] / f"{stem}.txt"
        image_path.write_bytes(buffer.tobytes())
        label_path.write_text("\n".join(labels))
        counts[split] += 1
        manifest.append(
            {
                "split": split,
                "level": level,
                "frame": frame,
                "center": [center_x, center_y],
                "objects": len(labels),
                "scene": scene,
                "source_group": f"{scene}:{frame}",
            }
        )
        sample_index += 1

    data_yaml = dataset_dir / "drone_flyby_exact.yaml"
    data_yaml.write_text(
        "path: {}\n".format(dataset_dir.resolve().as_posix())
        + "train: images/train\n"
        + "val: images/val\n"
        + "names:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(OBJECT_CLASSES))
    )
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        "Exact-view dataset written to {} (train={train}, val={val}, "
        "empty-skipped={empty})".format(dataset_dir, **counts)
    )
    return data_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an exact-view YOLO dataset.")
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--output-dir", type=Path, default=SRC_ROOT.parent / "training_artifacts")
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--neg-stride",
        type=int,
        default=5,
        help="Keep every Nth empty view; 0 keeps all negatives.",
    )
    parser.add_argument("--step-fraction", type=float, default=0.5)
    arguments = parser.parse_args()

    data_yaml = build_exact_view_dataset(
        scene=arguments.scene,
        output_dir=arguments.output_dir,
        levels=arguments.levels,
        val_fraction=arguments.val_fraction,
        neg_stride=arguments.neg_stride,
        step_fraction=arguments.step_fraction,
    )
    print(f"Data YAML: {data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
