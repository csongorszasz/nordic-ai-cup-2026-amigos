"""Build an exact-view YOLO dataset with copy-paste augmentation.

The supplied scene has 16 object instances across 25 frames, which is nowhere
near enough to train a small-object detector. This builder renders the same
evaluator-exact views as ``build_exact_view_dataset`` and then synthesises extra
training samples by pasting annotated object crops from the scene onto other
views.

Copy-paste is the standard answer to class scarcity in detection (and a core
component of weak/semi-supervised pipelines such as Point-Teaching), and it
matters especially here because the score is a **macro average**: rare classes
(``jammer``, ``mine_roller``, ``hangar``, ``medium_plane``) each carry 1/16 of
the score, so starving them is expensive.

The module is split so the augmentation is a pure, testable function:

* :func:`crop_objects_from_view` collects object crops from rendered views;
* :func:`paste_augment` pastes crops into a view and returns new labels;
* :func:`build_augmented_dataset` wires them into a YOLO dataset tree.

    python src/offline/build_augmented_dataset.py --copy-paste-per-view 2 \
        --copies-per-source 3 --output-dir training_artifacts/augmented
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dtos import OBJECT_CLASSES, TRANSMITTED_VIEW_SIZE  # noqa: E402
from offline.build_exact_view_dataset import (  # noqa: E402
    _annotation_to_yolo,
    grid_centers,
    render_view,
)
from utils import DEFAULT_SCENE, frame_numbers, load_annotations, load_frame, scene_directory  # noqa: E402
from offline.dataset_provenance import assert_training_source, mark_training_dataset, split_source_frames


VIEW_WIDTH, VIEW_HEIGHT = TRANSMITTED_VIEW_SIZE


@dataclass(slots=True)
class ObjectCrop:
    """One annotated object cropped from a rendered view."""

    class_index: int
    image: np.ndarray
    width: int
    height: int
    object_box: Optional[Tuple[float, float, float, float]] = None
    zoom_level: int = -1
    source_frame: int = -1


@dataclass
class AugmentationStats:
    """Counters for one dataset build."""

    train: int = 0
    val: int = 0
    empty_skipped: int = 0
    pasted_objects: int = 0
    bank_size: int = 0
    per_class: Dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Pure augmentation helpers
# --------------------------------------------------------------------------- #

def _feather_mask(height: int, width: int, border: int) -> np.ndarray:
    """A soft rectangular alpha mask so pasted crops do not show hard seams."""
    border = max(1, min(border, max(1, height // 2), max(1, width // 2)))
    ramp_y = np.minimum(np.arange(height), np.arange(height)[::-1]).astype(np.float32) / border
    ramp_x = np.minimum(np.arange(width), np.arange(width)[::-1]).astype(np.float32) / border
    return np.clip(np.minimum(ramp_y[:, None], ramp_x[None, :]), 0.0, 1.0)


def _iou_pixels(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _pixel_box(cx: float, cy: float, width: float, height: float) -> Tuple[float, float, float, float]:
    return (
        (cx - width / 2.0) * VIEW_WIDTH,
        (cy - height / 2.0) * VIEW_HEIGHT,
        (cx + width / 2.0) * VIEW_WIDTH,
        (cy + height / 2.0) * VIEW_HEIGHT,
    )


def crop_objects_from_view(
    view: np.ndarray,
    label_rows: Sequence[Tuple[int, float, float, float, float]],
    margin: float = 0.15,
    minimum_pixels: int = 4,
    zoom_level: int = -1,
    source_frame: int = -1,
) -> List[ObjectCrop]:
    """Crop each labelled object out of a rendered view.

    ``label_rows`` are ``(class_index, cx, cy, w, h)`` normalised rows. The crop
    includes a margin of surrounding context, which makes the pasted object look
    natural rather than cut out.
    """
    crops: List[ObjectCrop] = []
    for class_index, cx, cy, width, height in label_rows:
        x1, y1, x2, y2 = _pixel_box(cx, cy, width, height)
        tight_box = (x1, y1, x2, y2)
        pad_x = (x2 - x1) * margin
        pad_y = (y2 - y1) * margin
        x1 = max(0, math.floor(x1 - pad_x))
        y1 = max(0, math.floor(y1 - pad_y))
        x2 = min(VIEW_WIDTH, math.ceil(x2 + pad_x))
        y2 = min(VIEW_HEIGHT, math.ceil(y2 + pad_y))
        if x2 - x1 < minimum_pixels or y2 - y1 < minimum_pixels:
            continue
        crops.append(
            ObjectCrop(
                class_index=int(class_index),
                image=view[y1:y2, x1:x2].copy(),
                width=x2 - x1,
                height=y2 - y1,
                object_box=(tight_box[0] - x1, tight_box[1] - y1,
                            tight_box[2] - x1, tight_box[3] - y1),
                zoom_level=zoom_level,
                source_frame=source_frame,
            )
        )
    return crops


def rotate_object_crop(crop: ObjectCrop, quarter_turns: int) -> ObjectCrop:
    """Rotate pixels and the tight object box together, without interpolation."""
    turns = quarter_turns % 4
    image = np.rot90(crop.image, turns).copy()
    box = crop.object_box or (0.0, 0.0, float(crop.width), float(crop.height))
    width, height = crop.width, crop.height
    for _ in range(turns):
        x1, y1, x2, y2 = box
        box = (y1, width - x2, y2, width - x1)
        width, height = height, width
    return ObjectCrop(
        class_index=crop.class_index, image=image, width=width, height=height,
        object_box=box, zoom_level=crop.zoom_level, source_frame=crop.source_frame,
    )


def paste_augment(
    view: np.ndarray,
    label_rows: Sequence[Tuple[int, float, float, float, float]],
    bank: Sequence[ObjectCrop],
    rng: np.random.Generator,
    max_pastes: int = 1,
    scale_range: Tuple[float, float] = (0.7, 1.3),
    overlap_tolerance: float = 0.10,
    attempts: int = 12,
    zoom_level: Optional[int] = None,
    rotate_pastes: bool = False,
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    """Paste up to ``max_pastes`` crops onto a view and return it plus labels.

    A paste is accepted only when it fits fully inside the view and does not
    overlap an existing box by more than ``overlap_tolerance`` IoU. Duplicates
    of the same class near an existing instance are the main failure mode of
    naive copy-paste, so overlap is checked against *all* classes.
    """
    bank = [crop for crop in bank if zoom_level is None or crop.zoom_level in (-1, zoom_level)]
    if not bank or max_pastes <= 0:
        return view.copy(), list(label_rows)

    canvas = view.copy()
    labels = list(label_rows)
    boxes = [_pixel_box(cx, cy, w, h) for _, cx, cy, w, h in labels]
    by_class: Dict[int, List[ObjectCrop]] = {}
    for crop in bank:
        by_class.setdefault(crop.class_index, []).append(crop)
    classes = sorted(by_class)

    for _ in range(max_pastes):
        class_crops = by_class[classes[int(rng.integers(len(classes)))]]
        crop = class_crops[int(rng.integers(len(class_crops)))]
        if rotate_pastes:
            crop = rotate_object_crop(crop, int(rng.integers(4)))
        scale = float(rng.uniform(*scale_range))
        target_w = max(3, int(round(crop.width * scale)))
        target_h = max(3, int(round(crop.height * scale)))
        if target_w >= VIEW_WIDTH or target_h >= VIEW_HEIGHT:
            continue

        placed = False
        for _attempt in range(attempts):
            x = int(rng.integers(0, VIEW_WIDTH - target_w + 1))
            y = int(rng.integers(0, VIEW_HEIGHT - target_h + 1))
            candidate = (float(x), float(y), float(x + target_w), float(y + target_h))
            if any(
                max(0, min(candidate[2], box[2]) - max(candidate[0], box[0]))
                * max(0, min(candidate[3], box[3]) - max(candidate[1], box[1]))
                / max(1.0, (box[2] - box[0]) * (box[3] - box[1])) > overlap_tolerance
                or _iou_pixels(candidate, box) > overlap_tolerance
                for box in boxes
            ):
                continue

            resized = cv2.resize(
                crop.image, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
            alpha = _feather_mask(target_h, target_w, max(1, target_w // 8))
            region = canvas[y : y + target_h, x : x + target_w].astype(np.float32)
            blended = region * (1.0 - alpha[..., None]) + resized.astype(np.float32) * alpha[..., None]
            canvas[y : y + target_h, x : x + target_w] = np.clip(blended, 0, 255).astype(np.uint8)

            tight = crop.object_box or (0.0, 0.0, float(crop.width), float(crop.height))
            target_box = (
                x + tight[0] * target_w / crop.width,
                y + tight[1] * target_h / crop.height,
                x + tight[2] * target_w / crop.width,
                y + tight[3] * target_h / crop.height,
            )
            boxes.append(target_box)
            labels.append(
                (
                    crop.class_index,
                    (target_box[0] + target_box[2]) / 2.0 / VIEW_WIDTH,
                    (target_box[1] + target_box[3]) / 2.0 / VIEW_HEIGHT,
                    (target_box[2] - target_box[0]) / VIEW_WIDTH,
                    (target_box[3] - target_box[1]) / VIEW_HEIGHT,
                )
            )
            placed = True
            break
        if not placed:
            continue

    return canvas, labels


# --------------------------------------------------------------------------- #
# Dataset build
# --------------------------------------------------------------------------- #

def _annotation_rows(annotations: Sequence[Dict], source_region: Sequence[int]):
    """Convert source annotations to view-normalised YOLO rows."""
    return [row for annotation in annotations
            if (row := _annotation_to_yolo(annotation, source_region)) is not None]


def _write_sample(images_dir: Path, labels_dir: Path, stem: str, view: np.ndarray, rows) -> None:
    encoded, buffer = cv2.imencode(".png", view, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not encoded:
        raise RuntimeError(f"Could not encode {stem}")
    (images_dir / f"{stem}.png").write_bytes(buffer.tobytes())
    (labels_dir / f"{stem}.txt").write_text(
        "\n".join(
            f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cls, cx, cy, w, h in rows
        )
    )


def build_augmented_dataset(
    scene: str,
    output_dir: Path,
    levels: Sequence[int] = (0, 1, 2),
    val_fraction: float = 0.15,
    step_fraction: float = 0.5,
    copy_paste_per_view: int = 0,
    copies_per_source: int = 1,
    max_bank_per_class: int = 40,
    seed: int = 0,
    frames: Optional[Sequence[int]] = None,
    neg_stride: int = 5,
    rotate_pastes: bool = False,
) -> Path:
    """Render exact views, apply copy-paste augmentation, and write a dataset.

    Returns the path to the generated ``data.yaml``.
    """
    rng = np.random.default_rng(seed)
    assert_training_source(scene_directory(scene))
    frame_list = list(frames) if frames is not None else frame_numbers(scene)
    source_splits = split_source_frames(frame_list, val_fraction)
    dataset_dir = output_dir / "drone_flyby_augmented"
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

    # Pass 1: build the object bank from rendered views.
    bank: List[ObjectCrop] = []
    bank_groups: Dict[Tuple[int, int], List[ObjectCrop]] = {}
    bank_counts: Dict[Tuple[int, int], int] = {}

    for frame in frame_list:
        if source_splits[frame] != "train":
            continue
        image = load_frame(frame, scene)
        annotations = load_annotations(frame, scene)
        for level in levels:
            for center_x, center_y in grids[level]:
                view, source_region = render_view(image, center_x, center_y, level)
                rows = _annotation_rows(annotations, source_region)
                for crop in crop_objects_from_view(view, rows, zoom_level=level, source_frame=frame):
                    key = (crop.class_index, level)
                    group = bank_groups.setdefault(key, [])
                    bank_counts[key] = bank_counts.get(key, 0) + 1
                    if len(group) < max_bank_per_class:
                        group.append(crop)
                    else:
                        index = int(rng.integers(bank_counts[key]))
                        if index < max_bank_per_class:
                            group[index] = crop
    bank = [crop for group in bank_groups.values() for crop in group]

    if not bank:
        raise RuntimeError(
            "Object bank is empty; the scene produced no labelled crops to paste."
        )

    # Pass 2: write the dataset, optionally with augmented copies.
    stats = AugmentationStats(bank_size=len(bank))
    manifest = []
    for sample_index, (frame, level, center_x, center_y, view, rows) in enumerate(
        _render_samples(scene, frame_list, levels, grids)
    ):
        stem = f"L{level}_f{frame:06d}_x{center_x}_y{center_y}"
        if not rows and neg_stride > 0 and sample_index % neg_stride:
            stats.empty_skipped += 1
            continue

        split = source_splits[frame]
        _write_sample(dirs[split]["images"], dirs[split]["labels"], stem, view, rows)
        stats.train += split == "train"
        stats.val += split == "val"
        manifest.append({"stem": stem, "split": split, "frame": frame,
                         "level": level, "scene": scene, "objects": len(rows)})
        for cls, *_ in rows:
            name = OBJECT_CLASSES[cls]
            stats.per_class[name] = stats.per_class.get(name, 0) + 1

        if split == "train" and copy_paste_per_view > 0:
            for copy_index in range(copies_per_source):
                augmented, augmented_rows = paste_augment(
                    view, rows, bank, rng, max_pastes=copy_paste_per_view,
                    zoom_level=level, rotate_pastes=rotate_pastes,
                )
                if len(augmented_rows) == len(rows):
                    continue
                augmented_stem = f"{stem}_cp{copy_index}"
                _write_sample(dirs["train"]["images"], dirs["train"]["labels"], augmented_stem, augmented, augmented_rows)
                stats.train += 1
                manifest.append({"stem": augmented_stem, "split": "train", "frame": frame,
                                 "level": level, "scene": scene, "objects": len(augmented_rows),
                                 "augmentation": "copy-paste"})
                stats.pasted_objects += len(augmented_rows) - len(rows)
                for cls, *_ in augmented_rows[len(rows):]:
                    name = OBJECT_CLASSES[cls]
                    stats.per_class[name] = stats.per_class.get(name, 0) + 1

    data_yaml = dataset_dir / "drone_flyby_augmented.yaml"
    data_yaml.write_text(
        "path: {}\n".format(dataset_dir.resolve().as_posix())
        + "train: images/train\n"
        + "val: images/val\n"
        + "names:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(OBJECT_CLASSES))
    )
    (dataset_dir / "augmentation_stats.json").write_text(
        json.dumps(stats.__dict__, indent=2, sort_keys=True)
    )
    (dataset_dir / "manifest.json").write_text(json.dumps({
        "samples": manifest,
        "bank": [{"source_frame": crop.source_frame, "level": crop.zoom_level,
                  "class_index": crop.class_index} for crop in bank],
        "rotate_pastes": rotate_pastes,
    }, indent=2), encoding="utf-8")
    print(
        "Augmented exact-view dataset written to {} (train={train}, val={val}, "
        "empty-skipped={empty_skipped}, pasted={pasted_objects}, bank={bank_size})".format(
            dataset_dir, **stats.__dict__
        )
    )
    return data_yaml


def _render_samples(scene, frames, levels, grids):
    for frame in frames:
        image = load_frame(frame, scene)
        annotations = load_annotations(frame, scene)
        for level in levels:
            for center_x, center_y in grids[level]:
                view, source_region = render_view(image, center_x, center_y, level)
                yield frame, level, center_x, center_y, view, _annotation_rows(annotations, source_region)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a copy-paste augmented exact-view dataset.")
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--output-dir", type=Path, default=SRC_ROOT.parent / "training_artifacts")
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--step-fraction", type=float, default=0.5)
    parser.add_argument("--copy-paste-per-view", type=int, default=0)
    parser.add_argument("--copies-per-source", type=int, default=1)
    parser.add_argument("--max-bank-per-class", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rotate-pastes", action="store_true", help="Apply label-exact random quarter turns to pasted objects.")
    arguments = parser.parse_args()

    data_yaml = build_augmented_dataset(
        scene=arguments.scene,
        output_dir=arguments.output_dir,
        levels=arguments.levels,
        val_fraction=arguments.val_fraction,
        step_fraction=arguments.step_fraction,
        copy_paste_per_view=arguments.copy_paste_per_view,
        copies_per_source=arguments.copies_per_source,
        max_bank_per_class=arguments.max_bank_per_class,
        seed=arguments.seed,
        rotate_pastes=arguments.rotate_pastes,
    )
    print(f"Data YAML: {data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
