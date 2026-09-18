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
    MIN_VISIBLE_FRACTION,
    grid_centers,
    render_view,
)
from utils import DEFAULT_SCENE, frame_numbers, load_annotations, load_frame, source_bbox_to_view  # noqa: E402


CLASS_INDEX = {name: index for index, name in enumerate(OBJECT_CLASSES)}
VIEW_WIDTH, VIEW_HEIGHT = TRANSMITTED_VIEW_SIZE


@dataclass(slots=True)
class ObjectCrop:
    """One annotated object cropped from a rendered view."""

    class_index: int
    image: np.ndarray
    width: int
    height: int


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
) -> List[ObjectCrop]:
    """Crop each labelled object out of a rendered view.

    ``label_rows`` are ``(class_index, cx, cy, w, h)`` normalised rows. The crop
    includes a margin of surrounding context, which makes the pasted object look
    natural rather than cut out.
    """
    crops: List[ObjectCrop] = []
    for class_index, cx, cy, width, height in label_rows:
        x1, y1, x2, y2 = _pixel_box(cx, cy, width, height)
        pad_x = (x2 - x1) * margin
        pad_y = (y2 - y1) * margin
        x1 = max(0, int(round(x1 - pad_x)))
        y1 = max(0, int(round(y1 - pad_y)))
        x2 = min(VIEW_WIDTH, int(round(x2 + pad_x)))
        y2 = min(VIEW_HEIGHT, int(round(y2 + pad_y)))
        if x2 - x1 < minimum_pixels or y2 - y1 < minimum_pixels:
            continue
        crops.append(
            ObjectCrop(
                class_index=int(class_index),
                image=view[y1:y2, x1:x2].copy(),
                width=x2 - x1,
                height=y2 - y1,
            )
        )
    return crops


def paste_augment(
    view: np.ndarray,
    label_rows: Sequence[Tuple[int, float, float, float, float]],
    bank: Sequence[ObjectCrop],
    rng: np.random.Generator,
    max_pastes: int = 1,
    scale_range: Tuple[float, float] = (0.7, 1.3),
    overlap_tolerance: float = 0.10,
    attempts: int = 12,
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    """Paste up to ``max_pastes`` crops onto a view and return it plus labels.

    A paste is accepted only when it fits fully inside the view and does not
    overlap an existing box by more than ``overlap_tolerance`` IoU. Duplicates
    of the same class near an existing instance are the main failure mode of
    naive copy-paste, so overlap is checked against *all* classes.
    """
    if not bank or max_pastes <= 0:
        return view.copy(), list(label_rows)

    canvas = view.copy()
    labels = list(label_rows)
    boxes = [_pixel_box(cx, cy, w, h) for _, cx, cy, w, h in labels]

    for _ in range(max_pastes):
        crop = bank[int(rng.integers(0, len(bank)))]
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
            if any(_iou_pixels(candidate, box) > overlap_tolerance for box in boxes):
                continue

            resized = cv2.resize(
                crop.image, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
            alpha = _feather_mask(target_h, target_w, max(1, target_w // 8))
            region = canvas[y : y + target_h, x : x + target_w].astype(np.float32)
            blended = region * (1.0 - alpha[..., None]) + resized.astype(np.float32) * alpha[..., None]
            canvas[y : y + target_h, x : x + target_w] = np.clip(blended, 0, 255).astype(np.uint8)

            boxes.append(candidate)
            labels.append(
                (
                    crop.class_index,
                    (candidate[0] + candidate[2]) / 2.0 / VIEW_WIDTH,
                    (candidate[1] + candidate[3]) / 2.0 / VIEW_HEIGHT,
                    (candidate[2] - candidate[0]) / VIEW_WIDTH,
                    (candidate[3] - candidate[1]) / VIEW_HEIGHT,
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
    rows: List[Tuple[int, float, float, float, float]] = []
    for annotation in annotations:
        object_id = annotation["object_id"]
        if object_id not in CLASS_INDEX:
            continue
        x1, y1, x2, y2 = source_bbox_to_view(annotation["bbox"], source_region)
        clip_x1, clip_y1 = max(0.0, x1), max(0.0, y1)
        clip_x2, clip_y2 = min(1.0, x2), min(1.0, y2)
        if clip_x2 <= clip_x1 or clip_y2 <= clip_y1:
            continue
        full_area = (x2 - x1) * (y2 - y1)
        visible = (clip_x2 - clip_x1) * (clip_y2 - clip_y1)
        if full_area <= 0.0 or visible / full_area < MIN_VISIBLE_FRACTION:
            continue
        rows.append(
            (
                CLASS_INDEX[object_id],
                (clip_x1 + clip_x2) / 2.0,
                (clip_y1 + clip_y2) / 2.0,
                clip_x2 - clip_x1,
                clip_y2 - clip_y1,
            )
        )
    return rows


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
) -> Path:
    """Render exact views, apply copy-paste augmentation, and write a dataset.

    Returns the path to the generated ``data.yaml``.
    """
    rng = np.random.default_rng(seed)
    dataset_dir = output_dir / "drone_flyby_augmented"
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
    frame_list = list(frames) if frames is not None else frame_numbers(scene)

    # Pass 1: build the object bank from rendered views.
    bank: List[ObjectCrop] = []
    bank_counts: Dict[int, int] = {}
    samples: List[Tuple[str, int, int, int, np.ndarray, List]] = []

    for frame in frame_list:
        image = load_frame(frame, scene)
        annotations = load_annotations(frame, scene)
        for level in levels:
            for center_x, center_y in grids[level]:
                view, source_region = render_view(image, center_x, center_y, level)
                rows = _annotation_rows(annotations, source_region)
                samples.append((f"L{level}_f{frame:06d}_x{center_x}_y{center_y}", level, center_x, center_y, view, rows))
                for crop in crop_objects_from_view(view, rows):
                    if bank_counts.get(crop.class_index, 0) >= max_bank_per_class:
                        continue
                    bank.append(crop)
                    bank_counts[crop.class_index] = bank_counts.get(crop.class_index, 0) + 1

    if not bank:
        raise RuntimeError(
            "Object bank is empty; the scene produced no labelled crops to paste."
        )

    # Pass 2: write the dataset, optionally with augmented copies.
    stats = AugmentationStats(bank_size=len(bank))
    val_every = max(1, int(round(1.0 / val_fraction))) if val_fraction > 0 else 0
    sample_index = 0

    for stem, level, center_x, center_y, view, rows in samples:
        if not rows:
            stats.empty_skipped += 1
            continue

        split = "val" if val_every and sample_index % val_every == 0 else "train"
        _write_sample(dirs[split]["images"], dirs[split]["labels"], stem, view, rows)
        stats.train += split == "train"
        stats.val += split == "val"
        for cls, *_ in rows:
            name = OBJECT_CLASSES[cls]
            stats.per_class[name] = stats.per_class.get(name, 0) + 1

        if split == "train" and copy_paste_per_view > 0:
            for copy_index in range(copies_per_source):
                augmented, augmented_rows = paste_augment(
                    view, rows, bank, rng, max_pastes=copy_paste_per_view
                )
                if len(augmented_rows) == len(rows):
                    continue
                augmented_stem = f"{stem}_cp{copy_index}"
                _write_sample(dirs["train"]["images"], dirs["train"]["labels"], augmented_stem, augmented, augmented_rows)
                stats.train += 1
                stats.pasted_objects += len(augmented_rows) - len(rows)
                for cls, *_ in augmented_rows[len(rows):]:
                    name = OBJECT_CLASSES[cls]
                    stats.per_class[name] = stats.per_class.get(name, 0) + 1

        sample_index += 1

    data_yaml = dataset_dir / "drone_flyby_augmented.yaml"
    data_yaml.write_text(
        "path: {}\n".format(dataset_dir.as_posix())
        + "train: images/train\n"
        + "val: images/val\n"
        + "names:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(OBJECT_CLASSES))
    )
    (dataset_dir / "augmentation_stats.json").write_text(
        json.dumps(stats.__dict__, indent=2, sort_keys=True)
    )
    print(
        "Augmented exact-view dataset written to {} (train={train}, val={val}, "
        "empty-skipped={empty_skipped}, pasted={pasted_objects}, bank={bank_size})".format(
            dataset_dir, **stats.__dict__
        )
    )
    return data_yaml


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
    )
    print(f"Data YAML: {data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
