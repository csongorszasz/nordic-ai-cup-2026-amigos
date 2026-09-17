"""Train a YOLO detector from the prepared drone-flyby dataset.

This script is intentionally defensive: it can build a dataset manifest from the
supplied Helsinki annotations plus optional pseudo-labeled recordings, and it
only invokes Ultralytics if the dependency is available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

REPO_ROOT = SRC_ROOT.parent

from dtos import OBJECT_CLASSES


try:
    from ultralytics import YOLO  # type: ignore

    _ULTRALYTICS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    YOLO = None
    _ULTRALYTICS_AVAILABLE = False


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def _iter_images(images_dir: Path) -> Iterable[Path]:
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _xyxy_to_yolo(bbox: List[float], width: int, height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    cx = x1 + box_w / 2.0
    cy = y1 + box_h / 2.0
    return cx / width, cy / height, box_w / width, box_h / height


def _convert_hsv_object_id(object_id: str) -> int:
    try:
        return OBJECT_CLASSES.index(object_id)
    except ValueError as exc:
        raise ValueError(f"Unknown class {object_id!r}") from exc


def build_dataset_layout(
    output_dir: Path,
    helsinki_scene_dir: Path,
    pseudo_labeled_dir: Path | None = None,
    validation_split: float = 0.2,
) -> Path:
    """Build a YOLO-style dataset tree from the supplied resources."""
    dataset_dir = output_dir / "drone_flyby_dataset"
    images_train = dataset_dir / "images" / "train"
    labels_train = dataset_dir / "labels" / "train"
    images_val = dataset_dir / "images" / "val"
    labels_val = dataset_dir / "labels" / "val"

    for path in [images_train, labels_train, images_val, labels_val]:
        path.mkdir(parents=True, exist_ok=True)

    # Supplied Helsinki labels.
    images = list(_iter_images(helsinki_scene_dir / "images"))
    split_index = max(1, int(round(len(images) * (1.0 - validation_split))))
    train_images = images[:split_index]
    val_images = images[split_index:]

    for image_path in train_images:
        frame_index = int(image_path.stem.split("_")[-1])
        label_path = helsinki_scene_dir / "annotations" / f"frame_{frame_index:06d}.json"
        if not label_path.exists():
            continue

        destination_image = images_train / image_path.name
        destination_image.write_bytes(image_path.read_bytes())

        with open(label_path, "r", encoding="utf-8") as handle:
            frame_data = json.load(handle)

        lines: List[str] = []
        for annotation in frame_data.get("annotations", []):
            class_index = _convert_hsv_object_id(annotation["object_id"])
            cx, cy, w, h = _xyxy_to_yolo(annotation["bbox"], 3840, 2160)
            lines.append(f"{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        with open(labels_train / f"{image_path.stem}.txt", "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))

    for image_path in val_images:
        frame_index = int(image_path.stem.split("_")[-1])
        label_path = helsinki_scene_dir / "annotations" / f"frame_{frame_index:06d}.json"
        if not label_path.exists():
            continue

        destination_image = images_val / image_path.name
        destination_image.write_bytes(image_path.read_bytes())

        with open(label_path, "r", encoding="utf-8") as handle:
            frame_data = json.load(handle)

        lines: List[str] = []
        for annotation in frame_data.get("annotations", []):
            class_index = _convert_hsv_object_id(annotation["object_id"])
            cx, cy, w, h = _xyxy_to_yolo(annotation["bbox"], 3840, 2160)
            lines.append(f"{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        with open(labels_val / f"{image_path.stem}.txt", "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))

    # Optional pseudo-labeled validation frames.
    if pseudo_labeled_dir is not None and pseudo_labeled_dir.exists():
        for image_path in _iter_images(pseudo_labeled_dir / "images"):
            label_path = pseudo_labeled_dir / "labels" / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            (images_val / image_path.name).write_bytes(image_path.read_bytes())
            (labels_val / label_path.name).write_bytes(label_path.read_bytes())

    data_yaml = dataset_dir / "drone_flyby.yaml"
    with open(data_yaml, "w", encoding="utf-8") as handle:
        handle.write(
            "path: {}\n".format(dataset_dir.as_posix())
            + "train: images/train\n"
            + "val: images/val\n"
            + "names:\n"
            + "".join(f"  {idx}: {name}\n" for idx, name in enumerate(OBJECT_CLASSES))
        )

    return data_yaml


def train(
    data_yaml: Path,
    weights: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    multi_scale: bool = False,
    optimizer: str = "auto",
    lr0: float = 0.01,
    patience: int = 0,
    project: str | None = None,
    name: str = "train",
) -> None:
    if not _ULTRALYTICS_AVAILABLE:
        raise RuntimeError(
            "ultralytics is not installed. Install it to run YOLO training or use the dataset builder only."
        )

    yolo_cls = YOLO
    assert yolo_cls is not None
    model = yolo_cls(weights)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        multi_scale=multi_scale,
        optimizer=optimizer,
        lr0=lr0,
        close_mosaic=10,
        patience=patience,
        augment=True,
        project=project,
        name=name,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and train a YOLO detector for drone-flyby.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "training_artifacts")
    parser.add_argument("--helsinki-dir", type=Path, default=REPO_ROOT / "data" / "helsinki")
    parser.add_argument("--pseudo-labeled-dir", type=Path, default=None)
    parser.add_argument("--weights", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--multi-scale", action="store_true", help="Enable multi-scale training (higher VRAM).")
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default="train")
    parser.add_argument("--build-only", action="store_true")
    arguments = parser.parse_args()

    data_yaml = build_dataset_layout(arguments.output_dir, arguments.helsinki_dir, arguments.pseudo_labeled_dir)
    print(f"Dataset manifest written to {data_yaml}")

    if arguments.build_only:
        return 0

    train(
        data_yaml,
        arguments.weights,
        arguments.epochs,
        arguments.imgsz,
        arguments.batch,
        arguments.device,
        multi_scale=arguments.multi_scale,
        optimizer=arguments.optimizer,
        lr0=arguments.lr0,
        patience=arguments.patience,
        project=arguments.project,
        name=arguments.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
