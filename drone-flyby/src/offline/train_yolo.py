"""Train a YOLO detector from the prepared drone-flyby dataset.

This script is intentionally defensive: it can build a dataset manifest from the
supplied Helsinki annotations plus optional pseudo-labeled recordings, and it
only invokes Ultralytics if the dependency is available.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

REPO_ROOT = SRC_ROOT.parent

from dtos import OBJECT_CLASSES
from offline.dataset_provenance import assert_training_source, assert_training_yaml, mark_training_dataset, split_source_frames


try:
    from ultralytics import YOLO  # type: ignore

    _ULTRALYTICS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    YOLO = None
    _ULTRALYTICS_AVAILABLE = False


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def save_ap50_checkpoint(trainer) -> None:
    """Keep AP50's winner separately from the library's AP50:95 winner."""
    metric = float(trainer.metrics["metrics/mAP50(B)"])
    if not math.isfinite(metric):
        raise ValueError(f"Non-finite validation AP50 at epoch {trainer.epoch}")
    destination = Path(trainer.save_dir) / "weights" / "best_ap50.pt"
    metadata = destination.with_suffix(".json")
    previous = json.loads(metadata.read_text())["map50"] if metadata.is_file() else -1.0
    if metric <= previous:
        return
    shutil.copy2(trainer.last, destination)
    metadata.write_text(json.dumps({
        "epoch": trainer.epoch + 1, "map50": metric,
        "selection_scope": "source-frame development split, not competition AP",
    }, indent=2), encoding="utf-8")


def _optimizer_step_range(optimizer):
    steps = []
    for state in optimizer.state.values():
        if "step" not in state:
            continue
        value = state["step"]
        step = float(value.item() if hasattr(value, "item") else value)
        if not math.isfinite(step) or step < 0 or not step.is_integer():
            raise ValueError("Optimizer step counters must be finite nonnegative integers")
        steps.append(int(step))
    return {"min": min(steps), "max": max(steps), "parameters": len(steps)} if steps else None


def start_training_budget(trainer) -> None:
    trainer._drone_minibatches = 0
    trainer._drone_initial_steps = _optimizer_step_range(trainer.optimizer)
    trainer._drone_data_budget = {
        "training_samples": len(trainer.train_loader.dataset),
        "batches_per_epoch": len(trainer.train_loader),
    }


def count_training_batch(trainer) -> None:
    trainer._drone_minibatches += 1


def save_training_budget(trainer) -> None:
    report = {
        **trainer._drone_data_budget,
        "minibatches_processed_this_run": trainer._drone_minibatches,
        "optimizer_step_counters_before": trainer._drone_initial_steps,
        "optimizer_step_counters_after": _optimizer_step_range(trainer.optimizer),
        "batch_size": int(trainer.batch_size),
        "nominal_batch_size": int(trainer.args.nbs),
        "final_gradient_accumulation": int(trainer.accumulate),
        "start_epoch_index": int(trainer.start_epoch),
        "last_epoch_index": int(trainer.epoch),
        "epochs_requested": int(trainer.epochs),
        "optimizer": type(trainer.optimizer).__name__,
        "counter_scope": (
            "Per-process minibatches are observed separately from cumulative per-parameter optimizer "
            "step counters. Optimizer counters can include resumed history; null means unavailable."
        ),
    }
    destination = Path(trainer.save_dir) / "training_budget.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(f"Observed training budget: {json.dumps(report)}", flush=True)


def register_training_callbacks(model) -> None:
    model.add_callback("on_fit_epoch_end", save_ap50_checkpoint)
    model.add_callback("on_train_start", start_training_budget)
    model.add_callback("on_train_batch_end", count_training_batch)
    model.add_callback("on_train_end", save_training_budget)


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
    assert_training_source(helsinki_scene_dir)
    if pseudo_labeled_dir is not None:
        assert_training_source(pseudo_labeled_dir)
        if not pseudo_labeled_dir.is_dir():
            raise FileNotFoundError(f"Pseudo-label training directory does not exist: {pseudo_labeled_dir}")
    dataset_dir = output_dir / "drone_flyby_dataset"
    mark_training_dataset(dataset_dir)
    images_train = dataset_dir / "images" / "train"
    labels_train = dataset_dir / "labels" / "train"
    images_val = dataset_dir / "images" / "val"
    labels_val = dataset_dir / "labels" / "val"

    for path in [images_train, labels_train, images_val, labels_val]:
        path.mkdir(parents=True, exist_ok=True)

    # Supplied Helsinki labels.
    images = list(_iter_images(helsinki_scene_dir / "images"))
    splits = split_source_frames([int(image.stem.split("_")[-1]) for image in images], validation_split)
    train_images = [image for image in images if splits[int(image.stem.split("_")[-1])] == "train"]
    val_images = [image for image in images if splits[int(image.stem.split("_")[-1])] == "val"]

    for image_path in train_images:
        frame_index = int(image_path.stem.split("_")[-1])
        label_path = helsinki_scene_dir / "annotations" / f"frame_{frame_index:06d}.json"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing source annotations: {label_path}")

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
            raise FileNotFoundError(f"Missing source annotations: {label_path}")

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

    # Optional pseudo-labeled frames. These are training data, not validation:
    # routing them to val both starves training and makes any val metric a
    # measure of the pseudo-labeler rather than of the model.
    if pseudo_labeled_dir is not None and pseudo_labeled_dir.exists():
        for image_path in _iter_images(pseudo_labeled_dir / "images"):
            label_path = pseudo_labeled_dir / "labels" / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            # Prefix to avoid overwriting supplied frames that share a name.
            destination_image = images_train / f"pseudo_{image_path.name}"
            destination_label = labels_train / f"pseudo_{label_path.name}"
            destination_image.write_bytes(image_path.read_bytes())
            destination_label.write_bytes(label_path.read_bytes())

    data_yaml = dataset_dir / "drone_flyby.yaml"
    with open(data_yaml, "w", encoding="utf-8") as handle:
        handle.write(
            "path: {}\n".format(dataset_dir.resolve().as_posix())
            + "train: images/train\n"
            + "val: images/val\n"
            + "names:\n"
            + "".join(f"  {idx}: {name}\n" for idx, name in enumerate(OBJECT_CLASSES))
        )

    return data_yaml


def find_last_checkpoint(project: str | None, name: str) -> Path | None:
    """Locate a run's ``last.pt``, accounting for Ultralytics' run nesting.

    Ultralytics writes a relative ``--project`` under its ``runs/detect``
    directory, so the checkpoint may live at either ``<project>/<name>`` or
    ``runs/detect/<project>/<name>``. Absolute projects land directly at
    ``<project>/<name>``.
    """
    if not project:
        return None

    project_path = Path(project)
    candidates: List[Path] = []
    if project_path.is_absolute():
        candidates.append(project_path / name / "weights" / "last.pt")
    else:
        candidates.append(Path.cwd() / project_path / name / "weights" / "last.pt")
        candidates.append(Path.cwd() / "runs" / "detect" / project_path / name / "weights" / "last.pt")
    candidates.append(REPO_ROOT / project_path / name / "weights" / "last.pt")
    candidates.append(REPO_ROOT / "runs" / "detect" / project_path / name / "weights" / "last.pt")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


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
    cache: bool = False,
    resume: bool = False,
    resume_from: Path | None = None,
    augment_kwargs: dict | None = None,
    seed: int = 0,
    save_period: int = 10,
    warmup_epochs: float = 3.0,
) -> None:
    if not _ULTRALYTICS_AVAILABLE:
        raise RuntimeError(
            "ultralytics is not installed. Install it to run YOLO training or use the dataset builder only."
        )
    if not math.isfinite(warmup_epochs) or warmup_epochs < 0:
        raise ValueError("warmup_epochs must be finite and nonnegative")

    if resume:
        if resume_from is None or not Path(resume_from).is_file():
            raise FileNotFoundError(
                f"Cannot resume: no checkpoint found at {resume_from}. Run without "
                f"--resume to start a fresh training run."
            )
        assert_training_yaml(data_yaml)
        yolo_cls = YOLO
        assert yolo_cls is not None
        print(f"Resuming training from {resume_from}")
        resumed = yolo_cls(str(resume_from))
        checkpoint_data = resumed.ckpt.get("train_args", {}).get("data")
        if checkpoint_data:
            assert_training_yaml(Path(checkpoint_data))
        register_training_callbacks(resumed)
        # ``cache`` is forwarded explicitly so a run started with --cache can be
        # resumed without it; the saved dataloader cache is a suspected cause of
        # the mid-run stall.
        resumed.train(resume=True, device=device, cache=cache)
        return

    assert_training_yaml(data_yaml)
    yolo_cls = YOLO
    assert yolo_cls is not None
    model = yolo_cls(weights)
    if augment_kwargs and augment_kwargs.get("copy_paste", 0) > 0 and model.task == "detect":
        raise ValueError(
            "Ultralytics copy_paste requires segmentation masks and is inactive for box-only "
            "detection. Use build_augmented_dataset.py for labeled box copy-paste."
        )
    register_training_callbacks(model)
    train_kwargs = dict(
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
        cache=cache,
        project=project,
        name=name,
        seed=seed,
        deterministic=True,
        save_period=save_period,
        warmup_epochs=warmup_epochs,
    )
    # Dataset augmentation knobs (mosaic, mixup, copy-paste, geometry, colour)
    # are forwarded verbatim so the caller controls them without this script
    # hardcoding a policy. Copy-paste is especially useful for the rare classes
    # that dominate a macro-averaged score.
    if augment_kwargs:
        train_kwargs.update(augment_kwargs)
    model.train(**train_kwargs)


def build_augment_kwargs(arguments: argparse.Namespace) -> dict:
    """Assemble the dataset-augmentation kwargs from parsed CLI arguments."""
    return {
        "mosaic": arguments.mosaic,
        "mixup": arguments.mixup,
        "copy_paste": arguments.copy_paste,
        "degrees": arguments.degrees,
        "scale": arguments.scale,
        "translate": arguments.translate,
        "fliplr": arguments.fliplr,
        "flipud": arguments.flipud,
        "hsv_h": arguments.hsv_h,
        "hsv_s": arguments.hsv_s,
        "hsv_v": arguments.hsv_v,
        "erasing": arguments.erasing,
        "perspective": arguments.perspective,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and train a YOLO detector for drone-flyby.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "training_artifacts")
    parser.add_argument("--helsinki-dir", type=Path, default=REPO_ROOT / "data" / "helsinki")
    parser.add_argument("--pseudo-labeled-dir", type=Path, default=None)
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="Train directly on a prebuilt data YAML (e.g. the exact-view dataset) "
        "instead of rebuilding the full-resolution dataset.",
    )
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM during training.")
    # Box copy-paste is performed by the dataset builder, not this segmentation-only flag.
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--copy-paste", dest="copy_paste", type=float, default=0.0)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--hsv-h", dest="hsv_h", type=float, default=0.015)
    parser.add_argument("--hsv-s", dest="hsv_s", type=float, default=0.7)
    parser.add_argument("--hsv-v", dest="hsv_v", type=float, default=0.4)
    parser.add_argument("--erasing", type=float, default=0.4)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <project>/<name>/weights/last.pt if it exists.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Explicit checkpoint to resume from (implies --resume).",
    )
    arguments = parser.parse_args()

    if arguments.resume_from is not None:
        arguments.resume = True

    if arguments.resume:
        checkpoint = arguments.resume_from or find_last_checkpoint(arguments.project, arguments.name)
        if checkpoint is None:
            raise FileNotFoundError(
                f"No last.pt found for project={arguments.project!r} name={arguments.name!r}"
            )
        else:
            if arguments.data_yaml is None:
                raise ValueError("--resume requires --data-yaml for provenance validation")
            train(
                data_yaml=arguments.data_yaml,
                weights=arguments.weights,
                epochs=arguments.epochs,
                imgsz=arguments.imgsz,
                batch=arguments.batch,
                device=arguments.device,
                cache=arguments.cache,
                resume=True,
                resume_from=checkpoint,
            )
            return 0

    if arguments.data_yaml is not None:
        data_yaml = arguments.data_yaml
        print(f"Using prebuilt dataset: {data_yaml}")
    else:
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
        cache=arguments.cache,
        augment_kwargs=build_augment_kwargs(arguments),
        seed=arguments.seed,
        save_period=arguments.save_period,
        warmup_epochs=arguments.warmup_epochs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
