"""Source-level splits and an explicit evaluation-data quarantine."""

import json
import math
from pathlib import Path
from typing import Dict, Sequence

import yaml


def split_source_frames(frames: Sequence[int], val_fraction: float) -> Dict[int, str]:
    ordered = sorted(set(frames))
    if not ordered or not 0 <= val_fraction < 1:
        raise ValueError("A nonempty frame set and 0 <= val_fraction < 1 are required")
    if not val_fraction:
        return {frame: "train" for frame in ordered}
    if len(ordered) < 2:
        raise ValueError("At least two source frames are required for a validation split")
    val_count = min(len(ordered) - 1, max(1, math.ceil(len(ordered) * val_fraction)))
    # Whole-frame groups distributed across the flight avoid reserving almost
    # every appearance of late-entering classes for validation.
    validation = {ordered[(index + 1) * len(ordered) // (val_count + 1)]
                  for index in range(val_count)}
    return {frame: "val" if frame in validation else "train" for frame in ordered}


def assert_training_source(path: Path) -> None:
    resolved = Path(path).resolve()
    for parent in (resolved, *resolved.parents):
        marker = parent / "data_role.json"
        if parent.name in {"recorded_validation_data", "recordings"}:
            raise ValueError(f"Evaluation recording cannot be used for training: {path}")
        if marker.is_file():
            role = json.loads(marker.read_text(encoding="utf-8"))["data_role"]
            if role != "training-development":
                raise ValueError(f"Non-training data role {role!r}: {path}")
        if (parent / "metadata").is_dir() and (parent / "responses").is_dir():
            raise ValueError(f"Recorded evaluation sequence cannot be used for training: {path}")


def assert_training_yaml(path: Path) -> None:
    assert_training_source(path)
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = Path(data.get("path", path.parent))
    if not root.is_absolute():
        root = path.parent / root
    assert_training_source(root)
    for split in ("train", "val"):
        entries = data.get(split, [])
        for entry in entries if isinstance(entries, list) else [entries]:
            source = root / entry
            assert_training_source(source)
            if source.suffix == ".txt" and source.is_file():
                for image in source.read_text(encoding="utf-8").splitlines():
                    if image.strip():
                        assert_training_source(source.parent / image.strip())


def mark_training_dataset(directory: Path) -> None:
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Refusing to mix datasets in existing directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "data_role.json").write_text(json.dumps({
        "data_role": "training-development",
        "validation_scope": "held-out source frames; shared object instances, not independent generalization",
    }, indent=2), encoding="utf-8")
