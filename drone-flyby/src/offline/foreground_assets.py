"""Hash-pinned, explicitly reviewed alpha assets and correct resampling."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from dtos import OBJECT_CLASSES
from offline.dataset_provenance import assert_training_source


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Asset path escapes its authorized artifact root")
    return path


def load_reviewed_assets(review_path: Path, artifact_root: Path):
    review_bytes = Path(review_path).read_bytes()
    review = json.loads(review_bytes)
    if review.get("status") != "reviewed-for-composite-audit":
        raise ValueError("Foreground assets require an explicit completed visual review")
    if set(review.get("accepted_classes", [])) != set(OBJECT_CLASSES):
        raise ValueError("The audit must explicitly account for all challenge classes")
    manifests = {}
    sprites, provenance = {}, {}
    for name in OBJECT_CLASSES:
        source_name = review.get("class_sources", {}).get(name, review["default_source"])
        source = review["sources"][source_name]
        experiment = _inside(Path(artifact_root), source["experiment"])
        path = _inside(experiment, source["manifest"])
        assert_training_source(path)
        if source_name not in manifests:
            contents = path.read_bytes()
            if hashlib.sha256(contents).hexdigest() != source["sha256"].lower():
                raise ValueError(f"Reviewed source manifest changed: {source_name}")
            manifests[source_name] = json.loads(contents)
        entries = [entry for entry in manifests[source_name]["assets"] if entry["object_id"] == name]
        if len(entries) != 1 or entries[0].get("failure"):
            raise ValueError(f"No unique reviewed mask for {name}")
        entry = entries[0]
        image_path = _inside(path.parent, entry["file"])
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Reviewed alpha pixels changed: {name}")
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] != 4 or not image[:, :, 3].any():
            raise ValueError(f"Invalid reviewed alpha image: {name}")
        x1, y1, x2, y2 = entry["source"]["bbox"]
        if image.shape[:2] != (int(y2) - int(y1), int(x2) - int(x1)):
            raise ValueError(f"Source annotation canvas changed: {name}")
        sprites[name] = image
        provenance[name] = {
            **entry["source"], "asset_sha256": entry["sha256"],
            "source_experiment": source["experiment"], "review_status": review["status"],
            "review_sha256": hashlib.sha256(review_bytes).hexdigest(),
            "annotation_convention": review["annotation_convention"],
        }
    return sprites, provenance


def resize_sprite(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[2] == 3:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    if image.shape[2] != 4:
        raise ValueError("Sprites must be BGR or reviewed BGRA")
    alpha = image[:, :, 3:4].astype(np.float32) / 255
    premultiplied = np.concatenate((image[:, :, :3].astype(np.float32) * alpha, alpha), axis=2)
    return cv2.resize(premultiplied, (width, height), interpolation=cv2.INTER_LINEAR)


def composite_sprite(target: np.ndarray, sprite: np.ndarray) -> None:
    """Composite a BGR image or premultiplied float BGRA from resize_sprite."""
    if sprite.shape[:2] != target.shape[:2]:
        raise ValueError("Sprite and destination geometry must match")
    if sprite.shape[2] == 3:
        target[:] = sprite
    elif sprite.shape[2] == 4:
        if not np.issubdtype(sprite.dtype, np.floating) or np.any((sprite[:, :, 3] < 0) | (sprite[:, :, 3] > 1)):
            raise ValueError("RGBA must be premultiplied and normalized by resize_sprite")
        alpha = sprite[:, :, 3:4]
        target[:] = np.clip(sprite[:, :, :3] + target.astype(np.float32) * (1 - alpha), 0, 255).astype(np.uint8)
    else:
        raise ValueError("Invalid sprite channel count")
