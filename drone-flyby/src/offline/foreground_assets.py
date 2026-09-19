"""Hash-pinned, explicitly reviewed alpha assets and correct resampling."""

import hashlib
import json
import math
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


def resize_sprite(image: np.ndarray, width: int, height: int, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    if image.shape[2] == 3:
        return cv2.resize(image, (width, height), interpolation=interpolation)
    if image.shape[2] != 4:
        raise ValueError("Sprites must be BGR or reviewed BGRA")
    if image.dtype != np.uint8 or interpolation not in (cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_NEAREST):
        raise ValueError("RGBA resampling requires uint8 pixels and a non-overshooting interpolation kernel")
    alpha = image[:, :, 3:4].astype(np.float32) / 255
    premultiplied = np.concatenate((image[:, :, :3].astype(np.float32) * alpha, alpha), axis=2)
    resized = cv2.resize(premultiplied, (width, height), interpolation=interpolation)
    opacity = resized[:, :, 3]
    if not np.isfinite(opacity).all() or np.any((opacity < -1e-5) | (opacity > 1 + 1e-5)):
        raise ValueError("Unexpected alpha range after convex resampling")
    # OpenCV INTER_AREA can overshoot opaque alpha by a float32 rounding step.
    np.clip(opacity, 0.0, 1.0, out=opacity)
    return resized


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


def rotate_sprite_canvas(sprite: np.ndarray, degrees: float):
    """Rotate premultiplied pixels and the original annotation canvas together.

    The output canvas and its center are angle-independent. At zero degrees,
    parity-matched padding preserves pixels without an extra interpolation.
    """
    if (sprite.ndim != 3 or sprite.shape[2] != 4
            or min(sprite.shape[:2]) <= 0 or not np.issubdtype(sprite.dtype, np.floating)):
        raise ValueError("Rotation requires premultiplied floating-point BGRA")
    if not math.isfinite(degrees) or not np.isfinite(sprite).all():
        raise ValueError("Rotation angle and pixels must be finite")
    opacity = sprite[:, :, 3]
    if np.any((opacity < 0) | (opacity > 1)):
        raise ValueError("Rotation requires normalized alpha")
    if np.any(sprite[:, :, :3] < 0) or np.any(sprite[:, :, :3] > 255 * opacity[:, :, None] + 1e-3):
        raise ValueError("RGB must already be premultiplied by alpha")
    height, width = sprite.shape[:2]
    side = math.ceil(math.hypot(width, height)) + 4
    canvas_width = side + (side - width) % 2
    canvas_height = side + (side - height) % 2
    edge_transform = cv2.getRotationMatrix2D((width / 2, height / 2), degrees % 360, 1.0)
    edge_transform[:, 2] += ((canvas_width - width) / 2, (canvas_height - height) / 2)
    corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]])
    projected = corners @ edge_transform.T
    box = (*projected.min(axis=0), *projected.max(axis=0))
    # OpenCV transforms pixel centers; annotation coordinates describe edges.
    pixel_transform = edge_transform.copy()
    pixel_transform[:, 2] += 0.5 * (pixel_transform[:, :2].sum(axis=1) - 1)
    rotated = cv2.warpAffine(
        sprite, pixel_transform, (canvas_width, canvas_height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
    )
    alpha = rotated[:, :, 3]
    if np.any((alpha < -1e-5) | (alpha > 1 + 1e-5)):
        raise ValueError("Unexpected alpha range after rotation")
    np.clip(alpha, 0.0, 1.0, out=alpha)
    return rotated, tuple(float(value) for value in box)
