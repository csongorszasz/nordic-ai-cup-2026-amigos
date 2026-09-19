"""Propose foreground-only sprite masks from permitted training annotations.

Masks are unreviewed candidates, never automatically accepted training labels.
The preview and manifest must be reviewed before compositing or model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dtos import OBJECT_CLASSES
from offline.build_synthetic_sequence import harvest_sprites
from utils import load_frame


def alpha_asset(image: np.ndarray, mask: np.ndarray):
    if image.ndim != 3 or image.shape[2] != 3 or mask.shape != image.shape[:2]:
        raise ValueError("Mask must align exactly with the RGB/BGR image")
    foreground = mask.astype(bool)
    ys, xs = np.nonzero(foreground)
    if not len(xs):
        raise ValueError("Foreground mask is empty")
    alpha = foreground.astype(np.uint8) * 255
    image_with_alpha = np.dstack((image, alpha))
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return image_with_alpha, box


def checker_preview(image_with_alpha):
    height, width = image_with_alpha.shape[:2]
    yy, xx = np.indices((height, width))
    checker = np.where((xx // 8 + yy // 8) % 2, 180, 100).astype(np.uint8)
    background = np.repeat(checker[:, :, None], 3, axis=2)
    alpha = image_with_alpha[:, :, 3:4].astype(np.float32) / 255
    return (image_with_alpha[:, :, :3] * alpha + background * (1 - alpha)).astype(np.uint8)


def extract(scene, checkpoint: Path, output: Path, device="0", classes=None, center_prompt=False):
    if output.exists():
        raise FileExistsError(f"Use a new mask proposal directory: {output}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    selected = list(OBJECT_CLASSES) if classes is None else list(classes)
    if not selected or len(set(selected)) != len(selected) or any(name not in OBJECT_CLASSES for name in selected):
        raise ValueError("Select unique, known object classes")
    from ultralytics import SAM

    sprites, provenance = harvest_sprites(scene)
    model = SAM(str(checkpoint))
    output.mkdir(parents=True)
    entries = []
    contact = np.full((4 * 360, 4 * 440, 3), 235, dtype=np.uint8)
    frames = {}
    failures = []
    for index, name in enumerate(selected):
        source = provenance[name]
        frame = source["frame"]
        if frame not in frames:
            frames[frame] = load_frame(frame, scene)
        image = frames[frame]
        x1, y1, x2, y2 = map(int, source["bbox"])
        padding = max(16, max(x2 - x1, y2 - y1) // 2)
        left, top = max(0, x1 - padding), max(0, y1 - padding)
        right, bottom = min(image.shape[1], x2 + padding), min(image.shape[0], y2 + padding)
        context = image[top:bottom, left:right]
        prompt = [x1 - left, y1 - top, x2 - left, y2 - top]
        prompts = {}
        if center_prompt:
            height, width = context.shape[:2]
            prompts = {
                "points": [[[(prompt[0] + prompt[2]) / 2, (prompt[1] + prompt[3]) / 2],
                            [1, 1], [width - 2, 1], [1, height - 2], [width - 2, height - 2]]],
                "labels": [[1, 0, 0, 0, 0]],
            }
        results = model.predict(context, bboxes=[prompt], device=device, verbose=False, **prompts)
        entry = {"object_id": name, "source": source, "review_status": "unreviewed"}
        original = sprites[name]
        masked_preview = original.copy()
        if not results or results[0].masks is None or len(results[0].masks.data) != 1:
            entry["failure"] = "Expected exactly one prompted mask"
            failures.append(name)
        else:
            mask = results[0].masks.data[0].detach().cpu().numpy()
            if mask.shape != context.shape[:2]:
                raise ValueError(f"Mask/image geometry mismatch for {name}: {mask.shape} != {context.shape[:2]}")
            object_mask = mask[y1 - top:y2 - top, x1 - left:x2 - left] > 0.5
            if not object_mask.any():
                entry["failure"] = "Prompted mask has no foreground inside the annotation"
                failures.append(name)
            else:
                asset, box = alpha_asset(original, object_mask)
                path = output / f"{name}.png"
                if not cv2.imwrite(str(path), asset):
                    raise OSError(f"Could not write {path}")
                fraction = float(object_mask.mean())
                entry.update({
                    "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "alpha_bbox_xyxy": list(box), "foreground_fraction": fraction,
                    "rectangle_like": fraction >= 0.95,
                    "mask_score": float(results[0].boxes.conf[0].item()),
                })
                masked_preview = checker_preview(asset)
        entries.append(entry)
        cell_x, cell_y = (index % 4) * 440, (index // 4) * 360
        cv2.putText(contact, name, (cell_x + 8, cell_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        for panel, crop in enumerate((original, masked_preview)):
            scale = min(208 / crop.shape[1], 290 / crop.shape[0])
            resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            y, x = cell_y + 42, cell_x + 8 + panel * 216
            contact[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        print(f"{name}: {entry.get('failure', 'mask candidate; review required')}", flush=True)
    if not cv2.imwrite(str(output / "review.jpg"), contact, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError("Could not write mask review preview")
    with checkpoint.open("rb") as handle:
        checkpoint_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    (output / "manifest.json").write_text(json.dumps({
        "status": "unreviewed", "source_scene": scene,
        "segmentation_checkpoint": checkpoint.name, "segmentation_checkpoint_sha256": checkpoint_sha,
        "label_warning": "Predicted masks need visual review; confidence is not ground-truth quality.",
        "prompt_mode": "box-center-and-background-corners" if center_prompt else "box",
        "failures": failures, "assets": entries,
    }, indent=2), encoding="utf-8")
    (output / "data_role.json").write_text(json.dumps({"data_role": "training-development"}), encoding="utf-8")
    return len(failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="helsinki")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--classes", nargs="+", choices=OBJECT_CLASSES)
    parser.add_argument("--center-prompt", action="store_true",
                        help="Reviewable alternative with a positive center and negative context corners.")
    arguments = parser.parse_args()
    return 1 if extract(arguments.scene, arguments.checkpoint, arguments.output, arguments.device,
                        arguments.classes, arguments.center_prompt) else 0


if __name__ == "__main__":
    raise SystemExit(main())
