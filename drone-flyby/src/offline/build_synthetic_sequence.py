"""Render fully labeled moving episodes for controlled generalization diagnostics.

Backgrounds and trajectories are new; object appearances still come from the
supplied training frames. These episodes are not a competition-score estimate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH, OBJECT_CLASSES
from offline.dataset_provenance import assert_training_source, split_source_frames
from utils import frame_numbers, load_annotations, load_frame, scene_directory


def project_annotations(objects, offset, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    annotations = []
    for item in objects:
        x1, y1, x2, y2 = item["bbox"]
        box = (max(0, x1 - offset[0]), max(0, y1 - offset[1]),
               min(width, x2 - offset[0]), min(height, y2 - offset[1]))
        if box[0] < box[2] and box[1] < box[3]:
            annotations.append({
                "object_id": item["object_id"], "instance_id": item["instance_id"],
                "bbox": list(box),
            })
    return annotations


def harvest_sprites(scene):
    assert_training_source(scene_directory(scene))
    splits = split_source_frames(frame_numbers(scene), 0.15)
    sprites = {}
    provenance = {}
    for frame, split in splits.items():
        if split != "train":
            continue
        image = load_frame(frame, scene)
        for item in load_annotations(frame, scene):
            x1, y1, x2, y2 = map(int, item["bbox"])
            if x1 <= 0 or y1 <= 0 or x2 >= IMAGE_WIDTH or y2 >= IMAGE_HEIGHT:
                continue
            name = item["object_id"]
            crop = image[y1:y2, x1:x2]
            if crop.size and (name not in sprites or crop.size > sprites[name].size):
                sprites[name] = crop.copy()
                provenance[name] = {"scene": scene, "frame": frame, "bbox": item["bbox"]}
    missing = set(OBJECT_CLASSES) - sprites.keys()
    if missing:
        raise ValueError(f"No complete training sprite for classes: {sorted(missing)}")
    return sprites, provenance


def build_sequence(output: Path, scene="helsinki", frames=80, seed=101,
                   shift_x=0, shift_y=58, objects_per_frame=16,
                   purpose="development-evaluation", rotate=True):
    if frames < 2 or frames > 300 or objects_per_frame < 1 or abs(shift_x) > 200 or abs(shift_y) > 200:
        raise ValueError("Use 2..300 frames, positive density, and shifts within 200 source pixels")
    if output.exists():
        raise FileExistsError(f"Use a new episode directory: {output}")
    if purpose not in {"training-development", "development-evaluation"}:
        raise ValueError(f"Unknown data role: {purpose}")
    sprites, provenance = harvest_sprites(scene)
    rng = np.random.default_rng(seed)
    width = IMAGE_WIDTH + abs(shift_x) * (frames - 1)
    height = IMAGE_HEIGHT + abs(shift_y) * (frames - 1)
    if width * height > 120_000_000:
        raise ValueError("Episode strip exceeds the bounded 120-megapixel working set")
    coarse = rng.integers(35, 100, (math.ceil(height / 48), math.ceil(width / 48), 3), dtype=np.uint8)
    terrain = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_LINEAR)
    for x in range(250, width, 600):
        cv2.rectangle(terrain, (x, 0), (x + 35, height - 1), (105, 105, 105), -1)
    for y in range(300, height, 700):
        cv2.rectangle(terrain, (0, y), (width - 1, y + 30), (110, 110, 110), -1)
    for _ in range(width * height // 15000):
        x, y = int(rng.integers(width - 100)), int(rng.integers(height - 100))
        w, h = int(rng.integers(15, 95)), int(rng.integers(15, 95))
        color = tuple(int(v) for v in rng.integers(50, 155, 3))
        cv2.rectangle(terrain, (x, y), (x + w, y + h), color, -1)

    count = math.ceil(objects_per_frame * width * height / (IMAGE_WIDTH * IMAGE_HEIGHT))
    objects = []
    reserved = []
    for index in range(count):
        name = OBJECT_CLASSES[index % len(OBJECT_CLASSES)]
        sampled_rotation = int(rng.integers(4))
        rotation = sampled_rotation if rotate else 0
        sprite = np.rot90(sprites[name], rotation).copy()
        scale = float(rng.uniform(0.85, 1.15))
        sprite = cv2.resize(sprite, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        h, w = sprite.shape[:2]
        side = max(w, h)
        for _ in range(1000):
            square_x, square_y = int(rng.integers(width - side)), int(rng.integers(height - side))
            square = (square_x, square_y, square_x + side, square_y + side)
            x, y = square_x + (side - w) // 2, square_y + (side - h) // 2
            box = (x, y, x + w, y + h)
            if all(square[2] + 8 < old[0] or old[2] + 8 < square[0]
                   or square[3] + 8 < old[1] or old[3] + 8 < square[1] for old in reserved):
                break
        else:
            raise RuntimeError("Cannot place a labeled object without overlap")
        terrain[y:y + h, x:x + w] = sprite
        reserved.append(square)
        objects.append({
            "object_id": name, "instance_id": f"{seed}:{index}", "bbox": list(box),
            "rotation_quarters": rotation, "scale": scale, "placement": list(square),
        })

    (output / "images").mkdir(parents=True)
    (output / "annotations").mkdir()
    start_x, start_y = max(0, (frames - 1) * shift_x), max(0, (frames - 1) * shift_y)
    for frame in range(frames):
        offset = (start_x - frame * shift_x, start_y - frame * shift_y)
        image = terrain[offset[1]:offset[1] + IMAGE_HEIGHT, offset[0]:offset[0] + IMAGE_WIDTH]
        if not cv2.imwrite(str(output / "images" / f"frame_{frame:06d}.png"), image,
                           [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise OSError("Synthetic frame encoding failed")
        (output / "annotations" / f"frame_{frame:06d}.json").write_text(json.dumps({
            "frame": frame, "annotations": project_annotations(objects, offset),
        }), encoding="utf-8")
    metadata = {
        "seed": seed, "frames": frames, "shift": [shift_x, shift_y],
        "object_appearances": "shared training sprites; novel procedural background and placement",
        "sprites": provenance, "objects": objects,
        "rotate_objects": rotate,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "data_role.json").write_text(json.dumps({"data_role": purpose}), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-scene", default="helsinki")
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--shift-x", type=int, default=0)
    parser.add_argument("--shift-y", type=int, default=58)
    parser.add_argument("--objects-per-frame", type=int, default=16)
    parser.add_argument("--no-rotation", action="store_true", help="Paired orientation ablation with identical placements.")
    parser.add_argument("--purpose", choices=["training-development", "development-evaluation"],
                        default="development-evaluation")
    arguments = parser.parse_args()
    build_sequence(arguments.output, arguments.source_scene, arguments.frames, arguments.seed,
                   arguments.shift_x, arguments.shift_y, arguments.objects_per_frame, arguments.purpose,
                   rotate=not arguments.no_rotation)


if __name__ == "__main__":
    main()
