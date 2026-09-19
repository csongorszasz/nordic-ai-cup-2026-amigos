"""Render fully labeled moving episodes for controlled generalization diagnostics.

Backgrounds and trajectories are new; object appearances still come from the
supplied training frames. These episodes are not a competition-score estimate.
"""

from __future__ import annotations

import argparse
import hashlib
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
from offline.convert_nls_ortho import Image, read_georeference
from offline.foreground_assets import composite_sprite, load_reviewed_assets, resize_sprite
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


def orthophoto_background(path, provenance_path, width, height, rng, target_gsd=0.2395, origin=None):
    if provenance_path is None:
        raise ValueError("An orthophoto requires source/license/hash provenance")
    provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    for key in ("source_url", "license", "attribution", "sha256"):
        if not provenance.get(key):
            raise ValueError(f"Background provenance is missing {key}")
    with Path(path).open("rb") as handle:
        actual_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual_hash != provenance["sha256"].lower():
        raise ValueError("Background file does not match its provenance hash")
    geo = read_georeference(Path(path))
    if geo is None:
        raise ValueError("An orthophoto audit requires embedded verified georeferencing")
    if not math.isfinite(target_gsd) or target_gsd <= 0:
        raise ValueError("Target GSD must be positive and finite")
    source_width = max(1, int(round(width * target_gsd / geo["gsd_m"])))
    source_height = max(1, int(round(height * target_gsd / geo["gsd_m"])))
    with Image.open(path) as image:
        if image.width * image.height > 150_000_000:
            raise ValueError("Orthophoto exceeds the supported working set")
        if source_width > image.width or source_height > image.height:
            raise ValueError("The orthophoto cannot contain the complete requested flight strip")
        if origin is None:
            x = int(rng.integers(image.width - source_width + 1))
            y = int(rng.integers(image.height - source_height + 1))
        else:
            x, y = origin
            if x < 0 or y < 0 or x + source_width > image.width or y + source_height > image.height:
                raise ValueError("The reviewed background origin is outside the source image")
        crop = np.asarray(image.crop((x, y, x + source_width, y + source_height)).convert("RGB"))
    interpolation = cv2.INTER_AREA if source_width >= width else cv2.INTER_LINEAR
    terrain = cv2.resize(crop[:, :, ::-1].copy(), (width, height), interpolation=interpolation)
    return terrain, {
        **provenance, "kind": "orthophoto", "georeference": geo,
        "source_crop_xyxy": [x, y, x + source_width, y + source_height],
        "output_gsd_x_m": source_width * geo["gsd_m"] / width,
        "output_gsd_y_m": source_height * geo["gsd_m"] / height,
        "native_detail_note": "Resampling does not create detail beyond the source GSD.",
        "label_scope": "Inserted challenge sprites only; natural background objects are not annotated.",
    }


def build_sequence(output: Path, scene="helsinki", frames=80, seed=101,
                   shift_x=0, shift_y=58, objects_per_frame=16,
                   purpose="development-evaluation", rotate=True,
                   background=None, background_provenance=None, target_gsd=0.2395, background_origin=None,
                   sprite_review=None, sprite_artifact_root=None, allow_context_patches=False):
    if frames < 2 or frames > 300 or objects_per_frame < 1 or abs(shift_x) > 200 or abs(shift_y) > 200:
        raise ValueError("Use 2..300 frames, positive density, and shifts within 200 source pixels")
    if output.exists():
        raise FileExistsError(f"Use a new episode directory: {output}")
    if purpose not in {"training-development", "development-evaluation"}:
        raise ValueError(f"Unknown data role: {purpose}")
    if (sprite_review is None) != (sprite_artifact_root is None):
        raise ValueError("A sprite review and artifact root must be supplied together")
    if background is not None and sprite_review is None and not allow_context_patches:
        raise ValueError("Natural-background audits require reviewed alpha assets or explicit --allow-context-patches")
    sprites, provenance = (
        load_reviewed_assets(Path(sprite_review), Path(sprite_artifact_root))
        if sprite_review is not None else harvest_sprites(scene)
    )
    rng = np.random.default_rng(seed)
    width = IMAGE_WIDTH + abs(shift_x) * (frames - 1)
    height = IMAGE_HEIGHT + abs(shift_y) * (frames - 1)
    if width * height > 120_000_000:
        raise ValueError("Episode strip exceeds the bounded 120-megapixel working set")
    if background is None:
        background_info = {"kind": "procedural", "seed": seed}
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
    else:
        if purpose == "training-development":
            assert_training_source(Path(background))
        terrain, background_info = orthophoto_background(
            Path(background), background_provenance, width, height, rng, target_gsd, background_origin,
        )

    count = math.ceil(objects_per_frame * width * height / (IMAGE_WIDTH * IMAGE_HEIGHT))
    objects = []
    reserved = []
    for index in range(count):
        name = OBJECT_CLASSES[index % len(OBJECT_CLASSES)]
        sampled_rotation = int(rng.integers(4))
        rotation = sampled_rotation if rotate else 0
        sprite = np.rot90(sprites[name], rotation).copy()
        scale = float(rng.uniform(0.85, 1.15))
        sprite = resize_sprite(sprite, max(1, int(round(sprite.shape[1] * scale))),
                               max(1, int(round(sprite.shape[0] * scale))))
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
        composite_sprite(terrain[y:y + h, x:x + w], sprite)
        reserved.append(square)
        objects.append({
            "object_id": name, "instance_id": f"{seed}:{index}", "bbox": list(box),
            "rotation_quarters": rotation, "scale": scale, "placement": list(square),
        })

    (output / "images").mkdir(parents=True)
    (output / "annotations").mkdir()
    preview = cv2.resize(terrain, (min(1280, width), max(1, int(height * min(1280, width) / width))))
    if not cv2.imwrite(str(output / "overview.jpg"), preview):
        raise OSError("Synthetic overview encoding failed")
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
        "object_appearances": "shared training sprites; novel background and placement, not unseen object appearances",
        "sprites": provenance, "objects": objects,
        "rotate_objects": rotate,
        "background": background_info,
        "sprite_compositing": "reviewed-alpha" if sprite_review is not None else "rectangular-context-patch",
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
    parser.add_argument("--background", type=Path, help="Optional NLS GeoJP2 background, never a validation capture.")
    parser.add_argument("--background-provenance", type=Path, help="Required source/license/hash manifest.")
    parser.add_argument("--target-gsd", type=float, default=0.2395, help="Explicit simulated source-frame metres per pixel.")
    parser.add_argument("--background-origin", type=int, nargs=2, metavar=("X", "Y"),
                        help="Reviewed crop origin in native background-image pixels.")
    parser.add_argument("--sprite-review", type=Path, help="Explicit hash-pinned visual review specification.")
    parser.add_argument("--sprite-artifact-root", type=Path, help="Root containing the referenced mask experiments.")
    parser.add_argument("--allow-context-patches", action="store_true",
                        help="Explicit artifact-prone control only; not foreground-isolated transfer evidence.")
    parser.add_argument("--purpose", choices=["training-development", "development-evaluation"],
                        default="development-evaluation")
    arguments = parser.parse_args()
    build_sequence(arguments.output, arguments.source_scene, arguments.frames, arguments.seed,
                   arguments.shift_x, arguments.shift_y, arguments.objects_per_frame, arguments.purpose,
                   rotate=not arguments.no_rotation, background=arguments.background,
                   background_provenance=arguments.background_provenance, target_gsd=arguments.target_gsd,
                   background_origin=arguments.background_origin, sprite_review=arguments.sprite_review,
                   sprite_artifact_root=arguments.sprite_artifact_root,
                   allow_context_patches=arguments.allow_context_patches)


if __name__ == "__main__":
    main()
