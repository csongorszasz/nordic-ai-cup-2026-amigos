"""Build a held-out test flight with exact labels, to rank models without guessing.

Why this exists: our only honest test is 43 hand-labelled objects on one Copenhagen
flight, and the spread of the tune score between epochs of a single run is 0.068 --
larger than every difference we have tried to choose between. That set cannot rank
models. This one can, because the labels are exact by construction and there are as
many objects as we ask for.

What it measures: generalisation to **backgrounds no training run has seen**
(fetch_backgrounds.py --holdout). It does NOT measure generalisation to the
organisers' renderer -- the objects and the pasting are ours, so a model trained on
this pipeline is flattered. Every candidate is flattered equally, so the ranking
between them is fair; the absolute number means little.

A flight, not a pile of frames: the real drone drifts about 64 px of ground per frame
and the same object appears in many frames, so memory and tracking are exercised the
way they are live. Objects are pasted once onto a tall canvas at fixed positions, and
each frame is a window sliding down it.

    python training/holdout_scene.py                       # -> data/holdout/
    python training/holdout_scene.py --flights 4 --objects 60
    python training/run_policy.py --scene holdout --camera sweep --lag 1 --model <weights>

Output is a normal scene directory (images/frame_NNNNNN.png, annotations/*.json,
run_metadata.json), so run_policy.py and local_evaluator.py read it unchanged.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth_dataset as sd  # noqa: E402

ROOT = sd.ROOT
FRAME_W, FRAME_H = sd.SOURCE_W, sd.SOURCE_H
DRIFT_Y = 64          # ground pixels per frame, measured on the recorded validation flight
DRIFT_X_JITTER = 8    # the real flight wanders a few px sideways
EDGE_MARGIN = 120     # keep objects clear of the canvas sides


def build_canvas(tiles, frames, rng):
    """(tall background, open-ground mask) from held-out tiles stacked along the flight's drift.

    The mask matters: the first draft dropped tanks into open sea, which no model should be
    judged on. Each tile's <name>_ground.png is stacked and flipped with it, so the mask lines
    up with what it masks.
    """
    height = FRAME_H + DRIFT_Y * (frames - 1)
    canvas = np.zeros((height, FRAME_W, 3), np.uint8)
    ground = np.zeros((height, FRAME_W), np.uint8)
    y = 0
    while y < height:
        path = rng.choice(tiles)
        tile = cv2.imread(str(path))
        if tile is None:
            continue
        mask_path = path.with_name(path.stem + '_ground.png')
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if mask is None:
            continue   # no mask means we cannot tell land from water: skip the tile
        if tile.shape[1] != FRAME_W:
            tile = cv2.resize(tile, (FRAME_W, FRAME_H))
        if mask.shape[:2] != (FRAME_H, FRAME_W):
            mask = cv2.resize(mask, (FRAME_W, FRAME_H), interpolation=cv2.INTER_NEAREST)
        if rng.random() < 0.5:
            tile, mask = cv2.flip(tile, 1), cv2.flip(mask, 1)
        take = min(FRAME_H, height - y)
        canvas[y:y + take] = tile[:take]
        ground[y:y + take] = mask[:take]
        y += take
    return sd.grade(canvas, rng), ground > 127


def place_objects(canvas, ground, n_objects, sprites, model_sprites, box_scales, rng):
    """Paste objects at fixed canvas positions; return their boxes in canvas coordinates."""
    height = canvas.shape[0]
    names = [c for c in sd.OBJECT_CLASSES if c in sprites or c in model_sprites]
    placed, objects = [], []
    shade = rng.uniform(*sd.OBJECT_SHADE)
    shadow = (rng.randint(-14, 14), rng.randint(-14, 14))   # paste() wants (dx, dy)
    # paste() bounds-checks against the frame height; our canvas is a whole flight tall.
    frame_h, sd.SOURCE_H = sd.SOURCE_H, height
    for i in range(n_objects * 40):
        if len(objects) >= n_objects:
            break
        name = names[len(objects) % len(names)] if len(objects) < len(names) else rng.choice(names)
        use_model = name in model_sprites and (name not in sprites or rng.random() < 0.5)
        cx = rng.uniform(EDGE_MARGIN, FRAME_W - EDGE_MARGIN)
        cy = rng.uniform(EDGE_MARGIN, height - EDGE_MARGIN)
        # Lean depends on where the object sits within its FRAME, not the canvas, so use the
        # position it will have when the window is centred on it.
        frame_y = FRAME_H / 2
        if use_model:
            sprite, _, _, _ = sd.pick_model_sprite(model_sprites[name], cx, frame_y, rng)
        else:
            sprite = sd.transform_sprite(rng.choice(sprites[name]), rng, (cx, frame_y))
        if sprite is None:
            continue
        h, w = sprite.shape[:2]
        x, y = int(cx - w / 2), int(cy - h / 2)
        box = [x, y, x + w, y + h]
        if x < 0 or y < 0 or x + w >= FRAME_W or y + h >= height or sd.overlaps(box, placed):
            continue
        if not sd.on_ground(ground, box):
            continue   # keep them off water and roofs, as in the supplied scenes
        patch = canvas[y:y + h, x:x + w, :3]
        sd.paste(canvas, sd.match_lighting(sprite, patch, rng, shade), x, y, shadow, rng)
        placed.append(box)
        kind = 'model' if use_model else 'cutout'
        objects.append({'class': name, 'box': sd.scale_box(box, box_scales[kind].get(name, 1.0))})
    sd.SOURCE_H = frame_h
    return objects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backgrounds', default=str(ROOT / 'backgrounds' / 'holdout'))
    parser.add_argument('--out', default=str(ROOT / 'data' / 'holdout'))
    parser.add_argument('--flights', type=int, default=4)
    parser.add_argument('--frames', type=int, default=60, help='frames per flight')
    parser.add_argument('--objects', type=int, default=60, help='objects per flight')
    parser.add_argument('--min-ground', type=float, default=0.20,
                        help='skip tiles with less open ground than this')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Only tiles with real open ground: after water was excluded from the masks some coastal
    # tiles have almost none, and a canvas of those places nothing at all.
    tiles = []
    for path in sorted(Path(args.backgrounds).glob('*.jpg')):
        mask_path = path.with_name(path.stem + '_ground.png')
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None and (mask > 127).mean() >= args.min_ground:
            tiles.append(path)
    if not tiles:
        raise SystemExit(f'no usable backgrounds in {args.backgrounds} '
                         f'(need >= {args.min_ground:.0%} open ground; run fetch_backgrounds.py --holdout '
                         f'then ground_masks.py)')
    print(f'{len(tiles)} tiles with >= {args.min_ground:.0%} open ground')
    out = Path(args.out)
    for sub in ('images', 'annotations'):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    sprites = sd.load_sprites(ROOT / 'sprites', include_suspect=False, include_truncated=False)
    model_sprites = sd.load_model_sprites(ROOT / 'datasets' / 'model_sprites')
    box_scales = sd.label_scales(ROOT / 'sprites', ROOT / 'datasets' / 'model_sprites')

    frame_no, totals = 0, {}
    for flight in range(args.flights):
        canvas, ground = build_canvas(tiles, args.frames, rng)
        objects = place_objects(canvas, ground, args.objects, sprites, model_sprites, box_scales, rng)
        for name in (o['class'] for o in objects):
            totals[name] = totals.get(name, 0) + 1
        drift_x = 0.0
        for i in range(args.frames):
            top = DRIFT_Y * i
            drift_x = max(-EDGE_MARGIN, min(EDGE_MARGIN, drift_x + rng.uniform(-DRIFT_X_JITTER, DRIFT_X_JITTER)))
            left = int(round(drift_x))
            window = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
            sx1, sx2 = max(0, left), min(FRAME_W, left + FRAME_W)
            window[:, sx1 - left:sx2 - left] = canvas[top:top + FRAME_H, sx1:sx2]
            annotations = []
            for o in objects:
                x1, y1, x2, y2 = o['box']
                fx1, fy1, fx2, fy2 = x1 - left, y1 - top, x2 - left, y2 - top
                # Same rule as the supplied labels: an object is annotated while its centre
                # is inside the frame, and keeps its whole box even when the edge cuts it.
                if not (0 <= (fx1 + fx2) / 2 < FRAME_W and 0 <= (fy1 + fy2) / 2 < FRAME_H):
                    continue
                annotations.append({'object_id': o['class'], 'bbox': [fx1, fy1, fx2, fy2]})
            frame_no += 1
            cv2.imwrite(str(out / 'images' / f'frame_{frame_no:06d}.png'), window)
            (out / 'annotations' / f'frame_{frame_no:06d}.json').write_text(
                json.dumps({'frame': frame_no, 'annotations': annotations}))
        print(f'flight {flight + 1}/{args.flights}: {len(objects)} objects, {args.frames} frames')

    (out / 'run_metadata.json').write_text(json.dumps({
        'scene': 'holdout', 'frames': frame_no, 'flights': args.flights,
        'objects_per_flight': args.objects, 'objects_by_class': totals,
        'backgrounds': args.backgrounds, 'drift_y_px_per_frame': DRIFT_Y,
        'note': 'Synthetic held-out test flight. Backgrounds from fetch_backgrounds.py --holdout, '
                'used by no training run. Labels exact by construction. Ranks background '
                'generalisation between models; the absolute score is not comparable to the '
                'supplied scenes, because the objects and the pasting are ours.',
    }, indent=2))
    print(f'{frame_no} frames in {out}, {sum(totals.values())} objects over {len(totals)} classes')


if __name__ == '__main__':
    main()
