"""Build a synthetic training set: challenge sprites pasted onto aerial backgrounds.

Why: the supplied training scene is one Helsinki flight (forest and lake), so the
detector memorises it and calls city roads and roofs "hangar". The objects are the
same 3D models in every scene, so the fix is to keep the objects and vary the
ground under them.

The recorded validation frames are deliberately NOT used as backgrounds: they are
the only honest test set we have until the model has been submitted.

    python training/fetch_backgrounds.py                 # aerial backgrounds first
    python training/extract_sprites.py                   # sprite cut-outs (once)
    python training/synth_dataset.py                     # -> datasets/synth_yolo
    python training/synth_dataset.py --frames 400 --with-helsinki
    python training/synth_dataset.py --preview 3         # full frames to eyeball
    python training/model_sprites.py                     # renders from the painted 3D models
    python training/render_city.py --count 40            # Helsinki 3D-mesh backgrounds

Each synthetic frame is a 3840x2160 background with 8-30 objects pasted on it,
then cut into Level 0/1/2 views exactly like the evaluator sends them.

Objects come from two banks: the real cut-outs (sprites/), randomly rotated, and
renders of the painted 3D models (datasets/model_sprites/, --model-share of them).
A model render is chosen for the spot it is pasted at: tall objects lean away from the
camera's nadir point near the bottom of the frame (drone_camera.lean_at), so the render
whose tilt and lean match that spot is used, and it is not rotated afterwards.

Backgrounds with a <name>_ground.png (render_city.py) only get objects on open ground,
as in the supplied scenes. Every background is colour-graded towards the supplied
frames first: they are darker and far more saturated than the mesh or NAIP imagery.
"""

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))  # dtos.py / utils.py live in src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dtos import OBJECT_CLASSES  # noqa: E402
from make_dataset import (  # noqa: E402
    CLASS_INDEX,
    crop_view,
    labels_for_region,
    view_centres,
)
from utils import frame_numbers, load_annotations, load_frame  # noqa: E402
from drone_camera import METRES_PER_PIXEL, lean_at  # noqa: E402

SOURCE_W, SOURCE_H = 3840, 2160
# Classes that tend to appear together in the supplied scenes, so the detector sees
# the same context it will meet live (a hangar with planes beside it, and so on).
GROUPS = [
    ('hangar', 'small_plane', 'small_plane', 'jet_plane'),
    ('hangar', 'medium_plane', 'spacecraft'),
    ('large_tower', 'small_tower', 'jammer'),
    ('large_launcher', 'medium_launcher', 'small_launcher'),
    ('tank', 'tank', 'mine_roller', 'ta-ta'),
    ('helicopter', 'helicopter', 'condor'),
]
# Look of the supplied frames (grey mean ~96, saturation ~110 on 0-255, over their L0
# views); the mesh renders are ~144 and ~28, NAIP ~159 and ~31.
GRADE_MEAN = (80, 115)
GRADE_SATURATION = (80, 135)
MAX_GAIN = 1.4  # lighting match: more than this turned the near-black hangars neon yellow
MAX_SATURATION_BOOST = 2.5  # beyond this, colour noise and casts dominate
# Objects on a ground mask keep this far from anything that is not open ground (walls, trees),
# and all of their box must be on it: 70% let hangars run half into buildings.
GROUND_CLEARANCE_M = 3.0
GROUND_SHARE = 0.98
ALPHA_EDGE = 8  # alpha above this is the object; transform_sprite trims to it


def label_scales(sprite_dir: Path, model_dir: Path) -> dict:
    """{'cutout' | 'model': {class: factor}} from a pasted sprite's tight box to a supplied-style label.

    The supplied boxes are not tight: each class's is its object's box scaled by a steady
    factor (1.1 for the spacecraft, 1.7 for the jet, 2.5 for the launchers; the spread
    between sprites of a class is a few percent). A detector trained on tight synthetic
    boxes finds those objects at IoU ~0.4, under the 0.5 the score needs, so the synthetic
    labels are scaled the same way.

    Cut-outs: the median over the real cut-outs of sqrt(label area / object area).
    Model renders: the same factor, unless the renders are bigger than the cut-outs; then
    they show what GrabCut left out (the helicopter's rotor: 95 px against 63), and the
    factor is the label size over the render size, never below 1.
    """
    ratios, labels, cutouts = {}, {}, {}
    for entry in json.loads((sprite_dir / 'index.json').read_text()):
        if entry.get('truncated') or entry.get('suspect'):
            continue
        image = cv2.imread(str(sprite_dir / entry['file']), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape[2] != 4:
            continue
        ys, xs = np.nonzero(image[:, :, 3] > ALPHA_EDGE)
        if len(xs) < 4:
            continue
        x1, y1, x2, y2 = entry['bbox']
        tight = math.sqrt((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
        label = math.sqrt((x2 - x1) * (y2 - y1))
        ratios.setdefault(entry['class'], []).append(label / tight)
        labels.setdefault(entry['class'], []).append(label)
        cutouts.setdefault(entry['class'], []).append(tight)
    cutout = {name: float(np.median(values)) for name, values in ratios.items()}

    renders = {}
    index_path = model_dir / 'index.json'
    for entry in json.loads(index_path.read_text()) if index_path.exists() else []:
        image = cv2.imread(str(model_dir / entry['file']), cv2.IMREAD_UNCHANGED)
        if image is not None and image.shape[2] == 4:
            renders.setdefault(entry['class'], []).append(math.sqrt(image.shape[0] * image.shape[1]))
    model = {}
    for name, factor in cutout.items():
        if name in renders and np.median(renders[name]) > np.median(cutouts[name]):
            factor = min(factor, max(float(np.median(labels[name]) / np.median(renders[name])), 1.0))
        model[name] = factor
    return {'cutout': cutout, 'model': model}


def scale_box(box, factor: float):
    """The box scaled about its centre, kept inside the 4K frame."""
    x1, y1, x2, y2 = box
    cx, cy, half_w, half_h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * factor / 2, (y2 - y1) * factor / 2
    return [max(int(round(cx - half_w)), 0), max(int(round(cy - half_h)), 0),
            min(int(round(cx + half_w)), SOURCE_W), min(int(round(cy + half_h)), SOURCE_H)]


def load_sprites(sprite_dir: Path, include_suspect: bool, include_truncated: bool):
    """Sprite cut-outs by class, preferring hand-reviewed outlines."""
    index = json.loads((sprite_dir / 'index.json').read_text())
    manual = {}
    manual_path = sprite_dir / 'manual.json'
    if manual_path.exists():
        manual = json.loads(manual_path.read_text())

    by_class: dict[str, list[np.ndarray]] = {name: [] for name in OBJECT_CLASSES}
    for entry in index:
        if entry.get('truncated') and not include_truncated:
            continue
        if entry.get('suspect') and not include_suspect and entry['file'] not in manual:
            continue
        image = cv2.imread(str(sprite_dir / entry['file']), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape[2] != 4 or image.shape[0] < 4 or image.shape[1] < 4:
            continue
        by_class[entry['class']].append(image)
    return {name: sprites for name, sprites in by_class.items() if sprites}


def load_model_sprites(bank_dir: Path):
    """{class: [(RGBA, tilt, lean, yaw)]} from model_sprites.py, or {} when there is no bank."""
    index_path = bank_dir / 'index.json'
    if not index_path.exists():
        return {}
    by_class: dict[str, list] = {}
    for entry in json.loads(index_path.read_text()):
        image = cv2.imread(str(bank_dir / entry['file']), cv2.IMREAD_UNCHANGED)
        if image is not None and image.shape[2] == 4:
            by_class.setdefault(entry['class'], []).append((image, entry['tilt'], entry['lean'], entry['yaw']))
    return by_class


def pick_model_sprite(bank, cx: float, cy: float, rng: random.Random):
    """The (RGBA, tilt, lean, yaw) render, of a random few, whose tilt and lean best match frame position (cx, cy)."""
    tilt, lean = lean_at(cx, cy)
    candidates = rng.sample(bank, min(60, len(bank)))

    def cost(item):
        t, l = item[1], item[2]
        turn = abs((l - lean + 180) % 360 - 180)
        # Lean direction matters in proportion to how tilted the view is.
        return ((t - tilt) / 6) ** 2 + (turn / 25 * math.sin(math.radians(tilt))) ** 2

    return min(candidates, key=cost)


def transform_sprite(sprite: np.ndarray, rng: random.Random):
    """Random rotation, scale and flip. Returns the RGBA sprite, still tight around its alpha."""
    if rng.random() < 0.5:
        sprite = cv2.flip(sprite, 1)
    scale = rng.uniform(0.75, 1.35)
    angle = rng.uniform(0, 360)

    h, w = sprite.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    out_w, out_h = int(w * cos + h * sin) + 2, int(w * sin + h * cos) + 2
    matrix[0, 2] += out_w / 2 - w / 2
    matrix[1, 2] += out_h / 2 - h / 2
    rotated = cv2.warpAffine(sprite, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Trim back to the alpha, so the label is the object and not the rotation padding.
    ys, xs = np.nonzero(rotated[:, :, 3] > ALPHA_EDGE)
    if len(xs) == 0:
        return None
    return rotated[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def match_lighting(sprite: np.ndarray, background_patch: np.ndarray, rng: random.Random):
    """Nudge the sprite towards the local exposure, then jitter it a little."""
    rgb = sprite[:, :, :3].astype(np.float32)
    alpha = sprite[:, :, 3:4].astype(np.float32) / 255.0
    visible = alpha > 0.5
    if visible.sum() < 4:
        return sprite

    sprite_mean = float((rgb * alpha).sum() / max(alpha.sum() * 3, 1))
    background_mean = float(background_patch.mean())
    blend = rng.uniform(0.35, 0.75)  # partly match the scene, keep some of the model's own tone
    gain = (background_mean * blend + sprite_mean * (1 - blend)) / max(sprite_mean, 1e-3)
    gain = min(max(gain, 1 / MAX_GAIN), MAX_GAIN)
    rgb *= gain * rng.uniform(0.92, 1.08)
    rgb += rng.uniform(-10, 10)

    if rng.random() < 0.5:  # mild colour cast, the models are re-lit per scene
        rgb *= np.array([rng.uniform(0.94, 1.06) for _ in range(3)], np.float32)

    out = sprite.copy()
    out[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return out


def grade(background: np.ndarray, rng: random.Random):
    """Darken and saturate a background towards the supplied frames' look."""
    # Levels first: haze is an offset, and removing it restores colour a plain saturation
    # boost would turn into a cast (NAIP is bluish). One range for all channels: per-channel
    # ranges shift the white balance (sandy harbours went pink).
    small = background[::8, ::8].astype(np.float32)
    low, high = np.percentile(small, 0.5), np.percentile(small, 99.5)
    stretched = np.clip((background - low) / max(high - low, 1) * 255, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(stretched, cv2.COLOR_BGR2HSV).astype(np.float32)
    saturation = max(float(hsv[:, :, 1].mean()), 1.0)
    boost = min(rng.uniform(*GRADE_SATURATION) / saturation, MAX_SATURATION_BOOST)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * boost, 0, 255)
    value = max(float(hsv[:, :, 2].mean()) / 255, 0.05)
    target = rng.uniform(*GRADE_MEAN) / 255 * 1.15  # HSV value runs a little above grey
    gamma = math.log(min(target, 0.95)) / math.log(value)
    hsv[:, :, 2] = 255 * (hsv[:, :, 2] / 255) ** gamma
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def load_background(path: Path):
    """(BGR frame, open-ground mask shrunk by GROUND_CLEARANCE_M, or None) for one background."""
    mask_path = path.with_name(path.stem + '_ground.png')
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
    if mask is None:
        return cv2.imread(str(path)), None
    metres_per_px = METRES_PER_PIXEL * SOURCE_W / mask.shape[1]
    r = int(math.ceil(GROUND_CLEARANCE_M / metres_per_px))
    mask = cv2.erode((mask > 127).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return cv2.imread(str(path)), mask > 0


def pick_spot(ground, rng: random.Random):
    """(cx, cy) in the frame: anywhere, or on open ground when there is a mask."""
    if ground is None:
        return rng.uniform(0, SOURCE_W), rng.uniform(0, SOURCE_H)
    ys, xs = np.nonzero(ground)
    i = rng.randrange(len(xs))
    sx, sy = SOURCE_W / ground.shape[1], SOURCE_H / ground.shape[0]
    return (xs[i] + rng.random()) * sx, (ys[i] + rng.random()) * sy


def on_ground(ground, box, share: float = GROUND_SHARE) -> bool:
    """Whether most of the box lies on open ground (always, without a mask)."""
    if ground is None:
        return True
    sx, sy = ground.shape[1] / SOURCE_W, ground.shape[0] / SOURCE_H
    x1, y1, x2, y2 = box
    patch = ground[int(y1 * sy):int(y2 * sy) + 1, int(x1 * sx):int(x2 * sx) + 1]
    return patch.size > 0 and patch.mean() >= share


def class_size(name: str, sprites: dict, model_sprites: dict) -> float:
    """Median on-screen size (px, longest side) of a class's sprites."""
    images = [item[0] for item in model_sprites.get(name, [])] or sprites.get(name, [])
    return float(np.median([max(image.shape[:2]) for image in images])) if images else 0.0


def paste(canvas: np.ndarray, sprite: np.ndarray, x: int, y: int, shadow: tuple, rng: random.Random):
    """Alpha-blend a sprite at (x, y) with a soft drop shadow. Returns its box."""
    h, w = sprite.shape[:2]
    if x < 0 or y < 0 or x + w > SOURCE_W or y + h > SOURCE_H:
        return None
    alpha = (sprite[:, :, 3:4].astype(np.float32) / 255.0)

    dx, dy = shadow
    sx, sy = x + dx, y + dy
    if 0 <= sx and 0 <= sy and sx + w <= SOURCE_W and sy + h <= SOURCE_H:
        shadow_alpha = cv2.GaussianBlur(alpha[:, :, 0], (0, 0), 1.6)[:, :, None] * rng.uniform(0.25, 0.5)
        region = canvas[sy:sy + h, sx:sx + w].astype(np.float32)
        canvas[sy:sy + h, sx:sx + w] = (region * (1 - shadow_alpha)).astype(np.uint8)

    region = canvas[y:y + h, x:x + w].astype(np.float32)
    blended = sprite[:, :, :3].astype(np.float32) * alpha + region * (1 - alpha)
    canvas[y:y + h, x:x + w] = np.clip(blended, 0, 255).astype(np.uint8)
    return [x, y, x + w, y + h]


def overlaps(box, placed, margin: int = 6) -> bool:
    x1, y1, x2, y2 = box
    for px1, py1, px2, py2 in placed:
        if x1 < px2 + margin and px1 < x2 + margin and y1 < py2 + margin and py1 < y2 + margin:
            return True
    return False


def compose_frame(background: np.ndarray, sprites: dict, rng: random.Random, n_objects: int,
                  model_sprites: dict = None, model_share: float = 0.0, ground=None, classes: list = None,
                  box_scales: dict = None):
    """Paste objects onto a graded copy of the background. Returns (frame, annotations).

    `classes` fixes which objects to paste (scene3d.py asks for one of each); by default
    n_objects are drawn at random, partly in themed groups. `box_scales` (label_scales, per bank)
    turns each tight pasted box into a box like the supplied labels; without it they stay tight.
    """
    model_sprites = model_sprites or {}
    box_scales = box_scales or {}
    if ground is not None and not ground.any():
        ground = None
    canvas = grade(background, rng)
    if rng.random() < 0.6:  # the challenge imagery is soft; vary how soft ours is
        canvas = cv2.GaussianBlur(canvas, (0, 0), rng.uniform(0.3, 0.9))

    angle = rng.uniform(0, 2 * math.pi)  # one sun direction per frame
    distance = rng.uniform(2, 6)
    shadow = (int(round(math.cos(angle) * distance)), int(round(math.sin(angle) * distance)))

    annotations, placed = [], []
    wanted = list(classes or [])
    while len(wanted) < n_objects:
        if rng.random() < 0.45:  # a themed group, as the scenes have
            wanted.extend(name for name in rng.choice(GROUPS) if name in sprites or name in model_sprites)
        else:
            wanted.append(rng.choice(sorted(set(sprites) | set(model_sprites))))
    rng.shuffle(wanted)
    wanted = wanted[:n_objects]
    if ground is not None:  # big objects first, while there is still room for them on the open ground
        wanted.sort(key=lambda name: -class_size(name, sprites, model_sprites))

    for class_name in wanted:
        use_model = class_name in model_sprites and (class_name not in sprites or rng.random() < model_share)
        sprite = None if use_model else transform_sprite(rng.choice(sprites[class_name]), rng)
        if sprite is None and not use_model:
            continue
        for _ in range(30 if ground is None else 80):  # try a few spots before giving up on this object
            cx, cy = pick_spot(ground, rng)
            if use_model:  # the spot decides the pose: pick the matching render
                sprite, _, _, yaw = pick_model_sprite(model_sprites[class_name], cx, cy, rng)
            h, w = sprite.shape[:2]
            if h >= SOURCE_H or w >= SOURCE_W:
                break
            x, y = int(cx - w / 2), int(cy - h / 2)
            if x < 0 or y < 0 or x + w >= SOURCE_W or y + h >= SOURCE_H:
                continue
            box = [x, y, x + w, y + h]
            if overlaps(box, placed) or not on_ground(ground, box):
                continue
            lit = match_lighting(sprite, canvas[y:y + h, x:x + w, :3], rng)
            pasted = paste(canvas, lit, x, y, shadow, rng)
            if pasted is None:
                continue
            placed.append(pasted)
            annotations.append({'object_id': class_name, 'bbox': scale_box(pasted, box_scales.get('model' if use_model else 'cutout', {}).get(class_name, 1.0))})
            if use_model:  # the render's yaw: scene3d.py turns the 3D model to match
                annotations[-1]['yaw'] = yaw
            break

    if rng.random() < 0.7:  # sensor noise, so the pasted edges are not the only grain
        noise = rng.uniform(1.0, 3.5)
        canvas = np.clip(canvas.astype(np.float32) + np.random.normal(0, noise, canvas.shape), 0, 255).astype(np.uint8)
    return canvas, annotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backgrounds', nargs='*',
                        default=[str(ROOT / 'backgrounds' / 'naip'), str(ROOT / 'backgrounds' / 'helsinki3d_frames')])
    parser.add_argument('--model-sprites', default=str(ROOT / 'datasets' / 'model_sprites'))
    parser.add_argument('--model-share', type=float, default=0.5,
                        help='share of objects taken from the 3D-model renders when a class has both')
    parser.add_argument('--sprites', default=str(ROOT / 'sprites'))
    parser.add_argument('--out', default=str(ROOT / 'datasets' / 'synth_yolo'))
    parser.add_argument('--frames', type=int, default=300, help='synthetic 4K frames to compose')
    parser.add_argument('--min-objects', type=int, default=8)
    parser.add_argument('--max-objects', type=int, default=30)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--val-dir', help='validate on these images instead of a synthetic split (every synthetic '
                                          'frame then goes to train), e.g. the real Helsinki views of '
                                          'make_dataset.py --all-val')
    parser.add_argument('--with-helsinki', action='store_true', help='also cut views from the supplied scene')
    parser.add_argument('--include-suspect', action='store_true', help='use sprites flagged SUSPECT too')
    parser.add_argument('--include-truncated', action='store_true', help='use sprites cut by the frame edge')
    parser.add_argument('--l1-random', type=int, default=6)
    parser.add_argument('--l1-per-object', type=int, default=1)
    parser.add_argument('--l2-random', type=int, default=8)
    parser.add_argument('--l2-per-object', type=int, default=1)
    parser.add_argument('--preview', type=int, default=0, help='write N full composed frames and stop')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    background_paths = sorted(p for d in args.backgrounds for p in Path(d).glob('*.jpg'))
    if not background_paths:
        raise SystemExit(f'no backgrounds in {args.backgrounds}; run training/fetch_backgrounds.py first')
    sprites = load_sprites(Path(args.sprites), args.include_suspect, args.include_truncated)
    model_sprites = load_model_sprites(Path(args.model_sprites))
    box_scales = label_scales(Path(args.sprites), Path(args.model_sprites))
    for bank, factors in box_scales.items():
        print(f'label / {bank} box: ' + ', '.join(f'{name} {factor:.2f}' for name, factor in sorted(factors.items())))
    missing = [name for name in OBJECT_CLASSES if name not in sprites and name not in model_sprites]
    per_dir = ', '.join(f'{Path(d).name} {len(list(Path(d).glob("*.jpg")))}' for d in args.backgrounds)
    print(f'{len(background_paths)} backgrounds ({per_dir}); real cut-outs for {len(sprites)} classes, '
          f'model renders for {len(model_sprites)} ({sum(map(len, model_sprites.values()))} sprites)'
          + (f'; missing: {", ".join(missing)}' if missing else ''))

    out = Path(args.out)
    if args.preview:
        preview_dir = out.parent / 'synth_preview'
        preview_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.preview):
            background, ground = load_background(rng.choice(background_paths))
            frame, annotations = compose_frame(background, sprites, rng,
                                               rng.randint(args.min_objects, args.max_objects),
                                               model_sprites, args.model_share, ground, box_scales=box_scales)
            marked = frame.copy()
            for ann in annotations:
                x1, y1, x2, y2 = ann['bbox']
                cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(marked, ann['object_id'], (x1, max(y1 - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(str(preview_dir / f'synth_{i:02d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(preview_dir / f'synth_{i:02d}_boxes.jpg'), marked, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f'preview {i}: {len(annotations)} objects')
        print(f'previews in {preview_dir}')
        return

    if out.exists():
        shutil.rmtree(out)
    for split in ('train', 'val'):
        (out / 'images' / split).mkdir(parents=True)
        (out / 'labels' / split).mkdir(parents=True)

    counts = {'train': 0, 'val': 0}
    per_class = {name: 0 for name in OBJECT_CLASSES}

    def cut(frame, annotations, stem, split):
        plan = [
            (0, view_centres(0, annotations, 0, 0, rng)),
            (1, view_centres(1, annotations, args.l1_random, args.l1_per_object, rng)),
            (2, view_centres(2, annotations, args.l2_random, args.l2_per_object, rng)),
        ]
        for level, centres in plan:
            for i, (cx, cy) in enumerate(centres):
                view, region = crop_view(frame, level, cx, cy)
                lines = labels_for_region(annotations, region)
                name = f'{stem}_L{level}_{i:03d}_{cx}_{cy}'
                cv2.imwrite(str(out / 'images' / split / f'{name}.jpg'), view, [cv2.IMWRITE_JPEG_QUALITY, 95])
                (out / 'labels' / split / f'{name}.txt').write_text('\n'.join(lines))
                counts[split] += 1

    for n in range(args.frames):
        split = 'val' if not args.val_dir and rng.random() < args.val_fraction else 'train'
        background, ground = load_background(rng.choice(background_paths))
        frame, annotations = compose_frame(background, sprites, rng,
                                           rng.randint(args.min_objects, args.max_objects),
                                           model_sprites, args.model_share, ground, box_scales=box_scales)
        for ann in annotations:
            per_class[ann['object_id']] += 1
        cut(frame, annotations, f's{n:04d}', split)
        if (n + 1) % 25 == 0:
            print(f'{n + 1}/{args.frames} frames composed ({counts["train"]} train views)')

    if args.with_helsinki:
        for frame_no in frame_numbers():
            split = 'val' if frame_no in (6, 13, 21) else 'train'
            cut(load_frame(frame_no), load_annotations(frame_no), f'h{frame_no:03d}', split)

    names = '\n'.join(f'  {i}: {name}' for i, name in enumerate(OBJECT_CLASSES))
    val = Path(args.val_dir).resolve() if args.val_dir else 'images/val'  # YOLO takes an absolute val path as is
    (out / 'data.yaml').write_text(f'path: {out}\ntrain: images/train\nval: {val}\nnames:\n{names}\n')
    rare = ', '.join(f'{name} {count}' for name, count in sorted(per_class.items(), key=lambda kv: kv[1])[:5])
    print(f"wrote {counts['train']} train / {counts['val']} val views to {out}")
    print(f'objects pasted: {sum(per_class.values())} (rarest: {rare})')
    print(f'class index order matches {len(CLASS_INDEX)} challenge classes')


if __name__ == '__main__':
    main()
