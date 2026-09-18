"""Render 3D models top-down and match them to the class's real sprite cut-outs.

For each mesh found under models/<class>/, this renders it from above over a grid of
yaw and camera tilt, then for every Helsinki cut-out of that class finds the render
whose silhouette overlaps best, fits a per-channel colour correction, and reports how
close it gets. The output is a sheet to judge by eye (real sprite | best render per
model) plus models/<class>/match.json with the fitted size, yaws and colours, which
the sprite generator will use.

    python training/render_models.py hangar
    python training/render_models.py hangar --tilts 0 8 16 --yaw-step 3
    python training/render_models.py hangar --drop Cube.032     # leave out the model's concrete apron

Only silhouette, size and average colour survive at 20-180 px, so those are what is
matched; camo patterns and markings are approximated by the colour fit, not copied.

Sizes need no guessing: each render is scaled so its silhouette area equals the real
sprite's, and the implied real-world length is reported as a sanity check against the
Size column in models/MODELS.md.

Needs, on top of the training environment (kept out of requirements.txt because pyrender
pins old libraries that the IDUN setup would trip over):

    uv pip install --python .venv/bin/python --prerelease=allow trimesh pyrender "PyOpenGL>=3.1.7" "networkx>=3"

Meshes: .glb/.gltf/.obj load directly (trimesh). .fbx is not supported; convert it or
pick another format. glTF is Y-up by convention; use --up to override.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')  # headless rendering on the GPU

import cv2
import numpy as np
import pyrender
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))  # dtos.py lives in src/

from dtos import OBJECT_CLASSES  # noqa: E402

METRES_PER_PIXEL = 0.21
MESH_SUFFIXES = {'.glb', '.gltf', '.obj'}
RENDER_SIZE = 320     # render resolution; downscaled to the sprite's size afterwards
VIEW_HALF = 0.75      # orthographic half-width, in units of the mesh's longest horizontal side
SHEET_CELL = 200


# --------------------------------------------------------------------------- meshes

def find_meshes(class_dir: Path):
    meshes = sorted(p for p in class_dir.rglob('*') if p.suffix.lower() in MESH_SUFFIXES)
    # A glTF export may contain both a .gltf and its .bin/.glb twin; keep one per folder.
    seen, unique = set(), []
    for path in meshes:
        key = path.parent
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


TEXTURE_MAX_SIDE = 512  # the sprite is at most ~185 px; bigger textures only cost memory


def lighten_materials(meshes):
    """Keep only a small base-colour texture per material.

    Game-ready models ship 4K-8K normal, metallic and emissive maps; pyrender turns each
    into float arrays, and one 123 MB .glb reached 11 GB of RAM before this. None of
    those maps is visible in a colour render from 600 m, so drop them.
    """
    shrunk = {}
    for mesh in meshes:
        material = getattr(mesh.visual, 'material', None)
        if material is None:
            continue
        for attribute in ('normalTexture', 'metallicRoughnessTexture', 'emissiveTexture', 'occlusionTexture'):
            if getattr(material, attribute, None) is not None:
                setattr(material, attribute, None)
        # Without its texture the emissive factor alone would make the surface glow flat
        # white (glTF multiplies the two), so nothing may emit light.
        if getattr(material, 'emissiveFactor', None) is not None:
            material.emissiveFactor = np.zeros(3)
        for attribute in ('baseColorTexture', 'image'):  # PBR (glTF) and simple (OBJ) materials
            image = getattr(material, attribute, None)
            if image is None or not hasattr(image, 'size') or max(image.size) <= TEXTURE_MAX_SIDE:
                continue
            if id(image) not in shrunk:
                small = image.copy()
                small.thumbnail((TEXTURE_MAX_SIDE, TEXTURE_MAX_SIDE))
                shrunk[id(image)] = small
            setattr(material, attribute, shrunk[id(image)])


def part_name(mesh):
    return f"{mesh.metadata.get('name', '?')} ({mesh.metadata.get('node', '?')})"


def load_normalised(path: Path, up: str, drop=()):
    """Mesh list in a frame where +Z is up, centred, longest horizontal side = 1.

    `drop` removes parts whose geometry or node name contains any of the given strings,
    e.g. a concrete apron or display base the real object does not have.
    """
    loaded = trimesh.load(str(path), force='scene')
    meshes = [g for g in loaded.dump(concatenate=False) if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    dropped = [m for m in meshes if any(d in part_name(m) for d in drop)]
    meshes = [m for m in meshes if not any(d in part_name(m) for d in drop)]
    for mesh in dropped:
        print(f'    dropped part {part_name(mesh)}')
    if not meshes:
        raise ValueError(f'{path}: no triangle meshes left')
    lighten_materials(meshes)

    if up == 'auto':
        up = 'y' if path.suffix.lower() in {'.glb', '.gltf'} else 'z'
    if up == 'y':  # glTF convention: Y up, so rotate Y onto Z
        rotation = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        for mesh in meshes:
            mesh.apply_transform(rotation)

    bounds = np.array([m.bounds for m in meshes])
    low, high = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
    centre = (low + high) / 2
    centre[2] = low[2]  # stand it on the ground plane
    horizontal = max(high[0] - low[0], high[1] - low[1])
    for mesh in meshes:
        mesh.apply_translation(-centre)
        mesh.apply_scale(1.0 / horizontal)
    height = (high[2] - low[2]) / horizontal
    return meshes, height


def build_scene(meshes):
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.45, 0.45, 0.45])
    for mesh in meshes:
        try:
            scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
        except Exception:  # odd materials: fall back to the mesh's plain colour
            plain = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
            plain.visual.face_colors = [128, 128, 128, 255]
            scene.add(pyrender.Mesh.from_trimesh(plain, smooth=False))
    return scene


def render_views(meshes, height, yaws, tilts, sun_azimuth):
    """{(yaw, tilt): RGBA uint8} renders from above."""
    scene = build_scene(meshes)
    node_all = list(scene.mesh_nodes)
    camera = pyrender.OrthographicCamera(xmag=VIEW_HALF, ymag=VIEW_HALF, znear=0.01, zfar=20)
    camera_node = scene.add(camera, pose=np.eye(4))
    sun = pyrender.DirectionalLight(color=np.ones(3), intensity=3.5)
    sun_pose = trimesh.transformations.euler_matrix(np.radians(30), 0, np.radians(sun_azimuth), 'rzxy')
    scene.add(sun, pose=sun_pose)
    renderer = pyrender.OffscreenRenderer(RENDER_SIZE, RENDER_SIZE)

    views = {}
    for yaw in yaws:
        spin = trimesh.transformations.rotation_matrix(np.radians(yaw), [0, 0, 1])
        for node in node_all:
            scene.set_pose(node, spin)
        for tilt in tilts:
            pose = trimesh.transformations.rotation_matrix(np.radians(tilt), [1, 0, 0])
            pose = pose @ trimesh.transformations.translation_matrix([0, 0, 5 + height])
            scene.set_pose(camera_node, pose)
            color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            views[(yaw, tilt)] = color
    renderer.delete()
    return views


# --------------------------------------------------------------------------- matching

def load_sprites(class_name: str):
    index = json.loads((ROOT / 'sprites' / 'index.json').read_text())
    sprites = []
    for entry in index:
        if entry['class'] != class_name or entry.get('truncated'):
            continue
        image = cv2.imread(str(ROOT / 'sprites' / entry['file']), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape[2] != 4:
            continue
        sprites.append((entry['file'], cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)))
    return sprites


def fit_to_sprite(render: np.ndarray, sprite: np.ndarray):
    """Scale and centre a render onto the sprite's canvas so the silhouette areas match.

    Returns (placed RGBA, IoU, pixels per mesh unit) or None.
    """
    sprite_mask = sprite[:, :, 3] > 128
    render_mask = render[:, :, 3] > 0
    sprite_area, render_area = sprite_mask.sum(), render_mask.sum()
    if sprite_area < 8 or render_area < 8:
        return None

    scale = np.sqrt(sprite_area / render_area)
    ys, xs = np.nonzero(render_mask)
    crop = render[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    new_w = max(1, int(round(crop.shape[1] * scale)))
    new_h = max(1, int(round(crop.shape[0] * scale)))
    small = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    height, width = sprite.shape[:2]
    placed = np.zeros_like(sprite)
    sy, sx = np.nonzero(sprite_mask)
    cy, cx = sy.mean(), sx.mean()
    my, mx = np.nonzero(small[:, :, 3] > 64)
    if len(my) == 0:
        return None
    top = int(round(cy - my.mean()))
    left = int(round(cx - mx.mean()))
    y0, x0 = max(top, 0), max(left, 0)
    y1, x1 = min(top + new_h, height), min(left + new_w, width)
    if y1 <= y0 or x1 <= x0:
        return None
    placed[y0:y1, x0:x1] = small[y0 - top:y1 - top, x0 - left:x1 - left]

    placed_mask = placed[:, :, 3] > 64
    union = (placed_mask | sprite_mask).sum()
    iou = float((placed_mask & sprite_mask).sum() / union) if union else 0.0
    pixels_per_unit = RENDER_SIZE / (2 * VIEW_HALF) * scale
    return placed, iou, pixels_per_unit


def fit_colour(placed: np.ndarray, sprite: np.ndarray):
    """Per-channel gain and offset mapping the render's colours onto the sprite's."""
    both = (placed[:, :, 3] > 64) & (sprite[:, :, 3] > 128)
    if both.sum() < 8:
        return np.ones(3), np.zeros(3), 999.0
    source = placed[both][:, :3].astype(np.float64)
    target = sprite[both][:, :3].astype(np.float64)
    gains, offsets = np.ones(3), np.zeros(3)
    for c in range(3):
        spread = source[:, c].std()
        # A flat-shaded render can be nearly one colour, where gain and offset are not
        # separable; then only shift it. Either way the offset is set last, from the
        # clipped gain, so the average colour always matches the sprite.
        gain = np.cov(source[:, c], target[:, c])[0, 1] / spread ** 2 if spread > 2.0 else 1.0
        gains[c] = np.clip(gain, 0.2, 4.0)
        offsets[c] = target[:, c].mean() - gains[c] * source[:, c].mean()
    fitted = np.clip(source * gains + offsets, 0, 255)
    error = float(np.abs(fitted - target).mean())
    return gains, offsets, error


def apply_colour(placed: np.ndarray, gains, offsets):
    out = placed.copy()
    rgb = out[:, :, :3].astype(np.float64) * gains + offsets
    out[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return out


# --------------------------------------------------------------------------- sheet

def to_cell(rgba: np.ndarray, label_lines):
    background = np.full((SHEET_CELL, SHEET_CELL, 3), 235, np.uint8)
    if rgba is not None and rgba.size:
        h, w = rgba.shape[:2]
        scale = (SHEET_CELL - 50) / max(h, w)
        big = cv2.resize(rgba, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_NEAREST)
        alpha = big[:, :, 3:4] / 255.0
        y = 40 + (SHEET_CELL - 50 - big.shape[0]) // 2
        x = (SHEET_CELL - big.shape[1]) // 2
        region = background[y:y + big.shape[0], x:x + big.shape[1]].astype(np.float64)
        background[y:y + big.shape[0], x:x + big.shape[1]] = (big[:, :, :3] * alpha + region * (1 - alpha)).astype(np.uint8)
    for i, line in enumerate(label_lines):
        cv2.putText(background, line, (4, 14 + 13 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1)
    return cv2.cvtColor(background, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('class_name', choices=sorted(OBJECT_CLASSES))
    parser.add_argument('--yaw-step', type=int, default=5)
    parser.add_argument('--tilts', type=float, nargs='*', default=[0, 10, 20],
                        help='camera tilt from straight down, degrees (the imagery is near-nadir)')
    parser.add_argument('--sun-azimuth', type=float, default=135)
    parser.add_argument('--up', choices=['auto', 'y', 'z'], default='auto')
    parser.add_argument('--drop', nargs='*', default=[],
                        help='parts to leave out, matched against geometry/node names (e.g. Cube.032)')
    parser.add_argument('--max-sprites', type=int, default=8)
    parser.add_argument('--out', default=str(ROOT / 'datasets' / 'model_match'))
    args = parser.parse_args()

    class_dir = ROOT / 'models' / args.class_name
    mesh_paths = find_meshes(class_dir)
    if not mesh_paths:
        raise SystemExit(f'no .glb/.gltf/.obj under {class_dir}; download a model there first')
    sprites = load_sprites(args.class_name)
    if not sprites:
        raise SystemExit(f'no sprite cut-outs for {args.class_name}; run training/extract_sprites.py')
    sprites.sort(key=lambda item: -(item[1][:, :, 3] > 128).sum())
    sprites = sprites[:args.max_sprites]

    yaws = list(range(0, 360, args.yaw_step))
    print(f'{args.class_name}: {len(mesh_paths)} model(s), {len(sprites)} cut-outs, '
          f'{len(yaws)} yaws x {len(args.tilts)} tilts')

    results = {}
    for path in mesh_paths:
        name = str(path.parent.relative_to(class_dir)) if path.parent != class_dir else path.stem
        try:
            meshes, height = load_normalised(path, args.up, args.drop)
        except Exception as exc:
            print(f'  {name}: cannot load ({exc})')
            continue
        views = render_views(meshes, height, yaws, args.tilts, args.sun_azimuth)
        print(f'  {name}: {sum(len(m.faces) for m in meshes)} faces, height/length {height:.2f}, '
              f'{len(views)} renders')

        per_sprite = []
        for file, sprite in sprites:
            best = None
            for (yaw, tilt), render in views.items():
                fitted = fit_to_sprite(render, sprite)
                if fitted and (best is None or fitted[1] > best['iou']):
                    best = {'yaw': yaw, 'tilt': tilt, 'placed': fitted[0], 'iou': fitted[1],
                            'pixels_per_unit': fitted[2]}
            if best is None:
                per_sprite.append(None)
                continue
            gains, offsets, error = fit_colour(best['placed'], sprite)
            best.update(gains=gains, offsets=offsets, colour_error=error,
                        coloured=apply_colour(best['placed'], gains, offsets),
                        length_m=best['pixels_per_unit'] * METRES_PER_PIXEL, file=file)
            per_sprite.append(best)
        results[name] = per_sprite

        good = [b for b in per_sprite if b]
        if good:
            summary = {
                'mesh': str(path.relative_to(ROOT)),
                'dropped_parts': args.drop,
                'mean_iou': round(float(np.mean([b['iou'] for b in good])), 3),
                'mean_colour_error': round(float(np.mean([b['colour_error'] for b in good])), 1),
                'length_m': round(float(np.median([b['length_m'] for b in good])), 1),
                'tilt': float(np.median([b['tilt'] for b in good])),
                'gains': np.median([b['gains'] for b in good], axis=0).round(3).tolist(),
                'offsets': np.median([b['offsets'] for b in good], axis=0).round(1).tolist(),
                'per_sprite': [{'file': b['file'], 'yaw': b['yaw'], 'tilt': b['tilt'],
                                'iou': round(b['iou'], 3), 'colour_error': round(b['colour_error'], 1)}
                               for b in good],
            }
            match_path = class_dir / 'match.json'
            existing = json.loads(match_path.read_text()) if match_path.exists() else {}
            existing[name] = summary
            match_path.write_text(json.dumps(existing, indent=1, sort_keys=True))
            print(f'    mean IoU {summary["mean_iou"]:.3f}, colour error {summary["mean_colour_error"]:.1f}/255, '
                  f'implied length {summary["length_m"]} m')

    # Sheet: one row per real cut-out, real sprite first, then each model's best match.
    rows = []
    for i, (file, sprite) in enumerate(sprites):
        cells = [to_cell(sprite, ['REAL', Path(file).name])]
        for name, per_sprite in results.items():
            best = per_sprite[i]
            if best is None:
                cells.append(to_cell(None, [name[:26], 'no match']))
                continue
            cells.append(to_cell(best['coloured'], [
                name[:26],
                f"IoU {best['iou']:.2f}  colour err {best['colour_error']:.0f}",
                f"yaw {best['yaw']}  tilt {best['tilt']:.0f}  {best['length_m']:.1f} m",
            ]))
        rows.append(np.hstack(cells))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sheet_path = out / f'{args.class_name}_match.png'
    cv2.imwrite(str(sheet_path), np.vstack(rows))
    print(f'sheet -> {sheet_path}')


if __name__ == '__main__':
    main()
