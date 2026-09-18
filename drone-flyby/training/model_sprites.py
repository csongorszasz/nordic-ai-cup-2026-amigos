"""Render a bank of sprite cut-outs from the painted 3D models, for synth_dataset.py.

The real cut-outs show each object in the few poses the Helsinki flight happened to
catch. The painted models (render_models.py, datasets/model_match/_baked/) carry the
same shape and real colours, so they can be rendered in any pose: every class gets
new yaws, camera tilts and lean directions, at the size the real object has on screen.

    python training/model_sprites.py                    # -> datasets/model_sprites/
    python training/model_sprites.py --per-class 300 --classes tank helicopter

Per class it uses the model with the best fit in models/<class>/match.json that has a
painted .glb. Size: the fitted length in metres, over METRES_PER_PIXEL, jittered a
little. Tilt 0-45 deg and lean any direction; index.json records both, so the generator
can pick the render that matches where in the frame it pastes it. Output: RGBA PNGs
tight around the object, plus index.json in the same format as sprites/index.json.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_models as rm  # noqa: E402

ROOT = rm.ROOT
OUT = ROOT / 'datasets' / 'model_sprites'
SIZE_JITTER = 0.08   # relative; the fitted lengths vary about this much between frames
MAX_TILT = 45        # degrees off vertical at the far corners of the frame (drone_camera.lean_at)
# Rotor radius in model units (longest side = 1). The rotor was dropped from the fit
# because it spins; the real cut-outs show it as anything from invisible to a pale
# translucent disc, but the label box always spans the whole rotor (the Mi-28's rotor
# is about as wide as the fuselage is long), so it is drawn back here.
ROTORS = {'helicopter': 0.5}


def painted_model(class_name: str):
    """(name, fit) of the best-fitting model of a class that has a painted .glb, or None."""
    match_path = ROOT / 'models' / class_name / 'match.json'
    if not match_path.exists():
        return None
    fits = json.loads(match_path.read_text())
    ranked = sorted(fits.items(), key=lambda kv: -kv[1]['mean_iou'])
    for name, fit in ranked:
        if rm.painted_path(class_name, name, fit).exists():
            return name, fit
    return None


def load_painted(path: Path):
    """Painted meshes (Z up, normalised) with their texture rebuilt as render_models does.

    Rendering the glb's own PBR material comes out much darker in pyrender (88 -> 58 mean
    on the tank); a plain texture visual, as textured_meshes() makes, keeps the baked colours.
    """
    meshes, height = rm.load_normalised(path, 'y')
    for mesh in meshes:
        material = getattr(mesh.visual, 'material', None)
        image = getattr(material, 'baseColorTexture', None) or getattr(material, 'image', None)
        if image is not None and getattr(mesh.visual, 'uv', None) is not None:
            mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, image=image)
    return meshes, height


def project(renderer, point, yaw, tilt, lean):
    """Render pixel (x, y) of a point in the normalised model frame; inverse of Renderer.unproject."""
    world = rm.rotation_z(yaw) @ np.append(point, 1.0)
    camera = np.linalg.inv(renderer.camera_pose(tilt, lean)) @ world
    x = (camera[0] / renderer.view_half + 1) / 2 * rm.RENDER_SIZE
    y = (1 - camera[1] / renderer.view_half) / 2 * rm.RENDER_SIZE
    return x, y


def add_rotor(rgba, renderer, hub, radius, yaw, tilt, lean, rng):
    """Draw the spinning rotor as the real cut-outs show it.

    Always a faint disc, so the box spans the rotor as the real labels do; on top of it,
    30% of the time a pale translucent wedge (the blurred blades most real frames show)
    and 20% of the time a few thin blades.
    """
    cx, cy = project(renderer, hub, yaw, tilt, lean)
    r = radius * rm.RENDER_SIZE / (2 * renderer.view_half)
    centre, axes = (int(round(cx)), int(round(cy))), (int(round(r)), int(round(r)))
    colour = np.array([rng.uniform(160, 190), rng.uniform(185, 210), rng.uniform(150, 175)])  # pale green, RGB
    layer = np.zeros(rgba.shape[:2], np.float32)
    cv2.ellipse(layer, centre, axes, 0, 0, 360, rng.uniform(0.05, 0.08), -1, cv2.LINE_AA)
    mode = rng.random()
    if mode < 0.3:
        start = rng.uniform(0, 360)
        wedge = np.zeros_like(layer)
        cv2.ellipse(wedge, centre, axes, 0, start, start + rng.uniform(35, 70), rng.uniform(0.3, 0.55), -1, cv2.LINE_AA)
        layer = np.maximum(layer, cv2.GaussianBlur(wedge, (0, 0), max(r * 0.02, 0.5)))
    elif mode < 0.5:
        start, blades = rng.uniform(0, 360), rng.choice([2, 3, 5])
        for k in range(blades):
            angle = np.radians(start + 360 * k / blades)
            end = (int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle)))
            cv2.line(layer, centre, end, rng.uniform(0.5, 0.8), max(1, int(r * 0.04)), cv2.LINE_AA)
    body = rgba[:, :, 3:4].astype(np.float32) / 255
    a = layer[:, :, None]
    out = rgba.astype(np.float32)
    out[:, :, :3] = np.where(body > 0, out[:, :, :3] * (1 - a) + colour * a, colour)
    out[:, :, 3] = np.maximum(rgba[:, :, 3], layer * 255)
    return out.round().astype(np.uint8)


def render_sprite(renderer, height_px: float, yaw, tilt, lean, rotor=None, rng=None):
    """RGBA sprite, `height_px` = on-screen length of the model's longest side."""
    rgba, _ = renderer.render(yaw, tilt, lean)
    if rotor:
        rgba = add_rotor(rgba, renderer, *rotor, yaw, tilt, lean, rng)
    scale = height_px / (rm.RENDER_SIZE / (2 * renderer.view_half))
    small = cv2.resize(rgba, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ys, xs = np.nonzero(small[:, :, 3] > 8)
    if len(xs) < 4:
        return None
    return small[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', nargs='*', default=None)
    parser.add_argument('--per-class', type=int, default=200)
    parser.add_argument('--out', default=str(OUT))
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    index = []
    for class_name in args.classes or rm.OBJECT_CLASSES:
        found = painted_model(class_name)
        if not found:
            print(f'{class_name}: no painted model, skipped')
            continue
        name, fit = found
        meshes, height = load_painted(rm.painted_path(class_name, name, fit))
        # Recoloured models (recolour_models.py) keep their own texture, which carries no real
        # light: render them lit. Projection-painted ones carry the real image's light: flat.
        renderer = rm.Renderer(meshes, height, flat=fit.get('paint') != 'recoloured')
        rotor = None
        if class_name in ROTORS:  # hub: the top of the mast, the model's highest point
            vertices = np.vstack([m.vertices for m in meshes])
            top = vertices[vertices[:, 2] > vertices[:, 2].max() - 0.02]
            rotor = (top.mean(axis=0), ROTORS[class_name])
        length_px = fit['length_m'] / rm.METRES_PER_PIXEL
        # Tilt and lean depend on where the object sits in the frame (drone_camera.lean_at);
        # the bank covers the whole range and synth_dataset picks the pose for each spot.
        max_tilt = MAX_TILT
        (out / class_name).mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            yaw, tilt, lean = rng.uniform(0, 360), rng.uniform(0, max_tilt), rng.uniform(0, 360)
            sprite = render_sprite(renderer, length_px * rng.uniform(1 - SIZE_JITTER, 1 + SIZE_JITTER),
                                   yaw, tilt, lean, rotor, rng)
            if sprite is None:
                continue
            file = f'{class_name}/{name}_{i:04d}.png'
            # Renders are RGBA; the sprite files and cv2 are BGRA.
            cv2.imwrite(str(out / file), cv2.cvtColor(sprite, cv2.COLOR_RGBA2BGRA))
            index.append({'file': file, 'class': class_name, 'model': name,
                          'yaw': round(yaw, 1), 'tilt': round(tilt, 1), 'lean': round(lean, 1)})
        renderer.close()  # pyrender breaks with two offscreen renderers alive
        print(f'{class_name}: {args.per_class} renders of {name}, ~{length_px:.0f} px long, tilt up to {max_tilt:.0f}')
    (out / 'index.json').write_text(json.dumps(index, indent=1))
    print(f'{len(index)} sprites in {out}')


if __name__ == '__main__':
    main()
