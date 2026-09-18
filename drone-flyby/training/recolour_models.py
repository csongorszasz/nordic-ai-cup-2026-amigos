"""Give each fitted 3D model the real object's colours while keeping its own texture.

render_models.py used to paint a model by projecting the real top-down cut-out straight
down onto it: right from above, but the sides became smeared streaks and the model's own
detail was gone. This keeps the model's texture (or face colours) and only shifts its
colours towards the real object's: in Lab, per channel, the mean and spread of the
model's rendered pixels are matched to those of the real cut-outs. Measured on renders
at the fitted poses, lit as model_sprites.py renders them; three rounds, since the
renderer's response to a texture change is not linear.

    python training/recolour_models.py                  # every class with a fit
    python training/recolour_models.py tank helicopter

Writes datasets/model_match/_baked/<class>_<model>.glb (Y up, normalised as the fit) and
marks the fit with "paint": "recoloured" in models/<class>/match.json, which tells
model_sprites.py to render it lit instead of flat.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_models as rm  # noqa: E402

ROUNDS = 3
SPREAD_LIMITS = (0.4, 2.5)  # how far a channel's spread may be stretched or squeezed per round
MIN_ALPHA = 200             # pixels this opaque count; the cut-out edges are half ground
# A model whose render barely varies in colour has no real texture (the AT-AT's is a 2x2
# grey placeholder, the condor and the procedural tower use plain face colours): shifting
# its colours can only give one flat tone, so it keeps render_models.py's projected paint.
MIN_TEXTURE_SPREAD = 1.0    # Lab a/b standard deviation of the render


def best_fit(class_name: str):
    """(model name, fit) with the best mean IoU in models/<class>/match.json, or None."""
    path = rm.ROOT / 'models' / class_name / 'match.json'
    if not path.exists():
        return None
    fits = json.loads(path.read_text())
    return max(fits.items(), key=lambda kv: kv[1]['mean_iou'])


def lab_stats(pixels_rgb: np.ndarray):
    """Per-channel Lab (mean, std) of an N x 3 uint8 RGB array."""
    lab = cv2.cvtColor(pixels_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    return lab.mean(axis=0), lab.std(axis=0) + 1e-3


def opaque_pixels(rgba_images):
    return np.vstack([image[image[:, :, 3] >= MIN_ALPHA][:, :3] for image in rgba_images])


def transfer(rgb: np.ndarray, source, target) -> np.ndarray:
    """Map uint8 RGB colours (any shape ... x 3) so `source` Lab stats become `target`'s."""
    (source_mean, source_std), (target_mean, target_std) = source, target
    shape = rgb.shape
    lab = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    gain = np.clip(target_std / source_std, *SPREAD_LIMITS)
    lab = (lab - source_mean) * gain + target_mean
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_LAB2RGB)
    return out.reshape(shape)


def recolour_meshes(meshes, source, target):
    """Apply the colour transfer to every texture, base colour or face colour, in place."""
    from PIL import Image
    done = {}
    for mesh in meshes:
        visual = mesh.visual
        material = getattr(visual, 'material', None)
        for attribute in ('baseColorTexture', 'image'):
            image = getattr(material, attribute, None)
            if image is None:
                continue
            if id(image) not in done:
                pixels = np.asarray(image.convert('RGBA'))
                rgb = transfer(pixels[:, :, :3], source, target)
                done[id(image)] = Image.fromarray(np.dstack([rgb, pixels[:, :, 3]]))
            setattr(material, attribute, done[id(image)])
            break
        else:
            if material is not None and getattr(material, 'baseColorFactor', None) is not None:
                factor = np.array(material.baseColorFactor, dtype=np.float64)
                scale = 255.0 if factor.max() <= 1.0 else 1.0
                rgb = transfer(np.clip(factor[:3] * scale, 0, 255).astype(np.uint8)[None], source, target)[0]
                factor[:3] = rgb / scale
                material.baseColorFactor = factor
            elif isinstance(visual, trimesh.visual.ColorVisuals):
                colours = visual.face_colors.copy()
                colours[:, :3] = transfer(colours[:, :3], source, target)
                visual.face_colors = colours


def decimate_keep_uv(mesh, faces_wanted: int):
    """Fewer faces, each new vertex taking the uv of the nearest original one.

    trimesh's own decimation drops the texture, and fast_simplification's replay (which maps
    vertices exactly) returned stray coordinates on the condor, so uvs come by proximity.
    """
    import fast_simplification
    from scipy.spatial import cKDTree
    uv = getattr(mesh.visual, 'uv', None)
    if uv is None or len(mesh.faces) <= faces_wanted:
        return mesh
    vertices, faces = fast_simplification.simplify(mesh.vertices, mesh.faces, target_count=max(faces_wanted, 4))
    if not len(faces) or not np.isfinite(vertices).all():
        return mesh
    _, nearest = cKDTree(mesh.vertices).query(vertices)
    out = trimesh.Trimesh(vertices, faces, process=False)
    out.visual = trimesh.visual.TextureVisuals(uv=uv[nearest], material=mesh.visual.material)
    return out


def slim(meshes):
    """The meshes, decimated to about rm.MAX_EXPORT_FACES (the A-7 has 5.2M faces)."""
    total = sum(len(m.faces) for m in meshes)
    if total <= rm.MAX_EXPORT_FACES:
        return meshes
    keep = rm.MAX_EXPORT_FACES / total
    meshes = [decimate_keep_uv(m, int(len(m.faces) * keep)) if len(m.faces) > 100 else m for m in meshes]
    print(f'    decimated {total} -> {sum(len(m.faces) for m in meshes)} faces for export')
    return meshes


def export(meshes, path: Path):
    """Save as a Y-up .glb in the fit's normalised frame."""
    to_y_up = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    scene = trimesh.Scene()
    for i, mesh in enumerate(meshes):
        mesh = mesh.copy()
        mesh.apply_transform(to_y_up)
        scene.add_geometry(mesh, node_name=f'part_{i}')
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(path))


def render_stats(path: Path, fit):
    """Lab stats of the exported model, rendered as model_sprites.py does at the fitted poses."""
    from model_sprites import load_painted
    meshes, height = load_painted(path)
    renderer = rm.Renderer(meshes, height)  # lit: the recoloured textures carry no real light
    renders = [renderer.render(p['yaw'], p['tilt'], p['lean'])[0] for p in fit['per_sprite']]
    renderer.close()
    return lab_stats(opaque_pixels(renders))


def recolour(class_name: str):
    found = best_fit(class_name)
    if not found:
        print(f'{class_name}: no fit, skipped')
        return
    name, fit = found
    real = [sprite for file, sprite, _ in rm.load_sprites(class_name)
            if file in {p['file'] for p in fit['per_sprite']}] or [s for _, s, _ in rm.load_sprites(class_name)]
    target = lab_stats(opaque_pixels(real))
    meshes, _ = rm.load_normalised(rm.ROOT / fit['mesh'], fit['up'], fit.get('dropped_parts', []), fit.get('thicken', 1.0))
    textured = any((getattr(getattr(m.visual, 'material', None), 'baseColorTexture', None) or
                    getattr(getattr(m.visual, 'material', None), 'image', None)) is not None for m in meshes)
    if not textured:  # plain colours (the procedural tower): a box that projection paints exactly
        print(f'  {class_name}/{name}: no texture; keeps the projected paint')
        mark(class_name, name, 'projected')
        return
    meshes = slim(meshes)
    path = rm.BAKED / f'{class_name}_{name}.glb'
    trial = path.with_name(f'{path.stem}_recolour.glb')  # the projected paint stays until this is accepted
    export(meshes, trial)
    spread = render_stats(trial, fit)[1][1:].max()
    if spread < MIN_TEXTURE_SPREAD:
        trial.unlink()
        print(f'  {class_name}/{name}: no real texture (colour spread {spread:.2f}); keeps the projected paint')
        mark(class_name, name, 'projected')
        return
    path = trial
    for round_ in range(ROUNDS):
        source = render_stats(path, fit)
        error = np.abs(source[0] - target[0]).sum()
        print(f'  {class_name}/{name} round {round_}: Lab mean error {error:.1f}')
        recolour_meshes(meshes, source, target)
        export(meshes, path)
    source = render_stats(path, fit)
    final = rm.BAKED / f'{class_name}_{name}.glb'
    trial.replace(final)
    print(f'  {class_name}/{name}: Lab mean error {np.abs(source[0] - target[0]).sum():.1f} -> {final.name}')
    mark(class_name, name, 'recoloured')


def mark(class_name: str, name: str, paint: str):
    """Record in match.json how the model's .glb was painted (model_sprites.py lights it accordingly)."""
    match_path = rm.ROOT / 'models' / class_name / 'match.json'
    fits = json.loads(match_path.read_text())
    fits[name]['paint'] = paint
    match_path.write_text(json.dumps(fits, indent=1, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('classes', nargs='*', help='default: every class')
    args = parser.parse_args()
    unknown = set(args.classes) - set(rm.OBJECT_CLASSES)
    if unknown:
        parser.error(f'unknown classes: {", ".join(sorted(unknown))}')
    for class_name in args.classes or sorted(rm.OBJECT_CLASSES):
        recolour(class_name)


if __name__ == '__main__':
    main()
