"""Hand edits to the painted models, from the paint review notes (datasets/model_match/paint_review.json).

    python training/edit_models.py                    # every recipe -> _baked/<class>_<model>_proposed.glb
    python training/edit_models.py tank jet_plane
    python training/edit_models.py --promote tank      # the accepted proposal becomes the paint the data uses

Each recipe starts from the paint the training data uses now (render_models.py's projected
paint, or a rebuilt procedural model) and changes what the note asked for: parts painted
darker or in a plain colour, parts added (the tank's barrel), faces drawn from both sides
where the model's surfaces face inwards (the A-7's fuselage showed the ground through it).
Plain-coloured parts carry their shading baked in, since the sprites are rendered flat.

Next to each proposal, <...>_proposed.json records how much longer the model got (an added
barrel makes the longest side longer; model_sprites.py keeps the old parts' real size) and
what changed, which the paint review page shows. model_sprites.py --proposed renders them
to datasets/model_sprites_proposed/ for the page. --promote renames the proposal to
<class>_<model>_edited.glb and sets "paint": "edited" in models/<class>/match.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_sprites as ms  # noqa: E402
import procedural_models as pm  # noqa: E402
import render_models as rm  # noqa: E402

PROPOSALS = rm.ROOT / 'datasets' / 'model_match' / 'proposals.json'
SUN = np.array([np.cos(np.radians(135)), np.sin(np.radians(135)), 1.4])
SUN /= np.linalg.norm(SUN)


def shaded(mesh, colour, rng, variation=0.06):
    """The mesh in one plain colour, shaded by a fixed sun, each face varied a little."""
    mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    mesh.unmerge_vertices()  # one colour per face survives the glTF export only on its own vertices
    light = 0.6 + 0.4 * np.clip(mesh.face_normals @ SUN, 0, 1)
    light *= 1 + rng.uniform(-variation, variation, len(mesh.faces))
    colours = np.clip(np.outer(light, colour), 0, 255).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=np.hstack([colours, np.full((len(colours), 1), 255, np.uint8)]))
    return mesh


def retextured(mesh, change):
    """A copy of a painted mesh with its top-down texture changed by change(rgb, x, y) -> rgb.

    The texture is a planar projection (render_models.textured_meshes): pixel column and row
    map linearly to the model's x and y, over the square of side 2 * VIEW_HALF.
    """
    image = np.asarray(mesh.visual.material.image.convert('RGB')).astype(np.float32)
    h, w = image.shape[:2]
    u = (np.arange(w) + 0.5) / w
    v = 1 - (np.arange(h) + 0.5) / h
    x, y = np.meshgrid(u * 2 * rm.VIEW_HALF - rm.VIEW_HALF, v * 2 * rm.VIEW_HALF - rm.VIEW_HALF)
    image = np.clip(change(image, x, y), 0, 255).astype(np.uint8)
    visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, image=Image.fromarray(image))
    return trimesh.Trimesh(mesh.vertices, mesh.faces, visual=visual, process=False)


def darker(factor):
    return lambda rgb, x, y: rgb * factor


def two_sided(mesh):
    """Every face also drawn from the back: inward-facing surfaces no longer vanish."""
    visual = mesh.visual
    faces = np.vstack([mesh.faces, mesh.faces[:, ::-1]])
    out = trimesh.Trimesh(mesh.vertices, faces, process=False)
    if isinstance(visual, trimesh.visual.TextureVisuals):
        out.visual = trimesh.visual.TextureVisuals(uv=visual.uv, image=visual.material.image)
    elif visual.kind == 'face':  # plain colours (shaded()): the back faces take their front's colour
        out.visual = trimesh.visual.ColorVisuals(out, face_colors=np.vstack([visual.face_colors, visual.face_colors]))
    return out


def split(mesh, keep):
    """(faces where keep, the rest) as two meshes with the same visual."""
    parts = []
    for mask in (keep, ~keep):
        part = mesh.submesh([np.nonzero(mask)[0]], append=True)
        parts.append(part)
    return parts


# --------------------------------------------------------------------------- recipes
# Each: (meshes, rng) -> (meshes, what changed). Meshes are in the normalised frame (Z up,
# longest horizontal side 1, standing on z = 0), painted as the training data uses them now.

def tank(meshes, rng):
    """The Churchill's own gun is a stub; the real tank has a long thin barrel, about a third
    of the hull's length beyond its front (+Y)."""
    gun = meshes[3]  # the mantlet and stub, in front of the turret
    x = float(gun.vertices[:, 0].mean())
    z = float(np.percentile(gun.vertices[:, 2], 60))
    barrel = trimesh.creation.cylinder(radius=0.015, segment=[[x, 0.3, z], [x, 0.82, z]], sections=10)
    muzzle = trimesh.creation.cylinder(radius=0.022, segment=[[x, 0.76, z], [x, 0.82, z]], sections=10)
    meshes = meshes + [shaded(barrel, (70, 80, 45), rng), shaded(muzzle, (60, 68, 40), rng)]
    return meshes, 'long barrel added, a third of the hull beyond its front (dark olive)'


def jet_plane(meshes, rng):
    """Faces drawn from both sides (the fuselage showed the ground), the white a mid grey,
    the rear (+X, the fins and exhausts) a dark grey."""
    def paint(rgb, x, y):
        tail = np.clip((x - 0.2) / 0.15, 0, 1)[:, :, None]  # 0 ahead of x = 0.2, 1 behind 0.35
        grey = rgb.mean(axis=2, keepdims=True)
        rgb = (rgb * 0.5 + grey * 0.5) * 0.8                  # less white, a little less colour
        return rgb * (1 - tail) + rgb * 0.5 * tail
    meshes = [two_sided(retextured(m, paint)) for m in meshes]
    return meshes, 'no more see-through fuselage (faces drawn from both sides); mid grey, dark grey tail'


def large_tower(meshes, rng):
    """Light wooden planks on the cabin walls only; roof, legs and stairs darker."""
    mesh = meshes[0]
    top = mesh.vertices[:, 2].max()
    centres = mesh.triangles_center
    upright = np.abs(mesh.face_normals[:, 2]) < 0.35
    # The cabin: the upright faces between the top of the legs (3/4 of the height) and the roof.
    cabin = upright & (centres[:, 2] > top * 0.75) & (centres[:, 2] < top * 0.93)
    walls, rest = split(mesh, cabin)
    rest = retextured(rest, darker(0.6))
    planks = shaded(walls, (175, 118, 65), rng, variation=0.12)  # light orange-brown wood (Juan: not white)
    return [two_sided(rest), two_sided(planks)], 'cabin walls light orange-brown wooden planks; roof, legs and stairs darker'


def small_launcher(meshes, rng):
    """The launcher itself darker against the pale disc it stands on, so its shape shows."""
    disc, missiles, body = meshes
    return [retextured(disc, darker(0.85)), retextured(missiles, darker(0.75)), retextured(body, darker(0.5))], \
        'launcher body (the centre) much darker, missiles and the disc under it a little darker'


def small_tower(meshes, rng):
    """Rebuilt in plain colours: flat grey concrete base with a small second slab and a crate
    on it, light green legs and cabin, dark green roof with darker edges."""
    concrete, slab, crate = (180, 180, 172), (160, 160, 152), (140, 80, 55)
    light_green, roof, roof_edge = (125, 145, 80), (80, 100, 50), (55, 70, 35)
    height, side = pm.RECIPES['tower'][3:5]
    box = trimesh.creation.box
    parts = [(box((1.0, 1.0, 0.03)), (0, 0, 0.015), concrete),
             (box((0.3, 0.4, 0.025)), (0.65, -0.25, 0.0125), slab),       # the small second slab
             (box((0.1, 0.12, 0.08)), (0.62, -0.3, 0.065), crate)]         # and the crate on it
    leg = side / 2 - 0.03
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append((box((0.05, 0.05, height)), (sx * leg, sy * leg, height / 2), light_green))
    parts.append((box((side, side, 0.25)), (0, 0, height + 0.125), light_green))
    parts.append((box((side * 1.1, side * 1.1, 0.04)), (0, 0, height + 0.27), roof_edge))
    parts.append((box((side * 0.95, side * 0.95, 0.02)), (0, 0, height + 0.3), roof))
    out = []
    for mesh, centre, colour in parts:
        mesh.apply_translation(centre)
        out.append(shaded(mesh, colour, rng))
    return out, 'rebuilt in plain colours: grey concrete base + small second slab with a crate, light green cabin, dark green roof with darker edges'


def hangar(meshes, rng):
    """The door leaves, open at the shelter's mouth, still showed from above as two sticks
    (Juan, synthetic frame 2). They and the posts stand low beyond the end of the body; the
    frame above the mouth (higher up) stays, or the mouth looks notched."""
    # The body: the parts that run the whole length from the back (x < -0.4), the outer shell excepted.
    shell = max(meshes, key=lambda m: len(m.faces))
    body_end = max(m.bounds[1][0] for m in meshes if m is not shell and m.bounds[0][0] < -0.4
                   and len(m.faces) > 1000)
    door = [m for m in meshes if m.bounds[0][0] > body_end and m.bounds[1][2] < 0.13]
    kept = [m for m in meshes if not any(m is d for d in door)]
    return kept, f'door leaves and posts removed ({len(door)} low parts beyond the end of the body)'


RECIPES = {'hangar': hangar, 'tank': tank, 'jet_plane': jet_plane, 'large_tower': large_tower,
           'small_launcher': small_launcher, 'small_tower': small_tower}


def horizontal_extent(meshes):
    bounds = np.array([m.bounds for m in meshes])
    low, high = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
    return float(max(high[0] - low[0], high[1] - low[1]))


def export(meshes, path: Path):
    """As render_models.export_painted: the normalised frame, turned Y-up for glTF."""
    to_y_up = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    scene = trimesh.Scene()
    for i, mesh in enumerate(meshes):
        mesh = mesh.copy()
        mesh.apply_transform(to_y_up)
        scene.add_geometry(mesh, node_name=f'part_{i}')
    scene.export(str(path))


def propose(class_name: str):
    name, fit = ms.painted_model(class_name)
    meshes, _ = ms.load_painted(rm.painted_path(class_name, name, fit))
    before = horizontal_extent(meshes)
    meshes, what = RECIPES[class_name](meshes, np.random.default_rng(0))
    scale = horizontal_extent(meshes) / before
    path = rm.BAKED / f'{class_name}_{name}_proposed.glb'
    export(meshes, path)
    path.with_suffix('.json').write_text(json.dumps({'length_scale': round(scale, 3), 'change': what}, indent=1))
    proposals = json.loads(PROPOSALS.read_text()) if PROPOSALS.exists() else {}
    proposals[class_name] = what
    PROPOSALS.write_text(json.dumps(proposals, indent=1))
    print(f'{class_name}: {what}; longest side x{scale:.2f} -> {path.name} ({path.stat().st_size >> 20} MB)')


def promote(class_name: str):
    match_path = rm.ROOT / 'models' / class_name / 'match.json'
    fits = json.loads(match_path.read_text())
    name, fit = ms.painted_model(class_name)
    proposed = rm.BAKED / f'{class_name}_{name}_proposed.glb'
    meta = json.loads(proposed.with_suffix('.json').read_text())
    proposed.rename(rm.BAKED / f'{class_name}_{name}_edited.glb')
    proposed.with_suffix('.json').unlink()
    fit['paint'] = 'edited'
    fit['edits'] = meta['change']
    fit['length_m'] = round(fit['length_m'] * meta['length_scale'], 2)
    fits[name] = fit
    match_path.write_text(json.dumps(fits, indent=1))
    proposals = json.loads(PROPOSALS.read_text())
    proposals.pop(class_name, None)
    PROPOSALS.write_text(json.dumps(proposals, indent=1))
    print(f'{class_name}: {name} now uses the edited paint, length {fit["length_m"]} m')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('classes', nargs='*', help=f'default: all of {sorted(RECIPES)}')
    parser.add_argument('--promote', action='store_true', help='make the proposals the paint the data uses')
    args = parser.parse_args()
    for class_name in args.classes or sorted(RECIPES):
        (promote if args.promote else propose)(class_name)


if __name__ == '__main__':
    main()
