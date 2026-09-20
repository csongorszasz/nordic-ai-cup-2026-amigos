"""Models built from simple boxes, for classes no downloaded mesh matches.

    python training/procedural_models.py tower              # -> models/small_tower/procedural_tower/tower.glb
    python training/procedural_models.py tower --cabin-height 1.0
    python training/render_models.py small_tower            # then fit it like any other model

small_tower seen from the drone is a pale square platform with a smaller green cabin on
top, and the cabin sits off-centre on the platform: the camera sees the tower slightly
from the side, so its top leans away from the image centre. The downloaded watchtower's
roof covers its whole footprint, so its roof got matched to the platform. This builds
the shape as seen: a platform, four legs and a raised cabin with a roof. Being modelled
from scratch, it needs no licence.

Units are relative (platform side = 1); render_models.py scales by the real cut-outs.
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]


def box(size, centre, colour):
    mesh = trimesh.creation.box(extents=size)
    mesh.apply_translation(centre)
    mesh.visual.face_colors = [*colour, 255]
    return mesh


def tower(cabin_height: float, cabin_side: float):
    """Z-up tower: platform on the ground, legs, cabin and roof at `cabin_height`."""
    pale, wood, green, dark_green = (205, 205, 195), (120, 105, 85), (90, 115, 60), (70, 95, 45)
    parts = [box((1.0, 1.0, 0.03), (0, 0, 0.015), pale)]
    leg = cabin_side / 2 - 0.03
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(box((0.05, 0.05, cabin_height), (sx * leg, sy * leg, cabin_height / 2), wood))
    parts.append(box((cabin_side, cabin_side, 0.25), (0, 0, cabin_height + 0.125), green))
    parts.append(box((cabin_side * 1.1, cabin_side * 1.1, 0.04), (0, 0, cabin_height + 0.27), dark_green))
    return parts


RECIPES = {'tower': ('small_tower', 'procedural_tower', tower, 0.8, 0.6)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('recipe', choices=sorted(RECIPES))
    parser.add_argument('--cabin-height', type=float, help='relative to the platform side (default per recipe)')
    parser.add_argument('--cabin-side', type=float, help='relative to the platform side (default per recipe)')
    args = parser.parse_args()

    class_name, folder, build, cabin_height, cabin_side = RECIPES[args.recipe]
    parts = build(args.cabin_height or cabin_height, args.cabin_side or cabin_side)
    out = ROOT / 'models' / class_name / folder / f'{args.recipe}.glb'
    out.parent.mkdir(parents=True, exist_ok=True)
    # glTF is Y-up; render_models.py turns Y back to Z on load.
    to_y_up = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    scene = trimesh.Scene()
    for i, part in enumerate(parts):
        part.apply_transform(to_y_up)
        scene.add_geometry(part, node_name=f'part_{i}')
    scene.export(str(out))
    print(f'{out.relative_to(ROOT)}: {sum(len(p.faces) for p in parts)} faces')


if __name__ == '__main__':
    main()
