"""Turn one synthetic frame into a 3D scene to fly over: the city mesh with the objects in it.

synth_dataset.py pastes object sprites onto a background rendered by render_city.py. This
takes such a background (it needs the <name>_camera.json render_city.py writes), composes a
synthetic frame on it the same way, and finds where each pasted object stands in the city:
the ray through its box centre, dropped onto the mesh. The painted 3D model of its class goes
there, at its fitted size. The sprite review server shows the result at /scene3d.

    python training/scene3d.py                       # newest background with a camera file
    python training/scene3d.py --frame hel3d_25503147_6683431_330 --objects 40 --seed 2

Output, in datasets/scene3d/<frame>/: city.glb (the mesh around the frame, Y up, metres,
origin under the camera), scene.json (camera and objects) and synth.jpg (the 2D frame).
"""

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_city as rc  # noqa: E402
import synth_dataset as sd  # noqa: E402
from model_sprites import painted_model  # noqa: E402

ROOT = rc.ROOT
FRAMES = ROOT / 'backgrounds' / 'helsinki3d_frames'
OUT = ROOT / 'datasets' / 'scene3d'
TO_Y_UP = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])  # glTF and three.js are Y up
GROUND_RADIUS_M = 1.5  # mesh vertices this close (horizontally) give the ground height under an object


def ray(pose, u, v):
    """World origin and direction of the camera ray through 4K pixel (u, v)."""
    direction = pose[:3, :3] @ np.array([(u - rc.WIDTH / 2) / rc.FOCAL_PX, -(v - rc.HEIGHT / 2) / rc.FOCAL_PX, -1.0])
    return pose[:3, 3], direction / np.linalg.norm(direction)


def drop_on_ground(origin, direction, tree, heights, z0):
    """Where the ray meets the ground: intersect a level plane, re-read its height, repeat."""
    z = z0
    for _ in range(4):
        point = origin + direction * (z - origin[2]) / direction[2]
        near = tree.query_ball_point(point[:2], GROUND_RADIUS_M)
        if not near:
            break
        z = float(np.median(heights[near]))
    return origin + direction * (z - origin[2]) / direction[2]


def city_meshes(subs, centre):
    """Every L19 piece of the subtiles, moved so `centre` is the origin (Z up)."""
    meshes = []
    for sub in subs:
        offset = rc.origin(Path(sub).parent) - centre
        for obj in sorted(glob.glob(f'{sub}/*_L{rc.LEVEL}_*.obj')):
            mesh = trimesh.load(obj, force='mesh')
            if len(mesh.faces):
                mesh.apply_translation(offset)
                meshes.append(mesh)
    return meshes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', help='background name (default: the newest with a camera file)')
    parser.add_argument('--objects', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    cameras = sorted(FRAMES.glob('*_camera.json'), key=lambda p: p.stat().st_mtime)
    if args.frame:
        cameras = [FRAMES / f'{args.frame}_camera.json']
    if not cameras or not cameras[-1].exists():
        raise SystemExit('no background with a camera file; render one with training/render_city.py')
    name = cameras[-1].name.removesuffix('_camera.json')
    camera = json.loads(cameras[-1].read_text())
    pose = rc.camera_pose(camera['x'], camera['y'], camera['z'], camera['yaw'])

    # The synthetic frame, composed as synth_dataset.py does, with 3D-model sprites only.
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    background, ground = sd.load_background(FRAMES / f'{name}.jpg')
    bank = sd.load_model_sprites(ROOT / 'datasets' / 'model_sprites')
    frame, annotations = sd.compose_frame(background, {}, rng, args.objects, bank, 1.0, ground)

    # The city around it, origin on the ground under the camera.
    index = rc.build_index([rc.MESH_ROOT / t for t in camera['tiles']])
    reach = np.hypot(rc.WIDTH, rc.HEIGHT) / 2 * rc.METRES_PER_PIXEL + 50
    x, y = camera['x'], camera['y']
    subs = [s for s, b in index.items() if b[0] < x + reach and b[2] > x - reach and b[1] < y + reach and b[3] > y - reach]
    ground_z = float(np.median([index[s][4] for s in subs]))
    centre = np.array([x, y, ground_z])
    meshes = city_meshes(subs, centre)
    vertices = np.vstack([m.vertices for m in meshes])
    tree, heights = cKDTree(vertices[:, :2]), vertices[:, 2]
    print(f'{name}: {len(subs)} subtiles, {len(meshes)} pieces, {len(vertices) / 1e6:.1f}M vertices')

    local_pose = pose.copy()
    local_pose[:3, 3] -= centre
    objects = []
    for ann in annotations:
        x1, y1, x2, y2 = ann['bbox']
        origin, direction = ray(local_pose, (x1 + x2) / 2, (y1 + y2) / 2)
        spot = drop_on_ground(origin, direction, tree, heights, 0.0)
        found = painted_model(ann['object_id'])
        if not found:
            continue
        model, fit = found
        objects.append({
            'class': ann['object_id'], 'model': model, 'length_m': fit['length_m'],
            'url': f'/compare/_baked/{ann["object_id"]}_{model}.glb',
            'position': (TO_Y_UP[:3, :3] @ spot).round(2).tolist(),  # three.js coordinates
            'heading': rng.uniform(0, 360), 'bbox': ann['bbox'],
        })

    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    for i, mesh in enumerate(meshes):
        mesh.apply_transform(TO_Y_UP)
        scene.add_geometry(mesh, node_name=f'city_{i}')
    scene.export(str(out / 'city.glb'))

    marked = frame.copy()
    for ann in annotations:
        a, b, c, d = ann['bbox']
        cv2.rectangle(marked, (a, b), (c, d), (0, 255, 0), 2)
        cv2.putText(marked, ann['object_id'], (a, max(b - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(str(out / 'synth.jpg'), marked, [cv2.IMWRITE_JPEG_QUALITY, 88])
    (out / 'scene.json').write_text(json.dumps({
        'frame': name, 'city': f'/scene3d_data/{name}/city.glb', 'synth': f'/scene3d_data/{name}/synth.jpg',
        # The drone camera, Z up in metres around the origin; the page turns it Y up.
        'camera': {'pose': local_pose.round(4).tolist(), 'hfov': rc.HFOV, 'width': rc.WIDTH, 'height': rc.HEIGHT},
        'objects': objects,
    }, indent=1))
    size = (out / 'city.glb').stat().st_size / 2**20
    print(f'{len(objects)} objects; city.glb {size:.0f} MB -> {out}')


if __name__ == '__main__':
    main()
