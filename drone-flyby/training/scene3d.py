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
TEXTURE_PX = 256       # per piece in the atlas; the mesh's own are 512 px
KEEP_FACES = 1.0       # share of the city triangles kept: decimating tears the pieces open at their texture seams
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


def decimate(mesh, keep):
    """(vertices, faces, uv) with about `keep` of the faces; each surviving vertex keeps its uv."""
    import fast_simplification
    if keep >= 1 or len(mesh.faces) < 200:
        return mesh.vertices, mesh.faces, mesh.visual.uv
    _, _, collapses = fast_simplification.simplify(mesh.vertices, mesh.faces, target_reduction=1 - keep,
                                                   return_collapses=True)
    vertices, faces, mapping = fast_simplification.replay_simplification(mesh.vertices, mesh.faces, collapses)
    if np.isnan(vertices).any():  # happens on the odd piece; NaN bounds make the whole glb invalid
        return mesh.vertices, mesh.faces, mesh.visual.uv
    uv = np.zeros((len(vertices), 2))
    uv[mapping] = mesh.visual.uv
    return vertices, faces, uv


def subtile_mesh(sub, offset, texture_px, keep):
    """One subtile's pieces as a single mesh on one texture atlas (a draw call per subtile, not per piece)."""
    from io import BytesIO
    from PIL import Image
    pieces = []
    for obj in sorted(glob.glob(f'{sub}/*_L{rc.LEVEL}_*.obj')):
        mesh = trimesh.load(obj, force='mesh')
        image = getattr(getattr(mesh.visual, 'material', None), 'image', None)
        if len(mesh.faces) and image is not None and getattr(mesh.visual, 'uv', None) is not None:
            pieces.append((mesh, image))
    if not pieces:
        return None, None
    n = int(np.ceil(np.sqrt(len(pieces))))
    atlas = Image.new('RGB', (n * texture_px, n * texture_px))
    inset = 1 / texture_px  # keep a pixel's margin, or neighbouring cells bleed in at the seams
    vertices, faces, uvs, full, count = [], [], [], [], 0
    for k, (mesh, image) in enumerate(pieces):
        row, col = divmod(k, n)
        atlas.paste(image.convert('RGB').resize((texture_px, texture_px), Image.LANCZOS), (col * texture_px, row * texture_px))
        full.append(mesh.vertices + offset)
        v, f, uv = decimate(mesh, keep)
        uv = inset + np.clip(uv, 0, 1) * (1 - 2 * inset)
        uvs.append(np.stack([(col + uv[:, 0]) / n, (n - 1 - row + uv[:, 1]) / n], axis=1))  # uv v runs bottom-up
        vertices.append(v + offset)
        faces.append(f + count)
        count += len(v)
    buffer = BytesIO()
    atlas.save(buffer, 'JPEG', quality=85)  # a PIL image without a format would be stored as PNG
    merged = trimesh.Trimesh(np.vstack(vertices), np.vstack(faces), process=False)
    merged.visual = trimesh.visual.TextureVisuals(uv=np.vstack(uvs), image=Image.open(BytesIO(buffer.getvalue())))
    return merged, np.vstack(full)


def city_meshes(subs, centre, texture_px=TEXTURE_PX, keep=KEEP_FACES):
    """One mesh per subtile, moved so `centre` is the origin (Z up), and all full-detail vertices."""
    meshes, full = [], []
    for sub in subs:
        mesh, vertices = subtile_mesh(sub, rc.origin(Path(sub).parent) - centre, texture_px, keep)
        if mesh is not None:
            meshes.append(mesh)
            full.append(vertices)
    return meshes, np.vstack(full)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', help='background name (default: the newest with a camera file)')
    parser.add_argument('--objects', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--texture', type=int, default=TEXTURE_PX, help='city texture px per piece (512 = full)')
    parser.add_argument('--keep', type=float, default=KEEP_FACES, help='share of the city triangles kept')
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
    meshes, vertices = city_meshes(subs, centre, args.texture, args.keep)
    tree, heights = cKDTree(vertices[:, :2]), vertices[:, 2]
    shown = sum(len(m.vertices) for m in meshes)
    print(f'{name}: {len(meshes)} subtiles, {len(vertices) / 1e6:.1f}M vertices, {shown / 1e6:.2f}M kept for the viewer')

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
    scene.export(str(out / 'city.glb'), include_normals=False)  # shown unlit: normals only add weight

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
