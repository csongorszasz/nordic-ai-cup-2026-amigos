"""Render drone-view background frames from Helsinki's open 3D city mesh.

The competition frames look like a photorealistic 3D map seen from ~600 m: roofs lean
away from the image centre. Flat orthophotos cannot give that; this renders the City of
Helsinki reality mesh (2017, CC BY 4.0, "Helsinki 3D", https://kartta.hel.fi/3d/) with a
perspective camera matching the drone's geometry, at 3840x2160 and ~0.21 m/px.

    # tiles: 2 km zips from http://3d.hel.ninja/data/mesh/Helsinki3D-MESH_2017_OBJ_2km-250m_ZIP/
    # named <N km><E km> in GK25 (EPSG:3879), e.g. 676510 = N 6676 km, E 25510 km.
    # Only the L19 level (~0.1 m/px textures) is needed:
    #   unzip Helsinki3D_2017_OBJ_676510x2.zip '*_L19_*' metadata.xml -d backgrounds/helsinki3d/tile_676510
    python training/render_city.py --count 20                   # -> backgrounds/helsinki3d_frames/
    python training/render_city.py --count 4 --tiles tile_676510 --seed 1

Frames that are mostly sea or leave part of the view empty are skipped. Next to each
frame, <name>_ground.png (quarter resolution) marks open ground, from the depth buffer:
flat, level with its surroundings and not water. synth_dataset.py pastes objects only
there, as the supplied scenes do (fields, clearings, roads; never on trees or roofs).
"""

import argparse
import glob
import json
import os
import re
from functools import lru_cache
from pathlib import Path

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import cv2
import numpy as np
import pyrender
import trimesh

from drone_camera import FOCAL_PX, HEIGHT, HFOV, METRES_PER_PIXEL, PITCH_DEG, WIDTH, lean_at  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
MESH_ROOT = ROOT / 'backgrounds' / 'helsinki3d'
LEVEL = 19           # quadtree level with ~0.1 m/px textures; rendered down to 0.21
TEXTURE_SCALE = 0.5  # halve them on load (~0.18 m/px, still finer than the frame): a quarter of the memory
MAX_EMPTY = 0.02     # share of the frame the mesh may leave uncovered (cracks between pieces, filled in)
MAX_WATER = 0.5      # share of the frame that may be open water
MASK_SCALE = 4       # ground masks are saved at 1/4 of the frame size (~0.84 m per pixel)
GROUND_WINDOW_M = 80  # buildings narrower than this stand out above the ground under them
MAX_HEIGHT_M = 1.0   # open ground: at most this far above the local ground level...
MAX_ROUGHNESS_M = 0.3  # ...and this bumpy over ~4 m (tree canopy is far rougher)
MIN_GROUND = 0.05    # skip frames with less open ground than this
MIN_LAND_M = 1.0     # mesh heights are metres above sea level (N2000); below this is sea


def altitude():
    """Camera height above ground that gives METRES_PER_PIXEL at the image centre."""
    return WIDTH * METRES_PER_PIXEL / (2 * np.tan(np.radians(HFOV / 2)))


def origin(tile_dir: Path):
    text = (tile_dir / 'metadata.xml').read_text()
    return np.array([float(v) for v in re.search(r'<SRSOrigin>([^<]+)<', text).group(1).split(',')])


def build_index(tile_dirs):
    """{subtile folder: [xmin, ymin, xmax, ymax, zmin, zmax]} in GK25 metres, cached per tile."""
    index = {}
    for tile_dir in tile_dirs:
        cache = tile_dir / 'index.json'
        if cache.exists():
            index.update(json.loads(cache.read_text()))
            continue
        offset, entries = origin(tile_dir), {}
        for sub in sorted(p for p in tile_dir.iterdir() if p.is_dir()):
            lows, highs = [], []
            for obj in glob.glob(str(sub / f'*_L{LEVEL}_*.obj')):
                vertices = np.array([line.split()[1:4] for line in open(obj) if line.startswith('v ')], float)
                if len(vertices):
                    lows.append(vertices.min(axis=0))
                    highs.append(vertices.max(axis=0))
            if lows:
                low, high = np.min(lows, axis=0) + offset, np.max(highs, axis=0) + offset
                entries[str(sub)] = [low[0], low[1], high[0], high[1], low[2], high[2]]
        cache.write_text(json.dumps(entries))
        index.update(entries)
    return index


@lru_cache(maxsize=24)  # one frame touches up to ~20 subtiles; more than that ran a 16 GB laptop out of memory
def load_subtile(sub: str):
    """pyrender meshes of one subtile at LEVEL, placed in GK25 metres."""
    offset = origin(Path(sub).parent)
    meshes = []
    for obj in sorted(glob.glob(f'{sub}/*_L{LEVEL}_*.obj')):
        mesh = trimesh.load(obj, force='mesh')
        if not len(mesh.faces):
            continue
        mesh.apply_translation(offset)
        material = getattr(mesh.visual, 'material', None)
        if getattr(material, 'image', None) is not None:
            w, h = material.image.size
            material.image = material.image.resize((max(1, int(w * TEXTURE_SCALE)), max(1, int(h * TEXTURE_SCALE))))
        rendered = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        for primitive in rendered.primitives:
            # pyrender makes OBJ materials fully metallic, which ambient light does not reach.
            primitive.material.metallicFactor, primitive.material.roughnessFactor = 0.0, 1.0
        meshes.append(rendered)
    return meshes


def camera_pose(x, y, z, yaw):
    """At (x, y, z), image up rotated `yaw` degrees from north, pitched PITCH_DEG forward."""
    return trimesh.transformations.rotation_matrix(np.radians(yaw), [0, 0, 1], [x, y, 0]) @ \
        trimesh.transformations.translation_matrix([x, y, z]) @ \
        trimesh.transformations.rotation_matrix(np.radians(PITCH_DEG), [1, 0, 0])


def world_height(depth, pose):
    """World z of every pixel of a pyrender depth map (0 = nothing there -> NaN)."""
    h, w = depth.shape
    scale = w / WIDTH
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x = (u - w / 2) / (FOCAL_PX * scale) * depth
    y = -(v - h / 2) / (FOCAL_PX * scale) * depth
    z = pose[2, 0] * x + pose[2, 1] * y - pose[2, 2] * depth + pose[2, 3]
    return np.where(depth > 0, z, np.nan)


def ground_mask(depth, pose, rgb):
    """Quarter-resolution bool mask of open ground: flat, at ground level, not water."""
    small = (WIDTH // MASK_SCALE, HEIGHT // MASK_SCALE)
    z = world_height(cv2.resize(depth, small, interpolation=cv2.INTER_NEAREST), pose).astype(np.float32)
    filled = np.where(np.isnan(z), np.nanmedian(z), z)
    # Ground level: a grey opening removes anything narrower than the window (trees, buildings).
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(GROUND_WINDOW_M / (METRES_PER_PIXEL * MASK_SCALE)) | 1,) * 2)
    ground = cv2.GaussianBlur(cv2.morphologyEx(filled, cv2.MORPH_OPEN, k), (0, 0), 5)
    mean = cv2.blur(filled, (5, 5))
    rough = np.sqrt(np.maximum(cv2.blur(filled * filled, (5, 5)) - mean * mean, 0))
    water = cv2.resize(water_mask(rgb).astype(np.uint8), small, interpolation=cv2.INTER_NEAREST) > 0
    mask = (filled - ground < MAX_HEIGHT_M) & (rough < MAX_ROUGHNESS_M) & (filled > MIN_LAND_M) & ~water & ~np.isnan(z)
    # Drop specks: an object needs a few metres of open ground around it.
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0


def render_frame(renderer, index, x, y, yaw):
    """(RGB uint8 frame centred on GK25 (x, y), ground mask), or None when the mesh does not cover it."""
    reach = np.hypot(WIDTH, HEIGHT) / 2 * METRES_PER_PIXEL + 50
    subs = [s for s, b in index.items() if b[0] < x + reach and b[2] > x - reach and b[1] < y + reach and b[3] > y - reach]
    if not subs:
        return None
    ground = float(np.median([index[s][4] for s in subs]))
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
    for sub in subs:
        for mesh in load_subtile(sub):
            scene.add(mesh)
    yfov = 2 * np.arctan(np.tan(np.radians(HFOV / 2)) * HEIGHT / WIDTH)
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=WIDTH / HEIGHT,
                                     znear=300, zfar=1500)  # tight: the depth buffer gives the ground mask
    pose = camera_pose(x, y, ground + altitude(), yaw)
    scene.add(camera, pose=pose)
    colour, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    empty = colour[:, :, 3] == 0
    if empty.mean() > MAX_EMPTY:
        return None
    # pyrender reads the JPEG textures as sRGB and writes linear light; encode back to sRGB.
    rgb = (255 * (colour[:, :, :3] / 255.0) ** (1 / 2.2)).round().astype(np.uint8)
    if empty.any():  # thin cracks where neighbouring mesh pieces do not meet
        rgb = cv2.inpaint(rgb, empty.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    return rgb, ground_mask(depth, pose, rgb)


def water_mask(rgb):
    """Rough open-water mask: green-to-blue and nearly featureless.

    In the mesh the sea is a smooth grey-green (hue ~55-85, local detail ~1.2); land,
    even grass and forest, has detail well above 2.5.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    detail = cv2.blur(np.abs(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_32F)), (31, 31))
    return (hsv[:, :, 0] > 40) & (hsv[:, :, 0] < 110) & (detail < 2.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiles', nargs='*', help='folders under backgrounds/helsinki3d/ (default: all tile_*)')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=str(ROOT / 'backgrounds' / 'helsinki3d_frames'))
    args = parser.parse_args()

    tile_dirs = [MESH_ROOT / t for t in args.tiles] if args.tiles else sorted(MESH_ROOT.glob('tile_*'))
    index = build_index([t for t in tile_dirs if (t / 'metadata.xml').exists()])
    if not index:
        raise SystemExit(f'no tiles with L{LEVEL} data under {MESH_ROOT}')
    bounds = np.array(list(index.values()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    renderer = pyrender.OffscreenRenderer(WIDTH, HEIGHT)
    print(f'{len(index)} subtiles, camera {altitude():.0f} m above ground')

    written, tries = 0, 0
    while written < args.count and tries < args.count * 20:
        tries += 1
        # A random subtile, then a point in it: tiles can be kilometres apart.
        box = bounds[rng.integers(len(bounds))]
        x, y = rng.uniform(box[0], box[2]), rng.uniform(box[1], box[3])
        yaw = rng.uniform(0, 360)
        rendered = render_frame(renderer, index, x, y, yaw)
        if rendered is None:
            continue
        rgb, ground = rendered
        if water_mask(rgb).mean() > MAX_WATER or ground.mean() < MIN_GROUND:
            continue
        name = f'hel3d_{int(x)}_{int(y)}_{int(yaw):03d}'
        cv2.imwrite(str(out / f'{name}.jpg'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(out / f'{name}_ground.png'), ground.astype(np.uint8) * 255)
        written += 1
        print(f'  {name}: {ground.mean():.0%} open ground', flush=True)
    renderer.delete()
    print(f'{written} frames in {out} ({tries} tries)')


if __name__ == '__main__':
    main()
