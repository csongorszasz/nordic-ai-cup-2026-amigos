"""Render 3D models top-down and match them to the class's real sprite cut-outs.

For each mesh found under models/<class>/, this renders it from above over a grid of
yaw and camera tilt, then for every Helsinki cut-out of that class finds the render
whose silhouette overlaps best, fits a per-channel colour correction, and reports how
close it gets. The output is a sheet to judge by eye (real sprite | best render per
model) plus models/<class>/match.json with the fitted size, yaws and colours, which
the sprite generator will use.

    python training/render_models.py hangar
    python training/render_models.py hangar --tilts 0 8 16 --yaw-step 3
    python training/render_models.py helicopter --up z --drop rotor_Body  # skinned glTF, rotor spins
    python training/render_models.py hangar --drop Cube.032     # leave out the model's concrete apron
    python training/render_models.py large_tower --close-gaps 4 # lattice legs vs a cut-out that fills them
    python training/render_models.py large_tower --thicken 2    # beams twice as wide

Each cut-out is fitted for yaw, camera tilt and lean direction (the drone camera looks
slightly forward, so tall parts lean, mostly toward the image top) and scale. The score
is silhouette IoU plus how well light and dark parts line up (structure()), which is
what places a tower's cabin on its platform. Two colourings come out: a colour fit
(one gain, per-channel offsets) and the real colours, baked from the cut-outs onto the
model through the depth buffer (bake_texture), which carries camouflage and markings.
Painted models go to datasets/model_match/_baked/ for the 3D viewer.

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
import struct
import sys
from pathlib import Path

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')  # headless rendering on the GPU

import cv2
import numpy as np
import pyrender
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))  # dtos.py lives in src/

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH, OBJECT_CLASSES  # noqa: E402

METRES_PER_PIXEL = 0.21
MESH_SUFFIXES = {'.glb', '.gltf', '.obj'}
RENDER_SIZE = 320     # render resolution; downscaled to the sprite's size afterwards
VIEW_HALF = 0.75      # orthographic half-width, in units of the mesh's longest horizontal side
MAX_TILT = 40         # degrees; tall models get a wider view so a tilted render is not clipped
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


SLIM_CACHE = ROOT / 'datasets' / 'model_match' / '_slim'
BAKED = ROOT / 'datasets' / 'model_match' / '_baked'  # real-colour textures, one per class/model
MAX_EXPORT_FACES = 200_000  # painted models are decimated to about this for export (needs fast_simplification)
GLB_JSON, GLB_BIN = 0x4E4F534A, 0x004E4942


def slim_glb(path: Path) -> Path:
    """A cached copy of a .glb with only small base-colour textures, for models too big to load.

    trimesh decodes every embedded image before lighten_materials can drop it; a 552 MB
    model with 406 textures passed 9 GB that way. This rewrites the file one image at a
    time instead: maps other than base colour are unlinked and blanked, base colour is
    shrunk to TEXTURE_MAX_SIDE, and the binary chunk is repacked.
    """
    out = SLIM_CACHE / f'{path.parent.name}_{path.stem}.glb'
    if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
        return out
    data = path.read_bytes()
    json_length, = struct.unpack_from('<I', data, 12)
    gltf = json.loads(data[20:20 + json_length])
    bin_start = 20 + json_length + 8
    binary = memoryview(data)[bin_start:]
    if len(gltf.get('buffers', [])) != 1:
        return path

    keep = set()  # textures still used as base colour
    for material in gltf.get('materials', []):
        for key in ('normalTexture', 'occlusionTexture', 'emissiveTexture'):
            material.pop(key, None)
        material.pop('emissiveFactor', None)
        pbr = material.get('pbrMetallicRoughness', {})
        pbr.pop('metallicRoughnessTexture', None)
        if 'baseColorTexture' in pbr:
            keep.add(gltf['textures'][pbr['baseColorTexture']['index']]['source'])

    blank = cv2.imencode('.png', np.full((1, 1, 3), 128, np.uint8))[1].tobytes()
    replaced = {}
    for i, image in enumerate(gltf.get('images', [])):
        if 'bufferView' not in image:
            continue
        view = gltf['bufferViews'][image['bufferView']]
        if i not in keep:
            replaced[image['bufferView']] = blank
            image['mimeType'] = 'image/png'
            continue
        start = view.get('byteOffset', 0)
        pixels = cv2.imdecode(np.frombuffer(binary[start:start + view['byteLength']], np.uint8), cv2.IMREAD_UNCHANGED)
        if pixels is None or max(pixels.shape[:2]) <= TEXTURE_MAX_SIDE:
            continue
        factor = TEXTURE_MAX_SIDE / max(pixels.shape[:2])
        pixels = cv2.resize(pixels, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        replaced[image['bufferView']] = cv2.imencode('.png', pixels)[1].tobytes()
        image['mimeType'] = 'image/png'

    chunks, offset = [], 0
    for index, view in enumerate(gltf['bufferViews']):
        start = view.get('byteOffset', 0)
        chunk = replaced.get(index, binary[start:start + view['byteLength']])
        padding = (-offset) % 4
        chunks.append(b'\0' * padding)
        offset += padding
        view['byteOffset'], view['byteLength'] = offset, len(chunk)
        chunks.append(chunk)
        offset += len(chunk)
    new_binary = b''.join(chunks) + b'\0' * ((-offset) % 4)
    gltf['buffers'][0]['byteLength'] = offset

    header = json.dumps(gltf).encode()
    header += b' ' * ((-len(header)) % 4)
    total = 12 + 8 + len(header) + 8 + len(new_binary)
    SLIM_CACHE.mkdir(parents=True, exist_ok=True)
    out.write_bytes(struct.pack('<III', 0x46546C67, 2, total)
                    + struct.pack('<II', len(header), GLB_JSON) + header
                    + struct.pack('<II', len(new_binary), GLB_BIN) + new_binary)
    print(f'    slimmed {path.name}: {path.stat().st_size >> 20} MB -> {out.stat().st_size >> 20} MB')
    return out


def part_name(mesh):
    return f"{mesh.metadata.get('name', '?')} ({mesh.metadata.get('node', '?')})"


def thicken_beams(mesh, factor: float, min_aspect: float = 6.0):
    """Widen thin beams and planks in place, keeping their length.

    Each connected piece whose longest side is `min_aspect` times its next one is scaled
    by `factor` across its second principal axis, about its own centre. Low-poly models
    build legs and planks as such strips.
    """
    groups = trimesh.graph.connected_components(mesh.edges, nodes=np.arange(len(mesh.vertices)))
    vertices = mesh.vertices.copy()
    for group in groups:
        points = vertices[group]
        if len(points) < 3:
            continue
        centre = points.mean(axis=0)
        _, spread, axes = np.linalg.svd(points - centre, full_matrices=False)
        if spread[1] <= 0 or spread[0] < min_aspect * spread[1]:
            continue
        across = axes[1]
        offset = (points - centre) @ across
        vertices[group] = points + np.outer(offset * (factor - 1), across)
    mesh.vertices = vertices


def load_normalised(path: Path, up: str, drop=(), thicken: float = 1.0):
    """Mesh list in a frame where +Z is up, centred, longest horizontal side = 1.

    `drop` removes parts whose geometry or node name contains any of the given strings,
    e.g. a concrete apron or display base the real object does not have. `thicken`
    widens thin beams (thicken_beams()).
    """
    source = slim_glb(path) if path.suffix.lower() == '.glb' else path
    loaded = trimesh.load(str(source), force='scene')
    meshes = [g for g in loaded.dump(concatenate=False) if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    dropped = [m for m in meshes if any(d in part_name(m) for d in drop)]
    meshes = [m for m in meshes if not any(d in part_name(m) for d in drop)]
    for mesh in dropped:
        print(f'    dropped part {part_name(mesh)}')
    if not meshes:
        raise ValueError(f'{path}: no triangle meshes left')
    lighten_materials(meshes)
    if thicken != 1.0:
        for mesh in meshes:
            thicken_beams(mesh, thicken)

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


def build_scene(meshes, ambient=0.45):
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[ambient] * 3)
    for mesh in meshes:
        try:
            scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
        except Exception:  # odd materials: fall back to the mesh's plain colour
            plain = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
            plain.visual.face_colors = [128, 128, 128, 255]
            scene.add(pyrender.Mesh.from_trimesh(plain, smooth=False))
    return scene


ZNEAR, ZFAR = 0.01, 20.0


def rotation_z(degrees):
    return trimesh.transformations.rotation_matrix(np.radians(degrees), [0, 0, 1])


class Renderer:
    """Top-down orthographic renders of one model at any yaw, tilt and lean direction.

    Image up is +Y and image right is +X of the normalised model frame. `tilt` moves the
    camera `tilt` degrees off vertical without rotating the image; tall parts then lean
    away from the camera, toward image direction `lean` (degrees clockwise from image up,
    so 0 = tops lean up, 90 = right).
    """

    def __init__(self, meshes, height, sun_azimuth=135, flat=False):
        self.height = height
        # flat: full ambient light and no sun, so the render shows the texture's colours
        # as they are. For textures baked from real imagery, which already carry its light.
        self.scene = build_scene(meshes, ambient=1.0 if flat else 0.45)
        self.nodes = list(self.scene.mesh_nodes)
        # The footprint's corners reach 0.71 at any yaw; tilted, the top shifts by height * sin(tilt).
        self.view_half = max(VIEW_HALF, 0.72 + height * np.sin(np.radians(MAX_TILT)))
        camera = pyrender.OrthographicCamera(xmag=self.view_half, ymag=self.view_half, znear=ZNEAR, zfar=ZFAR)
        self.camera = self.scene.add(camera, pose=np.eye(4))
        if not flat:
            sun = pyrender.DirectionalLight(color=np.ones(3), intensity=3.5)
            self.scene.add(sun, pose=trimesh.transformations.euler_matrix(np.radians(30), 0, np.radians(sun_azimuth), 'rzxy'))
        self.renderer = pyrender.OffscreenRenderer(RENDER_SIZE, RENDER_SIZE)
        self.flags = pyrender.RenderFlags.RGBA
        self.yaw = None

    def camera_pose(self, tilt, lean):
        swing = rotation_z(-lean) @ trimesh.transformations.rotation_matrix(np.radians(tilt), [1, 0, 0]) @ rotation_z(lean)
        return swing @ trimesh.transformations.translation_matrix([0, 0, 5 + self.height])

    def render(self, yaw, tilt=0.0, lean=0.0):
        """(RGBA uint8, orthographic depth) at RENDER_SIZE; depth 0 where nothing is hit."""
        if yaw != self.yaw:
            for node in self.nodes:
                self.scene.set_pose(node, rotation_z(yaw))
            self.yaw = yaw
        self.scene.set_pose(self.camera, self.camera_pose(tilt, lean))
        color, depth = self.renderer.render(self.scene, flags=self.flags)
        # pyrender converts depth with the perspective formula even for orthographic
        # cameras; undo it to get back the (linear) orthographic depth.
        hit = depth > 0
        ndc = np.zeros_like(depth)
        ndc[hit] = (ZFAR + ZNEAR - 2 * ZNEAR * ZFAR / depth[hit]) / (ZFAR - ZNEAR)
        linear = np.where(hit, (ndc * (ZFAR - ZNEAR) + ZFAR + ZNEAR) / 2, 0)
        return color, linear

    def unproject(self, px, py, depth, yaw, tilt, lean):
        """Model-frame points (N x 3) under render pixels (px, py) with orthographic depth."""
        x = ((px + 0.5) / RENDER_SIZE * 2 - 1) * self.view_half
        y = (1 - (py + 0.5) / RENDER_SIZE * 2) * self.view_half
        camera_points = np.stack([x, y, -depth, np.ones_like(x)])
        world = self.camera_pose(tilt, lean) @ camera_points
        return (rotation_z(-yaw) @ world)[:3].T

    def close(self):
        self.renderer.delete()


# --------------------------------------------------------------------------- matching

def load_sprites(class_name: str):
    """[(file, RGBA cut-out, bbox in 4K pixels)] for the class's complete (uncut) examples."""
    index = json.loads((ROOT / 'sprites' / 'index.json').read_text())
    sprites = []
    for entry in index:
        if entry['class'] != class_name or entry.get('truncated'):
            continue
        image = cv2.imread(str(ROOT / 'sprites' / entry['file']), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape[2] != 4:
            continue
        sprites.append((entry['file'], cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA), entry['bbox']))
    return sprites


SCALE_STEPS = (0.85, 0.9, 0.95, 1.0, 1.05)  # around the area match, which runs large on cut-outs with a halo
TILTS = (0, 8, 16, 24)                     # camera degrees off vertical
LEANS = tuple(range(0, 360, 45))           # image direction tall parts lean toward
STRUCTURE_WEIGHT = 0.25                    # weight of light/dark agreement next to the IoU


def place(render: np.ndarray, transform, shape) -> np.ndarray:
    """Put a render onto a sprite-sized canvas with the transform fit_to_sprite found."""
    x0, y0, x1, y1, scale, left, top = transform
    crop = render[y0:y1, x0:x1]
    new_w = max(1, int(round(crop.shape[1] * scale)))
    new_h = max(1, int(round(crop.shape[0] * scale)))
    small = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    height, width = shape[:2]
    placed = np.zeros((height, width, render.shape[2]), render.dtype)
    py0, px0 = max(top, 0), max(left, 0)
    py1, px1 = min(top + new_h, height), min(left + new_w, width)
    if py1 > py0 and px1 > px0:
        placed[py0:py1, px0:px1] = small[py0 - top:py1 - top, px0 - left:px1 - left]
    return placed


def close_gaps(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Fill gaps up to about 2 * `pixels` wide, like the cut-out's mask does between a lattice's legs."""
    if pixels <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
    padded = np.pad(mask.astype(np.uint8), pixels)
    return cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)[pixels:-pixels, pixels:-pixels] > 0


def fit_to_sprite(render: np.ndarray, sprite: np.ndarray, scales=(1.0,), gaps=0):
    """Scale and centre a render onto the sprite's canvas; the scale with the best IoU wins.

    Scales are relative to the one that makes the silhouette areas equal. `gaps` closes
    the render's silhouette by that many sprite pixels before comparing (close_gaps()).
    Returns a dict with the placed RGBA, IoU and the transform (for place()), or None.
    """
    sprite_mask = sprite[:, :, 3] > 128
    render_mask = render[:, :, 3] > 0
    sprite_area, render_area = sprite_mask.sum(), render_mask.sum()
    if gaps and render_area:
        # Closed area measured at about sprite size: closing at render resolution needs a
        # kernel several times larger and is far too slow.
        f = max(sprite.shape[:2]) / render.shape[0]
        small = cv2.resize(render_mask.astype(np.uint8) * 255, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) > 64
        render_area = close_gaps(small, gaps).sum() / f ** 2
    if sprite_area < 8 or render_area < 8:
        return None
    ys, xs = np.nonzero(render_mask)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    sy, sx = np.nonzero(sprite_mask)
    best = None
    for factor in scales:
        scale = np.sqrt(sprite_area / render_area) * factor
        small = place(render, (*box, scale, 0, 0), (int((box[3] - box[1]) * scale) + 2, int((box[2] - box[0]) * scale) + 2))
        my, mx = np.nonzero(close_gaps(small[:, :, 3] > 64, gaps))
        if len(my) == 0:
            continue
        transform = (*box, scale, int(round(sx.mean() - mx.mean())), int(round(sy.mean() - my.mean())))
        placed = place(render, transform, sprite.shape)
        placed_mask = close_gaps(placed[:, :, 3] > 64, gaps)
        union = (placed_mask | sprite_mask).sum()
        iou = float((placed_mask & sprite_mask).sum() / union) if union else 0.0
        if best is None or iou > best['iou']:
            best = {'placed': placed, 'iou': iou, 'transform': transform}
    return best


def structure(placed: np.ndarray, sprite: np.ndarray) -> float:
    """Correlation of brightness between a placed render and the sprite, where both are solid.

    The silhouette cannot tell where the tower's cabin sits on its platform, or which end
    of a hangar is which; where light and dark parts are does. Camouflage the model lacks
    correlates with nothing, so it neither helps nor hurts.
    """
    both = (placed[:, :, 3] > 64) & (sprite[:, :, 3] > 128)
    if both.sum() < 8:
        return 0.0
    a = placed[both][:, :3].mean(axis=1)
    b = sprite[both][:, :3].mean(axis=1)
    if a.std() < 1 or b.std() < 1:
        return 0.0
    return float(max(np.corrcoef(a, b)[0, 1], 0.0))


def fit_sprite(renderer: Renderer, sprite: np.ndarray, yaws, tilts=TILTS, leans=LEANS, gaps=0):
    """Best yaw, tilt, lean and scale for one cut-out.

    Yaw comes first from the silhouette seen straight down (tilt barely changes it), then
    tilt and lean are searched on the best few yaws, scored by IoU plus structure(), and
    the scale is refined last.
    """
    by_yaw = []
    for yaw in yaws:
        fitted = fit_to_sprite(renderer.render(yaw)[0], sprite, gaps=gaps)
        if fitted:
            by_yaw.append((fitted['iou'], yaw))
    candidates = []
    for _, yaw in sorted(by_yaw, reverse=True)[:3]:
        for tilt in tilts:
            for lean in (leans if tilt else (0,)):
                fitted = fit_to_sprite(renderer.render(yaw, tilt, lean)[0], sprite, gaps=gaps)
                if fitted:
                    score = fitted['iou'] + STRUCTURE_WEIGHT * structure(fitted['placed'], sprite)
                    candidates.append((score, yaw, tilt, lean))
    best = None
    for _, yaw, tilt, lean in sorted(candidates, reverse=True)[:3]:
        fitted = fit_to_sprite(renderer.render(yaw, tilt, lean)[0], sprite, SCALE_STEPS, gaps)
        if not fitted:
            continue
        fitted['structure'] = structure(fitted['placed'], sprite)
        score = fitted['iou'] + STRUCTURE_WEIGHT * fitted['structure']
        if best is None or score > best['score']:
            best = {**fitted, 'score': score, 'yaw': yaw, 'tilt': tilt, 'lean': lean,
                    'pixels_per_unit': RENDER_SIZE / (2 * renderer.view_half) * fitted['transform'][4]}
    return best


# --------------------------------------------------------------------------- real colours

BAKE_SIZE = 256  # texture over the model's top-down square of side 2 * VIEW_HALF


def bake_texture(renderer: Renderer, fitted):
    """Top-down texture of the real colours, from cut-outs with their fits.

    Each fitted render's pixels are traced back to the model surface through the depth
    buffer, and the real cut-out's colour at that pixel is stored at the point's (x, y).
    Averaged over all cut-outs, this carries the real camouflage and markings onto the
    model; vertical sides take the colour of the edge above them. Returns RGB uint8.
    """
    total = np.zeros((BAKE_SIZE, BAKE_SIZE, 3))
    count = np.zeros((BAKE_SIZE, BAKE_SIZE))
    for sprite, fit in fitted:
        _, depth = renderer.render(fit['yaw'], fit['tilt'], fit['lean'])
        x0, y0, x1, y1, scale, left, top = fit['transform']
        py, px = np.nonzero(depth > 0)
        # Render pixel -> sprite pixel, the inverse of place().
        sx = np.floor((px + 0.5 - x0) * scale + left).astype(int)
        sy = np.floor((py + 0.5 - y0) * scale + top).astype(int)
        inside = (sx >= 0) & (sy >= 0) & (sx < sprite.shape[1]) & (sy < sprite.shape[0])
        px, py, sx, sy = px[inside], py[inside], sx[inside], sy[inside]
        # Only well inside the real silhouette: its edge mixes in the background.
        solid = cv2.erode((sprite[:, :, 3] > 128).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        keep = solid[sy, sx]
        px, py, sx, sy = px[keep], py[keep], sx[keep], sy[keep]
        points = renderer.unproject(px, py, depth[py, px], fit['yaw'], fit['tilt'], fit['lean'])
        tx = np.clip(((points[:, 0] + VIEW_HALF) / (2 * VIEW_HALF) * BAKE_SIZE).astype(int), 0, BAKE_SIZE - 1)
        ty = np.clip(((VIEW_HALF - points[:, 1]) / (2 * VIEW_HALF) * BAKE_SIZE).astype(int), 0, BAKE_SIZE - 1)
        np.add.at(total, (ty, tx), sprite[sy, sx, :3].astype(np.float64))
        np.add.at(count, (ty, tx), 1)
    if not count.any():
        return None
    texture = total / np.maximum(count, 1)[:, :, None]
    # Parts no cut-out showed take the nearest seen colour.
    from scipy.ndimage import distance_transform_edt
    _, (iy, ix) = distance_transform_edt(count == 0, return_indices=True)
    return texture[iy, ix].astype(np.uint8)


def export_painted(meshes, texture: np.ndarray, path: Path):
    """Save the model with the baked real colours as a .glb, in the fit's own frame.

    The geometry is the normalised one the fit used (dropped parts gone, --up applied), so
    the 3D viewer's drone view shows exactly the fitted pose. Showcase models (the A-7
    has 5.2M faces) are decimated to about MAX_EXPORT_FACES: the object is under 200 px
    in a frame, and the file must stay under GitHub's 100 MB limit to be committed.
    """
    total = sum(len(m.faces) for m in meshes)
    if total > MAX_EXPORT_FACES:
        keep = MAX_EXPORT_FACES / total
        meshes = [m.simplify_quadric_decimation(face_count=max(int(len(m.faces) * keep), 4))
                  if len(m.faces) > 100 else m for m in meshes]
        print(f'    decimated {total} -> {sum(len(m.faces) for m in meshes)} faces for export')
    to_y_up = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])  # glTF is Y-up
    scene = trimesh.Scene()
    for i, mesh in enumerate(textured_meshes(meshes, texture)):
        mesh.apply_transform(to_y_up)
        scene.add_geometry(mesh, node_name=f'part_{i}')
    scene.export(str(path))


def textured_meshes(meshes, texture: np.ndarray):
    """Copies of the meshes painted with a baked top-down texture (planar projection)."""
    from PIL import Image
    image = Image.fromarray(texture)
    out = []
    for mesh in meshes:
        uv = np.stack([(mesh.vertices[:, 0] + VIEW_HALF) / (2 * VIEW_HALF),
                       (mesh.vertices[:, 1] + VIEW_HALF) / (2 * VIEW_HALF)], axis=1)
        visual = trimesh.visual.TextureVisuals(uv=uv, image=image)
        out.append(trimesh.Trimesh(mesh.vertices, mesh.faces, visual=visual, process=False))
    return out


MIN_SHADING_CORRELATION = 0.3


def fit_colour(placed: np.ndarray, sprite: np.ndarray):
    """Gain and per-channel offset mapping the render's colours onto the sprite's."""
    both = (placed[:, :, 3] > 64) & (sprite[:, :, 3] > 128)
    if both.sum() < 8:
        return np.ones(3), np.zeros(3), 999.0
    source = placed[both][:, :3].astype(np.float64)
    target = sprite[both][:, :3].astype(np.float64)
    # One contrast gain for all three channels, from brightness, plus a per-channel
    # offset. Separate gains per channel bent the hue: the Churchill's green channel
    # alone got 3x, which turned its panel lines lime.
    #
    # The gain matches the spread of brightness rather than a per-pixel regression: at
    # 20-180 px the render sits a pixel or two off the sprite, and a regression slope
    # shrinks toward zero and washes out contrast (a TIE's black panels on a white body
    # came out mid-grey). But stretching only makes sense where the render's light and
    # dark parts line up with the sprite's (TIE, jet: correlation 0.5-0.6). A tank's
    # spread is camouflage the model does not have (correlation ~0), and its texture's
    # track and panel lines are not in the sprite either; there the regression slope
    # (near zero) is the honest answer and flattens that detail toward the mean colour.
    luma_source, luma_target = source.mean(axis=1), target.mean(axis=1)
    gain = 1.0  # a nearly flat render: gain and offset are not separable, only shift it
    if luma_source.std() > 2.0:
        correlation = np.corrcoef(luma_source, luma_target)[0, 1]
        ratio = luma_target.std() / luma_source.std()
        gain = ratio if correlation >= MIN_SHADING_CORRELATION else max(correlation, 0.0) * ratio
        gain = float(np.clip(gain, 0.2, 4.0))
    gains = np.full(3, gain)
    offsets = target.mean(axis=0) - gain * source.mean(axis=0)
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
    parser.add_argument('--tilts', type=float, nargs='*', default=list(TILTS),
                        help='camera tilt from straight down, toward the image centre, degrees')
    parser.add_argument('--sun-azimuth', type=float, default=135)
    parser.add_argument('--up', choices=['auto', 'y', 'z'], default='auto')
    parser.add_argument('--drop', nargs='*', default=[],
                        help='parts to leave out, matched against geometry/node names (e.g. Cube.032)')
    parser.add_argument('--close-gaps', type=int, default=0, metavar='PX',
                        help='fill gaps in the render silhouette up to ~2*PX sprite pixels before comparing, '
                             'for open structures whose cut-outs include the ground between their parts')
    parser.add_argument('--thicken', type=float, default=1.0, metavar='F',
                        help='widen thin beams and planks F times (legs of a lattice tower)')
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
            meshes, height = load_normalised(path, args.up, args.drop, args.thicken)
        except Exception as exc:
            print(f'  {name}: cannot load ({exc})')
            continue
        renderer = Renderer(meshes, height, args.sun_azimuth)
        print(f'  {name}: {sum(len(m.faces) for m in meshes)} faces, height/length {height:.2f}')

        per_sprite = []
        for file, sprite, bbox in sprites:
            best = fit_sprite(renderer, sprite, yaws, args.tilts, gaps=args.close_gaps)
            if best is None:
                per_sprite.append(None)
                continue
            gains, offsets, error = fit_colour(best['placed'], sprite)
            best.update(gains=gains, offsets=offsets, colour_error=error,
                        coloured=apply_colour(best['placed'], gains, offsets),
                        length_m=best['pixels_per_unit'] * METRES_PER_PIXEL, file=file)
            per_sprite.append(best)

        # Real colours: bake them onto the model and render each fit with them.
        fitted = [(sprite, best) for (_, sprite, _), best in zip(sprites, per_sprite) if best]
        texture = bake_texture(renderer, fitted)
        renderer.close()
        if texture is not None:
            BAKED.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(BAKED / f'{args.class_name}_{name}.png'), cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))
            export_painted(meshes, texture, BAKED / f'{args.class_name}_{name}.glb')
            painted = Renderer(textured_meshes(meshes, texture), height, flat=True)
            for (_, sprite, _), best in zip(sprites, per_sprite):
                if best:
                    render = painted.render(best['yaw'], best['tilt'], best['lean'])[0]
                    best['painted'] = place(render, best['transform'], sprite.shape)
            painted.close()
        results[name] = per_sprite

        good = [b for b in per_sprite if b]
        if good:
            summary = {
                'mesh': str(path.relative_to(ROOT)),
                'dropped_parts': args.drop,
                'up': args.up,
                'close_gaps': args.close_gaps,
                'thicken': args.thicken,
                'tilts': args.tilts,
                'mean_iou': round(float(np.mean([b['iou'] for b in good])), 3),
                'mean_colour_error': round(float(np.mean([b['colour_error'] for b in good])), 1),
                'length_m': round(float(np.median([b['length_m'] for b in good])), 1),
                'tilt': float(np.median([b['tilt'] for b in good])),
                'gains': np.median([b['gains'] for b in good], axis=0).round(3).tolist(),
                'offsets': np.median([b['offsets'] for b in good], axis=0).round(1).tolist(),
                'per_sprite': [{'file': b['file'], 'yaw': b['yaw'], 'tilt': b['tilt'], 'lean': b['lean'],
                                'structure': round(b['structure'], 3),
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
    for i, (file, sprite, _) in enumerate(sprites):
        cells = [to_cell(sprite, ['REAL', Path(file).name])]
        for name, per_sprite in results.items():
            best = per_sprite[i]
            if best is None:
                cells.append(to_cell(None, [name[:26], 'no match']))
                continue
            cells.append(to_cell(best['coloured'], [
                name[:26],
                f"IoU {best['iou']:.2f}  colour err {best['colour_error']:.0f}",
                f"yaw {best['yaw']} tilt {best['tilt']:.0f} lean {best['lean']} {best['length_m']:.1f} m",
            ]))
            if 'painted' in best:
                cells.append(to_cell(best['painted'], [name[:26], 'real colours baked on']))
        rows.append(np.hstack(cells))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sheet_path = out / f'{args.class_name}_match.png'
    cv2.imwrite(str(sheet_path), np.vstack(rows))
    print(f'sheet -> {sheet_path}')


if __name__ == '__main__':
    main()
