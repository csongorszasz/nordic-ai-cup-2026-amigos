"""Local web tool to review sprite cut-outs and redraw bad ones by hand.

    cd drone-flyby
    .venv/bin/python training/sprite_review/server.py      # http://localhost:8765

Decisions are stored in sprites/manual.json, keyed by sprite file:
    {"status": "accepted" | "rejected" | "manual", "polygon": [[x, y], ...] | null}
Polygons are in 4K source pixels. A manual polygon rewrites the sprite's alpha;
the GrabCut original is kept in sprites/_grabcut/ so "reset" can restore it.
"""

import json
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))  # dtos.py / utils.py live in src/

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH  # noqa: E402
from utils import load_frame  # noqa: E402

SPRITES = ROOT / 'sprites'
MANUAL = SPRITES / 'manual.json'
BACKUP = SPRITES / '_grabcut'
CONTEXT = 16  # source pixels of context shown around the box

app = FastAPI()
index = {e['file']: e for e in json.loads((SPRITES / 'index.json').read_text())}

# Real-vs-render comparisons written by training/compare_render.py, one folder per class.
MODEL_MATCH = ROOT / 'datasets' / 'model_match'
MODEL_MATCH.mkdir(parents=True, exist_ok=True)
app.mount('/compare', StaticFiles(directory=MODEL_MATCH, html=True), name='compare')


@app.get('/api/compare')
def compare_pages():
    """Classes that have a comparison page, for the sidebar."""
    return sorted(p.name.removesuffix('_compare') for p in MODEL_MATCH.glob('*_compare')
                  if (p / 'index.html').exists())


# 3D models under models/<class>/<model>/, shown by viewer3d.html.
MODELS = ROOT / 'models'
MESH_SUFFIXES = {'.glb', '.gltf', '.obj'}
app.mount('/models', StaticFiles(directory=MODELS), name='models')


@app.get('/viewer3d', response_class=HTMLResponse)
def viewer3d():
    return (Path(__file__).parent / 'viewer3d.html').read_text()


# Synthetic frames rebuilt in 3D by training/scene3d.py, flown over in scene3d.html.
SCENES = ROOT / 'datasets' / 'scene3d'
SCENES.mkdir(parents=True, exist_ok=True)
app.mount('/scene3d_data', StaticFiles(directory=SCENES), name='scene3d_data')


@app.get('/scene3d', response_class=HTMLResponse)
def scene3d():
    return (Path(__file__).parent / 'scene3d.html').read_text()


@app.get('/api/scenes')
def scenes_3d():
    """Scene folders, newest first."""
    found = sorted(SCENES.glob('*/scene.json'), key=lambda p: -p.stat().st_mtime)
    return [p.parent.name for p in found]


@app.get('/api/models')
def models_3d():
    """Every mesh with its fit from match.json, if render_models.py has run on it."""
    out = []
    for path in sorted(p for p in MODELS.rglob('*') if p.suffix.lower() in MESH_SUFFIXES):
        cls, name = path.relative_to(MODELS).parts[:2]
        # Prefer the texture-slimmed copy render_models.slim_glb() caches; the originals
        # can be hundreds of MB of 8K maps that the browser would have to decode.
        slim = MODEL_MATCH / '_slim' / f'{path.parent.name}_{path.stem}.glb'
        served = slim if slim.exists() else path
        url = ('/compare/' + slim.relative_to(MODEL_MATCH).as_posix() if slim.exists()
               else '/models/' + path.relative_to(MODELS).as_posix())
        # A .gltf is only the JSON; its .bin and textures sit beside it.
        size = (sum(f.stat().st_size for f in path.parent.rglob('*') if f.is_file())
                if served.suffix.lower() == '.gltf' else served.stat().st_size)
        item = {'key': f'{cls}/{name}', 'cls': cls, 'name': name, 'url': url,
                'size_mb': round(size / 2**20, 1)}
        painted = MODEL_MATCH / '_baked' / f'{cls}_{name}.glb'  # render_models.export_painted()
        if painted.exists():
            item['painted_url'] = '/compare/' + painted.relative_to(MODEL_MATCH).as_posix()
        match_path = MODELS / cls / 'match.json'
        fit = json.loads(match_path.read_text()).get(name) if match_path.exists() else None
        if fit and Path(fit['mesh']).name == path.name:
            per = fit.get('per_sprite', [])
            best = max(per, key=lambda s: s['iou']) if per else {}
            item.update(mean_iou=fit['mean_iou'], length_m=fit['length_m'],
                        dropped_parts=fit.get('dropped_parts', []),
                        yaw=best.get('yaw'), tilt=best.get('tilt', fit.get('tilt')), lean=best.get('lean', 0),
                        sprite=best.get('file'))
        out.append(item)
    return out


def load_manual() -> dict:
    return json.loads(MANUAL.read_text()) if MANUAL.exists() else {}


def crop_origin(bbox):
    x1, y1, x2, y2 = bbox
    return max(x1 - CONTEXT, 0), max(y1 - CONTEXT, 0), min(x2 + CONTEXT, IMAGE_WIDTH), min(y2 + CONTEXT, IMAGE_HEIGHT)


@lru_cache(maxsize=6)
def frame_image(frame: int):
    return load_frame(frame)


def entry_or_404(file: str) -> dict:
    if file not in index:
        raise HTTPException(404, 'unknown sprite')
    return index[file]


@app.get('/', response_class=HTMLResponse)
def page():
    return (Path(__file__).parent / 'index.html').read_text()


@app.get('/api/items')
def items():
    manual = load_manual()
    out = []
    for file, e in index.items():
        cx1, cy1, cx2, cy2 = crop_origin(e['bbox'])
        m = manual.get(file, {})
        out.append({**e, 'crop': [cx1, cy1, cx2, cy2], 'status': m.get('status', 'auto'), 'polygon': m.get('polygon')})
    return out


@app.get('/img/crop/{file:path}')
def crop(file: str):
    e = entry_or_404(file)
    cx1, cy1, cx2, cy2 = crop_origin(e['bbox'])
    ok, png = cv2.imencode('.png', frame_image(e['frame'])[cy1:cy2, cx1:cx2])
    return Response(png.tobytes(), media_type='image/png', headers={'Cache-Control': 'max-age=3600'})


@app.get('/img/sprite/{file:path}')
def sprite(file: str):
    entry_or_404(file)
    return FileResponse(SPRITES / file, headers={'Cache-Control': 'no-store'})


class Decision(BaseModel):
    file: str
    status: str                       # accepted | rejected | manual | auto
    polygon: Optional[List[List[float]]] = None


@app.post('/api/save')
def save(d: Decision):
    e = entry_or_404(d.file)
    if d.status not in ('accepted', 'rejected', 'manual', 'auto'):
        raise HTTPException(400, 'bad status')
    path, backup = SPRITES / d.file, BACKUP / d.file
    manual = load_manual()

    if d.status == 'manual':
        if not d.polygon or len(d.polygon) < 3:
            raise HTTPException(400, 'polygon needs 3+ points')
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
        x1, y1, x2, y2 = e['bbox']
        rgb = frame_image(e['frame'])[y1:y2, x1:x2]
        alpha = np.zeros(rgb.shape[:2], np.uint8)
        pts = np.round(np.array(d.polygon) - [x1, y1]).astype(np.int32)
        cv2.fillPoly(alpha, [pts], 255)
        cv2.imwrite(str(path), np.dstack([rgb, alpha]))
        manual[d.file] = {'status': 'manual', 'polygon': d.polygon}
    elif d.status == 'auto':
        if backup.exists():
            shutil.copy2(backup, path)
        manual.pop(d.file, None)
    else:
        manual[d.file] = {**manual.get(d.file, {}), 'status': d.status}

    MANUAL.write_text(json.dumps(manual, indent=1))
    return {'ok': True}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8765, log_level='warning')
