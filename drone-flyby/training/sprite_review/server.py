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
from datetime import datetime
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

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH, OBJECT_CLASSES  # noqa: E402
from utils import frame_numbers, load_annotations, load_frame  # noqa: E402

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


# Flight playback (flyby.html): the supplied scene and the recorded validation flight as a
# video, zoomed through the three levels by hand. Frames go out as JPEG, cached on disk.
RECORDED = ROOT / 'recordings' / 'validation_4k'
FLYBY_CACHE = ROOT / 'datasets' / 'flyby_cache'


def flyby_frames(source: str) -> dict:
    """{frame number: image path} of a playback source."""
    if source == 'helsinki':
        return {n: ROOT / 'data' / 'helsinki' / 'images' / f'frame_{n:06d}.png' for n in frame_numbers()}
    if source == 'validation_4k':
        return {int(p.stem.split('_')[-1]): p for p in sorted(RECORDED.glob('frame_*.jpg'))}
    raise HTTPException(404, 'unknown source')


@app.get('/flyby', response_class=HTMLResponse)
def flyby():
    return (Path(__file__).parent / 'flyby.html').read_text()


@app.get('/api/flyby/sources')
def flyby_sources():
    out = [{'name': 'helsinki', 'label': 'Helsinki (training scene, labelled)', 'frames': sorted(flyby_frames('helsinki')),
            'labelled': True}]
    if RECORDED.is_dir():
        coverage = RECORDED / 'coverage.json'  # rewritten by every rebuild: a new version busts browser caches
        labelled = (REVIEW / 'labels.json').exists()
        out.append({'name': 'validation_4k', 'frames': sorted(flyby_frames('validation_4k')), 'labelled': labelled,
                    'label': 'Copenhagen (recorded validation, ' + ('hand-checked labels, test only)' if labelled else 'no labels)'),
                    'version': int(coverage.stat().st_mtime) if coverage.exists() else 0})
    return out


@app.get('/flyby/img/{source}/{frame}.jpg')
def flyby_image(source: str, frame: int):
    path = flyby_frames(source).get(frame)
    if path is None:
        raise HTTPException(404, 'unknown frame')
    if path.suffix == '.png':  # the supplied frames are ~10 MB PNGs: convert once
        cached = FLYBY_CACHE / source / f'{frame:06d}.jpg'
        if not cached.exists():
            cached.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cached), cv2.imread(str(path)), [cv2.IMWRITE_JPEG_QUALITY, 92])
        path = cached
    return FileResponse(path, headers={'Cache-Control': 'max-age=86400'})


@app.get('/api/flyby/labels/{frame}')
def flyby_labels(frame: int, source: str = 'helsinki'):
    """Ground-truth boxes of a frame, in 4K source pixels: the supplied scene's, or for the
    recorded flight the ones built from the review (training/copenhagen_labels.py)."""
    if source == 'validation_4k':
        path = REVIEW / 'labels.json'
        if not path.exists():
            return []
        return [{'class': a['object_id'], 'bbox': a['bbox'], 'track': a['track']}
                for a in json.loads(path.read_text())['labels'].get(str(frame), [])]
    return [{'class': a['object_id'], 'bbox': a['bbox']} for a in load_annotations(frame)]


TRACES = ROOT / 'datasets' / 'policy_traces'  # training/run_policy.py


@app.get('/api/flyby/traces')
def flyby_traces():
    """Saved policy runs, newest first, without their steps."""
    out = []
    for path in sorted(TRACES.glob('*.json'), key=lambda p: -p.stat().st_mtime):
        trace = json.loads(path.read_text())
        trace.pop('steps', None)
        out.append(trace)
    return out


@app.get('/api/flyby/trace/{name}')
def flyby_trace(name: str):
    path = TRACES / f'{name}.json'
    if path.parent != TRACES or not path.exists():
        raise HTTPException(404, 'unknown trace')
    return FileResponse(path)


# Copenhagen review (review.html): accept or reject the detector's tracks on the recorded
# flight (training/copenhagen_candidates.py). TEST SET ONLY: never used for training.
REVIEW = ROOT / 'datasets' / 'copenhagen_test'
DECISIONS = REVIEW / 'decisions.json'


@lru_cache(maxsize=8)
def recorded_frame(frame: int):
    return cv2.imread(str(RECORDED / f'frame_{frame:04d}.jpg'))


@app.get('/review', response_class=HTMLResponse)
def review():
    return (Path(__file__).parent / 'review.html').read_text()


@app.get('/api/review/candidates')
def review_candidates():
    path = REVIEW / 'candidates.json'
    if not path.exists():
        raise HTTPException(404, 'no candidates yet: run training/copenhagen_candidates.py')
    data = json.loads(path.read_text())
    data['decisions'] = json.loads(DECISIONS.read_text()) if DECISIONS.exists() else {}
    data['classes'] = list(OBJECT_CLASSES)
    data['references'] = reference_sprites()
    data['sprites'] = {f: index[f]['bbox'] for fs in data['references'].values() for f in fs}
    return data


def reference_sprites(per_class: int = 3) -> dict:
    """Labelled Helsinki examples of each class to compare against: whole, clean, the biggest
    ones, from different frames. Served by /img/crop/{file}."""
    out = {}
    for cls in OBJECT_CLASSES:
        pool = [e for e in index.values() if e['class'] == cls and not e['truncated'] and not e['suspect']]
        pool = pool or [e for e in index.values() if e['class'] == cls]
        pool.sort(key=lambda e: -(e['bbox'][2] - e['bbox'][0]) * (e['bbox'][3] - e['bbox'][1]))
        picked, frames = [], set()
        for e in pool:
            if e['frame'] not in frames:
                picked.append(e['file'])
                frames.add(e['frame'])
            if len(picked) == per_class:
                break
        out[cls] = picked
    return out


@app.get('/review/crop/{frame}.jpg')
def review_crop(frame: int, box: str, side: int = 360):
    """A square crop around the box (three times its size, at least 200 px), box drawn in."""
    image = recorded_frame(frame)
    if image is None:
        raise HTTPException(404, 'unknown frame')
    return boxed_crop(image, [float(v) for v in box.split(',')], side)


@app.get('/flyby/refcrop/{file:path}')
def reference_crop(file: str, side: int = 200):
    """A labelled Helsinki object framed exactly as /review/crop frames a Copenhagen box, so the
    two can be compared side by side: how much room the official label leaves around it."""
    e = entry_or_404(file)
    return boxed_crop(frame_image(e['frame']), e['bbox'], side)


def boxed_crop(image, box, side: int):
    x1, y1, x2, y2 = (int(v) for v in box)
    half = max(max(x2 - x1, y2 - y1) * 1.5, 100)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = int(max(cx - half, 0)), int(max(cy - half, 0))
    c, d = int(min(cx + half, IMAGE_WIDTH)), int(min(cy + half, IMAGE_HEIGHT))
    crop = image[b:d, a:c].copy()
    scale = side / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    p = lambda x, y: (int((x - a) * scale), int((y - b) * scale))
    cv2.rectangle(crop, p(x1, y1), p(x2, y2), (0, 230, 255), 1 if side < 250 else 2)
    ok, jpg = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(jpg.tobytes(), media_type='image/jpeg', headers={'Cache-Control': 'max-age=3600'})


# Boxes drawn by hand in the playback page, for objects the detector never found (or found
# with a bad box). copenhagen_labels.py carries each through the flight like a reviewed track.
MANUAL_BOXES = REVIEW / 'manual_boxes.json'


def rebuild_labels():
    import subprocess
    done = subprocess.run([sys.executable, str(ROOT / 'training' / 'copenhagen_labels.py')],
                          capture_output=True, text=True, cwd=ROOT)
    if done.returncode != 0:
        raise HTTPException(500, 'label rebuild failed: ' + done.stderr[-500:])
    return done.stdout.splitlines()[0] if done.stdout else ''


def write_json(path: Path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)  # never a half-written file


@app.get('/api/flyby/references')
def flyby_references():
    refs = reference_sprites()
    return {'classes': list(OBJECT_CLASSES), 'references': refs,
            'sizes': {f: index[f]['bbox'] for fs in refs.values() for f in fs}}


class ManualBox(BaseModel):
    frame: int
    box: List[float]     # 4K source px
    cls: str


@app.post('/api/flyby/manual')
def flyby_manual(m: ManualBox):
    if m.cls not in OBJECT_CLASSES:
        raise HTTPException(400, 'bad class')
    x1, y1, x2, y2 = m.box
    if x2 - x1 < 4 or y2 - y1 < 4:
        raise HTTPException(400, 'box too small')
    boxes = json.loads(MANUAL_BOXES.read_text()) if MANUAL_BOXES.exists() else {}
    mid = f'm{m.frame:03d}_{int((x1 + x2) / 2):04d}_{int((y1 + y2) / 2):04d}'
    boxes[mid] = {'frame': m.frame, 'box': [round(v, 1) for v in m.box], 'class': m.cls,
                  'at': datetime.now().isoformat(timespec='seconds')}
    write_json(MANUAL_BOXES, boxes)
    return {'ok': True, 'id': mid, 'labels': rebuild_labels()}


class RemoveLabel(BaseModel):
    track: str


@app.post('/api/flyby/remove')
def flyby_remove(r: RemoveLabel):
    """Take an object out of the labels: a hand-drawn box is deleted, a reviewed track (and the
    duplicates merged into it) is marked rejected."""
    boxes = json.loads(MANUAL_BOXES.read_text()) if MANUAL_BOXES.exists() else {}
    if r.track in boxes:
        del boxes[r.track]
        write_json(MANUAL_BOXES, boxes)
        return {'ok': True, 'labels': rebuild_labels()}
    labels = json.loads((REVIEW / 'labels.json').read_text())
    obj = next((o for o in labels['objects'] if o['id'] == r.track), None)
    if obj is None:
        raise HTTPException(404, 'unknown object')
    decisions = json.loads(DECISIONS.read_text()) if DECISIONS.exists() else {}
    for tid in [obj['id']] + obj['merged']:
        d = decisions.get(tid, {'best_frame': obj['ref'], 'best_box': obj['box']})
        decisions[tid] = {**d, 'status': 'rejected', 'note': 'removed in the playback page',
                          'at': datetime.now().isoformat(timespec='seconds')}
    write_json(DECISIONS, decisions)
    # A note fix would override the rejection: drop it.
    fixes_path = REVIEW / 'note_fixes.json'
    if fixes_path.exists():
        fixes = json.loads(fixes_path.read_text())
        for tid in [obj['id']] + obj['merged']:
            fixes.pop(tid, None)
        write_json(fixes_path, fixes)
    return {'ok': True, 'labels': rebuild_labels()}


@app.get('/api/flyby/candidates/{frame}')
def flyby_candidates(frame: int):
    """The detector's tracks still waiting for a decision, with their box in this frame, so they can
    be reviewed in the playback page as well as in /review."""
    path = REVIEW / 'candidates.json'
    if not path.exists():
        return []
    decisions = json.loads(DECISIONS.read_text()) if DECISIONS.exists() else {}
    out = []
    for c in json.loads(path.read_text())['candidates']:
        if decisions.get(c['id'], {}).get('status') not in (None, 'unsure') or frame not in c['frames']:
            continue
        box = c['boxes'][c['frames'].index(frame)]
        out.append({'id': c['id'], 'class': c['class'], 'score': c['score'], 'bbox': box,
                    'best_frame': c['best_frame'], 'best_box': c['best_box']})
    return out


@app.post('/api/flyby/rebuild')
def flyby_rebuild():
    return {'ok': True, 'labels': rebuild_labels()}


class Relabel(BaseModel):
    track: str
    cls: str


@app.post('/api/flyby/relabel')
def flyby_relabel(r: Relabel):
    """Give an object in the labels another class: a hand-drawn box's, or the reviewed track's (and
    the duplicates merged into it)."""
    if r.cls not in OBJECT_CLASSES:
        raise HTTPException(400, 'bad class')
    boxes = json.loads(MANUAL_BOXES.read_text()) if MANUAL_BOXES.exists() else {}
    if r.track in boxes:
        boxes[r.track]['class'] = r.cls
        write_json(MANUAL_BOXES, boxes)
        return {'ok': True, 'labels': rebuild_labels()}
    obj = next((o for o in json.loads((REVIEW / 'labels.json').read_text())['objects'] if o['id'] == r.track), None)
    if obj is None:
        raise HTTPException(404, 'unknown object')
    decisions = json.loads(DECISIONS.read_text()) if DECISIONS.exists() else {}
    fixes_path = REVIEW / 'note_fixes.json'
    fixes = json.loads(fixes_path.read_text()) if fixes_path.exists() else {}
    for tid in [obj['id']] + obj['merged']:
        if tid in decisions:
            decisions[tid]['class'] = r.cls
        if 'class' in fixes.get(tid, {}):
            fixes[tid]['class'] = r.cls   # a note fix's class would win over the decision's
    write_json(DECISIONS, decisions)
    write_json(fixes_path, fixes)
    return {'ok': True, 'labels': rebuild_labels()}


class ReviewDecision(BaseModel):
    id: str
    status: str                  # accepted | rejected | unsure | note | undecided
    cls: Optional[str] = None    # corrected class, when the detector's was wrong
    note: Optional[str] = None   # free text for Claude to look at (status 'note', or alongside any other)
    best_frame: int
    best_box: List[float]


@app.post('/api/review/decide')
def review_decide(d: ReviewDecision):
    if d.status not in ('accepted', 'rejected', 'unsure', 'note', 'undecided'):
        raise HTTPException(400, 'bad status')
    if d.cls is not None and d.cls not in OBJECT_CLASSES:
        raise HTTPException(400, 'bad class')
    decisions = json.loads(DECISIONS.read_text()) if DECISIONS.exists() else {}
    if d.status == 'undecided':
        decisions.pop(d.id, None)
    else:
        # The box goes in too, so a decision can be matched to a track of a regenerated candidate set.
        decisions[d.id] = {'status': d.status, 'class': d.cls, 'best_frame': d.best_frame, 'best_box': d.best_box,
                           'note': d.note or None, 'at': datetime.now().isoformat(timespec='seconds')}
    tmp = DECISIONS.with_suffix('.tmp')
    tmp.write_text(json.dumps(decisions, indent=1))
    tmp.replace(DECISIONS)  # never a half-written file
    return {'ok': True, 'decided': len(decisions)}


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
        match_path = MODELS / cls / 'match.json'
        fit = json.loads(match_path.read_text()).get(name) if match_path.exists() else None
        # The paint the data uses (render_models.painted_path): projected or recoloured.
        suffix = '_recoloured' if fit and fit.get('paint') == 'recoloured' else ''
        painted = MODEL_MATCH / '_baked' / f'{cls}_{name}{suffix}.glb'
        if painted.exists():
            item['painted_url'] = '/compare/' + painted.relative_to(MODEL_MATCH).as_posix()
        if fit and Path(fit['mesh']).name == path.name:
            per = fit.get('per_sprite', [])
            best = max(per, key=lambda s: s['iou']) if per else {}
            item.update(mean_iou=fit['mean_iou'], length_m=fit['length_m'],
                        dropped_parts=fit.get('dropped_parts', []),
                        yaw=best.get('yaw'), tilt=best.get('tilt', fit.get('tilt')), lean=best.get('lean', 0),
                        sprite=best.get('file'))
        out.append(item)
    return out


# Paint review (paint.html): per class, the real object (Helsinki labels, Copenhagen labels)
# next to the 3D model three ways: as found, with the paint the training data uses now, and
# with a proposed new paint. Verdicts and notes go to datasets/model_match/paint_review.json.
MODEL_SPRITES = ROOT / 'datasets' / 'model_sprites'
PAINT_REVIEW = MODEL_MATCH / 'paint_review.json'
app.mount('/model_sprites', StaticFiles(directory=MODEL_SPRITES), name='model_sprites')


@app.get('/paint', response_class=HTMLResponse)
def paint_page():
    return (Path(__file__).parent / 'paint.html').read_text()


def baked_url(cls: str, name: str, suffix: str) -> Optional[str]:
    path = MODEL_MATCH / '_baked' / f'{cls}_{name}{suffix}.glb'
    return '/compare/' + path.relative_to(MODEL_MATCH).as_posix() if path.exists() else None


@app.get('/api/paint')
def paint_classes():
    import random
    models = {m['key']: m for m in models_3d()}
    sprites = json.loads((MODEL_SPRITES / 'index.json').read_text()) if (MODEL_SPRITES / 'index.json').exists() else []
    cph = json.loads((REVIEW / 'labels.json').read_text())['objects'] if (REVIEW / 'labels.json').exists() else []
    review = json.loads(PAINT_REVIEW.read_text()) if PAINT_REVIEW.exists() else {}
    helsinki = reference_sprites(8)
    out = []
    for cls in OBJECT_CLASSES:
        match_path = MODELS / cls / 'match.json'
        fits = json.loads(match_path.read_text()) if match_path.exists() else {}
        ranked = sorted(fits.items(), key=lambda kv: -kv[1]['mean_iou'])
        name, fit = next(((n, f) for n, f in ranked if baked_url(cls, n, '')), (None, None))
        item = {'cls': cls, 'model': name, 'review': review.get(cls)}
        if name:
            m = models.get(f'{cls}/{name}', {})
            used = '_recoloured' if fit.get('paint') == 'recoloured' else ''
            proposed = '_proposed' if baked_url(cls, name, '_proposed') else ('' if used else '_recoloured')
            item.update(original=m.get('url'), current=baked_url(cls, name, used),
                        current_label='recoloured' if used else 'projected from the Helsinki cut-outs',
                        proposed=baked_url(cls, name, proposed) if proposed != used else None,
                        proposed_label={'_proposed': 'proposed new paint', '_recoloured':
                                        'recoloured (model texture shifted to the real colours)',
                                        '': 'projected'}[proposed],
                        dropped_parts=m.get('dropped_parts', []), iou=fit['mean_iou'])
        pool = [e['file'] for e in sprites if e['class'] == cls]
        item['sprites'] = random.Random(cls).sample(pool, min(8, len(pool)))
        item['helsinki'] = helsinki[cls]
        item['copenhagen'] = [{'frame': o['ref'], 'box': o['box']} for o in cph if o['class'] == cls][:8]
        out.append(item)
    return out


class PaintVerdict(BaseModel):
    cls: str
    status: str            # proposed (use it) | current (keep) | note | undecided
    note: Optional[str] = None


@app.post('/api/paint/review')
def paint_review(v: PaintVerdict):
    if v.cls not in OBJECT_CLASSES or v.status not in ('proposed', 'current', 'note', 'undecided'):
        raise HTTPException(400, 'bad class or status')
    review = json.loads(PAINT_REVIEW.read_text()) if PAINT_REVIEW.exists() else {}
    if v.status == 'undecided' and not v.note:
        review.pop(v.cls, None)
    else:
        review[v.cls] = {'status': v.status, 'note': v.note or None, 'at': datetime.now().isoformat(timespec='seconds')}
    write_json(PAINT_REVIEW, review)
    return {'ok': True}


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
