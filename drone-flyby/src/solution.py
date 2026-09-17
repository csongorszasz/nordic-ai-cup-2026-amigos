"""Baseline v1: YOLO detector + per-sequence object memory + L1 sweep camera.

Detector: fine-tuned YOLO (training/train_yolo.py) run on the transmitted view.
Memory: detections are kept as tracks in source pixels, moved by their
estimated per-frame drift, and reported on later frames while the camera looks
elsewhere, because every frame is scored against the whole source frame.
Camera: snake sweep over six Level-1 positions that tile the frame.

Model path: $DRONE_MODEL, default runs/yolo11s_baseline/weights/best.pt.
"""

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from dtos import (
    ALLOWED_RESOLUTION_LEVELS,
    MAXIMUM_CENTER_DELTA_PIXELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    OBJECT_CLASSES,
    DroneFlybyPredictionDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
    RequestedViewDto,
)
from utils import center_bounds_for_level, clip_bbox_to_frame, decode_view

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]  # drone-flyby/ (runs/, recordings/, sprites/ live there)
MODEL_PATH = Path(os.environ.get('DRONE_MODEL', ROOT / 'runs' / 'yolo11s_baseline' / 'weights' / 'best.pt'))
DETECT_CONF = float(os.environ.get('DRONE_CONF', 0.05))
IMGSZ = 960
USE_MEMORY = os.environ.get('DRONE_MEMORY', '1') != '0'
CAMERA_POLICY = os.environ.get('DRONE_CAMERA', 'hybrid')  # 'hybrid', 'sweep' (L1 snake), 'l0' (always full view) or 'record'
# 'record': data collection only. No detection (fastest answers, scores 0), and a Level-2
# back-and-forth scan across the middle row so every object drifting through is captured
# at native resolution.
RECORD_LEVEL = int(os.environ.get('RECORD_LEVEL', 2))   # 2: row scan at native resolution, 0: full view every frame
RECORD_ROW_Y = int(os.environ.get('RECORD_ROW_Y', 1080))
RECORD_XS = list(range(480, 3361, 480))   # 7 L2 centres, 480 px apart (< 551 px move limit)

# Tracking
MATCH_DISTANCE_PX = 150       # max centre distance (source px) to match a detection to a track
MAX_UNSEEN_FRAMES = 40        # forget a track not re-detected for this long
MAX_MISSES_IN_VIEW = 2        # forget a track the camera looked at and did not find this many times
CONF_DECAY_PER_FRAME = 0.97   # remembered detections lose confidence as they age
LEVEL_CONF_WEIGHT = {0: 0.8, 1: 1.0, 2: 1.0}

# Hybrid planner weights (see choose_hybrid_view).
H_L0_WEIGHT = float(os.environ.get('H_L0_WEIGHT', 8.0))      # value of a full view once it is H_L0_STALE frames old
H_L0_STALE = int(os.environ.get('H_L0_STALE', 4))
H_CELL_STALE = 8                                              # frames until a cell counts as fully stale
H_UNCERTAIN_WEIGHT = float(os.environ.get('H_UNCERTAIN', 1.5))
H_UPGRADE_WEIGHT = float(os.environ.get('H_UPGRADE', 1.0))   # object never seen zoomed in
GRID_COLS, GRID_ROWS = 8, 4                                   # 480x540 cells

# Six Level-1 centres that tile the frame, visited as a snake.
SWEEP = [(960, 540), (1920, 540), (2880, 540), (2880, 1620), (1920, 1620), (960, 1620)]


# --------------------------------------------------------------------------- model

_model = None
_lock = threading.Lock()


def _load_model():
    global _model
    if not MODEL_PATH.exists():
        logger.error('No model at %s: serving empty detections', MODEL_PATH)
        return
    from ultralytics import YOLO

    _model = YOLO(str(MODEL_PATH))
    # Warm up so the first real frame is not the slow one.
    dummy = np.zeros((540, 960, 3), dtype=np.uint8)
    for _ in range(3):
        _model.predict(dummy, imgsz=IMGSZ, conf=DETECT_CONF, verbose=False, half=True)
    logger.info('Loaded %s', MODEL_PATH)


_load_model()


def run_detector(image: np.ndarray, region: Tuple[int, int, int, int]) -> List[Tuple[str, float, np.ndarray]]:
    """Detections on one view, as (class, confidence, box in source pixels)."""
    if _model is None:
        return []
    result = _model.predict(image, imgsz=IMGSZ, conf=DETECT_CONF, verbose=False, half=True)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    h, w = image.shape[:2]
    rx1, ry1, rx2, ry2 = region
    sx, sy = (rx2 - rx1) / w, (ry2 - ry1) / h
    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    out = []
    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes):
        box = np.array([rx1 + x1 * sx, ry1 + y1 * sy, rx1 + x2 * sx, ry1 + y2 * sy])
        out.append((OBJECT_CLASSES[cls], float(conf), box))
    return out


# --------------------------------------------------------------------------- memory

@dataclass
class Track:
    cls: str
    box: np.ndarray          # source pixels at last_frame
    conf: float
    last_frame: int
    velocity: Optional[np.ndarray] = None   # px/frame, own estimate
    misses: int = 0
    best_level: int = 0

    def centre(self, box=None):
        b = self.box if box is None else box
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


class SequenceState:
    def __init__(self):
        self.tracks: List[Track] = []
        self.velocity = np.zeros(2)   # global drift estimate, px/frame
        self.velocity_samples = 0
        self.sweep_index = 0
        self.fine_seen = np.full((GRID_ROWS, GRID_COLS), -10**6)   # last frame each cell was seen at L1/L2
        self.last_l0 = -10**6
        self.last_frame_seen = -1
        # The live service applies camera commands about a frame late: a request can still show
        # the old view while our last command is about to take effect. Remember it until seen.
        self.pending: Optional[Tuple[int, int, int]] = None

    def predicted_box(self, track: Track, frame: int) -> np.ndarray:
        v = track.velocity if track.velocity is not None else self.velocity
        dt = frame - track.last_frame
        return track.box + np.array([v[0], v[1], v[0], v[1]]) * dt

    def update(self, frame: int, level: int, region, detections):
        rx1, ry1, rx2, ry2 = region
        if level == 0:
            self.last_l0 = frame
        else:
            self.fine_seen[_cells_in(region)] = frame
        predicted = [self.predicted_box(t, frame) for t in self.tracks]

        # Greedy matching by centre distance, same class.
        pairs = []
        for di, (cls, conf, box) in enumerate(detections):
            c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            for ti, t in enumerate(self.tracks):
                if t.cls != cls:
                    continue
                d = np.linalg.norm(c - t.centre(predicted[ti]))
                if d < MATCH_DISTANCE_PX:
                    pairs.append((d, di, ti))
        pairs.sort()
        used_d, used_t = set(), set()
        drift_samples = []
        for _, di, ti in pairs:
            if di in used_d or ti in used_t:
                continue
            used_d.add(di)
            used_t.add(ti)
            cls, conf, box = detections[di]
            t = self.tracks[ti]
            dt = frame - t.last_frame
            if dt > 0:
                v = (t.centre(box) - t.centre()) / dt
                t.velocity = v if t.velocity is None else 0.5 * t.velocity + 0.5 * v
                drift_samples.append(v)
            t.box, t.last_frame, t.misses = box, frame, 0
            t.best_level = max(t.best_level, level)
            t.conf = max(conf * LEVEL_CONF_WEIGHT[level], t.conf * 0.5)

        if drift_samples:
            v = np.median(np.array(drift_samples), axis=0)
            a = 0.3 if self.velocity_samples else 1.0
            self.velocity = (1 - a) * self.velocity + a * v
            self.velocity_samples += 1

        for di, (cls, conf, box) in enumerate(detections):
            if di not in used_d:
                self.tracks.append(Track(cls, box, conf * LEVEL_CONF_WEIGHT[level], frame, best_level=level))

        # Tracks the camera looked straight at but did not find count a miss.
        for ti, t in enumerate(self.tracks):
            if ti in used_t or ti >= len(predicted):
                continue
            c = t.centre(predicted[ti])
            if level > 0 and rx1 <= c[0] <= rx2 and ry1 <= c[1] <= ry2:
                t.misses += 1

        self.tracks = [
            t for t in self.tracks
            if t.misses < MAX_MISSES_IN_VIEW
            and frame - t.last_frame <= MAX_UNSEEN_FRAMES
            and _inside_frame(self.predicted_box(t, frame))
        ]

    def report(self, frame: int) -> List[DroneFlybyPredictionDto]:
        out = []
        # One box per class-object: keep highest confidence when two tracks collide.
        candidates = []
        for t in self.tracks:
            if not USE_MEMORY and t.last_frame != frame:
                continue
            box = self.predicted_box(t, frame)
            conf = t.conf * CONF_DECAY_PER_FRAME ** (frame - t.last_frame)
            candidates.append((conf, t.cls, box))
        candidates.sort(key=lambda c: -c[0])
        kept: List[Tuple[float, str, np.ndarray]] = []
        for conf, cls, box in candidates:
            if any(k[1] == cls and _iou(k[2], box) > 0.3 for k in kept):
                continue
            kept.append((conf, cls, box))
        for conf, cls, box in kept[:500]:
            g = clip_bbox_to_frame((box[0] / IMAGE_WIDTH, box[1] / IMAGE_HEIGHT, box[2] / IMAGE_WIDTH, box[3] / IMAGE_HEIGHT))
            if g is None:
                continue
            out.append(DroneFlybyPredictionDto(object_id=cls, bbox=[round(v, 6) for v in g], confidence=round(float(min(max(conf, 0.0), 1.0)), 4)))
        return out


def _inside_frame(box) -> bool:
    return box[2] > 0 and box[3] > 0 and box[0] < IMAGE_WIDTH and box[1] < IMAGE_HEIGHT


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _cells_in(region):
    """Grid index slices of the cells a source region mostly covers."""
    cw, ch = IMAGE_WIDTH / GRID_COLS, IMAGE_HEIGHT / GRID_ROWS
    c1, c2 = int(round(region[0] / cw)), int(round(region[2] / cw))
    r1, r2 = int(round(region[1] / ch)), int(round(region[3] / ch))
    return slice(max(r1, 0), max(r2, r1 + 1)), slice(max(c1, 0), max(c2, c1 + 1))


_states: Dict[str, SequenceState] = {}


# --------------------------------------------------------------------------- recording

# Every received view is saved (the README allows keeping the validation sequence):
#   recordings/<sequence_id>/<frame>_L<level>_<cx>_<cy>.png  +  views.jsonl
RECORD_DIR = os.environ.get('DRONE_RECORD_DIR', str(ROOT / 'recordings'))
_recorder = None


def record_view(request: DroneFlybyPredictRequestDto) -> None:
    global _recorder
    if not RECORD_DIR:
        return
    if _recorder is None:
        from concurrent.futures import ThreadPoolExecutor
        _recorder = ThreadPoolExecutor(max_workers=1)
    v = request.view
    meta = {'frame': request.frame, 'frame_index': request.frame_index, 'level': v.resolution_level,
            'center_x': v.center_x, 'center_y': v.center_y, 'source_region_xyxy': list(v.source_region_xyxy)}
    image_b64 = v.image
    seq = ''.join(ch for ch in request.sequence_id if ch.isalnum() or ch in '-_')[:64] or 'unknown'

    def _write():
        import base64, json
        d = Path(RECORD_DIR) / seq
        d.mkdir(parents=True, exist_ok=True)
        name = f"{meta['frame']:04d}_L{meta['level']}_{meta['center_x']}_{meta['center_y']}.png"
        (d / name).write_bytes(base64.b64decode(image_b64))
        with open(d / 'views.jsonl', 'a') as f:
            f.write(json.dumps({**meta, 'file': name}) + '\n')

    _recorder.submit(_write)


# --------------------------------------------------------------------------- camera

def _legal(frm: Tuple[int, int, int], to: Tuple[int, int, int]) -> bool:
    """Would the evaluator accept moving from view `frm` to view `to`?"""
    fl, fx, fy = frm
    tl, tx, ty = to
    if tl not in ALLOWED_RESOLUTION_LEVELS[fl]:
        return False
    if tl == 0:
        return (tx, ty) == (1920, 1080)
    min_x, max_x, min_y, max_y = center_bounds_for_level(tl)
    if not (min_x <= tx <= max_x and min_y <= ty <= max_y):
        return False
    return float(np.hypot(tx - fx, ty - fy)) <= MAXIMUM_CENTER_DELTA_PIXELS[fl]


def _camera_states(request: DroneFlybyPredictRequestDto, state: SequenceState) -> List[Tuple[int, int, int]]:
    """Positions the camera may be in when our next command is applied."""
    v = request.view
    states = [(v.resolution_level, v.center_x, v.center_y)]
    if state.pending is not None and state.pending != states[0]:
        states.append(state.pending)
    return states


def _safe(request, state, view: Optional[RequestedViewDto], ok_from_all) -> Optional[RequestedViewDto]:
    """Drop a command that could be refused; prefer legal-from-everywhere, then legal-from-pending."""
    if view is None:
        return None
    target = (view.resolution_level, view.center_x, view.center_y)
    if ok_from_all(target):
        return view
    if state.pending is not None and _legal(state.pending, target):
        return view
    return None

def choose_next_view(request: DroneFlybyPredictRequestDto, state: SequenceState) -> Optional[RequestedViewDto]:
    constraints = request.camera_constraints
    current = request.view
    if CAMERA_POLICY == 'l0':
        return None if current.resolution_level == 0 else RequestedViewDto(resolution_level=0, center_x=1920, center_y=1080)
    if CAMERA_POLICY == 'hybrid':
        return choose_hybrid_view(request, state)
    if CAMERA_POLICY == 'record':
        return choose_record_view(request, state)
    if 1 not in constraints.allowed_resolution_levels:
        return None
    bounds = constraints.bounds_for_level(1)
    limit = constraints.maximum_center_delta

    # Advance through the sweep, skipping positions we cannot reach in one move.
    for step in range(1, len(SWEEP) + 1):
        idx = (state.sweep_index + step) % len(SWEEP)
        cx, cy = SWEEP[idx]
        cx = int(min(max(cx, bounds.minimum_center_x), bounds.maximum_center_x))
        cy = int(min(max(cy, bounds.minimum_center_y), bounds.maximum_center_y))
        if limit is None or np.hypot(cx - current.center_x, cy - current.center_y) <= limit:
            state.sweep_index = idx
            return RequestedViewDto(resolution_level=1, center_x=cx, center_y=cy)
    return None


def choose_record_view(request: DroneFlybyPredictRequestDto, state: SequenceState) -> Optional[RequestedViewDto]:
    current = request.view
    if RECORD_LEVEL == 0:
        return None if current.resolution_level == 0 else RequestedViewDto(resolution_level=0, center_x=1920, center_y=1080)
    if current.resolution_level == 0:
        return RequestedViewDto(resolution_level=1, center_x=960, center_y=RECORD_ROW_Y)
    if current.resolution_level == 1:
        return RequestedViewDto(resolution_level=2, center_x=RECORD_XS[0], center_y=RECORD_ROW_Y)
    # Ping-pong along the row; sweep_index walks 0..6..0.
    n = len(RECORD_XS)
    state.sweep_index = (state.sweep_index + 1) % (2 * (n - 1))
    i = state.sweep_index if state.sweep_index < n else 2 * (n - 1) - state.sweep_index
    return RequestedViewDto(resolution_level=2, center_x=RECORD_XS[i], center_y=RECORD_ROW_Y)


def _candidate_views(state: SequenceState, next_frame: int):
    centres = {
        1: [(x, y) for x in (960, 1440, 1920, 2400, 2880) for y in (540, 810, 1080, 1350, 1620)],
        2: [(x, y) for x in range(480, 3361, 480) for y in range(270, 1891, 405)],
    }
    # Also centre zoomed views on the predicted position of every remembered object.
    for t in state.tracks:
        c = t.centre(state.predicted_box(t, next_frame))
        centres[1].append((int(c[0]), int(c[1])))
        centres[2].append((int(c[0]), int(c[1])))
    yield 0, 1920, 1080
    for level in (1, 2):
        min_x, max_x, min_y, max_y = center_bounds_for_level(level)
        seen = set()
        for x, y in centres[level]:
            x = int(min(max(x, min_x), max_x))
            y = int(min(max(y, min_y), max_y))
            if (x, y) not in seen:
                seen.add((x, y))
                yield level, x, y


def choose_hybrid_view(request: DroneFlybyPredictRequestDto, state: SequenceState) -> Optional[RequestedViewDto]:
    """Pick the reachable view with the highest expected information for the next frame.

    Full view (L0): worth more the longer since we last saw everything.
    Zoomed views: stale grid cells they cover, plus remembered objects inside them
    that are uncertain or were only ever seen at a lower zoom.
    """
    current = request.view
    next_frame = request.frame + 1
    predicted = [(t, t.centre(state.predicted_box(t, next_frame))) for t in state.tracks]
    states = _camera_states(request, state)

    best, best_value = None, -1.0
    for level, cx, cy in _candidate_views(state, next_frame):
        if not all(_legal(st, (level, cx, cy)) for st in states):
            continue
        if level == 0:
            value = H_L0_WEIGHT * min(next_frame - state.last_l0, H_L0_STALE) / H_L0_STALE
        else:
            x1, y1, x2, y2 = (cx - (1920 >> (level - 1)) // 2, cy - (1080 >> (level - 1)) // 2,
                              cx + (1920 >> (level - 1)) // 2, cy + (1080 >> (level - 1)) // 2)
            stale = np.clip((next_frame - state.fine_seen[_cells_in((x1, y1, x2, y2))]) / H_CELL_STALE, 0, 1)
            value = float(stale.sum()) / (2 if level == 1 else 1) * 0.5
            for t, c in predicted:
                if x1 <= c[0] <= x2 and y1 <= c[1] <= y2:
                    value += H_UNCERTAIN_WEIGHT * (1.0 - min(t.conf, 1.0))
                    if t.best_level < level:
                        value += H_UPGRADE_WEIGHT
        if value > best_value:
            best, best_value = (level, cx, cy), value

    if best is None:
        return None
    level, cx, cy = best
    if len(states) == 1 and (level, cx, cy) == states[0]:
        return None
    return RequestedViewDto(resolution_level=level, center_x=int(cx), center_y=int(cy))


# --------------------------------------------------------------------------- entry point

def predict(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    if request.camera_command_feedback is not None:
        logger.warning('Camera command ignored: %s', request.camera_command_feedback.reason)

    try:
        record_view(request)
    except Exception:
        logger.exception('Recording failed on frame %s', request.frame)

    with _lock:
        state = _states.get(request.sequence_id)
        # A new attempt (or a replay reusing the id) must not inherit old memory.
        if state is None or request.frame_index == 0 or request.frame < state.last_frame_seen:
            state = _states[request.sequence_id] = SequenceState()
        state.last_frame_seen = request.frame
        v = request.view
        if state.pending == (v.resolution_level, v.center_x, v.center_y):
            state.pending = None
        annotations: List[DroneFlybyPredictionDto] = []
        requested_view = None
        try:
            image = None if CAMERA_POLICY == 'record' else decode_view(request.view)
            region = tuple(request.view.source_region_xyxy)
            detections = [] if CAMERA_POLICY == 'record' else run_detector(image, region)
            state.update(request.frame, request.view.resolution_level, region, detections)
            annotations = state.report(request.frame)
        except Exception:
            logger.exception('Detection failed on frame %s', request.frame)
        try:
            states = _camera_states(request, state)
            requested_view = _safe(request, state, choose_next_view(request, state),
                                   lambda target: all(_legal(st, target) for st in states))
            if requested_view is not None:
                state.pending = (requested_view.resolution_level, requested_view.center_x, requested_view.center_y)
        except Exception:
            logger.exception('Camera policy failed on frame %s', request.frame)

    return DroneFlybyPredictResponseDto(
        request_id=request.request_id,
        frame=request.frame,
        annotations=annotations,
        requested_view=requested_view,
    )
