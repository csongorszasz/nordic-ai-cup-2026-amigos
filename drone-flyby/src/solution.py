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

import cv2
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

# Version 2 ideas, each switchable (DRONE_V2=1 turns them all on):
#   V2_GMC     ground motion measured from the images (BoT-SORT's camera motion compensation)
#                instead of only from matched detections
#   V2_GROUND  the "where have we looked" map moves with the ground, so ground entering the
#                frame counts as never inspected (Visual Active Search: look where finds are likely)
#   V2_VOTES   one track per object whatever class the detector says this frame; the class is
#                a vote, and a close runner-up is also reported at lower confidence
#   V2_TRUNC   a box cut by the edge of the view (not of the frame) does not overwrite a whole
#                one, and makes the object worth a new look (SAHI's partial-detection handling)
# V2_GMC is on by default: on the recorded Copenhagen flight it cuts the error of the ground motion
# the memory uses from 14.9 to 3.1 px/frame (true motion from reconstruct_frames' homographies).
# The other three are off until the labelled Copenhagen frames can measure them (25 Helsinki
# frames are too few; there they were neutral or worse). V2_GMC=0 gives back v1 exactly.
_V2 = os.environ.get('DRONE_V2', '0')
V2_GMC = os.environ.get('V2_GMC', '1') != '0'
V2_GROUND = os.environ.get('V2_GROUND', _V2) != '0'
V2_VOTES = os.environ.get('V2_VOTES', _V2) != '0'
V2_TRUNC = os.environ.get('V2_TRUNC', _V2) != '0'
GMC_MIN_SCORE = 0.6          # template match (normalised correlation) needed to trust a zoomed view's shift
GMC_MAX_AGE = int(os.environ.get('GMC_MAX_AGE', 3))   # frames: an older full view has changed too much (perspective)
# The camera looks ahead at an angle: near ground (the bottom) moves faster than far ground and
# the picture spreads sideways, so the motion is fitted as a field v(p) = t + A (p - centre)
# to the recent measurements, not as one shift.
GMC_KEEP = 16                # measurements kept for the fit
GMC_RIDGE = float(os.environ.get('GMC_RIDGE', 2.0))   # pull of the prior on A, in samples' worth
GMC_PRIOR_WEIGHT = 1.0       # pull of the prior on t
# Starting field, measured with SIFT homographies on the Helsinki frames (training scene; fits
# its 9 test points to 0.7 px). The drone and camera are the same on every flight.
GMC_PRIOR_T = np.array([0.0, 66.3])
GMC_PRIOR_A = np.array([[7.3, -0.07], [0.36, 12.78]])
VOTE_LINK_IOU = 0.3          # a detection of another class continues a track when it overlaps it this much
RUNNER_UP_SHARE = 0.25       # report the second class too when it has this share of the votes
TRUNC_MARGIN = 3             # view px: a box this close to a view edge is cut by it
GRID_COLS, GRID_ROWS = (16, 8) if V2_GROUND else (8, 4)       # 240x270 or 480x540 cells

# Six Level-1 centres that tile the frame, visited as a snake.
SWEEP = [(960, 540), (1920, 540), (2880, 540), (2880, 1400), (1920, 1400), (960, 1400)]  # bottom row at 1400: Copenhagen tune 0.349 / check 0.365 vs 1620
if os.environ.get('DRONE_SWEEP'):   # e.g. "960,540;1920,540;2880,540;1920,540": Level-1 centres, in order
    SWEEP = [tuple(int(v) for v in p.split(',')) for p in os.environ['DRONE_SWEEP'].split(';')]


# --------------------------------------------------------------------------- model

_model = None
HALF = False
_lock = threading.Lock()


def _load_model():
    global _model
    if not MODEL_PATH.exists():
        logger.error('No model at %s: serving empty detections', MODEL_PATH)
        return
    import torch
    from ultralytics import YOLO

    global HALF
    HALF = torch.cuda.is_available()   # fp16 is a GPU speed-up; on CPU it is ~100x slower
    _model = YOLO(str(MODEL_PATH))
    # Warm up so the first real frame is not the slow one.
    dummy = np.zeros((540, 960, 3), dtype=np.uint8)
    for _ in range(3):
        _model.predict(dummy, imgsz=IMGSZ, conf=DETECT_CONF, verbose=False, half=HALF)
    logger.info('Loaded %s', MODEL_PATH)


_load_model()


def run_detector(image: np.ndarray, region: Tuple[int, int, int, int]) -> List[Tuple[str, float, np.ndarray]]:
    """Detections on one view, as (class, confidence, box in source pixels)."""
    if _model is None:
        return []
    result = _model.predict(image, imgsz=IMGSZ, conf=DETECT_CONF, verbose=False, half=HALF)[0]
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
    hits: int = 1
    truncated: bool = False  # only ever seen cut by the edge of a view
    votes: Optional[Dict[str, float]] = None

    def centre(self, box=None):
        b = self.box if box is None else box
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])

    def vote(self, cls: str, weight: float):
        self.votes = self.votes or {}
        self.votes[cls] = self.votes.get(cls, 0.0) + weight
        self.cls = max(self.votes, key=self.votes.get)


class SequenceState:
    def __init__(self):
        self.tracks: List[Track] = []
        self.velocity = np.zeros(2)   # global drift estimate, px/frame
        self.velocity_samples = 0
        self.gmc_samples = 0          # of those, measured from the images
        self.motion = []              # recent (position, motion px/frame, weight) measured from the images
        self.field = np.zeros((2, 2))  # A of the motion field, px/frame per 1000 px from the centre
        if V2_GMC:
            self.velocity, self.field = GMC_PRIOR_T.copy(), GMC_PRIOR_A.copy()
        self.sweep_index = 0
        self.fine_seen = np.full((GRID_ROWS, GRID_COLS), -10**6)   # last frame each cell was seen at L1/L2
        self.map_frame = None         # frame the ground map was last moved to
        self.map_shift = np.zeros(2)  # ground motion not yet applied to the map (less than a cell)
        self.l0_image = None          # last full view, grey, for measuring motion
        self.l0_image_frame = -1
        self.last_l0 = -10**6
        self.last_frame_seen = -1
        # The live service applies camera commands about a frame late: a request can still show
        # the old view while our last command is about to take effect. Remember it until seen.
        self.pending: Optional[Tuple[int, int, int]] = None

    def velocity_at(self, p) -> np.ndarray:
        return self.velocity + self.field @ ((np.asarray(p) - _CENTRE) / 1000)

    def predicted_box(self, track: Track, frame: int) -> np.ndarray:
        # With measured ground motion, a track's own estimate is only trusted once it has a few hits.
        own = track.velocity is not None and not (V2_GMC and track.hits < 3)
        v = track.velocity if own else self.velocity_at(track.centre())
        dt = frame - track.last_frame
        return track.box + np.array([v[0], v[1], v[0], v[1]]) * dt

    def _add_motion(self, p, v: np.ndarray, weight: float):
        """A measurement from the images; refit the motion field to the recent ones."""
        self.gmc_samples += 1
        self.velocity_samples += 1
        self.motion = (self.motion + [(np.asarray(p, float), v, weight)])[-GMC_KEEP:]
        rows, target, w = [], [], []
        for q, u, wt in self.motion:
            d = (q - _CENTRE) / 1000
            rows.append([1.0, d[0], d[1]])
            target.append(u)
            w.append(wt)
        X, Y, W = np.array(rows), np.array(target), np.diag(w)
        ridge = np.diag([GMC_PRIOR_WEIGHT, GMC_RIDGE, GMC_RIDGE])
        prior = np.vstack([GMC_PRIOR_T, GMC_PRIOR_A.T])
        coef = np.linalg.solve(X.T @ W @ X + ridge, X.T @ W @ Y + ridge @ prior)   # 3x2: t, then A's columns
        self.velocity, self.field = coef[0], coef[1:].T

    def _add_velocity(self, v: np.ndarray):
        a = 0.3 if self.velocity_samples else 1.0
        self.velocity = (1 - a) * self.velocity + a * v
        self.velocity_samples += 1

    def measure_motion(self, frame: int, level: int, region, image: np.ndarray):
        """Ground motion from the pictures: a full view against the previous one, a zoomed
        view against the last full view."""
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        prev, prev_frame = self.l0_image, self.l0_image_frame
        if level == 0:
            self.l0_image, self.l0_image_frame = grey, frame
        if prev is None or frame <= prev_frame:
            return
        dt = frame - prev_frame
        if level == 0:
            (dx, dy), response = cv2.phaseCorrelate(prev, grey, _hann(grey.shape))
            if response > 0.05:   # one shift for the whole view: the motion around the centre
                self._add_motion(_CENTRE, np.array([dx, dy]) * 4 / dt, 3.0)
            return
        if dt > GMC_MAX_AGE:
            return
        # The zoomed view at Level-0 scale (1/2 of a Level-1 view, 1/4 of a Level-2 one), looked for
        # near where the ground motion puts it.
        f = 2 if level == 1 else 1
        small = cv2.resize(grey, (grey.shape[1] * f // 4, grey.shape[0] * f // 4), interpolation=cv2.INTER_AREA)
        if small.std() < 8:
            return   # water, haze: nothing to match
        centre = np.array([(region[0] + region[2]) / 2, (region[1] + region[3]) / 2])
        v = self.velocity_at(centre)
        ex, ey = (region[0] - v[0] * dt) / 4, (region[1] - v[1] * dt) / 4
        if ex < 0 or ey < 0 or ex + small.shape[1] > prev.shape[1] or ey + small.shape[0] > prev.shape[0]:
            return   # part of this ground was not in the full view yet: it would match somewhere wrong
        margin = int(min(40 + (0 if self.velocity_samples else 20 * dt), 150))
        x1, y1 = max(int(ex) - margin, 0), max(int(ey) - margin, 0)
        x2 = min(int(ex) + small.shape[1] + margin, prev.shape[1])
        y2 = min(int(ey) + small.shape[0] + margin, prev.shape[0])
        if x2 - x1 <= small.shape[1] or y2 - y1 <= small.shape[0]:
            return
        scores = cv2.matchTemplate(prev[y1:y2, x1:x2], small, cv2.TM_CCOEFF_NORMED)
        _, best, _, (mx, my) = cv2.minMaxLoc(scores)
        if best >= GMC_MIN_SCORE:
            self._add_motion(centre, np.array([region[0] - 4 * (x1 + mx), region[1] - 4 * (y1 + my)]) / dt, 1.0)

    def move_map(self, frame: int):
        """Slide the look map with the ground; cells coming in at the edge were never looked at."""
        if self.map_frame is not None:
            self.map_shift += self.velocity * (frame - self.map_frame)
        self.map_frame = frame
        cw, ch = IMAGE_WIDTH / GRID_COLS, IMAGE_HEIGHT / GRID_ROWS
        for axis, size in ((0, ch), (1, cw)):
            k = int(self.map_shift[1 - axis] / size)   # whole cells; x shift moves columns (axis 1)
            if k == 0:
                continue
            self.map_shift[1 - axis] -= k * size
            self.fine_seen = np.roll(self.fine_seen, k, axis=axis)
            index = [slice(None), slice(None)]
            index[axis] = slice(0, k) if k > 0 else slice(k, None)
            self.fine_seen[tuple(index)] = -10**6

    def update(self, frame: int, level: int, region, detections, image: Optional[np.ndarray] = None):
        rx1, ry1, rx2, ry2 = region
        if V2_GMC and image is not None:
            self.measure_motion(frame, level, region, image)
        if V2_GROUND:
            self.move_map(frame)
        if level == 0:
            self.last_l0 = frame
        else:
            self.fine_seen[_cells_in(region)] = frame
        predicted = [self.predicted_box(t, frame) for t in self.tracks]
        cut = [V2_TRUNC and _cut_by_view(box, region, level) for _, _, box in detections]

        # Greedy matching by centre distance; same class, or (voting) any class that overlaps well.
        pairs = []
        for di, (cls, conf, box) in enumerate(detections):
            c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            for ti, t in enumerate(self.tracks):
                d = np.linalg.norm(c - t.centre(predicted[ti]))
                if d >= MATCH_DISTANCE_PX:
                    continue
                if t.cls == cls or (V2_VOTES and _iou(box, predicted[ti]) >= VOTE_LINK_IOU):
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
            if cut[di] and not t.truncated:
                box = predicted[ti]   # a cut box would shrink a whole one: keep the whole one where it should be
            else:
                dt = frame - t.last_frame
                if dt > 0 and not cut[di] and not t.truncated:
                    v = (t.centre(box) - t.centre()) / dt
                    t.velocity = v if t.velocity is None else 0.5 * t.velocity + 0.5 * v
                    drift_samples.append(v)
                t.truncated = cut[di]
            t.box, t.last_frame, t.misses, t.hits = box, frame, 0, t.hits + 1
            t.best_level = max(t.best_level, level)
            t.conf = max(conf * LEVEL_CONF_WEIGHT[level], t.conf * 0.5)
            if V2_VOTES:
                t.vote(cls, conf * LEVEL_CONF_WEIGHT[level])

        if drift_samples and not V2_GMC:   # with V2_GMC the field (prior + images) is better
            self._add_velocity(np.median(np.array(drift_samples), axis=0))

        for di, (cls, conf, box) in enumerate(detections):
            if di not in used_d:
                t = Track(cls, box, conf * LEVEL_CONF_WEIGHT[level], frame, best_level=level, truncated=cut[di])
                if V2_VOTES:
                    t.vote(cls, t.conf)
                self.tracks.append(t)

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
            # An unsure class: the runner-up too, at its share. Costs little if wrong (mAP ranks it low).
            if t.votes and len(t.votes) > 1:
                total = sum(t.votes.values())
                cls2, v2 = sorted(t.votes.items(), key=lambda kv: -kv[1])[1]
                if v2 / total >= RUNNER_UP_SHARE:
                    candidates.append((conf * v2 / total, cls2, box))
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


_CENTRE = np.array([IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2])
_hann_cache: Dict[Tuple[int, int], np.ndarray] = {}


def _hann(shape) -> np.ndarray:
    if shape not in _hann_cache:
        _hann_cache[shape] = cv2.createHanningWindow((shape[1], shape[0]), cv2.CV_32F)
    return _hann_cache[shape]


def _cut_by_view(box, region, level: int) -> bool:
    """Does the box touch an edge of the view that is not also an edge of the frame?"""
    m = TRUNC_MARGIN * (4 >> level)   # view px -> source px (4, 2, 1 at L0, L1, L2)
    rx1, ry1, rx2, ry2 = region
    return ((box[0] <= rx1 + m and rx1 > 0) or (box[1] <= ry1 + m and ry1 > 0)
            or (box[2] >= rx2 - m and rx2 < IMAGE_WIDTH) or (box[3] >= ry2 - m and ry2 < IMAGE_HEIGHT))


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
    # Plan from where the camera will be: the live service applies a command about a frame late,
    # so our last command may still be pending. Planning from the view we were sent made every
    # other move unreachable and the sweep bounced between two top positions (Copenhagen with a
    # one-frame delay: 0.198 mAP50, 165 moves in 248 frames).
    base = state.pending or (current.resolution_level, current.center_x, current.center_y)
    if base[0] == 1:   # continue after the sweep position the camera is at (or nearest to)
        state.sweep_index = min(range(len(SWEEP)), key=lambda i: np.hypot(SWEEP[i][0] - base[1], SWEEP[i][1] - base[2]))

    # Advance through the sweep, skipping positions we cannot reach in one move.
    for step in range(1, len(SWEEP) + 1):
        idx = (state.sweep_index + step) % len(SWEEP)
        cx, cy = SWEEP[idx]
        cx = int(min(max(cx, bounds.minimum_center_x), bounds.maximum_center_x))
        cy = int(min(max(cy, bounds.minimum_center_y), bounds.maximum_center_y))
        if _legal(base, (1, cx, cy)):
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
            if V2_GROUND:   # same scale as v1 on the finer grid: up to 2 for a Level-1 view, 1 for Level 2
                value = float(stale.mean()) * (2 if level == 1 else 1)
            else:
                value = float(stale.sum()) / (2 if level == 1 else 1) * 0.5
            for t, c in predicted:
                if x1 <= c[0] <= x2 and y1 <= c[1] <= y2:
                    value += H_UNCERTAIN_WEIGHT * (1.0 - min(t.conf, 1.0))
                    if t.best_level < level or t.truncated:
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
            state.update(request.frame, request.view.resolution_level, region, detections, image)
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
