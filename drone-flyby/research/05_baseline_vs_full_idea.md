# Drone-Flyby — Baseline Implementation vs Original Full Idea

This note compares the current first-pass implementation in `drone-flyby/` with the broader system described in the research notes.

## 1. What is implemented now

### `config.py`

- Central config dataclass for detector, tracker, camera policy, and geometry constants.
- Current defaults favor the working baseline over the aspirational stack.

### `dtos.py`

- Strict request/response models.
- Camera constraints and source-frame normalization are enforced here.
- This is the protocol layer both the baseline and the future system must obey.

### `utils.py`

- Image decode/encode helpers.
- Coordinate transforms between view-local, source-pixel, and normalized full-frame boxes.
- Response validation and camera rejection checks.

### `core/interfaces.py`

- Shared contracts for detector, tracker, and camera policy.
- Keeps the modules swappable.

### `core/pipeline.py`

- Orchestrates detection, tracking, and camera steering for each request.
- Handles session reset and exception containment.

### `core/tracker.py`

- `WorldMapTracker` is implemented.
- Uses global 4K coordinates, IoU matching, ego-motion compensation, out-of-view persistence, and class-aware NMS.
- This is already close to the original memory-system goal.

### `core/camera_policy.py`

- `SurveyAndZoomPolicy` is implemented.
- It queues unscanned clusters, steps through `L0 -> L1 -> L2`, and returns legally through `L2 -> L1 -> L0`.
- It is a practical first version of the active exploration policy.

### `core/detector.py`

- Current baseline detector is still the Canny contour detector.
- A new `TemplateBankDetector` is added as an initial data-driven detector.
- A `YoloDetector` scaffold is present for later fine-tuned weights.

### `tests/`

- Covers scaffold, tracker, camera policy, and detector behavior.
- Real Helsinki frames are used for tracker, camera-policy, and detector tests.

## 2. What the original full idea called for

The original architecture aimed for:

- a strong aerial object detector, ideally YOLOv11s or similar
- SAHI or multi-resolution inference at low zoom
- a trained detector backed by pseudo-labeling and offline fine-tuning
- a richer survey/zoom strategy with more explicit cluster targeting
- a tracker that behaves like a world map and keeps out-of-view objects alive
- TensorRT or another optimized deployment path

## 3. Main differences

### Detection

**Current baseline:**

- template bank plus a simple Canny fallback
- no trained deep detector yet
- no SAHI runtime yet

**Full idea:**

- fine-tuned aerial detector
- multi-scale inference
- optimized deployment

### Camera policy

**Current baseline:**

- enough to drive exploration legally and repeatedly
- simple queueing and zoom progression

**Full idea:**

- more adaptive cluster ranking
- deeper notion of scene coverage and uncertainty
- stronger use of the free reset and quadrant scheduling

### Tracking

**Current baseline:**

- already relatively close to the target design
- persistent memory, motion compensation, and NMS are present

**Full idea:**

- further tuning of association, pruning, confidence decay, and cross-resolution refinement
- possible refinement with detector quality improvements

## 4. Practical interpretation

The current repository is no longer just scaffolding.
It now has a usable first version of the three key online modules:

1. detector
2. tracker
3. camera policy

The remaining gap to the full design is mostly on the detector side:

- better class-specific detection
- better generalization to unseen scenes
- training or fine-tuning on the challenge domain

## 5. Recommended next phase

1. Keep the tracker and camera policy stable.
2. Use the new template-bank detector as the bridge to a trained detector.
3. Replace or augment it with a fine-tuned YOLO-based backend.
4. Add offline data harvesting and pseudo-label generation.

In short: the first pass is now a working system, while the original idea still points to a stronger detector stack and later optimization work.
