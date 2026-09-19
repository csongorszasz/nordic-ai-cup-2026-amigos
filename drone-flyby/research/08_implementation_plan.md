# Drone Flyby — Implementation Plan (Architecture A+B, staged)

> Historical implementation record. The current score-driven loop repairs
> measurement and data integrity before promotion, advances coverage with terrain,
> and compares actual realtime AP against simpler baselines. Static coverage
> and the all-visible oracle below are not sufficient deployment gates.

> Companion to `07_new_research.md`. This document is the plan **and** the record
> of what has been implemented. Training is **not** run here; the offline
> training/data pipeline is prepared and tested, ready to execute on IDUN.

---

## 1. Objective and chosen architecture

The current pipeline scores zero because it holds the camera at Level 0 (objects
2–13 px) and answers every frame from the current crop alone. The chosen fix is
the research's recommended spine:

- **A. Coverage-first acquisition + persistent world map** — the deployable core.
- **B. Belief-map value-of-information (next-best-view) planner** — the evolution
  of A, implemented alongside it and selected by configuration.
- **C. Coarse-to-fine detection** — folded into A: L1 is the recall workhorse,
  L2 is spent only on verification targets.
- **D and E** remain future work (learned policy; detector architecture). The
  detector training pipeline is prepared but not prioritised over the loop.

Everything is behind the existing interfaces (`BaseDetector`, `BaseTracker`,
`BaseCameraPolicy`), so detector swaps remain orthogonal.

---

## 2. What was implemented

### 2.1 System architecture

| Component | File | Role |
|---|---|---|
| `TrackBelief` + `TrackerSummary.track_beliefs` | `src/core/interfaces.py` | Decouples the planner from tracker internals; carries existence, confidence, best zoom, position spread per track |
| `EgoMotionEstimator` / `EgoMotionResult` | `src/core/ego_motion.py` | Phase-correlation and ECC estimators; source-pixel scaling, camera-delta correction, gap normalisation, rejection envelope |
| `BeliefField` / `CandidateView` | `src/core/belief.py` | 32×18 coverage/interest grid; unobserved fraction, track value, value-of-information, deterministic exploration targets, scored candidate generation |
| `BeliefVoIPolicy` | `src/core/camera_policy.py` | L1-first planner: always has an exploration candidate (never idles), zooms L2 only on an in-view, rate-limited, higher-value verification target |
| `CameraConstraintGuard` (hardened) | `src/core/camera_policy.py` | Now produces legal L2→L1 moves even when the current centre lies outside the target level's bounds |
| `WorldMapTracker` (extended) | `src/core/tracker.py` | Populates `track_beliefs`; uses the ego-motion estimator; no longer kills age-old confirmed tracks |
| Config + defaults | `src/config.py` | Defaults now select `world_map` + `belief_voi`; new ego-motion and belief settings; all env-overridable |
| Pipeline | `src/core/pipeline.py` | Supplies the view grayscale at every level (not only L0) so ego-motion works while zoomed |

### 2.2 Behavioural changes and the bugs they fixed

1. **Zero-score default.** `TRACKER_TYPE` and `POLICY_TYPE` now default to the
   recommended architecture. The previous `passthrough`/`hold` pair is retained
   and still selectable for baseline comparisons.
2. **Single-hit tracks were pruned by age.** `_prune` removed any track with
   `hits < min_hits_to_confirm` after 4 frames — including tracks that
   `_is_confirmed` had accepted on a single confident hit. Every L0-only object
   died the moment the camera moved away. The simulator exposed this immediately.
3. **Illegal L2→L1 moves near the frame edge.** Scaling an over-long move toward
   the target and then clamping to the target bounds could push the centre back
   outside the distance limit when the current centre was outside those bounds
   (e.g. L2 centre x=3360 → L1). The guard now falls back to the nearest legal
   in-bounds point. Regression test added.
4. **Ego-motion only at L0.** With a moving camera the loop needed ego-motion at
   every level. The estimator now works on same-scale consecutive views and adds
   back the known camera-centre delta; level changes are rejected rather than
   mis-measured.

### 2.3 Offline training and evaluation tooling (prepared, not run)

| Tool | File | Purpose |
|---|---|---|
| Copy-paste augmented dataset | `src/offline/build_augmented_dataset.py` | Renders exact views, harvests annotated object crops, pastes them onto other views to expand the 16-instance dataset; rare-class friendly |
| Training augmentation controls | `src/offline/train_yolo.py` | CLI knobs for mosaic/mixup/copy-paste/geometry/colour and multi-scale; forwarded to Ultralytics |
| Closed-loop simulator | `src/offline/camera_simulator.py` | Replays camera+tracker+policy offline with a view-limited oracle detector; reports coverage, instances seen, mAP; supports a detectability gate |

---

## 3. Verification performed

- **Unit/integration tests: 137 passed** (`python -m pytest src/tests`).
  New coverage: `test_belief.py`, `test_ego_motion.py`, `test_belief_policy.py`,
  `test_camera_simulator.py`, `test_build_augmented_dataset.py`, plus additions
  to `test_tracker.py`, `test_camera_policy.py`, `test_train_yolo_resume.py`.
- **Memory gate:** `python src/offline/oracle_memory_benchmark.py --tracker world_map`
  → **COCO mAP@0.50 = 1.0000**. With ground truth as detections the memory emits
  every object perfectly; the memory layer is sound.
- **Closed-loop simulator** on the 25-frame Helsinki scene (oracle view detector,
  `--min-view-pixels 16`):
  - `hold`: L1 coverage 0%, mAP 0.371
  - `belief_voi` + `world_map`: L1 coverage 100%, mAP 0.303

The simulator result is deliberately reported honestly: coverage is solved, but
closed-loop mAP is still limited by track persistence under camera motion
(ego-motion accepted only between same-scale consecutive views; association drift
while zoomed). That is the next tuning lever and the reason the simulator exists.
It does **not** block P0: the real validator's detector will re-anchor tracks
whenever it sees an object, and coverage is now guaranteed.

---

## 4. How to run

### 4.1 Serve the recommended architecture

```bash
cd drone-flyby
python src/api.py        # defaults: world_map + belief_voi
python src/local_evaluator.py --oracle        # harness sanity, expect 1.000
python src/local_evaluator.py --realtime      # real measure
```

Environment overrides (all optional):

```bash
DRONE_FLYBY_TRACKER_TYPE=world_map
DRONE_FLYBY_POLICY_TYPE=belief_voi
DRONE_FLYBY_EGO_MOTION_METHOD=phase_correlation   # or ecc
DRONE_FLYBY_BELIEF_CELL_SIZE=120
DRONE_FLYBY_BELIEF_L2_MIN_INTERVAL=4
DRONE_FLYBY_MIN_EXISTENCE=0.20
```

Baselines for comparison:

```bash
DRONE_FLYBY_POLICY_TYPE=hold DRONE_FLYBY_TRACKER_TYPE=passthrough python src/api.py
DRONE_FLYBY_POLICY_TYPE=active_coverage python src/api.py
```

### 4.2 Offline policy/architecture evaluation (no endpoint)

```bash
python src/offline/camera_simulator.py --policy belief_voi --tracker world_map \
    --min-view-pixels 16 --frames 25
# Compare: --policy hold, --policy active_coverage, --ego-motion ecc
```

### 4.3 Training data preparation (no training)

```bash
# 1. Baseline exact views (already exists)
python src/offline/build_exact_view_dataset.py \
    --scene helsinki --output-dir training_artifacts/exact_views

# 2. Copy-paste augmented views
python src/offline/build_augmented_dataset.py \
    --scene helsinki --output-dir training_artifacts/augmented \
    --copy-paste-per-view 2 --copies-per-source 3 --seed 0

# 3. Inspect stats
cat training_artifacts/augmented/drone_flyby_augmented/augmentation_stats.json
```

### 4.4 Training on IDUN (run later, by hand)

```bash
# From drone-flyby/, with the dataset present locally:
bash idun/submit.sh run python src/offline/train_yolo.py \
    --data-yaml training_artifacts/augmented/drone_flyby_augmented/drone_flyby_augmented.yaml \
    --weights yolo11s_drone_flyby.pt --imgsz 1280 --batch 32 --epochs 150 \
    --multi-scale --project runs/augmented --name yolo11s_aug

bash idun/submit.sh fetch       # retrieve runs/augmented/.../weights/best.pt
```

Then calibrate and serve:

```bash
python src/offline/calibrate.py ...        # per-class thresholds, macro-AP oriented
DRONE_FLYBY_YOLO_WEIGHTS_PATH=runs/augmented/yolo11s_aug/weights/best.pt \
    python src/api.py
python src/local_evaluator.py --realtime
```

---

## 5. Staged plan and exit criteria

| Stage | Work | Exit criterion | Status |
|---|---|---|---|
| **P0** | Enable world_map + belief_voi; harden guard and prune; view grayscale at every level | Non-zero local mAP; no illegal camera moves; oracle gate 1.0 | **Done** |
| **P1** | Tune ego-motion acceptance and association; add `min_view_pixels`-aware simulator sweeps | Closed-loop simulator mAP ≥ `hold` and rising | Next |
| **P2** | Build augmented dataset; train YOLO11s/m at 1280 with copy-paste + multi-scale on IDUN | Per-class AP improvement on recorded validation, rare classes included | Prepared |
| **P3** | Calibrate per-class thresholds; TTA/ensemble inside the latency budget | `--realtime` score stable and higher | Prepared |
| **P4** | Belief-map tuning (weights, cell size, L2 interval) via simulator | Coverage stays 100% with fewer wasted L2 frames | In tooling |
| **P5** | Offline simulator trajectories → imitation/RL camera policy | Learned policy beats heuristic offline | Future |

### Ablation matrix (via config/env)

| Variable | Baseline | Variants |
|---|---|---|
| Tracker | passthrough | world_map |
| Policy | hold | active_coverage, belief_voi |
| Ego-motion | phase_correlation | ecc |
| Belief cell size | 120 | 96, 160 |
| L2 interval | 4 | 2, 6, 10 |
| Miss decay | flat | in-view vs out-of-view (implemented) |

---

## 6. Known limitations and risks

- **Closed-loop track persistence** is the open problem: the simulator shows
  coverage is solved but mAP under a moving camera trails the L0 memory ceiling.
  Ego-motion is accepted only between same-scale consecutive frames; a homography
  or ECC refinement plus association tuning is the next step. ECC is available
  but measured worse on this data, so phase correlation remains the default.
- **Detector model** is unchanged (YOLO11s on exact views). The augmented dataset
  and training knobs are prepared but training has not been run.
- **Simulator oracle** is a model, not a detector; use `--min-view-pixels` when
  comparing policies, and never read its absolute mAP as a competition estimate.
- **One-shot evaluation**: keep the `_predict_only` fallback ladder and warm-up
  invariants; never deploy an unverified configuration.

---

## 7. Files touched

Added:

- `src/core/ego_motion.py`
- `src/core/belief.py`
- `src/offline/camera_simulator.py`
- `src/offline/build_augmented_dataset.py`
- `src/tests/test_belief.py`, `test_ego_motion.py`, `test_belief_policy.py`,
  `test_camera_simulator.py`, `test_build_augmented_dataset.py`
- `research/08_implementation_plan.md`

Modified:

- `src/core/interfaces.py` (TrackBelief, TrackerSummary)
- `src/core/tracker.py` (beliefs, ego-motion, prune fix, factory)
- `src/core/camera_policy.py` (BeliefVoIPolicy, guard fix, factory)
- `src/core/pipeline.py` (view grayscale at every level)
- `src/config.py` (settings, defaults, env parsing)
- `src/offline/train_yolo.py` (augmentation kwargs and CLI)
- `src/tests/test_tracker.py`, `test_camera_policy.py`, `test_train_yolo_resume.py`
