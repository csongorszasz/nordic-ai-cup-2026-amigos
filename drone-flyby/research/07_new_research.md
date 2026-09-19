# Drone Flyby — Architecture Research: Why the Current Pipeline Scores Zero, and What to Build Instead

> Historical hypothesis, not the current conclusion. Subsequent reproducible
> IDUN experiments show that L0 can score well, that proposal thresholds and
> actual inference resolution matter substantially, and that the original
> active policy trailed hold/passthrough. The 3333 ms timeout is not a loss-free
> latency budget. Use the current README and persisted experiment results;
> the claims and measurements below describe the earlier system.

> Scope: system-first architecture research. Detector model-level choices are covered as a
> secondary section because the current zero score is not caused by the detector.
> This document is independent of the earlier `01`–`06` research notes.

---

## 0. Executive summary

The current pipeline does not score *low*. It scores **zero**, and the cause is
architectural, not a tuning problem:

1. The deployed configuration holds the camera at **resolution level 0** for the
   entire attempt (`POLICY_TYPE="hold"`).
2. At level 0 the source frame is downsampled 4× into 960×540. The objects the
   challenge asks for are **2–13 px** at that scale. The information needed to
   recognise them was never transmitted. Cropping or upscaling the received image
   cannot recover it.
3. The deployed tracker is `passthrough`: it returns only what the detector found
   *in the current crop*. Because the evaluator scores **the whole source frame on
   every frame**, an object that is outside the camera view, or simply missed,
   yields an empty contribution — there is no carried belief.
4. Result: nearly every frame is answered with zero annotations. COCO mAP over
   empty predictions is 0.

The detector itself is not the problem. The fine-tuned YOLO11s reaches
**mAP@0.50 ≈ 0.995** on its own exact-view validation split (rendered level
0/1/2 views of the supplied Helsinki scene), and the server runs at **~25 ms per
frame against a 3333 ms budget**. The pixels, once zoomed, are detectably
separable, and there is ~130× latency headroom. What is missing is a **closed
loop**: a camera policy that spends frames acquiring the information, and a
memory that accumulates it and answers for the whole frame.

The rest of this document analyses that closed loop as an architecture problem,
surveys the credible designs, and recommends a staged build:

- **Spine (deploy now):** deterministic coverage at level 1 with a persistent
  world map and level-2 verification — the WAMI / persistent-surveillance pattern.
- **Evolution:** replace the fixed schedule with a **belief-map, information-gain
  ("next-best-view") planner**.
- **Endgame:** train the camera policy offline against a **self-built simulator**
  that renders any legal view from the 4K source, so the policy can be learned on
  IDUN without ever touching the evaluator.

---

## 1. Root-cause analysis

### 1.1 The observed failure

Two validation recordings are retained under
`recorded_validation_data/`. The run log
`logs/validation_20260918_151420.log` shows the behaviour plainly: across 246
answered requests, only a handful of frames contain any detection, and those
never exceed 2 boxes.

| Measurement | Value | Source |
|---|---|---|
| Frames answered | 246 / 249 | `index.jsonl` |
| Frames with ≥1 detection | 43 / 246 | `index.jsonl` |
| Maximum detections in any frame | 2 | `index.jsonl` |
| Median request latency | ~24 ms | `index.jsonl` |
| Resolution level of every request | **0** | `metadata/*.json` |
| Exact-view YOLO11s validation mAP@0.5 | **0.995** | `runs/exact_runs/yolo11s_exact150/results.csv` |

### 1.2 The causal chain

```
camera held at L0                          (config: POLICY_TYPE="hold")
        │
        ▼
960×540 image covers the whole 3840×2160 frame
        │
        ▼
objects are 2–13 px (median object ≈ 13×12 px at L0)
        │
        ▼
the 4× downsample discarded the detail a detector needs
        │
        ├──────────────► detector output ≈ empty
        │
        ▼
passthrough tracker returns only current-frame detections
        │
        ▼
empty annotations for the whole frame
        │
        ▼
the evaluator scores every frame against every object
        │
        ▼
mAP@0.50 = 0
```

Each arrow is verifiable in the code and data:

- **Camera never moves.** `config.py:44` defaults `POLICY_TYPE = "hold"`; the
  factory returns `HoldCameraPolicy`, whose `decide_next_view` is `return None`
  (`camera_policy.py:108`). Every recorded metadata file reports
  `resolution_level: 0`.
- **Objects are tiny at L0.** Parsing the 25 supplied Helsinki annotation files:
  259 boxes total; the smallest (`spacecraft`) is 49×8 source px → **12×2 px at
  L0**; the smallest `ta-ta` is 9×3.5 px. Median is 53×50 source px → 13×12 px.
- **The pixels are gone, not merely small.** The protocol explicitly says cropping
  and upscaling a level-0 image "does not recover what a level-1 or level-2 view
  would have shown. That detail was never sent" (`README.md`). This is an
  information bound, not a detector weakness.
- **Passthrough has no memory.** `TRACKER_TYPE = "passthrough"`
  (`config.py:35`) selects `PassthroughTracker`, which returns detections from the
  current view only (`tracker.py:462`). The evaluator, meanwhile, scores the
  **complete source frame** every frame (`README.md`, Scoring). Detections made
  while zoomed on one object cannot be reported for objects elsewhere unless a
  memory carries them.
- **Whole-frame scoring converts misses into recall loss.** Skipped or
  unanswered frames "contribute no detections" while their ground truth still
  counts. With a hold policy there is no mechanism to look elsewhere, so a large
  fraction of the 16 instances is never acquired.

### 1.3 Why the detector is not the culprit

The exact-view training run renders the protocol-exact 960×540 views for a grid
of legal camera positions at each level and fine-tunes YOLO11s at 960 px. Its
validation `mAP50` is **0.995** and `mAP50-95` is **0.231** (`results.csv`). Two
caveats matter, but neither changes the conclusion:

- That validation split shares the *same 16 object instances* as its training
  split (one scene, 25 frames), so 0.995 is a harness number, not a
  generalization estimate — see §1.4.
- The `mAP50-95` gap (0.231) shows localisation is only loosely correct at higher
  IoU, but the competition scores at **IoU 0.50**, precisely where the model is
  strong.

The point stands: when the detector is fed zoomed pixels it finds the objects.
At level 0 it is fed a view in which the objects are a few pixels; no detector
should be expected to succeed there.

### 1.4 The real bottleneck: it is the closed loop

The challenge is not image detection. It is **active perception under a moving
sensor**: a partially observable system where the agent chooses where to look,
must build a belief about a static scene from sequential partial views, and is
graded on the full scene on every step. Formally this is a POMDP:

| POMDP component | Drone-flyby realisation |
|---|---|
| State | True object set in the 4K source frame (static across frames) |
| Observation | 960×540 PNG of one crop, plus camera geometry |
| Action | Next `(resolution_level, center_x, center_y)` |
| Transition | Camera moves (bounded); scene is static relative to the ground, moves in sensor coordinates by ego-motion |
| Belief | What the system currently believes about all 16 objects |
| Reward | COCO mAP@0.50 over all frames, macro-averaged over classes |

Both failures of the current pipeline map directly onto missing POMDP pieces: no
**action policy** (camera held) and no **belief state** (passthrough). An
architecture for this problem must therefore specify three things explicitly —
**acquisition** (where to look), **fusion** (what we now believe), and
**emission** (what to answer) — and the interfaces between them.

---

## 2. Constraints that shape the architecture

Any proposed design must live inside these numbers. They are not incidental; they
eliminate otherwise-attractive options (e.g. heavy feature-memory video models).

| Constraint | Value | Architectural consequence |
|---|---|---|
| Frame cadence | 333 ms (3 fps) | At most one camera move per frame |
| Per-request budget | 3333 ms | ~130× the current 25 ms: expensive inference is affordable |
| Camera move limit | 551 / 1102 / 2203 px at L2/L1/L0 | Coverage must be planned as a path, not per-frame jumps |
| Level transitions | one step at a time; L0↔L2 needs two frames | L0 is expensive to revisit; L1 is the natural search altitude |
| L0 reset | free, must be exactly (1920, 1080) | Survey returns are cheap but cost one frame |
| Objects | static in the world, 16 classes, one of each in Helsinki | Registration + track fusion is viable; no motion model needed |
| Ego-motion | drone flies forward ~13.9 m/frame (≈58 source px down) | Track coordinates must be advanced between observations |
| Scoring | whole frame, every frame; no NMS by the scorer | Emission must be de-duplicated and global |
| Evaluation | one completed attempt | The system must be robust, not tuned per-run |
| Training | IDUN (A100/H100), no network from compute nodes | Offline loop; endpoint served elsewhere |
| Honest test set | recorded validation frames must not become training ground truth | Pseudo-labels may be used but the recording is the benchmark |

---

## 3. Architecture decision space

Five designs are credible. Each is described with the mechanism that makes it
work, the evidence for it, and how it fits the constraints above.

### A. Coverage-first acquisition + world-map fusion

**Mechanism.** Stop treating the received image as "the frame". Treat it as one
observation of a persistent world state. A deterministic overlapping sweep at
level 1 guarantees that every part of the frame is seen at a resolution where
objects are detectable; the tracker fuses every detection into cumulative
source-frame coordinates; every response reports the entire accumulated map;
level 2 is spent only to verify/classify a track the map already suspects.

**Evidence / prior art.**

- Wide-Area Motion Imagery (WAMI) persistent surveillance is the classical
  version of exactly this problem: small objects, moving sensor, registration
  required, track fusion downstream. The standard pipeline is global-motion
  compensation (homography) followed by detection and multi-target fusion
  (GM-PHD). See Teutsch & Grinberg, *Robust Detection of Moving Vehicles in
  Wide Area Motion Imagery* (CVPRW 2016); Prokaj & Medioni, *Persistent Tracking
  for Wide Area Aerial Surveillance* (CVPR 2014).
- Context R-CNN (Beery et al., arXiv:1912.03538) is the strongest evidence for
  the *memory* half: on static-camera detection it lifts mAP@0.5 by **+17.9 mAP**
  over a single-frame baseline by attending over a long-term per-camera memory
  bank. Our objects are static; the camera moves — equivalent by registration.
- Slicing-aided inference (SAHI, Akyön et al., ICIP 2022, arXiv:2202.06934)
  confirms that operating on smaller windows is the reliable route to small-object
  AP (+5–15 AP on VisDrone/xView). Note the cost: one study measures a drop from
  ~27 fps to ~1 fps when tiling (`IEEE Access 2024`, RT-DETR-X + SAHI). SAHI is
  *not* a fit at inference here — but the principle is already realised by the
  camera levels, and the cost data justifies spending the latency budget on
  camera moves instead of tiles of an already-degraded image.

**Fit.** Excellent. Every component exists in the repo (`WorldMapTracker`,
`ActiveCoveragePolicy`) and the design is bounded and deterministic — ideal for a
one-shot evaluation. **This is the recommended spine.**

**Weakness.** Fixed sweeps are oblivious to where objects actually are: it may
spend frames on empty ground while a detected-but-ambiguous object waits. That
motivates B.

### B. Belief-map next-best-view (information-gain) planner

**Mechanism.** Maintain an explicit 2D belief field over the 3840×2160 frame:
per-cell probability that an undetected object is present, per-cell uncertainty,
and per-track class/confidence. Each frame, choose the camera pose that maximises
expected information gain minus travel cost, subject to the movement rules. Where
A asks "what is the next cell in the sweep", B asks "which observation would
reduce our uncertainty about the score the most".

**Evidence / prior art.**

- Active perception as a POMDP with learned or greedy view selection is
  well-established. *Active Classification of Moving Targets with Learned Control
  Policies* (arXiv:2212.03068) formalises belief updates and a control policy over
  informative viewpoints; its observation representation is literally target
  position + belief entropy — a compact template for our belief map.
- *Learning to View: Decision Transformers for Active Object Detection* (ICRA
  2023) shows a policy trained to obtain views that maximise detection quality
  beats passive detection.
- Next-best-view planners with coverage/uncertainty rewards (GenNBV,
  arXiv:2402.16174; Hestia, arXiv:2508.01014) demonstrate the reward design:
  improvement in coverage/uncertainty per step, penalised by travel.
- For our scale the cheap, transparent version is a **value-of-information
  heuristic**: `VoI(cell) = P(object) × zoom_deficit × visibility ×
  exp(−travel_cost)`, which needs no training and is auditable.

**Fit.** Strong and generalises A. The catch is calibration: VoI depends on a
trustworthy `P(object)` and uncertainty. It should be built *on top of* A's map,
once A is measured, not instead of it.

**Weakness.** If the belief field is miscalibrated the planner can thrash between
targets. Requires confidence calibration per class (already scaffolded by
`offline/calibrate.py`).

### C. Coarse-to-fine detector cascade fused in map coordinates

**Mechanism.** One model (or two heads) produces *recall-oriented* candidates at
L0/L1 — accepting false positives — and a *precision-oriented* pass at L2
confirms class and tightens the box. Candidates are associated in world
coordinates, not image coordinates, so a low-resolution candidate at L1 and a
high-resolution confirmation at L2 are the same track.

**Evidence / prior art.** This is the classic coarse-to-fine detection cascade,
and it is also how Context R-CNN separates "propose" (single-frame model) from
"classify with context". The small-object literature repeatedly shows the head
resolution is what binds tiny-object AP: P2/extra-small heads and multi-scale
fusion (HRFNet, *Chinese J. Aeronautics* 2025; SO-DETR, arXiv:2504.11470;
Dome-DETR, ACM MM 2025) all push gains through higher-resolution features.
At **IoU 0.50** the competition threshold, a L2-confirmed tight box is easy; the
hard part is recall, which the L1 cascade supplies.

**Fit.** Good. In practice C is not a separate system: it is the detector split
inside A/B (L1 = generalist, L2 = specialist). Keeping both detector tiers in one
deployable artifact (one YOLO with two confidence profiles, or two checkpoints)
is an implementation detail behind `BaseDetector`.

**Weakness.** Two passes per frame costs latency — affordable here, but the
memory fusion must keep the two tiers' coordinate conventions straight.

### D. Learned camera policy via a self-built offline simulator

**Mechanism.** We can render the evaluator-exact view for **any** legal camera
pose from the supplied 4K frames and recorded sequences. That makes the camera
control problem fully simulatable offline: an agent can be dropped into a known
scene, take the same bounded moves, receive the same 960×540 crops, and be scored
with the same mAP, all on IDUN and entirely disconnected from the evaluator. The
policy can then be trained by imitation of a greedy oracle or by RL, and frozen
as a small network or a lookup table at serving time.

**Evidence / prior art.** Decision-Transformer active detection (ICRA 2023) and
RL next-best-view (GenNBV; arXiv:2508.01014) are the templates. Crucially, we do
not need the RL machinery of D to benefit: the **simulator alone is valuable**,
because it enables offline search for good exploration schedules (e.g. "is L1
serpentine with 50% overlap optimal for 249 frames?") without burning validation
attempts.

**Fit.** This is the highest-ceiling design and uniquely enabled by the fact that
we possess the ground-truth scene. It is also the highest-effort and riskiest for
a one-shot evaluation; it belongs after A/B are measured.

**Weakness.** Simulator/real gap (recorded sequences differ from Helsinki),
reward misspecification, and the policy may overfit the one scene we can render.
Mitigate by training on recorded sequences as well.

### E. Model-level detector options (secondary)

Given a good crop, which detector architecture is best? Ranked by fit:

1. **Higher-capacity YOLO at high imgsz with a P2 (stride-4) head.** The exact
   pattern behind most VisDrone gains. Train YOLO11m/l at 960–1536 px with
   multi-scale augmentation and a P2 head so objects of ~10–30 px survive the
   feature pyramid. Fits the existing Ultralytics toolchain and TensorRT export.
2. **RT-DETR / D-FINE / Dome-DETR.** Strong small-object results (e.g. D-FINE,
   Dome-DETR on VisDrone/AI-TOD). They would require venturing outside the
   Ultralytics training loop, and their advantage over a well-tuned high-res YOLO
   at IoU 0.50 is smaller than the dataset-quality advantage left on the table.
3. **TTA (flips/scales) and small checkpoint ensembles.** Cheap since latency is
   abundant (~130× headroom); can be added behind the detector interface.
4. **Super-resolution front-end.** Tempting for L0, but honest assessment: SR
   *hallucinates* detail; it cannot restore the information the 4× downsample
   discarded, and false texture can create confident false positives. Do not use
   SR to substitute for zooming. It may be worth a controlled experiment on L1
   crops only.
5. **SAHI-style tiling at inference.** Not useful on the received image: tiling a
   960×540 L0 crop does not create pixels. Its correct analogue is **training on
   exact views** (already done) and **spending frames on camera moves** (A/B).

Data quality dominates model choice here: the exact-view dataset is currently
873 train / 147 val images generated from just **25 source frames and 16
instances**. That is the largest single gap, and §5 addresses it.

---

## 4. Recommended architecture

**A as the spine, B as its evolution, C folded into A, D as the endgame.** This
ordering is deliberate: A and B use components that already exist and can be
verified against the local evaluator within days, while D requires a simulator and
a training campaign that should not gate the first non-zero score.

### 4.1 Component diagram

```
                       ┌──────────────────────────────────────────┐
                       │            PipelineOrchestrator           │
                       │  - session reset / stale-frame guard      │
                       │  - latency budget guard                   │
                       └───────────────┬──────────────────────────┘
                                       │ per request
        ┌──────────────────────────────┼───────────────────────────────┐
        ▼                              ▼                               ▼
┌────────────────┐          ┌────────────────────┐          ┌────────────────────┐
│   DETECTOR     │          │   WORLD MAP        │          │  CAMERA PLANNER     │
│  (BaseDetector)│          │  (BaseTracker)     │          │ (BaseCameraPolicy)  │
│                │          │                    │          │                     │
│ L1 generalist  │  dets    │ - ego-motion       │  belief  │ A: coverage schedule│
│ L2 specialist  │─────────►│   registration     │─────────►│    (L1 serpentine,  │
│ per-zoom conf  │          │ - world-frame      │          │     L2 verify)      │
│ calibration    │          │   track fusion     │          │                     │
│                │          │ - belief/uncertain │          │ B: VoI / NBV over   │
│                │          │ - emission (NMS,   │          │    belief map       │
│                │          │   global coords)   │          │                     │
└────────────────┘          └────────────────────┘          └─────────┬──────────┘
        ▲                              │                             │
        │                              ▼                             ▼
     view crop                 full-frame annotations        next (level, cx, cy)
                                       │                    validated by
                                       │                CameraConstraintGuard
                                       ▼
                              DroneFlybyPredictResponseDto
```

The existing interfaces (`core/interfaces.py`) already match this decomposition;
the architecture work is in the *implementations* selected and the contract
between map and planner.

### 4.2 Per-frame dataflow

1. **Decode** the 960×540 crop; note `resolution_level` and
   `source_region_xyxy`.
2. **Register.** Estimate ego-motion from consecutive L0/L1 frames (see §4.4) and
   advance every track by the frame-gap-normalised shift. A gap of *k* frames
   advances by *k×* the per-frame shift.
3. **Detect** at the current zoom with the appropriate confidence profile and
   calibration; convert every detection to source pixels via
   `view_bbox_to_source`.
4. **Associate and fuse** detections to tracks in world coordinates (class-aware,
   motion-gated IoU + centre distance, Hungarian assignment).
5. **Decay or kill** tracks that were inside the current crop yet not detected
   (strong negative evidence) versus tracks outside the crop (weak decay). This
   asymmetry is the heart of a correct memory: "not seen because not looked at"
   must not erode a track.
6. **Emit** every confirmed, sufficiently-likely track as a global annotation,
   class-aware NMS, capped at 500. *Emission never depends on the current frame
   alone.*
7. **Plan** the next view. From L0: enter L1 toward the highest-VoI target or
   next coverage cell. From L1: move along the sweep, or drop to L2 if a target
   is inside the current view and due for verification. From L2: always step back
   to L1 (one step), optionally using the free L0 reset to re-survey.
8. **Guard** the camera command through the constraint validator, and never let a
   planning error drop the detections already computed.

### 4.3 Belief state (the central data structure)

The world map is the architecture. Proposed schema per track (extending
`TrackedObject`):

| Field | Meaning |
|---|---|
| `bbox_4k` | Current best box in source pixels |
| `class_scores` | Accumulated per-class evidence (not one label) |
| `confidence` | Best observed detector confidence |
| `existence` | P(track is a real object), updated by hits and in-view misses |
| `position_std` | Positional uncertainty, grows when unseen, shrinks on observation |
| `best_zoom` | Deepest level at which it was confirmed (drives VoI) |
| `hits`, `frames_since_seen`, `last_seen_frame` | Track lifecycle |
| `velocity` | Optional, for ego-motion residual |

The map owns **all** knowledge; the detector is a stateless sensor. This is what
makes whole-frame scoring tractable: the answer for an object last seen 40 frames
ago is still in the map, registered into the current frame by ego-motion.

### 4.4 Ego-motion: from phase correlation to registration

The current `WorldMapTracker` predicts ego-motion with `cv2.phaseCorrelate` on L0
grayscale frames (`tracker.py:164`). This is a reasonable first approximation but
is fragile under zoom changes (L0 frames are unavailable while zoomed) and under
large inter-frame gaps. WAMI practice is explicit **registration**: estimate a
homography between frames (feature matching + RANSAC) or an ECC alignment. For a
planar-ish nadir scene at fixed altitude the transform is close to a pure
translation, so an ECC/phase correlation on a *synthesised L0-equivalent* is
acceptable; but the architecture should leave room for homography so that
parallax from tall structures (towers, hangars) does not bias track positions.

Crucially, ego-motion must be valid regardless of which level the camera is on.
The map should maintain a **global reference L0 mosaic / last-known L0 frame**,
not depend on receiving L0 every frame.

### 4.5 Camera policy: from schedule to value of information

Stage 1 (A): deterministic overlapping L1 grid (serpentine, 50% overlap),
interleaved with L2 verification when a confirmed track has `best_zoom < 2` and
lies inside the current L1 view. Level transitions obey the one-step rule; L2→L0
is always two moves.

Stage 2 (B): replace the "next cell" rule with a VoI ranking over the belief map:

```
VoI(pose) = Σ_tracks  P(exists_t) · zoom_deficit_t · visibility_t(pose)
                    · exp(−travel_cost(pose) / λ)
          + λ_explore · expected_new_object_gain(unseen_cells, pose)
```

The exploration term is what prevents the policy from fixating on known tracks
while missing entire regions. It should decay as coverage grows.

### 4.6 Emission policy and the fallback ladder

A hard requirement from the scoring rules: **never answer a frame with an empty
list if the map holds confirmed tracks.** The fallback ladder, in order:

1. Full pipeline succeeds → emit map.
2. Detector fails → emit map from `predict_only()` (already implemented).
3. Tracker fails → emit last known good annotation list.
4. Any uncaught error → emit a valid, possibly empty response — never HTTP 500,
   because a 500 forfeits the frame's detections and still counts the ground
   truth.

### 4.7 Latency budget allocation

The 3333 ms budget is enormous relative to the ~25 ms detector. Allocate it
deliberately instead of ignoring it:

| Stage | Budget | Use |
|---|---|---|
| Decode + register | 50 ms | ECC/homography |
| Detect L1 | 200 ms | primary recall pass |
| Detect L2 (when applicable) | 200 ms | verification |
| TTA / ensemble | up to 500 ms | optional precision |
| Map update + plan + validate | 100 ms | |
| **Cutoff for planning** | 90% of budget | hold camera, still return detections |

The existing pipeline already checks `elapsed_ms < budget_ms * 0.90` before
planning (`pipeline.py:132`). Keep that invariant; it converts a slow detector
into a held camera rather than a lost frame.

---

## 5. Offline training and data architecture

The closed-loop architecture and the detector are only as good as the data. The
offline loop is:

```
data/helsinki 4K frames + annotations
        │
        ├─► render evaluator-exact L0/L1/L2 views  (build_exact_view_dataset.py)
        │        │
        │        ├─► geometric + photometric augmentation
        │        ├─► copy-paste of object crops across backgrounds
        │        └─► multi-scale (960–1536 px) training set
        │
recorded_validation_data/  (honest test set — NOT ground truth)
        │
        ├─► pseudo-label with current best model  (pseudo_label.py)
        ├─► include ONLY as unlabeled/pseudo data, never as human GT
        └─► use the recording itself only for scoring
        │
        ▼
   IDUN: train YOLO11{m,l} @ high imgsz, P2 head, exact views
        │
        ▼
   calibrate per-class thresholds  (calibrate.py)
        │
        ▼
   local_evaluator.py  (--oracle sanity, --realtime for score, --verbose)
        │
        ▼
   deploy weights + TensorRT export (export_tensorrt.py)
```

### 5.1 Data-scarcity techniques worth using

The exact-view set derives from only 25 frames / 16 instances. The literature on
learning detectors from few labels is directly applicable:

- **Copy-paste augmentation.** Extract annotated object crops and paste them onto
  other scene backgrounds with physically plausible scales. This is the standard
  answer to class imbalance / rarity in detection and is specifically used in
  weak/semi-supervised pipelines (Point-Teaching, arXiv:2206.00274).
- **Hallucination-free photometric + geometric augmentation.** Flips, rotations,
  scale jitter, brightness/contrast, noise, and blur matching the 4× downsample.
  The evaluator uses `INTER_AREA` downsampling; augmentation must reproduce the
  same resampling statistics or the model learns the wrong texture.
- **Self-training / pseudo-labeling.** Soft Teacher (arXiv:2106.09018) and
  Unbiased Teacher (arXiv:2102.09480) both show large gains from teacher–student
  pseudo-labeling when labels are scarce. For us the "unlabeled" pool is the 249
  recorded validation frames. Respect the honesty rule: use them only as
  unlabeled data for teacher–student training, and keep the recording as the
  score benchmark. Pseudo-label quality should be validated per class.
- **Synthetic multi-view expansion.** Because we can render any crop from the 4K
  frames, we can massively expand the exact-view set (more camera poses, more
  level/centre combinations, including partial-object-at-crop-edge examples).
  Partial-object handling matters: the exact-view builder already discards boxes
  with <30% visible area (`build_exact_view_dataset.py:52`) — that threshold
  should be tuned against recall of objects entering/leaving the view.

### 5.2 Detector training recipe (secondary, but required)

- Base: YOLO11m or YOLO11l (the current YOLO11s is capacity-limited for 16
  classes at high resolution and a P2 head).
- Input: 960–1536 px multi-scale; train on the exact-view renders, not generic
  4K resizes.
- Head: add a P2 (stride-4) detection head so ~10 px objects survive.
- Augmentation: the copy-paste and photometric suite above.
- Calibration: per-class confidence thresholds chosen to maximise macro AP, not
  global accuracy — because scoring is macro-averaged, rare classes (`jammer`,
  `mine_roller`, `hangar`, `medium_plane`) dominate the average and must not be
  suppressed.
- Export: TensorRT FP16 for the final endpoint; verify with `--oracle` and
  `--realtime`.

---

## 6. Migration and experiment plan

The work is staged so that each stage produces a measurable non-zero score before
the next is attempted. The score benchmark is `local_evaluator.py` on the
recorded sequences, plus the platform's validation run (which the platform
allows repeatedly).

| Stage | Change | Exit criterion |
|---|---|---|
| **P0** | Select `world_map` + `active_coverage`; verify no regressions; calibrate thresholds | Detections appear; map carries tracks across frames; non-zero local mAP |
| **P1** | Ego-motion hardening (ECC/homography), in-view vs out-of-view miss decay, per-class NMS | Track survival across a full sequence; no ghost tracks; recall rises |
| **P2** | Retrain on expanded exact-view set with copy-paste + multi-scale, YOLO11m/l + P2 | Per-class AP improvement on recorded validation, especially rare classes |
| **P3** | Belief-map VoI planner (replace fixed sweep) | Same or better mAP with fewer wasted L2 moves |
| **P4** | Simulator + imitation/RL camera policy | Policy beats heuristic on held-out simulated sequences |
| **P5** | TTA / ensemble / TensorRT; final robustness pass | Stable score under `--realtime` |

### 6.1 Ablations to run

| Variable | Baseline | Variants |
|---|---|---|
| Tracker | passthrough | world_map |
| Policy | hold | sweep, deterministic_l1, active_coverage, VoI |
| Detector | YOLO11s exact | YOLO11m/l, P2, high imgsz |
| Miss handling | flat decay | in-view vs out-of-view |
| Augmentation | none | +copy-paste, +multi-scale |
| Ego-motion | phase correlation | ECC / homography |

### 6.2 Metrics that matter

- **Macro mAP@0.50** — the competition score.
- **Per-class AP** — macro-averaging means one dead class costs ~1/16 of the
  score; report every class.
- **Frames-answered** and **detections/frame** — detect the empty-response
  failure mode directly.
- **Track recall/lifetime** — whether memory survives gaps.
- **Frames with zero detections** — should trend to zero once the loop works.

---

## 7. Score-preservation checklist (architectural invariants)

Response-format rules from the challenge remain mandatory and are easy to violate
when adding state:

- Boxes are normalised to the **whole source frame**, never the received view.
  Use `view_bbox_to_global` for detections, `source_bbox_to_global` for tracks.
- A box must satisfy `0 ≤ x1 < x2 ≤ 1` and `0 ≤ y1 < y2 ≤ 1` strictly; drop
  degenerate boxes (`clip_bbox_to_frame` returns `None`).
- `requested_view` coordinates must be **Python ints**; round before returning.
- Echo `request_id` and `frame` exactly; no unknown fields.
- `object_id` must be an exact member of `dtos.OBJECT_CLASSES`.
- Never return HTTP 500 for an answered request; catch everything and fall back
  to memory.
- Warm the model before the attempt; the first inference is the slowest.

Architecture-specific risks to watch:

- **Track drift** when ego-motion is misestimated; bounded by registration
  quality and by re-observing at L1/L2.
- **Class confusion** among visually similar classes (`ta-ta` vs `tank`,
  launchers at different scales); mitigated by class-evidence accumulation and
  L2 verification rather than single-frame labels.
- **Overfitting to Helsinki**; mitigated by augmentation, pseudo-labeling on the
  recorded sequence, and testing on both recorded sequences.
- **One-shot evaluation**; mitigated by keeping the fallback ladder and never
  deploying an unverified policy.

---

## 8. Conclusion

The zero score is not a detector failure. It is the predictable result of an
architecture with no action policy and no belief state, forced to answer
whole-frame questions from a single, information-starved, held view. The
detector already reaches ~0.995 mAP@0.50 on zoomed views and runs ~130× under
budget. The correct fix is a closed-loop, active-perception architecture:

1. **Acquire** — a camera policy that guarantees detectable views (coverage),
   then spends excess frames on information gain (VoI/NBV).
2. **Fuse** — a persistent world map in source coordinates, registered for
   ego-motion, that survives missed frames and off-view periods.
3. **Emit** — answer the whole frame from the map every frame, with a fallback
   ladder that never yields an empty response while belief exists.
4. **Improve** — train the detector on evaluator-exact views with copy-paste and
   multi-scale augmentation, and learn the camera policy offline against a
   simulator built from the 4K source.

Stage P0 uses code that already exists and should move the score from 0
immediately; P2–P4 are where the large gains live.

---

## 9. References

Small / aerial object detection

- Akyön, Altinuc, Temizel. *Slicing Aided Hyper Inference and Fine-Tuning for
  Small Object Detection.* ICIP 2022. arXiv:2202.06934. https://github.com/obss/sahi
- Li et al. *Improved RT-DETR Algorithm for Small Object Detection in Drone
  Aerial Imagery.* ICAISIS 2025.
- Chen et al. *Hybrid receptive field network for small object detection on drone
  view (HRFNet).* Chinese Journal of Aeronautics 38(2), 2025.
- Zhang et al. *SO-DETR: Leveraging Dual-Domain Features and Knowledge
  Distillation for Small Object Detection.* arXiv:2504.11470.
- Hu et al. *Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for
  Efficient Tiny Object Detection.* ACM MM 2025. arXiv:2505.05741.
- *GLDS-YOLO: An Improved Lightweight Model for Small Object Detection in UAV
  Aerial Imagery.* Electronics 14(19):3831, 2025.
- Muzammul et al. *Enhancing UAV Aerial Image Analysis: Integrating Advanced SAHI
  Techniques With Real-Time Detection Models on the VisDrone Dataset.* IEEE
  Access 2024. (latency cost of slicing: ~27 fps → ~1 fps)

Video / temporal fusion and persistent surveillance

- Beery et al. *Context R-CNN: Long Term Temporal Context for Per-Camera Object
  Detection.* CVPR 2020. arXiv:1912.03538.
- Xiao & Lee. *Object Detection with an Aligned Spatial-Temporal Memory.* ECCV 2018.
- Sun et al. *MAMBA: Multi-level Aggregation via Memory Bank for Video Object
  Detection.* arXiv:2401.09923.
- *Object Detection in Video with Spatial-temporal Context Aggregation.*
  arXiv:1907.04988.
- Teutsch & Grinberg. *Robust Detection of Moving Vehicles in Wide Area Motion
  Imagery.* CVPRW 2016.
- Prokaj & Medioni. *Persistent Tracking for Wide Area Aerial Surveillance.*
  CVPR 2014. (registration + track fusion for WAMI)
- Negin et al. *Transforming Temporal Embeddings to Keypoint Heatmaps for
  Detection of Tiny Vehicles in Wide Area Motion Imagery (WAMI).* CVPRW 2022.

Active perception / next-best-view / RL

- *Active Classification of Moving Targets with Learned Control Policies.*
  arXiv:2212.03068.
- Ding et al. *Learning to View: Decision Transformers for Active Object
  Detection.* ICRA 2023.
- *GenNBV: Generalizable Next-Best-View Policy for Active 3D Reconstruction.*
  CVPR 2024. arXiv:2402.16174.
- *Hestia: Hierarchical Next-Best-View Exploration.* arXiv:2508.01014.

Semi-supervised / pseudo-labeling

- Xu et al. *End-to-End Semi-Supervised Object Detection with Soft Teacher.*
  ICCV 2021. arXiv:2106.09018.
- Liu et al. *Unbiased Teacher for Semi-Supervised Object Detection.* ICLR 2021.
  arXiv:2102.09480.
- Ge et al. *Point-Teaching: Weakly Semi-Supervised Object Detection with Point
  Annotations.* arXiv:2206.00274.
- Li et al. *Rethinking Pseudo Labels for Semi-supervised Object Detection.*
  AAAI 2022.

Repository-internal references

- `src/core/pipeline.py`, `src/core/tracker.py`, `src/core/detector.py`,
  `src/core/camera_policy.py`, `src/config.py`, `src/dtos.py`,
  `src/offline/build_exact_view_dataset.py`, `src/offline/pseudo_label.py`,
  `src/offline/calibrate.py`, `src/local_evaluator.py`.
- `recorded_validation_data/*/` and `logs/validation_20260918_151420.log`
  (evidence for the held-L0 / empty-response failure mode).
- `runs/exact_runs/yolo11s_exact150/results.csv` (detector mAP on exact views).
