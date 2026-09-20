# Drone flyby: where things stand (2026-09-19 19:30 CEST, branch `waffles`)

For anyone (person or agent) picking this up. Final submission deadline: 2026-09-20 16:00 CEST.

## Rules

- **Never start a validation or the final evaluation yourself.** A person presses the button on
  cases.nordicaicup.com; the final evaluation is a single attempt.
- `datasets/copenhagen_test` and `recordings/validation_4k` (the recorded validation flight) are
  **test only**: never train on them. They are the only honest measure we have.
- Serve from a public IP (the Azure VM, `deploy_vm.sh`). Tunnels (cloudflared, ngrok) lost 20-60 of the
  249 frames and cost 0.03-0.16 mAP.

## Best so far: live 0.517 (247/249 frames answered)

```bash
bash deploy_vm.sh <vm ip> runs/synth400_11s_neighbours_0919-1604/weights/epoch15_int8_openvino_model DRONE_SMALL_IMGSZ=1280 V2_TRUNC=1 DRONE_MEMORY_LEAD=0.1
```

Camera timing live is noisy: commands land on the next frame, a frame later or never, in shares that change
from run to run (network). Planning from the view received instead of the last command
(`DRONE_SWEEP_FROM_VIEW=1`) scored 0.357 live in a run that also lost 27 frames to request gaps: keep the
default planner. `run_policy.py --live-timing NEXT,LATER` replays such timing, but it underrates the default
planner (0.459 replayed vs 0.589 for the real live answers), so do not choose camera changes on it alone.

- Detector: YOLO11s trained on 400 synthetic 4K frames for 30 epochs (`idun/submit.sh train-synth`), epoch 15,
  OpenVINO int8 on a 4-vCPU VM (~183 ms per request; keep it well under the 333 ms frame interval).
- `DRONE_SMALL_IMGSZ=1280`: a second, enlarged pass keeping only ta-ta / small / medium launcher.
- `V2_TRUNC=1`: a box cut by the view edge does not overwrite a whole one (hangar 0.53 -> 0.67 offline).
- Camera: snake sweep over six Level-1 positions, memory reports every tracked object every frame,
  boxes carried through the ground-motion field (`V2_GMC`, `V2_INTEGRATE`, on by default).

| Live history | Live | Offline (Copenhagen) |
|---|---|---|
| synthetic v1 | 0.273 | 0.30 |
| + repainted models, per-scene darkness | 0.377 | 0.431 |
| + int8, second small-object pass | 0.423 | 0.485 |
| + motion integration | 0.441 | 0.524 |
| + neighbour placement data, V2_TRUNC | 0.474 | 0.574 |
| + labels cleaned, memory lead 0.1 | **0.517** | **0.640** |

## Scoring offline

Assets are not in git. From the GitHub releases:
- `drone-data-denmark/copenhagen_validation_4k.zip`: `recordings/validation_4k/` (249 rebuilt 4K frames) +
  `datasets/copenhagen_test/labels.json` (43 hand-checked objects). Unzip inside `drone-flyby/`.
- `drone-data-denmark/drone_weights_sprites_0919.zip`: the live weights, `sprites/` and `datasets/model_sprites/`
  (repainted model renders).
- `drone-data-denmark/drone_backgrounds_0919.zip`: `backgrounds/helsinki3d_frames/` and `backgrounds/naip/`.

```bash
python training/run_policy.py --scene validation_4k --camera sweep --lag 1 \
    --model runs/synth400_11s_neighbours_0919-1604/weights/epoch15_int8_openvino_model \
    --set DRONE_SMALL_IMGSZ=1280 V2_TRUNC=1
```

It prints mAP50 overall, per class, and on frames 1-125 ("tune": choose checkpoints and settings here) and
126-249 ("check": confirm). The supplied Helsinki scene (25 frames) did not predict live or Copenhagen scores:
do not select on it.

Per class for the live model's 0.640 (offline, measured 2026-09-20 01:10): ta-ta 0.00, mine_roller 0.29,
medium_launcher 0.40, small_launcher 0.46, hangar 0.56, small_tower 0.69, large_tower 0.76,
large_launcher 0.79, tank 0.79, jet_plane 0.86, helicopter 0.89, small_plane 0.91, medium_plane 0.91.

## ta-ta is 0.00 because of the camera, not the detector

Run the live detector straight at the three ta-ta in the 4K Copenhagen frame and it finds all three at
**conf 0.72-0.78** with boxes that match the labels. Feed it the Level-1 view the drone actually transmits
(1920x1080 of source into 960x540, so a 2x downscale) and it finds nothing, or 0.07. A ta-ta is 19x44 px in
the source, so it arrives as 9x22 px. The model knows the class; the pixels never arrive. Only a Level-2
visit would fix it, and L2 dives cost more coverage than they win (0.568).

It is worse than a miss. Over the 249 frames the run reports **353 ta-ta boxes, and not one reaches IoU 0.5
with any of the three real ones** (best 0.06). They land on dark bushes and a circular field feature. So
ta-ta contributes 0.00 and adds false positives on top.

Raising the second small-object pass from 1280 to 1600 gives 0.640 -> **0.648**, all of it from
medium_launcher (0.40 -> 0.48) and small_launcher (0.46 -> 0.48); ta-ta stays 0.00. It costs about 35% more
inference (960+1280 = 237 ms on the laptop, 960+1600 = 320 ms; the VM ran 960+1280 in 183 ms, so expect
~247 ms against a 333 ms frame interval). Not deployed: +0.008 is not worth risking dropped frames without
measuring on the VM first.

## What to submit (2026-09-20 morning)

**Submit the model that is already serving.** Deploy command, unchanged since 0.5171:

```bash
bash vm_up.sh          # if the VM is deallocated; prints the IP
bash deploy_vm.sh 9.160.106.215 runs/synth400_11s_neighbours_0919-1604/weights/epoch15_int8_openvino_model \
     DRONE_SMALL_IMGSZ=1280 V2_TRUNC=1 DRONE_MEMORY_LEAD=0.1
# submit http://9.160.106.215:9053/predict
```

Four live validations, the only measurement that can rank models:

| served | live |
|---|---|
| **neighbours e15, 1280 small pass** | **0.5171** |
| bigbg e15 (240 sharp Danish backgrounds) | 0.4953 |
| allbg last | 0.4926 |
| neighbours e15 + 1600 small pass | 0.4594 |

Every challenger lands in 0.46-0.50; the incumbent is alone at 0.5171, and the 0.022 gap to
the next best is wider than the spread among the challengers.

Dead ends closed with measurements, do not redo them:
- **Bigger models**: 11s and 11m on one recipe gave tune 0.410 and 0.411, no effect from size.
  The VM cannot be scaled either -- the DSv2 quota is 4 vCPUs and all 4 are in use.
  Measured cost ratio 11m/11s on one machine is 1.95x, so against the VM's known 183 ms for
  960+1280: 11m with the small pass is ~357 ms (over the 333 ms interval), 11m without it
  ~145 ms (fits). So 11m is servable only by dropping the small pass, which is two changes at
  once and gives up the launcher classes the pass exists for.
- **Serving from a laptop GPU** (port-forward, no tunnel): 0.3453. Not compute -- localhost
  round-trip was 63 ms against the VM's 180. Each request is a 1.5 MB base64 PNG, so 3 fps
  needs ~36 Mbit/s sustained; the line measured 112 Mbit/s to one host and 18 to another.
- **The 1600 small pass**: genuinely better offline, 836 ms per request on the VM, two frames
  in three skipped.

## The offline score is too noisy to rank models. Read this before chasing a number.

`oldgen` re-ran the live model's **exact generator code** (git 510418e) with its exact recipe.
By epoch: 0.587, 0.620, **0.516**, 0.564, 0.552, 0.469 on the tune half. The live model's
epoch 15 scored 0.636. Same code, same command, a different draw.

Spread of the tune score across epochs **within one run**, over 17 runs: median **0.068**,
up to 0.151. Every difference argued about during the night -- dk 0.622, sharp 0.609,
allbg 0.635, bigbg 0.636 against the live model's 0.636 -- sits inside that.

So: the live model's 0.636 is one lucky epoch of one run, not a reproducible property of
its recipe. Nothing in the generator "broke" tonight; `ctrl` at 0.476 was the same noise.
43 objects over 13 classes cannot separate models a few points apart, and picking the best
of N noisy checkpoints selects mostly for luck, which is exactly how allbg reached a
validation and lost 0.025 live.

What the offline score is still good for: catching a model that is plainly broken (0.45 vs
0.62), and per-class diagnosis. What it cannot do: choose between two decent models. For
that, only a live validation counts -- 249 frames against the real ground truth.

## Choose on frames 1-125 only. The overall number cost us a validation.

`allbg` last scored **0.663** overall against the live model's 0.640, was deployed, and came
back from validation at **0.4926** against 0.5171. Its halves explain it:

| | frames 1-125 (tune) | frames 126-249 (check) | overall | live |
|---|---|---|---|---|
| live model | **0.636** | 0.715 | 0.640 | **0.5171** |
| allbg last | 0.635 | 0.747 | **0.663** | 0.4926 |

On the half we are allowed to choose on they are tied; the entire gain sat in the half kept
back for confirmation. Ranking by the overall number is ranking partly by the confirmation
half, and it picked a model that is 0.025 worse live. `overnight.sh` now flags a candidate
only when its tune half beats 0.636; anything better overall but not on tune is recorded in
`overnight/CHECK_HALF_ONLY.txt` and not proposed for deployment.

Offline still ranks big differences correctly (0.55 vs 0.64 is real). It cannot separate
0.64 from 0.66 -- 43 objects over 13 classes is too small a test for that.

## A quarter of what we report is for classes that are not there

Counting every box the live model reports over the 249 Copenhagen frames:

| class | in our labels | boxes reported | max conf | fires on |
|---|---|---|---|---|
| jammer | **0** | **1255** | 0.78 | sheds and bushes in fields, parked trucks, cars in yards |
| spacecraft | **0** | 439 | 0.77 | one dark angular vehicle, over and over |
| condor | **0** | 177 | 0.86 | **boats in a marina**, a rooftop |
| ta-ta | 3 | 353 | 0.73 | bushes, a circular field feature (never the real ones) |

1871 of 7765 boxes, 24%, are for three classes with no instance in the scene. They cost nothing offline
(the scorer only averages classes present in the ground truth) but the supplied Helsinki scene has all 16,
so the real evaluation probably does too, and then these decide those classes' AP.

On the Helsinki scene the same model scores condor 0.87, jammer 0.74, spacecraft 0.90, so the classes are
not broken; the model has simply never seen a marina, a Danish shed or a city yard and reads them as
objects. That is the same finding as the background experiment from the other direction, and it is the
argument for more and more varied sharp photo backgrounds rather than more sprite work.

## The recipe with no flags is still the one to beat

The live model at 0.640 was generated by `python training/synth_dataset.py --frames 400` with **no options
at all**. Nine later recipes, each adding flags meant to fix something real, all score below it:

| Run | Change | Copenhagen |
|---|---|---|
| `dk` | sharp Danish backgrounds added, cut-outs only for the 3 small classes | 0.628 |
| `sharp` | Danish + NAIP only, no Helsinki mesh | 0.625 |
| `allbg` | all three background sets, model helicopters, ground masks | 0.616 |
| `cutouts` | cut-outs only, every class | 0.604 |
| `fix2` | overlap limit, new hangar | 0.597 |

Sharp photo backgrounds are worth about +0.025 over the Helsinki 3D-mesh renders ("van Gogh" ones), which
is the clearest single signal. Nothing else has separated from noise.

Open: the open-ground masks (`training/ground_masks.py`, added 23:12) keep grass, fields and sand only,
**11-16% of a frame**, empty on some. That forbids the tarmac and concrete real objects stand on at an
airbase. `allbg` uses them and still reached 0.616, so they are not obviously fatal. `--ignore-ground`
pastes anywhere again; runs `ctrl` (defaults), `dkplus` (defaults + Danish backgrounds) and `trimsng`
(hand-trimmed cut-outs, lean-aware) test it.

`overnight.sh` scores every IDUN checkpoint against Copenhagen unattended, detached from any editor:
`sort -k3 -rn overnight/scores.tsv`, and `overnight/BEATS_LIVE.txt` lists anything past 0.640.

## Open question: why live is ~0.83-0.87x offline

The scorer macro-averages AP over the classes present in the ground truth (README "Scoring"). The supplied
Helsinki scene has all 16 classes; our Copenhagen labels have 13 (no condor, jammer, spacecraft).
13/16 x 0.574 = 0.466, live 0.474. If the real ground truth has all 16, those three classes are worth
3/16 of the score, plus ta-ta another 1/16.

Being tested:
1. `DRONE_LOG_RESPONSES=/tmp/responses.jsonl` on the VM logs every live answer; `training/score_live_log.py`
   scores them against our labels. ~0.57 means the gap is objects/classes our labels lack; ~0.47 means serving.
2. Ground never seen at Level 2 in any recording: the bottom strip of frames 1-6 and the top strip of
   frames 240-249 (`reconstruct_frames.py --level-maps`). `DRONE_CAMERA=record RECORD_EDGES=1` records those.

## Tried, did not help

Test-time flips (-0.02), averaging two models (+0.002 on tune), 50 epochs, 600 frames, class balancing,
the hybrid planner (about half the sweep's score). Camera changes chosen on replay timing both lost live:
planning from the view received (0.357) and a 2-frame sweep hold (0.425). Class hedging (+0.003), L2 dives
(0.568), memory-forgetting tweaks (0.529-0.535).

## Where the code is

- `src/solution.py`: detector, memory, camera; every setting is an environment variable (see the top).
- `training/synth_dataset.py`: synthetic frames (3D-mesh backgrounds + model renders + real cut-outs).
- `training/run_policy.py`, `src/local_evaluator.py`: offline replay and scoring.
- `training/sprite_review/server.py`: review site (http://localhost:8765/flyby: Helsinki, Copenhagen and
  synthetic frames with labels).
- `deploy_vm.sh`, `vm_up.sh`: Azure VM serving.
