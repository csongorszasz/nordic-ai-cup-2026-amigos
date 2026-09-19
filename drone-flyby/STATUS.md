# Drone flyby: where things stand (2026-09-19 19:30 CEST, branch `waffles`)

For anyone (person or agent) picking this up. Final submission deadline: 2026-09-20 16:00 CEST.

## Rules

- **Never start a validation or the final evaluation yourself.** A person presses the button on
  cases.nordicaicup.com; the final evaluation is a single attempt.
- `datasets/copenhagen_test` and `recordings/validation_4k` (the recorded validation flight) are
  **test only**: never train on them. They are the only honest measure we have.
- Serve from a public IP (the Azure VM, `deploy_vm.sh`). Tunnels (cloudflared, ngrok) lost 20-60 of the
  249 frames and cost 0.03-0.16 mAP.

## Best so far: live 0.474 (248/249 frames answered)

```bash
bash deploy_vm.sh <vm ip> runs/synth400_11s_neighbours_0919-1604/weights/epoch15_int8_openvino_model DRONE_SMALL_IMGSZ=1280 V2_TRUNC=1
```

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
| + neighbour placement data, V2_TRUNC | **0.474** | **0.574** |

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

Per class, best config (offline): ta-ta 0.00, medium_launcher 0.27, small_launcher 0.30, large_launcher 0.45,
mine_roller 0.45, medium_plane 0.65, small_tower 0.66, large_tower 0.67, hangar 0.67, tank 0.79,
helicopter 0.81, small_plane 0.85, jet_plane 0.88.

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
the hybrid planner (about half the sweep's score), a 1600 second pass on the VM (too slow: frames skipped).

## Where the code is

- `src/solution.py`: detector, memory, camera; every setting is an environment variable (see the top).
- `training/synth_dataset.py`: synthetic frames (3D-mesh backgrounds + model renders + real cut-outs).
- `training/run_policy.py`, `src/local_evaluator.py`: offline replay and scoring.
- `training/sprite_review/server.py`: review site (http://localhost:8765/flyby: Helsinki, Copenhagen and
  synthetic frames with labels).
- `deploy_vm.sh`, `vm_up.sh`: Azure VM serving.
