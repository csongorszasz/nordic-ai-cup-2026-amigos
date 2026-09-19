# Reproducible IDUN experiments

Experiments request **one A100/H100 with 80 GB VRAM** on `GPUQ`, billed to
`share-ie-idi` by default. Both job definitions enforce the same constraint.
`preflight.py` checks CUDA, the actual device, snapshot hashes, and package
versions before executing a command.
Optional external experiment tracking and Ultralytics telemetry are disabled
in the snapshot's own settings. HTTP benchmarks verify their launched server's
run nonce before sending any frames.

SSH access requires an authorized NTNU connection and an existing `idun` SSH
alias. Do not put passwords, private keys, or competition API keys in scripts.

## Windows

From `drone-flyby`, using PowerShell 7, OpenSSH, and tar:

```powershell
.\idun\submit.ps1 -Action sync -Experiment baseline-001
.\idun\submit.ps1 -Action setup -Experiment baseline-001
.\idun\submit.ps1 -Action run -Experiment baseline-001 -RunCommand 'python -m pytest src/tests -q'
.\idun\submit.ps1 -Action status -Experiment baseline-001
.\idun\submit.ps1 -Action fetch -Experiment baseline-001
```

`sync` creates a **new immutable code snapshot** under
`~/nordic-cup/drone-score-loop/experiments/<experiment>`. It refuses to overwrite
an existing snapshot. Shell scripts are normalized to LF and syntax-checked.
The manifest contains hashes of the actual uploaded content, including data.
`run` and `test` claim the snapshot once before submitting, preventing duplicate
submissions after a client restart. A failed submission needs a new experiment
ID; do not blindly resubmit an ID whose result is unknown.

`setup` creates a task-owned virtual environment, never updates other conda
environments, and writes a package lock. Use `-Environment env-v2` consistently
for setup and submission when dependencies change. The same environment can be
reused by subsequent snapshots with compatible requirements.

`test` submits a small GPU forward-pass check instead of a custom command.
`fetch` retrieves logs and results under `runs\idun\<experiment>`.
On Windows, deeply nested recordings can exceed OpenSSH SCP's path limit.
Use `-FetchDirectory "$env:TEMP\drone-flyby\baseline-001"` to select a shorter
final experiment directory. It still receives the `runs` and `logs` folders;
this option is valid only for `fetch`. `fetch-status.json` records completion
or failure, so a partial download is not mistaken for missing producer data.
An experiment failure is not a zero-score result: inspect Slurm's state and
exit code, the error log, and any `failure.json`.

## Bash

The existing `submit.sh` entry point also uses per-experiment snapshots:

```bash
EXPERIMENT_ID=baseline-001 bash idun/submit.sh setup
EXPERIMENT_ID=baseline-002 bash idun/submit.sh run python -m pytest src/tests -q
EXPERIMENT_ID=baseline-002 bash idun/submit.sh fetch
```

Setup and run each create their own snapshot in the Bash launcher; reuse the
environment, not a snapshot ID. `REMOTE`, `REMOTE_ROOT`, `DRONE_ENV_PATH`, and
`SLURM_ACCOUNT` override the defaults. This entry point requires tar; fetching
results additionally uses rsync. No upload performs `rsync --delete`.

## Data boundaries

Uploads are allowlisted to source, job scripts, requirements, tests, the supplied
Helsinki scene, and optional `weights`. Neither `recorded_validation_data` nor
`recordings` is uploaded implicitly.

The recorder marks captures with `data_role.json` containing
`"data_role": "evaluation-only"`. Authorized evaluation-only transfers must retain
that marker and remain separate from training inputs. Builders, pseudo-labeling,
and training reject these inputs. Never train on validation recordings or their
derived crops/labels.

The exact-view builder splits by **source frame before generating views**.
Helsinki still shares object instances between splits: its validation AP is a
development diagnostic, not independent generalization evidence.

## Comparisons and serving

```bash
python src/offline/experiment_runner.py --mode oracle \
  --min-view-pixels 16 --output runs/oracle-diagnostic
python src/offline/experiment_runner.py --mode detector \
  --weights /path/to/task-checkpoint.pt --output runs/detector-comparison
python src/offline/experiment_runner.py --mode http \
  --weights /path/to/task-checkpoint.pt --output runs/realtime-comparison
```

Oracle size gates are diagnostic assumptions, not measured detector accuracy.
`http` starts an owned local endpoint for each configuration and measures the
real frame clock; it never contacts the competition. Raw predictions, per-class
AP, failures, and manifests are retained.

The intended competition endpoint also runs on an 80 GB IDUN allocation.
External access requires an authorized ingress/proxy route; an internal compute
node port alone does not establish public reachability. Keep serving and training
isolated, warm up the chosen checkpoint, and do not change it during an attempt.
**Official validation and evaluation remain human-triggered and approval-gated.**

## Session-scoped public prediction endpoint

Use the existing `cloudflared` binary and a task-trained checkpoint with its
verified SHA-256. From PowerShell, for example:

```powershell
.\idun\serve.ps1 -Experiment serving-001 `
    -WeightsPath /cluster/home/dominiba/nordic-cup/drone-score-loop/experiments/alpha-training-002/runs/training/alpha/weights/best_ap50.pt `
    -WeightsSha256 fa665aa57c9622ffb41aea90202fb00ed96133aa8968f75fcedbdbae18837bde
```

The launcher creates an immutable snapshot and stays attached to an SSH
pseudo-terminal running `srun`. It requests one A100/H100 80 GB GPU on `GPUQ`,
with a **12-hour maximum walltime**. Keep this command attached to the CLI
session, not a detached job. Closing the session interrupts the owned
allocation; the supervisor stops its API and tunnel children together.
Other jobs and listeners are never killed or reused.

The fixed provisional profile is YOLO standard + world-map + hold, FP16,
3200-pixel inference, a 0.001 L0 proposal floor, and at most 300 detector
proposals. Inherited `DRONE_FLYBY_*` overrides are cleared before applying this
profile; recording is opt-in and calibration is disabled. The example alpha checkpoint
improved the foreground-only development audit but is **not competition
validated**, and it is not an automatic promotion of `profiles\champion.json`.

After source/device/checkpoint checks, one Uvicorn worker binds
`0.0.0.0:9052`. This does not change `python src/api.py`'s existing 9053 default.
The supervisor verifies the warmed model and its run nonce before launching
`cloudflared` on the same compute node, pointed at `http://127.0.0.1:9052`.
It uses HTTP/2 over outbound TCP 7844; cluster-approved DNS and HTTPS egress
are also needed. An isolated cloudflared home avoids changing an existing
configuration, and metrics bind only to loopback.

The printed `TUNNEL_URL` includes `/predict`. It is a temporary, unauthenticated
Quick Tunnel hostname, **not proof that an external prediction succeeded**.
Verify `/stats` against `runs/serving/deployment.json`, then send a protocol-valid
development-frame POST from outside IDUN before handing over the URL. Do not
send smoke-test sequences during a competition run: the model's state is
sequence-scoped. This launcher never contacts competition submission APIs.

Logs, PIDs, checkpoint/configuration provenance, node, job expiry, and the
assigned URL are in the snapshot's `runs/serving/`. The local session log is
under `runs\idun\<experiment>`. Quick Tunnels have no uptime guarantee; a restart
gets a new hostname. Either child exiting fails the service rather than
silently leaving a stale URL. A new launch requires a new snapshot ID.
An abrupt SSH disconnect can make Slurm kill the entire step before the
supervisor flushes its final manifest. An assigned URL in an old artifact is
not a liveness check: consult Slurm and verify the live endpoint's nonce.

The supervisor also probes the public hostname using non-mutating `/stats`
requests and verifies its nonce. `public_health` in the deployment manifest
distinguishes reachable, unreachable, and stalled/inconclusive checks. DNS
failure does not silently rotate the URL or stop the origin. Probes are bounded
to one outstanding daemon thread so a stalled resolver cannot stall child
supervision. An origin-node check is not proof of every external client's RTT.

### Evaluation-only diagnostic capture

Add `-RecordValidation` to `serve.ps1` to enable capture **without changing the
model/tracker/camera profile**. Use a new immutable snapshot and isolated
diagnostic allocation rather than altering an active submission service.
Captures live under its `runs/recorded_validation_data/<sequence_id>`.
The original serving allocation stays separate, and no second GPU experiment
may compete with a diagnostic endpoint while it is receiving an attempt.

Each received request has a unique stem shared by its exact input PNG,
complete DTO metadata, response, and pipeline diagnostics. Retries cannot
overwrite earlier inputs; response and diagnostic identity are checked.
Diagnostics distinguish raw post-YOLO/pre-tracker detections, final fresh/memory
rank provenance, timing, observed frame-index gaps, cached responses, and
fallbacks. They never add fields to the competition response.

Image hashes and serving provenance accompany the recording. Queue drops,
failed writes, missing pairs, and failed predictions are explicit in status
files and `/stats`; a generated response is **not** evidence of evaluator
acceptance. With the known validation denominator:

```bash
python src/offline/summarize_recording.py \
  --dir runs/recorded_validation_data/<sequence_id> \
  --expected-frames 249 --output-json runs/capture-summary.json
```

The summary does not invent AP without ground truth or infer missing optical
views from a transmitted crop. Modern metadata can be reconstructed exactly
with `load_recorded_request`; legacy recordings missing protocol fields are
rejected for exact replay rather than filled with guessed defaults.
All captures are marked **evaluation-only** and excluded from training,
pseudo-labeling, augmentation, and calibration. Keep images, logs and temporary
URLs out of Git. Only the user triggers official competition attempts.

For fixed-observation detector/tracker comparisons on an owned experimental GPU:

```bash
export DRONE_FLYBY_CONF_L0=0.001
export DRONE_FLYBY_INFERENCE_HALF=true
python src/offline/experiment_runner.py --mode recording \
  --recording /path/to/evaluation-only/sequence --expected-frames 249 \
  --weights /path/to/task-checkpoint.pt --imgsz 3200 \
  --policies hold --trackers passthrough world_map --output runs/recorded-comparison
```

Set model overrides explicitly to match the captured provenance when reproducing
a baseline; different overrides represent a new candidate, not a reproduction.
This validates image hashes, identities, producer status, serving provenance,
and original processing order before loading the model. Every received crop and
frame-index gap is preserved. It reports response differences and coverage, not
AP without ground truth; a fixed crop stream cannot evaluate a different camera
trajectory or supply frames that never arrived.
Exact equality is reported without rounding. Numeric comparisons separately
report aligned classes/order/camera, confidence differences, and bounding-box
drift in source pixels: cross-node tracking arithmetic can differ below a
pixel even when raw detections match. Do not hide the exact verdict or assume
that any tolerance implies equal AP. Each result stores its effective runtime
configuration rather than only the matrix's pre-override defaults.

For an owned development HTTP replay, `experiment_runner.py --mode http
--capture-inputs ...` verifies the same capture path and its overhead without
contacting the competition.

### Observed training budgets

Training writes `training_budget.json` with the observed minibatch count and
the optimizer's per-parameter step-counter range before and after the run.
Gradient accumulation means these are not interchangeable: nominal batch 64
with physical batch 32 uses accumulation, so 972 minibatches must not be
reported as 972 optimizer updates. Null counters are explicitly unavailable,
not assumed zero; resumed counters can include prior training history.

### Matched non-right-angle audit

`build_synthetic_sequence.py --rotation-offset-degrees <angle>` requires the
reviewed alpha assets. Pair **explicit `0`** with offsets such as `22.5` or `45`,
keeping the seed and other arguments identical. These controls use the same
angle-independent padded canvases, centers, sampled quarter turns, scales,
placements, and background crop. Zero offset preserves the source pixels.
Do not compare a padded offset scene against an older scene without this option:
the older placement reservations differ.

Labels enclose the transformed **original annotation canvas**, not a tightened
mask-support box. Rotation uses premultiplied alpha and accounts for pixel-center
versus annotation-edge coordinates. The audit shares source object appearances
and tests a specific orientation/resampling shift; it is not independent
competition validation or evidence of a training improvement by itself.

`zoom_diagnostics.py` measures conditional recall on fully-contained objects at
real optical L0/L1/L2 crops and records the measured tensor shape/precision.
Different neural sizes must be compared explicitly; enlarging an L0 tensor is
not equivalent to acquiring L1/L2 detail. These repeated crop observations do
not measure a legal camera policy, full-frame AP, or new independent objects.
