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
