# Training on IDUN

IDUN is NTNU's GPU cluster: A100/H100-class GPUs through SLURM jobs, up to 80 GB of VRAM,
versus the 8 GB on the training laptop. It cannot serve the endpoint (compute nodes are not
reachable from the internet), so the loop is: **train on IDUN, fetch the weights, serve locally.**

Adapted from Juan's thesis harness (`thesis-prep/src/idun/`).

## One-time setup

**Network.** IDUN only accepts SSH from **eduroam** or the **NTNU VPN** (Cisco Secure Client or
eduVPN). Off campus, connect the VPN first.

**Account.** IDUN logins are personal NTNU (Feide) accounts; access is requested by a
supervisor through the NTNU IT service desk / IDUN project allocation.

**SSH key.** On your laptop:

```bash
ssh-keygen -t ed25519 -C "<ntnu-username>@ntnu.no"
ssh-copy-id -i ~/.ssh/id_ed25519.pub <ntnu-username>@idun.hpc.ntnu.no   # asks for the NTNU password once
```

and add an alias the scripts use:

```bash
mkdir -p ~/.ssh && cat >> ~/.ssh/config <<'EOF'

Host idun
    HostName idun.hpc.ntnu.no
    User <ntnu-username>
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
ssh idun hostname        # should print an idun-login node
```

**Environment on IDUN.** From `drone-flyby/`:

```bash
bash idun/submit.sh setup     # conda env "drone": torch 2.5.1 cu121 + ultralytics, ~10 min
bash idun/submit.sh test      # 20-min GPU job; the log should end with TEST OK
```

## Everyday use

All commands run from `drone-flyby/` on your laptop. Every submit first rsyncs the code.

```bash
bash idun/submit.sh train-exact 150 32     # exact L0/L1/L2 views + 150-epoch YOLO11s@960 run
bash idun/submit.sh run python src/offline/train_yolo.py --data-yaml <yaml> --epochs 60 --batch 32
bash idun/submit.sh queue                  # PD = waiting for a GPU, R = running
bash idun/submit.sh logs                   # tail the newest job log (Ctrl+C stops tailing, not the job)
bash idun/submit.sh fetch                  # copy runs/ back: runs/exact_runs/<name>/weights/best.pt
```

`train-exact` reproduces the local run inside one GPU job: it renders evaluator-exact
960x540 L0/L1/L2 views from `data/helsinki/` (873 train / 147 val), then fine-tunes
YOLO11s at 960 px starting from `yolo11s.pt`. Tune it with `bash idun/submit.sh
train-exact <epochs> <batch>`.

Then serve the new weights locally:

```bash
DRONE_FLYBY_YOLO_WEIGHTS_PATH=runs/exact_runs/yolo11s_exact150/weights/best.pt \
  python src/api.py
```

## What goes up and what stays

| Uploaded | Stays on the laptop |
|---|---|
| `src/`, `idun/`, `requirements.txt` | `.venv/` (IDUN has its own env) |
| `data/helsinki/` (~400 MB, first sync only) | `training_artifacts/` (rebuilt on IDUN) |
| `weights/` (the fine-tuned starting checkpoint) | `runs/` (comes back with `fetch`), `recordings/` |

`yolo11s.pt` (the COCO start for the sanity test) is downloaded on IDUN by `setup`;
`train-exact` starts instead from `weights/yolo11s_drone_flyby.pt`, which the sync
uploads. Override with `BASE_WEIGHTS=... bash idun/submit.sh train-exact`.

The code lives in **`~/nordic-cup/drone-flyby`** on IDUN, apart from anything else in the
home directory, because the sync uses `rsync --delete`. Change it with `REMOTE_DIR=...`.

## Job settings

| | `job.slurm` | `job_test.slurm` |
|---|---|---|
| Partition | `GPUQ` | `GPUQ` |
| GPU | 1, with 32/40/80 GB | 1, any |
| Time limit | 12 h | 20 min |
| CPU / RAM | 8 / 48 GB | 4 / 16 GB |
| Account | `share-ie-idi` (override with `SLURM_ACCOUNT=`) | same |

With 32+ GB of VRAM, raise `--batch` (32 at 960 px fits comfortably; the laptop needs 8).
Queue time varies from minutes to hours; `queue` shows the reason a job is waiting.

## Rules that still apply

- **Nothing on IDUN talks to the competition.** Jobs train; a human validates and submits
  from the web form. No scripts or AI tools submit, ever.
- The team API key never goes on IDUN, in job scripts, or in logs.
- The recorded validation frames (`recordings/`) are not uploaded and must not become training
  data: they are our only honest test set.
