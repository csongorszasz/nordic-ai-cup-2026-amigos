# IDUN handoff — survival-simulator

This runbook lets csongor's agent run the survival-simulator training/benchmark
and serve the agent endpoint on NTNU **IDUN**, using dominic's IDUN login and
`share-ie-idi` allocation via a delegated SSH key. Follow the steps in order.

Everything runs from **csongor's machine**. The only human-gated step is
**Step 1** (dominic authorizes csongor's public key on IDUN). No passwords or
private keys are ever shared.

---

## 0. Prerequisites

- NTNU **VPN / eduroam** active (IDUN is only reachable from the NTNU network).
- `git`, `ssh`, `rsync`, and `curl` installed.
- Python 3.10–3.13 locally is *not* required — the environment is built on IDUN.

---

## 1. [HUMAN GATE · dominic] Authorize csongor's key

1. csongor generates a dedicated key and prints the **public** half:

   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/idun_key -N ""
   cat ~/.ssh/idun_key.pub
   ```

2. csongor sends that one line to dominic (chat/email is fine — it is a public key).

3. dominic authorizes it on IDUN for user `dominiba`:

   ```bash
   ssh idun 'mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys' <<'KEY'
   <paste csongor's public key line here>
   KEY
   ```

   If NTNU's IDUN requires keys to be registered through the account portal
   instead of `authorized_keys`, dominic adds the same public key there.

4. Revoke at any time by deleting that line (or portal entry).

---

## 2. csongor: configure SSH and verify

```bash
cat >> ~/.ssh/config <<'CFG'
Host idun
    HostName idun.hpc.ntnu.no
    User dominiba
    IdentityFile ~/.ssh/idun_key
CFG

ssh idun "echo ok && hostname && whoami"      # expect: ok, idun-login1, dominiba
```

---

## 3. Get the code

```bash
git clone https://github.com/csongorszasz/nordic-ai-cup-2026-amigos.git
cd nordic-ai-cup-2026-amigos
git checkout survival-simulator/csongor-idun
cd survival-simulator
```

All harness commands below run from the `survival-simulator` directory.

---

## 4. Build the environment on IDUN (once)

```bash
bash idun/submit.sh setup
```

This rsyncs the project to `~/nordic-survival` on IDUN and creates the
`survival` conda env (Python 3.12, `requirements.txt`, `torch==2.8.0`,
`cma`, `matplotlib`), then runs a headless Pygame smoke test.

---

## 5. Run experiments (submit SLURM jobs)

```bash
# Controller search (CMA, CPU). Tune size with --set; see configs/search.json.
bash idun/submit.sh search --set search.candidates=16 --set search.worlds=4

# Throughput profile (CPU).
bash idun/submit.sh profile

# Neural training on a GPU (imitation or PPO).
bash idun/submit.sh train configs/imitation-gru.json imitation
bash idun/submit.sh train configs/ppo-gru.json ppo
#   warm start:  bash idun/submit.sh train configs/ppo-gru.json ppo --set checkpoint=training-results/imitation-<id>/checkpoint.pt
#   resume:      bash idun/submit.sh train configs/ppo-gru.json ppo --resume training-results/ppo-<id>/checkpoint.pt

# Benchmark / compare (CPU).
bash idun/submit.sh benchmark run --policy src.policies.runtime:create_policy \
    --config configs/controller.json --suite quick --output benchmark-results/controller-quick
bash idun/submit.sh compare --reference benchmark-results/random-standard \
    --candidate benchmark-results/controller-standard --output benchmark-results/cmp-1

# Unit tests.
bash idun/submit.sh test
```

To request an **80 GB GPU** for PPO, run on IDUN from `~/nordic-survival`
after syncing the project:

```bash
mkdir -p logs
sbatch --constraint=gpu80g idun/job_train.slurm configs/ppo-gru.json ppo-80gb
```

This overrides the training script's GPU constraint for this submission only.
It does not change the application's `resources.max_vram_mb` memory limit.

Monitor and collect:

```bash
bash idun/submit.sh queue            # list your jobs
bash idun/submit.sh logs <jobid>     # tail a job log
bash idun/submit.sh pull             # copy training-results/ and benchmark-results/ back
```

Notes:
- Benchmark `--output` directories are never overwritten; use a new path per run.
- Search cannot `--resume`; size `search.candidates`/`search.worlds` to finish
  within the 12 h walltime.
- `resources.workers` is set from `--cpus-per-task - 1`; edit the SLURM files to
  change CPU/GPU sizing.

### Evaluate each new checkpoint without stopping training

On IDUN, submit the watcher directly from `~/nordic-survival`:

```bash
sbatch --exclude=<training-node> idun/job_watch.slurm \
  --training-job <training-job-id> \
  --run training-results/<training-run> \
  --reference benchmark-results/<completed-random-quick-run> \
  --output benchmark-results/<new-monitor-directory>
```

The CPUQ job captures the current checkpoint and checks for new generations every
five seconds. Copies are published only when the weights, checksum sidecar, and
policy descriptor agree; update numbers come from the saved training state.
One evaluator runs full-horizon `quick` benchmarks and paired comparisons against
the existing reference. Capture continues while evaluation is busy, so snapshots
can queue without blocking training. The output contains `status.json`,
`snapshots/update-NNNN`, per-update logs, benchmarks, and comparison reports.

The watcher stops capturing when the training job leaves the queue, then finishes
pending evaluations, subject to its own 12-hour walltime. It records missed update
numbers and evaluation failures instead of silently treating them as successes.
Restart with the same output to process preserved snapshots; completed and failed
evaluations are not rerun. Earlier overwritten checkpoints cannot be recovered.
Quick-suite results remain exploratory; reserve holdout worlds for final selection.

For an already-running training job, copy only `watch_checkpoints.py` and
`job_watch.slurm` into the remote `idun` directory. Do not use `submit.sh` to deploy
the watcher: its resync can change live sources and delete remote logs. Exclude
the training node to keep benchmark compute separate; no GPU is requested.

---

## 6. Serve the endpoint (for validation / evaluation)

```bash
bash idun/submit.sh serve configs/controller.json
# or a trained policy:
bash idun/submit.sh serve training-results/ppo-<id>/policy.json
```

The job prints `PUBLIC_URL: https://<...>.trycloudflare.com`. Submit that URL on
the competition site. The endpoint stays up until the SLURM time limit; restart
the job (not the tunnel) to change policy. The tunnel URL is **ephemeral** — if
the job restarts, re-read the URL and re-submit it.

The evaluation runs **three attempts back to back**, so leave the serve job
running through all three.

---

## 7. Conventions and caveats

- `REMOTE_DIR=~/nordic-survival`, `SLURM_ACCOUNT=share-ie-idi`, `CONDA_ENV=survival`
  can be overridden via environment variables before calling `submit.sh`.
- Jobs run under dominic's IDUN account and charge the `share-ie-idi` allocation.
- Do not commit keys or credentials. The delegated private key stays at
  `~/.ssh/idun_key` on csongor's machine.
- The remote sync excludes `.git`, `training-results/`, `benchmark-results/`.
  Benchmark/training provenance may therefore report git as unavailable on IDUN;
  run the reference benchmark locally if exact git provenance is required.
