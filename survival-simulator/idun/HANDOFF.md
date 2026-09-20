# IDUN handoff — survival-simulator

This runbook lets csongor's agent run the survival-simulator training/benchmark
and serve the agent endpoint on NTNU **IDUN**, using dominic's IDUN login and
`share-ie-idi` allocation via a delegated SSH key. Follow the steps in order.

The **user launches every real job** (including training, teacher collection,
profiles and evaluations). An assistant may prepare code, synthetic tests, saved
plots and dry-run plans, but must not execute SSH/SLURM experiments. Step 1 also
requires the account owner's authorization. No passwords or private keys are shared.

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

# Neural training requires a reviewed preflight; see the staged procedure below.

# Benchmark / compare (CPU).
bash idun/submit.sh benchmark run --policy src.policies.runtime:create_policy \
    --config configs/controller.json --suite quick --output benchmark-results/controller-quick
bash idun/submit.sh compare --reference benchmark-results/random-standard \
    --candidate benchmark-results/controller-standard --output benchmark-results/cmp-1

# Unit tests.
bash idun/submit.sh test
```

### Teacher, preflight, then manual launch

The exact teacher is `configs/controller-turnaway-wall-aware.json`, pinned by hash.
Both learner presets start with cadence 1 and deliberately small **diagnostic**
budgets. Their numerical settings are provisional, not claims of optimality.
See `docs/pipeline-overview.md` and `configs/evidence-diagnostic.json`.

On a configured machine, generate a plan without constructing a simulator or GPU:

```bash
python train.py --config configs/imitation-gru.json --preflight \
  --evidence configs/evidence-diagnostic.json --output plans/bc
```

The user reviews `plans/bc/run-plan.json`, its exact settings, evidence and total
episode/storage budgets, then launches:

```bash
bash idun/submit.sh train configs/imitation-gru.json bc --run-plan plans/bc/run-plan.json
```

This path copies code to `~/nordic-survival/releases/<plan-id>` once, rather than
rsyncing over active sources. Training/benchmark outputs remain shared through
links; logs belong to that release. Reusing the plan reuses frozen code. Changes
to effective overrides, teacher, weights, or sources require a new preflight.
Keep paths without spaces when choosing remote root/release names.

To request an **80 GB A100/H100-class allocation**, the user can submit from the
printed frozen release with the cluster's available constraint:

```bash
mkdir -p logs
sbatch --constraint=gpu80g idun/job_train.slurm configs/imitation-gru.json bc-80gb \
  --run-plan run-plan.json
```

An 80 GB allocation does not automatically change `resources.max_vram_mb`, workers,
batch sizes, or model width. The script no longer overrides worker count from
`SLURM_CPUS_PER_TASK`. Effective settings must match preflight and fit allocated
CPUs/GPU memory/host memory. Measure a user-launched pilot before increasing them.
The application no longer imposes the laptop's 8192 MiB upper validation limit.

After BC, enable DAgger explicitly with `--set imitation.dagger_rounds=1` in a new
plan and launch. For warm-start PPO, specify the recorded checkpoint in both
preflight and launch. The scratch PPO preset has no checkpoint and zero teacher
regularization; it does not query the teacher during collection. No stage
automatically launches its successor.

Extended run evidence must contain each parameter group's exact resolved values,
measured-pilot artifact paths, units, alternatives, and uncertainty. Dry-run
derivations include native-step bounds, GAE/BPTT horizons, scheduled checkpoints,
and total evaluation episodes including the teacher. References alone are not
task-specific proof.

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
- Choose worker and learner-thread settings deliberately and request enough CPUs;
  the launcher must not change reviewed hyperparameters.

### Evaluate scheduled checkpoints and view the learning curve

The user starts a separate CPU watcher **from the same frozen release**. Exclude
the training node so benchmark compute does not distort learner throughput:

```bash
sbatch --exclude=<training-node> idun/job_watch.slurm \
  --training-job <training-job-id> \
  --run training-results/<training-run>
```

The trainer publishes slim, immutable snapshots initially, every five completed
updates by default, and at the final update. The watcher processes durable ready
manifests, not whichever overwritten latest file a poll happens to catch.
It evaluates the pinned teacher and each checkpoint on the same full-episode
development cases, serially, with explicit per-episode timeouts.

Every completed evaluation refreshes:

- `training-results/<run>/progress/learning-curve.png`
- `learning-curve-ticks.png`, `learning-curve.csv`, `evaluations.jsonl`, `status.json`
- immutable episode records under the run's `evaluation/` directory

The line is mean native final episode score. The band is observed world-seed
range, not a confidence interval. Queued/failed/missing updates are visible;
losses and partial training rewards never stand in for an evaluation.

Queue and storage caps are reviewed configuration. If the pending queue fills,
the trainer pauses with a saved completed-update checkpoint (exit 3); it does not
discard checkpoints or spawn extra jobs. The user drains the existing queue
with the watcher or `python -m idun.watch_checkpoints --run ... --once`, then
creates a new reviewed `--resume` plan/output. Historical score points and actual
native-step counters are retained. The watcher drains only scheduled requests
inside the authorized episode/attempt budget when the trainer stops.

Failures are not automatically retried. `python -m src.training.evaluation retry
--run ... --update N --reason "..."` enables an explicit retry only if the original
plan reserved another attempt with `evaluation.max_attempts`. Successful cases
are not rerun. Default attempts are one.

Plot-only inspection is safe and starts no simulations:

```bash
python -m src.training.progress --run training-results/<run>
```

Legacy rolling-checkpoint monitoring remains available only with
`--legacy-latest --output ... --reference ... --training-job ...`; it can miss
overwritten updates and does not provide the scheduled pipeline's guarantees.
Do not update or resync live sources. Earlier overwritten checkpoints cannot
be recovered. Reserve holdout for a final frozen policy.

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
