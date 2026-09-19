"""Evaluate frozen checkpoints while a separate SLURM training job runs."""

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Event

from src.benchmarking.artifacts import load_run
from src.benchmarking.config import PROJECT_ROOT, read_json
from src.training.artifacts import file_hash, load_checkpoint, write_json


def snapshot_latest(run_path: Path, output: Path, known_digest: str | None = None) -> dict | None:
    try:
        descriptor = read_json(run_path / "policy.json")
    except FileNotFoundError:
        return None
    digest = descriptor["checkpoint_sha256"]
    if digest == known_digest:
        return None
    snapshots = output / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pending-", dir=snapshots) as temporary:
        staging = Path(temporary)
        try:
            for filename in ("checkpoint.pt", "checkpoint.pt.json"):
                shutil.copyfile(run_path / filename, staging / filename)
        except FileNotFoundError:
            return None
        checkpoint = staging / "checkpoint.pt"
        if (file_hash(checkpoint) != digest
                or read_json(staging / "checkpoint.pt.json")["sha256"] != digest):
            return None
        loaded = load_checkpoint(checkpoint, device="cpu", expected_sha256=digest)
        update = loaded.training_state["next_update"]
        if type(update) is not int or update < 1:
            raise ValueError("Checkpoint must identify a completed training update.")
        destination = (snapshots / f"update-{update:04d}").resolve()
        if destination.exists():
            metadata = read_json(destination / "snapshot.json")
            if metadata["sha256"] != digest:
                raise ValueError(f"Update {update} already has a different checkpoint.")
            return metadata
        descriptor.update(checkpoint=str(destination / "checkpoint.pt"), device="cpu", torch_threads=1)
        metadata = {
            "update": update, "sha256": digest, "path": str(destination),
            "source": str(run_path.resolve()),
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(staging / "policy.json", descriptor)
        write_json(staging / "snapshot.json", metadata)
        staging.rename(destination)
        return metadata


def evaluate_snapshot(snapshot: dict, output: Path, reference: Path) -> dict:
    update = snapshot["update"]
    candidate = output / f"update-{update:04d}"
    comparison = output / f"compare-{update:04d}"
    checkpoint = Path(snapshot["path"])
    environment = {
        **os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1",
        "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
        "PYGAME_HIDE_SUPPORT_PROMPT": "1", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }
    with (output / f"update-{update:04d}.log").open("a", encoding="utf-8") as log:
        if not (candidate / "summary.json").exists():
            subprocess.run([
                sys.executable, "benchmark.py", "run",
                "--policy", "src.policies.runtime:create_policy",
                "--label", f"ppo-gru-update-{update}",
                "--config", str(checkpoint / "policy.json"),
                "--artifact", str(checkpoint / "checkpoint.pt"),
                "--suite", "quick", "--output", str(candidate),
            ], cwd=PROJECT_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not (comparison / "summary.json").exists():
            subprocess.run([
                sys.executable, "benchmark.py", "compare", "--reference", str(reference),
                "--candidate", str(candidate), "--output", str(comparison), "--no-plots",
            ], cwd=PROJECT_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = read_json(candidate / "summary.json")
    report = read_json(comparison / "summary.json")
    paired = report["comparisons"][0]
    return {
        "mean_score": summary["score"]["mean"],
        "mean_survival_seconds": summary["survival_seconds"]["mean"],
        "mean_delta_vs_random": paired["mean_delta"],
        "wins": paired["wins"], "cases": paired["case_count"],
        "report": str(comparison / "report.md"),
    }


def training_active(job_id: int) -> bool:
    result = subprocess.run([
        "squeue", "--noheader", "--user", getpass.getuser(), "--format=%i",
    ], capture_output=True, text=True, check=True, timeout=30)
    return str(job_id) in result.stdout.split()


def monitor(run_path: Path, output: Path, reference: Path, job_id: int,
            poll_seconds: float = 5, scheduler_seconds: float = 30) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    status = read_json(status_path) if status_path.exists() else {
        "training_job": job_id, "training_run": str(run_path), "reference": str(reference),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "updates": {}, "missing_updates": [],
    }
    if (status["training_job"] != job_id or status["reference"] != str(reference)
            or status["training_run"] != str(run_path)):
        raise ValueError("Monitor output belongs to another training run or reference.")
    status["state"] = "watching"
    records = status["updates"]
    pending = {}
    known_digest = None
    last_update = None
    next_scheduler_check = 0.0
    active = True
    wakeup = Event()

    def persist():
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(status_path, status)

    def record_result(future):
        update = pending.pop(future)
        record = records[str(update)]
        try:
            record.update(state="complete", result=future.result())
            print(json.dumps({"event": "evaluated", "update": update, **record["result"]}), flush=True)
        except Exception as exc:
            record.update(state="failed", error=str(exc))
            print(json.dumps({"event": "evaluation_failed", "update": update, "error": str(exc)}), flush=True)
        persist()

    with ThreadPoolExecutor(max_workers=1) as evaluator:
        def enqueue(snapshot):
            update = snapshot["update"]
            record = records.setdefault(str(update), dict(snapshot))
            if record.get("state") not in ("complete", "failed"):
                record["state"] = "queued"
                pending[evaluator.submit(evaluate_snapshot, snapshot, output, reference)] = update

        for metadata_path in sorted((output / "snapshots").glob("update-*/snapshot.json")):
            snapshot = read_json(metadata_path)
            enqueue(snapshot)
            last_update, known_digest = snapshot["update"], snapshot["sha256"]
        while True:
            for future in list(pending):
                if future.done():
                    record_result(future)
            if time.monotonic() >= next_scheduler_check:
                try:
                    active = training_active(job_id)
                    status.pop("scheduler_error", None)
                except (subprocess.SubprocessError, OSError) as exc:
                    status["scheduler_error"] = str(exc)
                next_scheduler_check = time.monotonic() + scheduler_seconds
            snapshot = snapshot_latest(run_path, output, known_digest)
            if snapshot is not None:
                update = snapshot["update"]
                if last_update is not None and update > last_update + 1:
                    missing = list(range(last_update + 1, update))
                    status["missing_updates"].extend(missing)
                    print(json.dumps({"event": "missed_updates", "updates": missing}), flush=True)
                status.setdefault("first_observed_update", update)
                enqueue(snapshot)
                last_update, known_digest = update, snapshot["sha256"]
                print(json.dumps({"event": "captured", "update": update, "sha256": known_digest}), flush=True)
            persist()
            if not active:
                break
            wakeup.wait(poll_seconds)
        status["state"] = "draining"
        persist()
        for future in as_completed(list(pending)):
            record_result(future)
    failed = any(record["state"] == "failed" for record in records.values())
    status["state"] = "finished_with_errors" if failed or status["missing_updates"] or not records else "complete"
    persist()
    return status


def monitor_scheduled(run_path: Path, job_id: int | None, *, poll_seconds: float = 5, once: bool = False) -> dict:
    """User-launched monitor for durable requests; never submits another job."""
    from src.training.evaluation import evaluate_pending, pending_updates, schedule_path
    from src.training.progress import render_progress, progress_data

    while True:
        active = training_active(job_id) if job_id is not None and not once else False
        manifest = run_path / "manifest.json"
        if not schedule_path(run_path).exists():
            if once or not active:
                raise ValueError("Training has not published an evaluation schedule.")
        else:
            evaluate_pending(run_path)
            render_progress(run_path)
            data = progress_data(run_path)
            print(json.dumps({
                "event": "evaluation_progress", "latest_evaluated": data["latest_evaluated_update"],
                "latest_published": data["latest_published_update"],
            }), flush=True)
            status = read_json(manifest)["status"] if manifest.exists() else "running"
            if once or not active or status in ("complete", "failed", "interrupted", "paused"):
                # The trainer can publish its final checkpoint while an earlier
                # snapshot list is being evaluated. Drain that durable tail too.
                if not once:
                    while pending_updates(run_path):
                        evaluate_pending(run_path)
                    render_progress(run_path)
                    data = progress_data(run_path)
                failed = any(point["state"] == "failed" for point in data["points"])
                missing = any(point["state"] != "complete" for point in data["points"])
                return {"state": "finished_with_errors" if failed else "incomplete" if missing else "complete",
                        "training_state": status}
        Event().wait(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--training-job", type=int)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--once", action="store_true", help="Evaluate only published requests, then exit.")
    parser.add_argument("--legacy-latest", action="store_true", help="Explicit opt-in for old rolling-checkpoint runs.")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if not args.legacy_latest:
        if args.output is not None or args.reference is not None:
            parser.error("Scheduled evaluation writes inside the run and uses its pinned teacher reference.")
        if args.training_job is None and not args.once:
            parser.error("Provide --training-job or --once.")
        from src.training.evaluation import evaluation_lock
        run_path = args.run.resolve()
        with evaluation_lock(run_path.parent / ".watcher-locks" / run_path.name):
            status = monitor_scheduled(args.run.resolve(), args.training_job,
                                       poll_seconds=args.poll_seconds, once=args.once)
        return 0 if status["state"] == "complete" else 1
    if args.output is None or args.reference is None or args.training_job is None:
        parser.error("Legacy mode requires --output, --reference and --training-job.")
    import fcntl
    import torch

    torch.set_num_threads(1)
    reference = load_run(args.reference)
    if (reference.manifest.status != "complete" or reference.manifest.suite.name != "quick"
            or reference.manifest.repeats != 1 or reference.manifest.max_steps is not None
            or reference.manifest.fixed_policy_seed is not None):
        raise ValueError("Reference must be a complete, uncapped quick-suite run with default policy seeds.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = monitor(args.run.resolve(), output, args.reference.resolve(), args.training_job, args.poll_seconds)
    return 0 if status["state"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())