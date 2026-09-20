"""Durable evaluation requests, independent of the learner and of SLURM."""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from src.benchmarking.artifacts import RunWriter, build_manifest, load_run
from src.benchmarking.config import (
    PROJECT_ROOT, EpisodeCase, EpisodeResult, PolicySpec, SimulationSettings,
    content_hash, load_suite, make_cases, read_json,
)
from src.benchmarking.comparison import compare_runs
from src.policies.config import ExperimentConfig, RuntimeConfig
from src.training.artifacts import file_hash, save_checkpoint, write_json
from src.training.preflight import scheduled_updates, source_fingerprint
from src.training.teacher import resolve_teacher


class EvaluationBackpressure(RuntimeError):
    """A durable checkpoint exists; collection pauses rather than dropping requests."""


@contextmanager
def evaluation_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".evaluation.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            if not lock.read(1):
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def schedule_path(run: Path) -> Path:
    return run / "evaluation" / "schedule.json"


def load_schedule(run: Path) -> dict:
    schedule = read_json(schedule_path(run))
    payload = {key: value for key, value in schedule.items() if key != "schedule_id"}
    if schedule.get("version") != 1 or schedule.get("schedule_id") != content_hash(payload):
        raise ValueError("Evaluation schedule checksum/version mismatch.")
    return schedule


def initialize_schedule(run: Path, config: ExperimentConfig, manifest: dict, *, lineage: dict | None = None) -> dict:
    teacher = resolve_teacher(config)
    if config.teacher is None:
        raise ValueError("Scheduled evaluation requires a pinned teacher.")
    schedule = {
        "version": 1, "run_id": manifest["run_id"], "mode": config.mode,
        "stage": ("bc-dagger" if config.imitation.dagger_rounds else "bc") if config.mode == "imitation"
        else ("ppo-warm" if config.checkpoint else "ppo-teacher-guided"
              if config.ppo.imitation_weight else "ppo-scratch"),
        "teacher": teacher.provenance, "teacher_policy": teacher.provenance["descriptor"],
        "suite": load_suite(config.evaluation.suite).model_dump(mode="json"),
        "settings": SimulationSettings().model_dump(mode="json"),
        "evaluation": config.evaluation.model_dump(mode="json"),
        "source": source_fingerprint(), "updates": scheduled_updates(config),
        "lineage": lineage, "action_repeat": 1, "inference": "deterministic",
        "episode_budget": config.budget.max_evaluation_episodes if config.budget else 0,
    }
    schedule["protocol_id"] = content_hash({
        key: schedule[key] for key in ("teacher", "suite", "settings", "source", "action_repeat", "inference")
    } | {"repeats": config.evaluation.repeats})
    destination = schedule_path(run)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if lineage is not None and lineage["kind"] == "resume":
        previous_run = Path(lineage["parent_run"])
        previous = load_schedule(previous_run)
        if previous["protocol_id"] != schedule["protocol_id"]:
            raise ValueError("Resume evaluation protocol changed.")
        if pending_updates(previous_run):
            raise EvaluationBackpressure("Drain the previous run's evaluation queue before resuming training.")
        history = previous_run / "evaluation" / "history.json"
        rows = read_json(history) if history.exists() else []
        for ready in sorted((previous_run / "evaluation" / "checkpoints").glob("update-*/ready.json")):
            row = read_json(ready.parent / "result.json")
            if row["update"] <= lineage["resume_update"]:
                rows.append(row)
        write_json(run / "evaluation" / "history.json", rows)
        reference = previous_run / "evaluation" / "teacher" / "result.json"
        if reference.exists():
            # Preserve a portable baseline summary for the historical curve.
            write_json(run / "evaluation" / "historical-teacher.json", read_json(reference))
        schedule["stage"] = previous["stage"]
        schedule["updates"] = [update for update in schedule["updates"] if update > lineage["resume_update"]]
    elif lineage is not None and lineage["kind"] == "fork":
        first = lineage["start_optimizer_step"]
        if type(first) is not int or not 0 <= first <= config.updates:
            raise ValueError("Fork evaluation start must be within the optimizer-step budget.")
        schedule["updates"] = sorted({first, *(update for update in schedule["updates"] if update > first)})
        if lineage.get("data_comparison"):
            arm = lineage["data_comparison"]["arm"]
            if arm not in ("bc-control", "bc-dagger"):
                raise ValueError("Unknown matched-data comparison arm.")
            schedule["stage"] = arm
    schedule["schedule_id"] = content_hash(schedule)
    if destination.exists() and load_schedule(run) != schedule:
        raise ValueError("An existing evaluation schedule cannot be replaced.")
    write_json(destination, schedule)
    return schedule


def snapshot_directory(run: Path, update: int) -> Path:
    return run / "evaluation" / "checkpoints" / f"update-{update:08d}"


def pending_updates(run: Path) -> list[int]:
    pending = []
    for path in sorted((run / "evaluation" / "checkpoints").glob("update-*/ready.json")):
        result = path.parent / "result.json"
        if not result.exists() or read_json(result).get("state") not in ("complete", "failed"):
            pending.append(read_json(path)["update"])
    return pending


def check_capacity(run: Path, config: ExperimentConfig) -> None:
    if len(pending_updates(run)) >= config.evaluation.max_pending:
        raise EvaluationBackpressure(
            "Evaluation queue is full; the latest training checkpoint is preserved. "
            "Run the user-launched evaluator, then resume with a reviewed run plan."
        )


def publish_snapshot(run: Path, network, config: ExperimentConfig, state: dict, schedule: dict) -> dict | None:
    update = state["next_update"]
    for name in ("next_update", "native_ticks_total", "prior_native_ticks", "optimizer_steps"):
        if type(state.get(name, 0)) is not int or state.get(name, 0) < 0:
            raise ValueError(f"Snapshot {name} must be a nonnegative integer.")
    if update not in schedule["updates"]:
        return None
    parent = snapshot_directory(run, update)
    weights_hash = hashlib.sha256()
    for name, value in sorted(network.state_dict().items()):
        weights_hash.update(name.encode())
        weights_hash.update(str((value.dtype, tuple(value.shape))).encode())
        weights_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    model_digest = weights_hash.hexdigest()
    if parent.exists():
        record = read_json(parent / "ready.json")
        if (record["update"] != update or record["protocol_id"] != schedule["protocol_id"]
                or record["model_sha256"] != model_digest
                or record["native_ticks"] != state.get("native_ticks_total", 0) + state.get("prior_native_ticks", 0)
                or record["optimizer_steps"] != state.get("optimizer_steps", 0)):
            raise ValueError("Conflicting evaluation snapshot.")
        return record
    parent.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pending-", dir=parent.parent) as temporary:
        staging = Path(temporary)
        slim_state = {key: state.get(key, 0) for key in (
            "next_update", "optimizer_steps", "native_ticks_total", "prior_native_ticks",
        )}
        digest = save_checkpoint(staging / "checkpoint.pt", network, config,
                                 training_state=slim_state, inference_only=True)
        size = sum(path.stat().st_size for path in staging.iterdir())
        if size > config.evaluation.max_snapshot_mb * 1024**2:
            raise ValueError("Inference snapshot exceeds its declared size budget.")
        used = sum(path.stat().st_size for path in parent.parent.rglob("*") if path.is_file())
        if used > config.evaluation.max_storage_mb * 1024**2:
            raise EvaluationBackpressure("Evaluation snapshot storage budget exceeded; checkpoint preserved.")
        record = {
            "version": 1, "update": update, "sha256": digest, "protocol_id": schedule["protocol_id"],
            "model_sha256": model_digest,
            "run_id": schedule["run_id"], "stage": schedule["stage"],
            "native_ticks": state.get("native_ticks_total", 0) + state.get("prior_native_ticks", 0),
            "optimizer_steps": state.get("optimizer_steps", 0), "size_bytes": size,
        }
        write_json(staging / "ready.json", record)
        actual_size = sum(path.stat().st_size for path in staging.iterdir())
        if actual_size > config.evaluation.max_snapshot_mb * 1024**2:
            raise ValueError("Inference snapshot and ready metadata exceed the declared size budget.")
        # Directory publication makes the checkpoint, checksum and ready marker visible together.
        staging.rename(parent)
    return record


def _execute_episode(case: EpisodeCase, descriptor: dict, output: Path, timeout: float) -> EpisodeResult:
    request = output.with_suffix(".request.json")
    write_json(request, {"case": case.model_dump(mode="json"), "policy": descriptor})
    environment = {
        **os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYGAME_HIDE_SUPPORT_PROMPT": "1",
        "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
    }
    with output.with_suffix(".log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "src.training.evaluation", "episode", "--request", str(request),
             "--result", str(output)], cwd=PROJECT_ROOT, env=environment,
            stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if code and not output.exists():
        raise RuntimeError(f"Episode worker exited with code {code}; see {output.with_suffix('.log')}.")
    return EpisodeResult.model_validate(read_json(output))


def _case_results(directory: Path, cases, descriptor, schedule, executor) -> list[EpisodeResult]:
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    for case in cases:
        output = directory / f"{case.case_id}.json"
        attempted = directory / f"{case.case_id}.attempt.json"
        if output.exists():
            result = EpisodeResult.model_validate(read_json(output))
        else:
            if attempted.exists():
                raise RuntimeError(f"Interrupted episode attempt: {case.case_id}; explicit retry is required.")
            write_json(attempted, {"case": case.model_dump(mode="json"), "state": "started"})
            result = executor(case, descriptor, output, schedule["evaluation"]["episode_timeout_seconds"])
            write_json(output, result.model_dump(mode="json"))
        if result.case != case or result.status != "ok":
            raise ValueError(f"Cannot aggregate failed, truncated or mismatched case {case.case_id}.")
        results.append(result)
    return results


def _benchmark(directory: Path, cases, results, descriptor, schedule) -> Path:
    from src.benchmarking.config import Suite

    destination = directory / "benchmark"
    if destination.exists():
        saved = load_run(destination)
        if saved.manifest.status == "complete":
            if saved.manifest.policy.config != descriptor or list(saved.episodes) != list(results):
                raise ValueError("Saved benchmark does not match the immutable policy and episode records.")
            return destination
        # A crashed aggregation is not a second simulator attempt. Preserve it for diagnosis.
        index = 1
        while directory.joinpath(f"aggregation-interrupted-{index}").exists():
            index += 1
        destination.rename(directory / f"aggregation-interrupted-{index}")
    manifest = build_manifest(
        PROJECT_ROOT, Suite.model_validate(schedule["suite"]),
        PolicySpec(reference="src.policies.runtime:create_policy",
                   label="teacher" if descriptor["policy"] == "heuristic" else schedule["stage"],
                   config=descriptor), repeats=schedule["evaluation"]["repeats"],
        policy_sources=[PROJECT_ROOT / "src" / "policies" / "runtime.py"],
        extra_artifacts=[Path(descriptor["checkpoint"])] if descriptor.get("checkpoint") else (),
    )
    writer = RunWriter(destination, manifest)
    for result in results:
        writer.append(result)
    writer.finalize("complete")
    return destination


def evaluate_pending(run: Path, *, executor=None, refresh: bool = True) -> dict:
    """Called only by a user-launched evaluator. Default executor starts CPU episodes."""
    from src.benchmarking.config import Suite

    run = Path(run).resolve(strict=True)
    executor = executor or _execute_episode
    schedule = load_schedule(run)
    if schedule["source"] != source_fingerprint():
        raise ValueError("Evaluator code differs from the frozen training sources.")
    if refresh:
        from src.benchmarking.plots import require_plotting
        require_plotting()
    cases = make_cases(Suite.model_validate(schedule["suite"]), schedule["evaluation"]["repeats"])
    if (len(schedule["updates"]) + 1) * len(cases) * schedule["evaluation"]["max_attempts"] > schedule["episode_budget"]:
        raise ValueError("Evaluation schedule exceeds the authorized episode budget.")
    completed = []
    with evaluation_lock(run / "evaluation"):
        reference_dir = run / "evaluation" / "teacher"
        reference_result = reference_dir / "result.json"
        if reference_result.exists() and read_json(reference_result).get("state") == "failed":
            raise ValueError("Teacher evaluation failed; inspect it and explicitly retry before continuing.")
        try:
            reference = _case_results(reference_dir, cases, schedule["teacher_policy"], schedule, executor)
            reference_path = _benchmark(reference_dir, cases, reference, schedule["teacher_policy"], schedule)
            write_json(reference_result, {"state": "complete", "protocol_id": schedule["protocol_id"],
                                         "scores": [value.score for value in reference]})
        except BaseException as exc:
            write_json(reference_result, {"state": "failed", "error": str(exc),
                                         "protocol_id": schedule["protocol_id"]})
            if refresh:
                from src.training.progress import render_progress
                render_progress(run)
            raise
        for ready in sorted((run / "evaluation" / "checkpoints").glob("update-*/ready.json")):
            record = read_json(ready)
            destination = ready.parent
            result_path = destination / "result.json"
            if result_path.exists() and read_json(result_path).get("state") in ("complete", "failed"):
                continue
            try:
                if record["update"] not in schedule["updates"] or record["protocol_id"] != schedule["protocol_id"]:
                    raise ValueError("Snapshot update/protocol does not match the authorized schedule.")
                weights = destination / "checkpoint.pt"
                if file_hash(weights) != record["sha256"]:
                    raise ValueError("Snapshot checksum mismatch.")
                descriptor = RuntimeConfig(policy="neural", checkpoint=str(weights.resolve()),
                                           checkpoint_sha256=record["sha256"]).model_dump(mode="json")
                results = _case_results(destination / "episodes", cases, descriptor, schedule, executor)
                benchmark = _benchmark(destination, cases, results, descriptor, schedule)
                report = compare_runs(load_run(reference_path), [load_run(benchmark)])
                result = {
                    **record, "state": "complete", "scores": [value.score for value in results],
                    "world_seeds": [case.world_seed for case in cases],
                    "mean_score": sum(value.score for value in results) / len(results),
                    "teacher_mean": sum(value.score for value in reference) / len(reference),
                    "paired": report["comparisons"][0],
                    "benchmark": str(benchmark.relative_to(run)),
                }
                write_json(result_path, result)
                completed.append(record["update"])
            except BaseException as exc:
                write_json(result_path, {**record, "state": "failed", "error": str(exc)})
                print(f"Evaluation update {record['update']} failed: {exc}", file=sys.stderr)
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
            finally:
                if refresh:
                    from src.training.progress import render_progress
                    render_progress(run)
    return {"completed_updates": completed, "pending": pending_updates(run)}


def retry_failed(run: Path, update: str, reason: str) -> None:
    """Explicit user retry within the already reviewed per-episode attempt cap."""
    if not reason.strip():
        raise ValueError("An explicit retry reason is required.")
    run = Path(run).resolve(strict=True)
    schedule = load_schedule(run)
    directory = run / "evaluation" / "teacher" if update == "teacher" else snapshot_directory(run, int(update))
    with evaluation_lock(run / "evaluation"):
        state_path = directory / "result.json"
        state = read_json(state_path)
        if state["state"] != "failed":
            raise ValueError("Only a failed evaluation can be explicitly retried.")
        episodes = directory if update == "teacher" else directory / "episodes"
        moves = []
        from src.benchmarking.config import Suite
        for case in make_cases(Suite.model_validate(schedule["suite"]), schedule["evaluation"]["repeats"]):
            attempted = episodes / f"{case.case_id}.attempt.json"
            output = episodes / f"{case.case_id}.json"
            if output.exists():
                result = EpisodeResult.model_validate(read_json(output))
                if result.case == case and result.status == "ok":
                    continue
            if not attempted.exists():
                continue
            number = len(list((episodes / "history").glob(f"{case.case_id}-attempt-*.json"))) + 1
            if number >= schedule["evaluation"]["max_attempts"]:
                raise ValueError("Retry exceeds the reviewed max_attempts; create a separately reviewed evaluation plan.")
            moves.append((attempted, output, number))
        if not moves:
            raise ValueError("No retryable episode attempts; resolve the recorded protocol/artifact error separately.")
        history = episodes / "history"
        history.mkdir(exist_ok=True)
        for attempted, output, number in moves:
            case_name = attempted.name.removesuffix(".attempt.json")
            attempted.rename(history / f"{case_name}-attempt-{number}.json")
            if output.exists():
                output.rename(history / f"{case_name}-result-{number}.json")
        archived = directory / "failed-evaluations"
        archived.mkdir(exist_ok=True)
        write_json(archived / f"{len(list(archived.glob('*.json'))):04d}.json", {**state, "retry_reason": reason})
        write_json(state_path, {**state, "state": "queued", "retry_reason": reason})


def _episode(request: Path, result: Path) -> None:
    from src.benchmarking.runner import EpisodeExecutionError, run_episode
    from src.policies.runtime import create_policy

    payload = read_json(request)
    try:
        outcome = run_episode(EpisodeCase.model_validate(payload["case"]), create_policy, payload["policy"])
    except EpisodeExecutionError as exc:
        write_json(result, exc.result.model_dump(mode="json"))
        raise
    write_json(result, outcome.model_dump(mode="json"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    once = commands.add_parser("once", help="User-launched evaluation of durable scheduled checkpoints.")
    once.add_argument("--run", type=Path, required=True)
    plot = commands.add_parser("plot", help="Regenerate plots from saved records; never runs simulations.")
    plot.add_argument("--run", type=Path, required=True)
    retry = commands.add_parser("retry", help="Explicitly retry failed cases within the planned attempt cap.")
    retry.add_argument("--run", type=Path, required=True)
    retry.add_argument("--update", required=True, help="Update number or teacher.")
    retry.add_argument("--reason", required=True)
    episode = commands.add_parser("episode", help=argparse.SUPPRESS)
    episode.add_argument("--request", type=Path, required=True)
    episode.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "episode":
        _episode(args.request, args.result)
    elif args.command == "plot":
        from src.training.progress import render_progress
        render_progress(args.run)
    elif args.command == "retry":
        retry_failed(args.run, args.update, args.reason)
    else:
        evaluate_pending(args.run)
        failures = list((args.run / "evaluation").rglob("result.json"))
        return int(any(read_json(path).get("state") == "failed" for path in failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
