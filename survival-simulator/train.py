"""Configuration-driven controller search, profiling, imitation, and PPO."""

import argparse
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from src.benchmarking.config import PROJECT_ROOT, EpisodeCase
from src.benchmarking.runner import run_episode
from src.core import SimulationCore
from src.policies.config import ExperimentConfig, RuntimeConfig, load_experiment
from src.training.artifacts import TrainingRun, write_json, write_json_gzip
from src.training.seeds import training_seeds


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, required=True, help="Versioned experiment JSON.")
    cli.add_argument("--output", type=Path, required=True, help="New run directory; never overwritten.")
    cli.add_argument("--set", action="append", default=[], help="Typed dotted override: model.memory=none")
    cli.add_argument("--resume", type=Path, help="Resume optimizer/RNG/update state from a checkpoint.")
    cli.add_argument("--preflight", action="store_true", help="Write a run plan without simulation or training.")
    cli.add_argument("--evidence", type=Path, help="Hyperparameter rationale/evidence JSON for preflight.")
    cli.add_argument("--run-plan", type=Path, help="Previously reviewed preflight JSON, required for learning.")
    return cli


class _ProgressSimulation(SimulationCore):
    def __init__(self, **kwargs):
        super().__init__(rendering=False, **kwargs)
        self._progress_ticks = 0

    def step(self, actions):
        state = super().step(actions)
        self._progress_ticks += 1
        if self._progress_ticks % 1000 == 0:
            print(
                f"world={self.seed} ticks={self._progress_ticks} "
                f"time={state['sim_time']:.1f} agents={state['num_agents']} score={state['score']:.2f}",
                flush=True,
            )
        return state


def _search_episode(arguments):
    from src.policies.heuristic import create_policy

    world_seed, options = arguments
    print(f"world={world_seed} initializing", flush=True)
    case = EpisodeCase(case_id=f"world-{world_seed}-repeat-0", world_seed=world_seed,
                       repeat_index=0, policy_seed=world_seed)
    return run_episode(case, create_policy, options, simulation_factory=_ProgressSimulation)


def run_search(config: ExperimentConfig, run: TrainingRun) -> dict:
    if config.search.objective == "survival":
        return run_survival_search(config, run)
    from src.training.search import aggregate_world_scores, optimize_controller

    worlds = training_seeds(config.seed, config.search.worlds)
    run.emit({"event": "training_worlds", "seeds": worlds, "full_horizon": True})
    evaluation_index = 0
    candidate_metrics = {}

    def evaluate_many(configurations):
        nonlocal evaluation_index
        start = evaluation_index
        evaluation_index += len(configurations)
        jobs = [
            (start + offset, seed, options.model_dump(mode="json"))
            for offset, options in enumerate(configurations) for seed in worlds
        ]
        scores = {index: {} for index in range(start, evaluation_index)}

        def record(job, result):
            index, seed, options = job
            if result.score is None or result.case.world_seed != seed:
                raise ValueError("Controller search cannot rank missing or mismatched episode scores.")
            scores[index][seed] = result.score
            run.emit({"event": "search_episode", "candidate": index,
                      "config": options, "result": result.model_dump(mode="json")})
            print(f"candidate={index} world={seed} score={result.score:.6f}", flush=True)

        run.emit({"event": "search_batch", "candidate_start": start,
                  "candidates": len(configurations), "episodes": len(jobs),
                  "workers": config.resources.workers})
        if config.resources.workers == 1:
            for job in jobs:
                record(job, _search_episode(job[1:]))
        else:
            from src.training.workers import _cpu_spawn_environment

            with ExitStack() as stack:
                with _cpu_spawn_environment():
                    pool = stack.enter_context(ProcessPoolExecutor(
                        max_workers=min(config.resources.workers, len(jobs)),
                    ))
                    pending = {pool.submit(_search_episode, job[1:]): job for job in jobs}
                for future in as_completed(pending):
                    job = pending[future]
                    try:
                        record(job, future.result())
                    except Exception as exc:
                        exc.add_note(f"Controller search candidate {job[0]}, world {job[1]}.")
                        for queued in pending:
                            queued.cancel()
                        raise
        objectives = []
        for index in range(start, evaluation_index):
            if set(scores[index]) != set(worlds):
                raise ValueError("Controller search requires every world before ranking a candidate.")
            metrics = aggregate_world_scores(
                [scores[index][seed] for seed in worlds],
                lower_tail_fraction=config.search.lower_tail_fraction,
                lower_tail_weight=config.search.lower_tail_weight,
            )
            candidate_metrics[index] = metrics
            objectives.append(metrics.objective)
            run.emit({
                "event": "search_candidate_summary", "candidate": index,
                "mean_score": metrics.mean, "lower_tail_score": metrics.lower_tail,
                "objective": metrics.objective,
            })
            print(
                f"candidate={index} mean_score={metrics.mean:.6f} "
                f"lower_tail={metrics.lower_tail:.6f} objective={metrics.objective:.6f}",
                flush=True,
            )
        return objectives

    result = optimize_controller(
        config.heuristic, config.search, config.seed, lambda options: evaluate_many([options])[0],
        evaluate_many=evaluate_many,
    )
    selected = RuntimeConfig(policy="heuristic", heuristic=result.best_config)
    write_json(run.path / "policy.json", selected.model_dump(mode="json"))
    best_trial = max(result.trials, key=lambda trial: (trial.score, -trial.index))
    best_metrics = candidate_metrics[best_trial.index]
    return {
        "mode": "search", "method": result.method, "trials": len(result.trials),
        "worlds": worlds, "best_training_objective": result.best_score,
        "best_training_mean_score": best_metrics.mean,
        "best_training_lower_tail_score": best_metrics.lower_tail,
        "objective": {
            "lower_tail_fraction": config.search.lower_tail_fraction,
            "lower_tail_weight": config.search.lower_tail_weight,
        },
        "selected_config": selected.model_dump(mode="json"),
        "ranking_scope": "training worlds only; not held-out evidence",
    }


def _survival_search_episode(job):
    from src.benchmarking.telemetry import EpisodeTelemetry
    from src.policies.heuristic import create_policy

    index, world_seed, options, policy_seed, root, sample_every, max_trace_mb = job
    case = EpisodeCase(
        case_id=f"world-{world_seed}-repeat-0", world_seed=world_seed, repeat_index=0,
        policy_seed=world_seed if policy_seed is None else policy_seed,
        policy_seed_mode="derived" if policy_seed is None else "fixed",
    )
    telemetry = EpisodeTelemetry(
        Path(root), case, candidate=index, sample_every=sample_every, max_trace_mb=max_trace_mb,
    )
    return run_episode(
        case, create_policy, options, simulation_factory=_ProgressSimulation, telemetry=telemetry,
    )


def run_survival_search(config: ExperimentConfig, run: TrainingRun) -> dict:
    from src.benchmarking.config import read_json
    from src.benchmarking.jobs import JobExecutionError, run_jobs
    from src.benchmarking.progress import render_progress
    from src.benchmarking.survival import OBJECTIVE_VERSION, survival_metrics
    from src.training.search import optimize_controller

    if config.search.progress:
        from src.benchmarking.plots import require_plotting

        require_plotting()
    worlds = training_seeds(config.seed, config.search.worlds)
    write_json(run.path / "search-worlds.json", {"version": 1, "name": "search", "seeds": worlds})
    run.emit({
        "event": "training_worlds", "seeds": worlds, "full_horizon": True,
        "objective_version": OBJECTIVE_VERSION,
        "max_episodes": config.search.candidates * len(worlds),
        "max_native_ticks": config.search.candidates * len(worlds) * 30001,
        "max_trace_bytes": config.search.candidates * len(worlds) * config.search.max_trace_mb * 1024**2,
    })
    index = 0
    metrics_by_candidate = {}
    last_refresh = 0.0

    def refresh():
        nonlocal last_refresh
        if config.search.progress and time.monotonic() - last_refresh >= 10.0:
            render_progress(run.path)
            last_refresh = time.monotonic()

    def evaluate_many(configurations):
        nonlocal index
        start = index
        index += len(configurations)
        jobs = [
            (start + offset, seed, options.model_dump(mode="json"),
             config.search.fixed_policy_seed, str(run.path), config.search.sample_every,
             config.search.max_trace_mb)
            for offset, options in enumerate(configurations) for seed in worlds
        ]
        records = {candidate: {} for candidate in range(start, index)}
        run.emit({
            "event": "search_batch", "candidate_start": start, "candidates": len(configurations),
            "episodes": len(jobs), "workers": config.resources.workers,
        })
        try:
            for job, result in run_jobs(
                jobs, _survival_search_episode, workers=config.resources.workers,
                timeout_seconds=config.search.episode_timeout_seconds, progress=refresh,
            ):
                candidate, seed, options = job[:3]
                run.emit({
                    "event": "search_episode", "candidate": candidate, "config": options,
                    "result": result.model_dump(mode="json"),
                })
                if result.case.world_seed != seed or result.status != "ok":
                    raise ValueError("Survival search received an invalid episode identity or outcome.")
                records[candidate][seed] = result
                print(
                    f"candidate={candidate} world={seed} survival={result.survival_seconds:.1f} "
                    f"completed={result.completed} score={result.score:.3f}", flush=True,
                )
        except JobExecutionError as error:
            run.emit({
                "event": "search_worker_failed", "candidate": error.job[0], "world_seed": error.job[1],
                "error": str(error),
                "result": error.result.model_dump(mode="json") if error.result is not None else None,
            })
            raise
        objectives = []
        expected = [f"world-{seed}-repeat-0" for seed in worlds]
        for candidate in range(start, index):
            health = {}
            for seed, record in records[candidate].items():
                path = run.path / "telemetry" / f"candidate-{candidate:05d}" / record.case.case_id / "summary.json"
                summary = read_json(path)
                if summary["result"] != record.model_dump(mode="json"):
                    raise ValueError("Search telemetry disagrees with its completed episode.")
                sample = summary["last_sample"]
                health[record.case.case_id] = (sample["young_reproductive"], sample["energy_p10"] or 0.0)
            metrics = survival_metrics(
                list(records[candidate].values()), expected_case_ids=expected,
                tail_fraction=config.search.lower_tail_fraction, terminal_health=health,
            )
            metrics_by_candidate[candidate] = metrics
            objectives.append(tuple(metrics["rank"]))
            run.emit({"event": "search_candidate_summary", "candidate": candidate, **metrics})
            print(
                f"candidate={candidate} completed={metrics['completed_worlds']}/{metrics['worlds']} "
                f"tail={metrics['lower_tail_seconds']:.1f} worst={metrics['worst_seconds']:.1f}",
                flush=True,
            )
        refresh()
        return objectives

    result = optimize_controller(
        config.heuristic, config.search, config.seed, lambda options: evaluate_many([options])[0],
        evaluate_many=evaluate_many,
    )
    selected = RuntimeConfig(policy="heuristic", heuristic=result.best_config)
    write_json(run.path / "policy.json", selected.model_dump(mode="json"))
    write_json(run.path / "survival-summary.json", {
        "objective_version": OBJECTIVE_VERSION, "candidates": metrics_by_candidate,
    })
    return {
        "mode": "search", "method": result.method, "trials": len(result.trials),
        "worlds": worlds, "best_survival_rank": result.best_score,
        "selected_config": selected.model_dump(mode="json"),
        "ranking_scope": "Fixed search worlds only; not held-out or infinite-horizon evidence.",
        "objective_version": OBJECTIVE_VERSION,
    }


def run_profile(config: ExperimentConfig, run: TrainingRun) -> dict:
    from src.policies.features import encode_step
    from src.policies.heuristic import build_policy
    from src.training.env import EnvironmentAdapter
    from src.training.workers import EnvironmentPool

    workers = config.resources.workers
    seeds = iter(training_seeds(config.seed, workers * (config.rollout_steps + 1)))
    policies = [build_policy(config.seed + index, config.heuristic) for index in range(workers)]
    started = time.perf_counter()
    with ExitStack() as stack:
        pool_kwargs = {"timeout_seconds": config.resources.worker_timeout_seconds}
        if config.resources.action_repeat != 1:
            pool_kwargs["action_repeat"] = config.resources.action_repeat
        pool = stack.enter_context(EnvironmentPool(
            workers, **pool_kwargs,
        )) if workers > 1 else None
        adapter = (
            EnvironmentAdapter()
            if workers == 1 and config.resources.action_repeat == 1
            else EnvironmentAdapter(action_repeat=config.resources.action_repeat)
            if workers == 1 else None
        )
        observations = pool.reset([next(seeds) for _ in range(workers)]) if pool else [adapter.reset(next(seeds))]
        initialization = time.perf_counter() - started
        policy_seconds = encoding_seconds = 0.0
        peak_tokens = transitions = native_transitions = episodes = 0
        stepping_started = time.perf_counter()
        for _ in range(config.rollout_steps):
            t0 = time.perf_counter()
            features = [encode_step(observation) for observation in observations]
            encoding_seconds += time.perf_counter() - t0
            peak_tokens = max(peak_tokens, *(len(batch.agent_ids) + len(batch.entities) for batch in features))
            t0 = time.perf_counter()
            actions = [policy.act(observation) for policy, observation in zip(policies, observations)]
            policy_seconds += time.perf_counter() - t0
            results = pool.step(actions) if pool else [adapter.step(actions[0])]
            observations = [result.observation for result in results]
            transitions += workers
            native_transitions += sum(result.native_ticks for result in results)
            for index, result in enumerate(results):
                if result.terminated:
                    episodes += 1
                    observations[index] = pool.reset_at(index, next(seeds)) if pool else adapter.reset(next(seeds))
                    reset = getattr(policies[index], "reset", None)
                    if callable(reset):
                        reset()
        stepping = time.perf_counter() - stepping_started
    result = {
        "mode": "profile", "workers": workers, "transitions": transitions,
        "action_repeat": config.resources.action_repeat,
        "native_transitions": native_transitions,
        "initialization_seconds": initialization, "collection_seconds": stepping,
        "transitions_per_second": transitions / stepping,
        "native_transitions_per_second": (
            native_transitions / stepping
        ),
        "policy_seconds": policy_seconds, "encoding_seconds": encoding_seconds,
        "mean_policy_batch_ms": 1000 * policy_seconds / transitions,
        "peak_frame_tokens": peak_tokens, "completed_episodes": episodes,
        "measurement": (
            "RNG-equivalent headless training core, includes any mid-collection "
            "resets; not reference-benchmark or HTTP latency"
        ),
    }
    run.emit({"event": "profile", **result})
    return result


def run_neural(config: ExperimentConfig, run: TrainingRun, resume: Path | None = None) -> dict:
    import torch
    from src.policies.networks import PolicyNetwork
    from src.training.artifacts import (
        load_checkpoint, restore_rng, save_checkpoint, verify_resume_provenance,
    )
    from src.training.learner import run_learning
    from src.training.evaluation import initialize_schedule, publish_snapshot, check_capacity

    device = config.resources.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Choose resources.device=cpu explicitly.")
    torch.set_num_threads(config.resources.torch_threads)
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    if config.resources.workers + config.resources.torch_threads > allocated_cpus:
        raise ValueError("Reviewed environment workers plus learner threads exceed the CPU allocation.")
    if config.budget and "SLURM_MEM_PER_NODE" in os.environ:
        allocated_memory = int(os.environ["SLURM_MEM_PER_NODE"])
        if allocated_memory and config.budget.host_memory_mb > allocated_memory:
            raise ValueError("The reviewed host-memory budget exceeds the SLURM allocation.")
    if device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        limit = config.resources.max_vram_mb * 1024**2
        if limit > total:
            raise ValueError("Configured VRAM limit exceeds the available device capacity.")
        torch.cuda.set_per_process_memory_fraction(limit / total)
        torch.cuda.reset_peak_memory_stats()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    source = resume or (Path(config.checkpoint) if config.checkpoint else None)
    if resume is not None:
        verify_resume_provenance(resume, run.manifest)
    loaded = load_checkpoint(source, expected_model=config.model, device=device) if source else None
    network = loaded.network if loaded else PolicyNetwork(config.model).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.optimizer.learning_rate)
    state = None
    start_update = 0
    prior_native_ticks = 0
    lineage = None
    if loaded is not None:
        recorded_ticks = loaded.training_state.get(
            "native_ticks_total", loaded.training_state.get("collector", {}).get("native_tick_count"),
        )
        if recorded_ticks is None:
            raise ValueError("Checkpoint lacks actual native-step accounting; cannot claim matched-compute lineage.")
        prior_native_ticks = int(recorded_ticks) + loaded.training_state.get("prior_native_ticks", 0)
        lineage = {"kind": "resume" if resume else "warm_start", "checkpoint_sha256": loaded.sha256,
                   "prior_native_ticks": prior_native_ticks,
                   "resume_update": loaded.training_state.get("next_update", 0),
                   "parent_run": str(source.resolve().parent)}
    if resume is not None:
        if loaded.optimizer_state is None:
            raise ValueError("The checkpoint has no optimizer state for resume.")
        optimizer.load_state_dict(loaded.optimizer_state)
        for group in optimizer.param_groups:
            group["lr"] = config.optimizer.learning_rate
        restore_rng(loaded.rng_state)
        state = loaded.training_state
        start_update = state.get("next_update", 0)
        if loaded.config.mode != config.mode:
            raise ValueError("Use checkpoint warm start, not --resume, when changing training modes.")
    if start_update >= config.updates:
        raise ValueError("The target update count is already complete; increase updates to resume.")
    run.emit({
        "event": "learner_start", "device": device, "parameters": sum(p.numel() for p in network.parameters()),
        "start_update": start_update, "source": str(source) if source else None,
        "resume": resume is not None, "worlds_restart_on_resume": resume is not None,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "gpu_total_mb": torch.cuda.get_device_properties(0).total_memory // 1024**2 if device == "cuda" else None,
    })

    def checkpoint(training_state: dict):
        # Publish due inference weights first: a durable resume checkpoint must
        # never refer to a scheduled update whose weights were already lost.
        if schedule is not None:
            publish_snapshot(run.path, network, config, training_state, schedule)
        target = run.path / "checkpoint.pt"
        digest = save_checkpoint(target, network, config, optimizer, training_state)
        descriptor = RuntimeConfig(
            policy="neural", checkpoint=os.path.relpath(target, PROJECT_ROOT),
            checkpoint_sha256=digest, device="cpu",
        )
        write_json(run.path / "policy.json", descriptor.model_dump(mode="json"))

    schedule = None
    if config.evaluation.enabled:
        schedule = initialize_schedule(run.path, config, run.manifest, lineage=lineage)
        initial = {
            "next_update": start_update,
            "native_ticks_total": state.get("native_ticks_total", 0) if state else 0,
            "prior_native_ticks": state.get("prior_native_ticks", 0) if state else prior_native_ticks,
            "optimizer_steps": state.get("optimizer_steps", 0) if state else 0,
        }
        if start_update == 0:
            publish_snapshot(run.path, network, config, initial, schedule)
        check_capacity(run.path, config)

    def update_callback(training_state: dict):
        if schedule is not None:
            if training_state["next_update"] < config.updates:
                check_capacity(run.path, config)

    def emit(event: dict):
        run.emit(event)
        if "update" in event:
            print(f"update={event['update']} event={event.get('event', 'learning')}", flush=True)

    result = run_learning(
        config, network, optimizer, emit=emit, checkpoint=checkpoint,
        start_update=start_update, training_state=state,
        dataset_callback=lambda dataset: write_json_gzip(run.path / "dataset.json.gz", dataset),
        update_callback=update_callback, prior_native_ticks=prior_native_ticks,
    )
    if device == "cuda":
        result["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        result["peak_cuda_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024**2
        if result["peak_cuda_reserved_mb"] > config.resources.max_vram_mb:
            raise RuntimeError("Training exceeded its configured VRAM limit.")
    return {**result, "device": device, "model_ready_for_submission": False,
            "selection_required": "Compare full-horizon native scores and the real HTTP path."}


def execute(
    config: ExperimentConfig, output: Path, resume: Path | None = None, *, run_plan: Path | None = None,
) -> int:
    if resume is not None and config.mode not in ("imitation", "ppo"):
        raise ValueError("--resume is only valid for learning modes.")
    if resume is not None and config.checkpoint is not None:
        raise ValueError("Choose --resume or a warm-start checkpoint, not both.")
    plan = None
    if config.mode in ("imitation", "ppo"):
        from src.training.preflight import verify_run_plan
        if run_plan is None:
            raise ValueError("Learning requires --run-plan from a reviewed --preflight; no run was started.")
        plan = verify_run_plan(run_plan, config, resume=resume)
    recorded_config = config.model_copy(update={"checkpoint": str(resume.resolve())}) if resume else config
    run = TrainingRun(output, recorded_config)
    if plan is not None:
        write_json(run.path / "run-plan.json", plan)
    try:
        if config.mode == "search":
            result = run_search(config, run)
        elif config.mode == "profile":
            result = run_profile(config, run)
        else:
            result = run_neural(config, run, resume)
        run.finish("complete", result)
        if config.mode == "search" and config.search.objective == "survival" and config.search.progress:
            from src.benchmarking.progress import render_progress

            render_progress(run.path)
    except KeyboardInterrupt:
        run.finish("interrupted", {"error": "Interrupted; completed checkpoint updates are preserved."})
        return 130
    except Exception as exc:
        if config.mode in ("imitation", "ppo"):
            from src.training.evaluation import EvaluationBackpressure
            if isinstance(exc, EvaluationBackpressure):
                run.finish("paused", {"reason": str(exc), "resume_requires_user_launch": True})
                print(str(exc), file=sys.stderr)
                return 3
        # Record the training boundary failure and propagate it; never manufacture a result.
        if run.manifest["status"] == "running":
            run.finish("failed", {"error_type": type(exc).__name__, "error": str(exc),
                                  "traceback": traceback.format_exc()})
        raise
    print(f"Saved {config.mode} run to {run.path}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = load_experiment(args.config, args.set)
        if args.preflight:
            from src.training.preflight import build_run_plan
            from src.benchmarking.config import read_json
            if args.evidence is None or args.run_plan is not None:
                raise ValueError("--preflight requires --evidence and cannot launch --run-plan.")
            plan = build_run_plan(config, read_json(args.evidence), resume=args.resume)
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / "run-plan.json", plan)
            write_json(args.output / "resolved-config.json", config.model_dump(mode="json"))
            print(f"Plan {plan['plan_id']}: {plan['accounting']}. Nothing was launched.")
            return 0
        if args.evidence is not None:
            raise ValueError("--evidence belongs to --preflight, not a launch.")
        return execute(config, args.output, args.resume, run_plan=args.run_plan)
    except (OSError, ValueError, ImportError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
