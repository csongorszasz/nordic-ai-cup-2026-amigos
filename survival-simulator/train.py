"""Configuration-driven controller search, profiling, imitation, and PPO."""

import argparse
import os
import random
import statistics
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
    return cli


class _ProgressSimulation(SimulationCore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
    from src.training.search import optimize_controller

    worlds = training_seeds(config.seed, config.search.worlds)
    run.emit({"event": "training_worlds", "seeds": worlds, "full_horizon": True})
    evaluation_index = 0

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
        means = []
        for index in range(start, evaluation_index):
            if set(scores[index]) != set(worlds):
                raise ValueError("Controller search requires every world before ranking a candidate.")
            score = statistics.mean(scores[index][seed] for seed in worlds)
            means.append(score)
            print(f"candidate={index} mean_score={score:.6f}", flush=True)
        return means

    result = optimize_controller(
        config.heuristic, config.search, config.seed, lambda options: evaluate_many([options])[0],
        evaluate_many=evaluate_many,
    )
    selected = RuntimeConfig(policy="heuristic", heuristic=result.best_config)
    write_json(run.path / "policy.json", selected.model_dump(mode="json"))
    return {
        "mode": "search", "method": result.method, "trials": len(result.trials),
        "worlds": worlds, "best_training_mean_score": result.best_score,
        "selected_config": selected.model_dump(mode="json"),
        "ranking_scope": "training worlds only; not held-out evidence",
    }


def run_profile(config: ExperimentConfig, run: TrainingRun) -> dict:
    from src.policies.features import encode_step
    from src.policies.heuristic import HeuristicPolicy
    from src.training.env import EnvironmentAdapter
    from src.training.workers import EnvironmentPool

    workers = config.resources.workers
    seeds = iter(training_seeds(config.seed, workers * (config.rollout_steps + 1)))
    policies = [HeuristicPolicy(config.seed + index, config.heuristic) for index in range(workers)]
    started = time.perf_counter()
    with ExitStack() as stack:
        pool = stack.enter_context(EnvironmentPool(
            workers, timeout_seconds=config.resources.worker_timeout_seconds,
        )) if workers > 1 else None
        adapter = EnvironmentAdapter() if workers == 1 else None
        observations = pool.reset([next(seeds) for _ in range(workers)]) if pool else [adapter.reset(next(seeds))]
        initialization = time.perf_counter() - started
        policy_seconds = encoding_seconds = 0.0
        peak_tokens = transitions = episodes = 0
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
            for index, result in enumerate(results):
                if result.terminated:
                    episodes += 1
                    observations[index] = pool.reset_at(index, next(seeds)) if pool else adapter.reset(next(seeds))
        stepping = time.perf_counter() - stepping_started
    result = {
        "mode": "profile", "workers": workers, "transitions": transitions,
        "initialization_seconds": initialization, "collection_seconds": stepping,
        "transitions_per_second": transitions / stepping,
        "policy_seconds": policy_seconds, "encoding_seconds": encoding_seconds,
        "mean_policy_batch_ms": 1000 * policy_seconds / transitions,
        "peak_frame_tokens": peak_tokens, "completed_episodes": episodes,
        "measurement": "unprofiled reference engine, includes any mid-collection resets; not HTTP latency",
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

    device = config.resources.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Choose resources.device=cpu explicitly.")
    torch.set_num_threads(config.resources.torch_threads)
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
    })

    def checkpoint(training_state: dict):
        target = run.path / "checkpoint.pt"
        digest = save_checkpoint(target, network, config, optimizer, training_state)
        descriptor = RuntimeConfig(
            policy="neural", checkpoint=os.path.relpath(target, PROJECT_ROOT),
            checkpoint_sha256=digest, device="cpu",
        )
        write_json(run.path / "policy.json", descriptor.model_dump(mode="json"))

    def emit(event: dict):
        run.emit(event)
        if "update" in event:
            print(f"update={event['update']} event={event.get('event', 'learning')}", flush=True)

    result = run_learning(
        config, network, optimizer, emit=emit, checkpoint=checkpoint,
        start_update=start_update, training_state=state,
        dataset_callback=lambda dataset: write_json_gzip(run.path / "dataset.json.gz", dataset),
    )
    if device == "cuda":
        result["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        result["peak_cuda_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024**2
        if result["peak_cuda_reserved_mb"] > config.resources.max_vram_mb:
            raise RuntimeError("Training exceeded its configured VRAM limit.")
    return {**result, "device": device, "model_ready_for_submission": False,
            "selection_required": "Compare full-horizon native scores and the real HTTP path."}


def execute(config: ExperimentConfig, output: Path, resume: Path | None = None) -> int:
    if resume is not None and config.mode not in ("imitation", "ppo"):
        raise ValueError("--resume is only valid for learning modes.")
    if resume is not None and config.checkpoint is not None:
        raise ValueError("Choose --resume or a warm-start checkpoint, not both.")
    recorded_config = config.model_copy(update={"checkpoint": str(resume.resolve())}) if resume else config
    run = TrainingRun(output, recorded_config)
    try:
        if config.mode == "search":
            result = run_search(config, run)
        elif config.mode == "profile":
            result = run_profile(config, run)
        else:
            result = run_neural(config, run, resume)
        run.finish("complete", result)
    except KeyboardInterrupt:
        run.finish("interrupted", {"error": "Interrupted; completed checkpoint updates are preserved."})
        return 130
    except Exception as exc:
        # Record the training boundary failure and propagate it; never manufacture a result.
        run.finish("failed", {"error_type": type(exc).__name__, "error": str(exc),
                              "traceback": traceback.format_exc()})
        raise
    print(f"Saved {config.mode} run to {run.path}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = load_experiment(args.config, args.set)
        return execute(config, args.output, args.resume)
    except (OSError, ValueError, ImportError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
