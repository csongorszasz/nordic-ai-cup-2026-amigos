"""Deterministic, whole-team DAgger collection with immutable expert retention."""

import argparse
import copy
import gzip
import json
import multiprocessing
import random
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from src.benchmarking.config import SimulationSettings, content_hash, read_json
from src.benchmarking.policies import validate_actions
from src.policies.actions import sample_actions
from src.policies.config import RuntimeConfig
from src.policies.features import encode_step, feature_version
from src.policies.heuristic import build_policy
from src.policies.memory import PolicyMemory
from src.training.artifacts import file_hash, load_checkpoint, save_checkpoint, write_json
from src.training.bc_corpus_fit import _DEFINITION_FIELDS, _verify_continuation
from src.training.env import EnvironmentAdapter
from src.training.offline_bc import migrate_peer_context, reset_peer_context_optimizer
from src.training.preflight import source_fingerprint
from src.training.seeds import training_seeds
from src.training.teacher import resolve_teacher


def prepare_anchor(checkpoint: Path, expert: Path, output: Path, *, peer_context: bool):
    loaded = load_checkpoint(checkpoint)
    previous = read_json(checkpoint.parent / "offline-definition.json")
    definition = {key: previous[key] for key in _DEFINITION_FIELDS}
    if definition["dataset_manifest_sha256"] != file_hash(expert / "manifest.json"):
        raise ValueError("Anchor must use the parent's unchanged expert corpus.")
    network = migrate_peer_context(loaded.network) if peer_context and not loaded.network.config.peer_context else loaded.network
    definition["model"] = network.config.model_dump(mode="json")
    source = source_fingerprint()
    _verify_continuation(checkpoint, loaded, definition, source, fork=True)
    optimizer = torch.optim.Adam(network.parameters(), lr=loaded.config.optimizer.learning_rate)
    optimizer.load_state_dict(copy.deepcopy(loaded.optimizer_state))
    if network.config.peer_context and not loaded.config.model.peer_context:
        reset_peer_context_optimizer(network, optimizer)
    output.mkdir(parents=True, exist_ok=False)
    identity = content_hash(definition)
    record = {
        **definition, "definition_id": identity, "source": source,
        "feature_version": feature_version(network.config.public_context, network.config.peer_context),
        "anchor_parent_sha256": loaded.sha256,
        "start_optimizer_step": loaded.training_state["optimizer_steps"],
    }
    write_json(output / "offline-definition.json", record)
    state = {**copy.deepcopy(loaded.training_state), "offline_definition_id": identity,
             "offline_definition_sha256": file_hash(output / "offline-definition.json"),
             "offline_source_sha256": source["sha256"]}
    config = loaded.config.model_copy(update={"model": network.config})
    digest = save_checkpoint(output / "checkpoint.pt", network, config, optimizer, state, rng_state=loaded.rng_state)
    write_json(output / "policy.json", RuntimeConfig(
        policy="neural", checkpoint=str((output / "checkpoint.pt").resolve()), checkpoint_sha256=digest,
    ).model_dump(mode="json"))
    write_json(output / "anchor.json", {
        "parent_checkpoint": str(checkpoint.resolve()), "parent_sha256": loaded.sha256,
        "checkpoint_sha256": digest, "expert_manifest_sha256": file_hash(expert / "manifest.json"),
        "optimizer_steps": state["optimizer_steps"], "peer_context": network.config.peer_context,
        "migration": ("zero newly activated Agent columns/moments; initial predictions preserved"
                      if network.config.peer_context != loaded.network.config.peer_context else "none"),
    })
    return digest


def collect_recovery_episode(job, *, environment_factory=EnvironmentAdapter):
    seed, destination, checkpoint, probability, max_frames = job
    loaded = load_checkpoint(Path(checkpoint), device="cpu")
    torch.set_num_threads(1)
    path = Path(destination) / "train" / f"world-{seed}"
    if path.exists():
        result = read_json(path / "episode.json")
        if (result["sha256"] != file_hash(path / "frames.jsonl.gz")
                or result["behavior_sha256"] != loaded.sha256
                or result["teacher_probability"] != probability or result["seed"] != seed):
            raise ValueError("Recovery shard checksum mismatch.")
        return result
    network = loaded.network.eval()
    teacher = build_policy(seed, resolve_teacher(loaded.config).heuristic)
    env = environment_factory(action_repeat=1)
    step = env.reset(seed)
    memory = PolicyMemory()
    mixture = random.Random(seed ^ 0xDA66E2)  # Separate mixture choices from world randomness.
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    frames = 0
    with tempfile.TemporaryDirectory(prefix=".pending-", dir=path.parent) as temporary:
        staging = Path(temporary)
        with gzip.open(staging / "frames.jsonl.gz", "wt", encoding="utf-8") as stream, torch.no_grad():
            while not env.done:
                if frames >= max_frames:
                    raise RuntimeError("Recovery exceeded its full-horizon bound; do not publish truncated episodes.")
                ordered = step.model_copy(update={"agent_status": sorted(step.agent_status, key=lambda agent: agent.agent_id)})
                batch = encode_step(ordered, memory.previous_actions,
                                    public_context=network.config.public_context,
                                    peer_context=network.config.peer_context)
                hidden = memory.prepare(batch.agent_ids, network.config.hidden_size, "cpu")
                prediction = network(batch, hidden)
                proposal = sample_actions(prediction, ordered, deterministic=True).actions
                labels = validate_actions(teacher.act(ordered), batch.agent_ids)
                expert_turn = mixture.random() < probability
                executed = labels if expert_turn else proposal
                transition = env.step(executed)
                memory.commit(batch.agent_ids, prediction.next_hidden, executed)
                stream.write(json.dumps({
                    "step": ordered.model_dump(mode="json"),
                    "actions": [action.model_dump(mode="json") for action in labels],
                    "executed_actions": [action.model_dump(mode="json") for action in executed],
                    "teacher_executed": expert_turn, "reward": transition.reward,
                    "terminated": transition.terminated,
                }, separators=(",", ":"), allow_nan=False) + "\n")
                counts["agent_labels"] += len(labels)
                counts["spawn_labels"] += sum(action.spawn_agent for action in labels)
                counts["executed_spawns"] += sum(action.spawn_agent for action in executed)
                counts["idle_labels"] += sum(action.move_distance == 0 for action in labels)
                counts["teacher_team_ticks"] += expert_turn
                counts["student_team_ticks"] += not expert_turn
                counts["predator_frames"] += any(
                    value["type"] == "Predator" for agent in ordered.agent_status for value in agent.observations
                )
                counts["descendant_frames"] += any(agent.agent_id >= 5 for agent in ordered.agent_status)
                counts["late_frames"] += ordered.sim_time >= 120
                frames += 1
                step = transition.observation
                if frames % 2000 == 0:
                    print(f"recovery world={seed} frames={frames} score={step.score:.2f}", flush=True)
        metadata = {
            "seed": seed, "split": "train", "origin": "recovery", "frames": frames,
            "native_ticks": frames + 1, "score": step.score, "sim_time": step.sim_time,
            "coverage": dict(counts), "terminated": True, "teacher_probability": probability,
            "behavior_sha256": loaded.sha256, "path": path.relative_to(destination).as_posix(),
            "sha256": file_hash(staging / "frames.jsonl.gz"),
        }
        write_json(staging / "episode.json", metadata)
        staging.rename(path)
    return metadata


def collect_recovery(checkpoint, expert, output, *, episodes=2, probability=0.5, workers=2):
    if episodes < 1 or workers < 1 or not 0 <= probability <= 1:
        raise ValueError("Invalid recovery count, workers or teacher mixture.")
    original = read_json(expert / "manifest.json")
    if original["status"] != "complete":
        raise ValueError("Expert corpus must be complete.")
    loaded = load_checkpoint(checkpoint)
    if loaded.config.teacher.sha256 != original["teacher"]["sha256"]:
        raise ValueError("Recovery and expert teacher differ.")
    excluded = set(original["train_seeds"]) | set(original["validation_seeds"])
    seeds = [seed for seed in training_seeds(loaded.config.seed, len(excluded) + episodes)
             if seed not in excluded][:episodes]
    if len(seeds) != episodes:
        raise ValueError("Cannot allocate disjoint recovery worlds.")
    settings = SimulationSettings()
    if original["simulation"] != settings.model_dump(mode="json"):
        raise ValueError("Recovery settings must match the expert simulation exactly.")
    max_frames = math_ceil_ticks(settings.time_limit, settings.dt)
    definition = {
        "version": 1, "kind": "deterministic-team-dagger", "behavior_sha256": loaded.sha256,
        "expert_manifest_sha256": file_hash(expert / "manifest.json"),
        "teacher": resolve_teacher(loaded.config).provenance,
        "source": source_fingerprint(), "train_seeds": seeds, "validation_seeds": [],
        "teacher_probability": probability, "mixture_unit": "whole_team_tick",
        "student_actions": "deterministic", "max_frames_per_episode": max_frames,
        "simulation": settings.model_dump(mode="json"),
    }
    definition["dataset_id"] = content_hash(definition)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "definition.json").exists() and read_json(output / "definition.json") != definition:
        raise ValueError("Recovery output belongs to another policy or protocol.")
    write_json(output / "definition.json", definition)
    jobs = [(seed, str(output), str(checkpoint.resolve()), probability, max_frames) for seed in seeds]
    results = []
    if workers == 1:
        iterator = map(collect_recovery_episode, jobs)
        for result in iterator:
            results.append(result)
            write_json(output / "manifest.json", {**definition, "status": "collecting", "episodes": results})
    else:
        from src.training.workers import _cpu_spawn_environment
        with _cpu_spawn_environment(), ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            futures = [pool.submit(collect_recovery_episode, job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())
                write_json(output / "manifest.json", {**definition, "status": "collecting", "episodes": results})
    totals = Counter()
    for row in results:
        totals.update(row["coverage"])
    manifest = {**definition, "status": "complete", "episodes": sorted(results, key=lambda row: row["seed"]),
                "coverage": {"train": dict(totals)}, "native_ticks": sum(row["native_ticks"] for row in results)}
    write_json(output / "manifest.json", manifest)
    return manifest


def math_ceil_ticks(time_limit, dt):
    import math
    return math.ceil(time_limit / dt) + 2


def aggregate_corpora(expert: Path, recovery: Path, output: Path):
    base, extra = read_json(expert / "manifest.json"), read_json(recovery / "manifest.json")
    if base["status"] != "complete" or extra["status"] != "complete":
        raise ValueError("Aggregate only complete corpora.")
    if (extra["expert_manifest_sha256"] != file_hash(expert / "manifest.json")
            or base["teacher"]["sha256"] != extra["teacher"]["sha256"]
            or base["simulation"] != extra["simulation"]
            or (set(base["train_seeds"]) | set(base["validation_seeds"])) & set(extra["train_seeds"])):
        raise ValueError("Recovery provenance or seed separation does not match the expert corpus.")
    expert_frames = sum(row["frames"] for row in base["episodes"] if row["split"] == "train")
    recovery_frames = sum(row["frames"] for row in extra["episodes"])
    if recovery_frames > expert_frames:
        raise ValueError("This diagnostic must retain at least half expert frames; reduce recovery rounds or add explicit balanced sampling.")
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for source, manifest, origin in ((expert, base, "expert"), (recovery, extra, "recovery")):
        for row in manifest["episodes"]:
            old = source / row["path"]
            if file_hash(old / "frames.jsonl.gz") != row["sha256"]:
                raise ValueError("Input shard checksum mismatch.")
            relative = Path(origin) / row["path"]
            target = output / relative
            target.mkdir(parents=True)
            shutil.copy2(old / "frames.jsonl.gz", target / "frames.jsonl.gz")
            if file_hash(target / "frames.jsonl.gz") != row["sha256"]:
                raise ValueError("Copied demonstration shard checksum mismatch.")
            new = {**row, "origin": origin, "path": relative.as_posix()}
            write_json(target / "episode.json", new)
            results.append(new)
    coverage = {}
    for split in ("train", "validation"):
        counts = Counter()
        for row in results:
            if row["split"] == split:
                counts.update(row["coverage"])
        coverage[split] = dict(counts)
    definition = {
        "version": 1, "kind": "expert-plus-dagger", "teacher": base["teacher"],
        "simulation": base["simulation"], "source": source_fingerprint(),
        "expert_corpus": str(expert.resolve()), "expert_manifest_sha256": file_hash(expert / "manifest.json"),
        "recovery_corpus": str(recovery.resolve()), "recovery_manifest_sha256": file_hash(recovery / "manifest.json"),
        "train_seeds": [*base["train_seeds"], *extra["train_seeds"]],
        "validation_seeds": base["validation_seeds"],
        "expert_frame_fraction": expert_frames / (expert_frames + recovery_frames),
    }
    definition["dataset_id"] = content_hash(definition)
    manifest = {**definition, "status": "complete", "episodes": sorted(results, key=lambda row: row["seed"]),
                "coverage": coverage, "native_ticks": base["native_ticks"] + extra["native_ticks"]}
    write_json(output / "definition.json", definition)
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    anchor = commands.add_parser("anchor")
    anchor.add_argument("--checkpoint", type=Path, required=True)
    anchor.add_argument("--expert", type=Path, required=True)
    anchor.add_argument("--output", type=Path, required=True)
    anchor.add_argument("--peer-context", action="store_true")
    collect = commands.add_parser("collect")
    collect.add_argument("--checkpoint", type=Path, required=True)
    collect.add_argument("--expert", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--episodes", type=int, default=2)
    collect.add_argument("--teacher-probability", type=float, default=0.5)
    collect.add_argument("--workers", type=int, default=2)
    merge = commands.add_parser("aggregate")
    merge.add_argument("--expert", type=Path, required=True)
    merge.add_argument("--recovery", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "anchor":
        print(prepare_anchor(args.checkpoint, args.expert, args.output, peer_context=args.peer_context))
    elif args.command == "collect":
        print(json.dumps(collect_recovery(args.checkpoint, args.expert, args.output,
                                         episodes=args.episodes, probability=args.teacher_probability,
                                         workers=args.workers)["coverage"]))
    else:
        print(json.dumps(aggregate_corpora(args.expert, args.recovery, args.output)["coverage"]))


if __name__ == "__main__":
    main()
