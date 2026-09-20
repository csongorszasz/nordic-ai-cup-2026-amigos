"""Immutable episode-sharded, public-observation teacher demonstrations."""

import argparse
import gzip
import json
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.benchmarking.config import PROJECT_ROOT, SimulationSettings, content_hash, read_json
from src.benchmarking.policies import validate_actions
from src.policies.config import load_experiment
from src.policies.features import encode_step
from src.policies.heuristic import build_policy
from src.training.artifacts import file_hash, write_json
from src.training.env import EnvironmentAdapter
from src.training.preflight import source_fingerprint
from src.training.seeds import training_seeds
from src.training.teacher import resolve_teacher
from src.utils.DTOs import ActionRequest, StepResponse


def collect_episode(job):
    seed, split, destination, teacher_config = job
    from src.policies.config import HeuristicConfig

    path = Path(destination) / split / f"world-{seed}"
    if path.exists():
        result = read_json(path / "episode.json")
        if result["seed"] != seed or result["sha256"] != file_hash(path / "frames.jsonl.gz"):
            raise ValueError("Existing demonstration shard is inconsistent.")
        return result
    path.parent.mkdir(parents=True, exist_ok=True)
    env = EnvironmentAdapter(action_repeat=1)
    teacher = build_policy(seed, HeuristicConfig.model_validate(teacher_config))
    step = env.reset(seed)
    counts = Counter()
    frames = 0
    with tempfile.TemporaryDirectory(prefix=".pending-", dir=path.parent) as temporary:
        staging = Path(temporary)
        with gzip.open(staging / "frames.jsonl.gz", "wt", encoding="utf-8") as stream:
            while not env.done:
                actions = validate_actions(teacher.act(step), [a.agent_id for a in step.agent_status])
                transition = env.step(actions)
                stream.write(json.dumps({
                    "step": step.model_dump(mode="json"),
                    "actions": [action.model_dump(mode="json") for action in actions],
                    "reward": transition.reward, "terminated": transition.terminated,
                }, separators=(",", ":"), allow_nan=False) + "\n")
                counts["agent_labels"] += len(actions)
                counts["spawn_labels"] += sum(action.spawn_agent for action in actions)
                counts["idle_labels"] += sum(action.move_distance == 0 for action in actions)
                counts["predator_frames"] += any(
                    obs["type"] == "Predator" for agent in step.agent_status for obs in agent.observations
                )
                counts["descendant_frames"] += any(a.agent_id >= 5 for a in step.agent_status)
                counts["late_frames"] += step.sim_time >= 120
                frames += 1
                step = transition.observation
                if frames % 2000 == 0:
                    print(f"world={seed} split={split} frames={frames} score={step.score:.2f}", flush=True)
        result = {
            "seed": seed, "split": split, "frames": frames, "native_ticks": frames + 1,
            "score": step.score, "sim_time": step.sim_time, "terminated": True,
            "coverage": dict(counts), "path": str(path.relative_to(destination).as_posix()),
            "sha256": file_hash(staging / "frames.jsonl.gz"),
        }
        write_json(staging / "episode.json", result)
        staging.rename(path)
    return result


def collect_corpus(config, destination: Path, *, train_worlds: int, validation_worlds: int, workers: int):
    if min(train_worlds, validation_worlds, workers) < 1:
        raise ValueError("Corpus splits and worker count must be positive.")
    teacher = resolve_teacher(config)
    seeds = training_seeds(config.seed, train_worlds + validation_worlds)
    definition = {
        "version": 1, "teacher": teacher.provenance, "source": source_fingerprint(),
        "simulation": SimulationSettings().model_dump(mode="json"),
        "train_seeds": seeds[:train_worlds], "validation_seeds": seeds[train_worlds:],
    }
    definition["dataset_id"] = content_hash(definition)
    destination.mkdir(parents=True, exist_ok=True)
    definition_path = destination / "definition.json"
    if definition_path.exists() and read_json(definition_path) != definition:
        raise ValueError("Cannot replace a corpus with a changed teacher, source or split.")
    write_json(definition_path, definition)
    jobs = [
        (seed, split, str(destination), teacher.heuristic.model_dump(mode="json"))
        for split in ("train", "validation") for seed in definition[f"{split}_seeds"]
    ]
    results = []

    def record(result):
        results.append(result)
        write_json(destination / "manifest.json", {
            **definition, "status": "collecting", "episodes": sorted(results, key=lambda row: row["seed"]),
        })
        print(f"completed world={result['seed']} split={result['split']} score={result['score']:.2f}", flush=True)

    if workers == 1:
        for job in jobs:
            record(collect_episode(job))
    else:
        from src.training.workers import _cpu_spawn_environment
        with _cpu_spawn_environment():
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(collect_episode, job) for job in jobs]
                for future in as_completed(futures):
                    record(future.result())
    coverage = {}
    for split in ("train", "validation"):
        totals = Counter()
        for result in results:
            if result["split"] == split:
                totals.update(result["coverage"])
        coverage[split] = dict(totals)
    manifest = {**definition, "status": "complete", "episodes": sorted(results, key=lambda row: row["seed"]),
                "coverage": coverage, "native_ticks": sum(row["native_ticks"] for row in results)}
    write_json(destination / "manifest.json", manifest)
    return manifest


def read_episode(corpus: Path, metadata: dict):
    shard = corpus / metadata["path"] / "frames.jsonl.gz"
    if file_hash(shard) != metadata["sha256"]:
        raise ValueError(f"Demonstration checksum mismatch: {shard}")
    frames = []
    previous = {}
    with gzip.open(shard, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            step = StepResponse.model_validate(row["step"])
            actions = validate_actions(
                [ActionRequest.model_validate(action) for action in row["actions"]],
                [agent.agent_id for agent in step.agent_status],
            )
            executed = validate_actions(
                [ActionRequest.model_validate(action) for action in row.get("executed_actions", row["actions"])],
                [agent.agent_id for agent in step.agent_status],
            )
            frames.append((step, tuple(actions), previous))
            previous = {action.agent_id: action for action in executed}
    if len(frames) != metadata["frames"]:
        raise ValueError("Demonstration frame count mismatch.")
    return frames


def legacy_replay(checkpoint: Path):
    from src.training.artifacts import load_checkpoint
    from src.training.imitation import ImitationDataset

    loaded = load_checkpoint(checkpoint)
    replay = ImitationDataset(loaded.config.imitation.max_frames)
    replay.load_state_dict(loaded.training_state["imitation_dataset"])
    # Recover actual previous actions from ordered records, not obsolete hidden states.
    episodes, current, previous, identity = [], [], {}, None
    for record in sorted(replay.frames, key=lambda value: (value.env_index, value.world_seed, value.episode_step)):
        key = (record.env_index, record.world_seed)
        if key != identity or record.episode_start:
            if current:
                episodes.append(current)
            current, previous, identity = [], {}, key
        current.append((record.step, record.teacher_actions, previous))
        previous = {action.agent_id: action for action in record.actions}
    if current:
        episodes.append(current)
    return episodes, loaded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-worlds", type=int, default=8)
    parser.add_argument("--validation-worlds", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = collect_corpus(load_experiment(args.config), args.output.resolve(),
                            train_worlds=args.train_worlds, validation_worlds=args.validation_worlds,
                            workers=args.workers)
    print(json.dumps({"dataset_id": result["dataset_id"], "coverage": result["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
