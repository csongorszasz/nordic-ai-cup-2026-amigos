"""Resumable offline BC on complete, episode-disjoint teacher demonstrations.

--fork preserves the parent's optimizer, scheduler, RNG and sampler, but freezes
new source/evaluation provenance without merging histories. Its only permitted
model-input change is --peer-context (public-v2 to public-peer-v3); --resume
requires the unchanged definition and sources. --steps is an absolute counter.
"""

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.benchmarking.config import content_hash, read_json
from src.policies.config import EvaluationConfig, ModelConfig, RunBudget, RuntimeConfig
from src.policies.features import feature_version
from src.training.artifacts import (
    TrainingRun, file_hash, load_checkpoint, restore_rng, save_checkpoint, write_json,
)
from src.training.demonstrations import read_episode
from src.training.evaluation import initialize_schedule, publish_snapshot, check_capacity, EvaluationBackpressure
from src.training.imitation import checked_optimizer_step
from src.training.offline_bc import (
    cache_frames, copying_metrics, forward_sequence, migrate_peer_context, pack_sequence,
    render_curve, reset_peer_context_optimizer, sequence_loss,
)
from src.training.preflight import source_fingerprint


_DEFINITION_FIELDS = (
    "dataset_id", "dataset_manifest_sha256", "model", "sequence_length", "burn_in",
    "batch_sequences", "validate_every", "initial_lr", "loss", "seed",
)


def _verify_continuation(checkpoint, loaded, definition, source, *, fork):
    """Validate the frozen parent before allowing any output or optimizer work."""
    path = checkpoint.parent / "offline-definition.json"
    if not path.is_file():
        raise ValueError("A corpus continuation requires offline-definition.json beside its checkpoint.")
    previous = read_json(path)
    if not isinstance(previous, dict) or not all(key in previous for key in _DEFINITION_FIELDS):
        raise ValueError("The parent offline definition is incomplete.")
    recorded = {key: previous[key] for key in _DEFINITION_FIELDS}
    state = loaded.training_state
    if (previous.get("definition_id") != content_hash(recorded)
            or state.get("offline_definition_id") != previous["definition_id"]
            or (state.get("offline_definition_sha256") is not None
                and state["offline_definition_sha256"] != file_hash(path))):
        raise ValueError("Parent offline definition checksum does not match its checkpoint.")
    parent_source = previous.get("source")
    if (not isinstance(parent_source, dict) or not isinstance(parent_source.get("files"), dict)
            or parent_source.get("sha256") != content_hash(parent_source["files"])
            or (state.get("offline_source_sha256") is not None
                and state["offline_source_sha256"] != parent_source["sha256"])):
        raise ValueError("Parent source signature is missing or inconsistent.")
    parent_model = ModelConfig.model_validate(recorded["model"])
    if parent_model != loaded.config.model:
        raise ValueError("Parent offline model does not match its checkpoint.")
    parent_schema = feature_version(parent_model.public_context, parent_model.peer_context)
    if previous.get("feature_version", parent_schema) != parent_schema:
        raise ValueError("Parent offline feature schema does not match its checkpoint.")
    comparable = {**recorded, "model": parent_model.model_dump(mode="json")}
    if fork:
        requested = ModelConfig.model_validate(definition["model"])
        if requested.peer_context != parent_model.peer_context:
            if not parent_model.public_context or parent_model.peer_context or not requested.peer_context:
                raise ValueError("A fork only permits enabling peer_context on a public-context model.")
            comparable["model"]["peer_context"] = True
        if comparable != definition:
            raise ValueError("Fork cannot change corpus, model/training parameters, sampler, loss or reporting protocol.")
    elif (state["offline_definition_id"] != content_hash(definition)
          or comparable != definition or parent_source != source):
        raise ValueError("Resume requires unchanged corpus, architecture, sampler, loss, protocol and sources; use --fork.")
    required = ("sampler", "scheduler", "scheduler_epoch", "lr_reductions",
                "optimizer_steps", "prior_native_ticks", "best_validation_loss")
    if (loaded.optimizer_state is None or not all(key in state for key in required)
            or not all(key in loaded.rng_state for key in ("python", "torch", "cuda"))):
        raise ValueError("Continuation requires the saved optimizer, sampler, scheduler, counters and RNG.")
    if (type(state["optimizer_steps"]) is not int or state["optimizer_steps"] < 0
            or state.get("next_update") != state["optimizer_steps"]
            or not isinstance(state["sampler"], dict) or not state["sampler"]
            or not isinstance(state["scheduler"], dict) or not state["scheduler"]):
        raise ValueError("Continuation has invalid optimizer/sampler/scheduler counters.")
    return previous


def _verify_data_fork(checkpoint, loaded, definition, source, corpus):
    """Permit only retained-expert data augmentation, never an implicit resume."""
    old = read_json(checkpoint.parent / "offline-definition.json")
    parent = {key: old[key] for key in _DEFINITION_FIELDS}
    parent["model"] = loaded.config.model.model_dump(mode="json")
    previous = _verify_continuation(checkpoint, loaded, parent, source, fork=True)
    comparable = {**definition, "dataset_id": parent["dataset_id"],
                  "dataset_manifest_sha256": parent["dataset_manifest_sha256"]}
    if comparable != parent:
        raise ValueError("Data comparison cannot change model, optimizer, loss, sequences or reporting.")
    if definition["dataset_manifest_sha256"] != parent["dataset_manifest_sha256"]:
        manifest = corpus.manifest
        if manifest.get("kind") != "expert-plus-dagger":
            raise ValueError("Changed data must be an explicitly retained expert-plus-DAgger corpus.")
        expert = Path(manifest["expert_corpus"])
        if (file_hash(expert / "manifest.json") != parent["dataset_manifest_sha256"]
                or manifest["expert_manifest_sha256"] != parent["dataset_manifest_sha256"]):
            raise ValueError("DAgger comparison does not retain the source expert corpus.")
        reference = read_json(expert / "manifest.json")
        retained = {(row["seed"], row["split"], row["sha256"]) for row in manifest["episodes"]
                    if row.get("origin") == "expert"}
        expected = {(row["seed"], row["split"], row["sha256"]) for row in reference["episodes"]}
        if retained != expected or manifest["validation_seeds"] != reference["validation_seeds"]:
            raise ValueError("Expert episodes or validation split were altered.")
    return previous


class EpisodeStore:
    def __init__(self, path, model_config):
        self.path = Path(path)
        self.manifest = read_json(self.path / "manifest.json")
        self.manifest_sha256 = file_hash(self.path / "manifest.json")
        if self.manifest["status"] != "complete":
            raise ValueError("Offline BC requires a complete immutable corpus.")
        train = self.manifest["train_seeds"]
        validation = self.manifest["validation_seeds"]
        if set(train) & set(validation):
            raise ValueError("Training/validation episode seeds overlap.")
        for split in ("train", "validation"):
            for event in ("idle_labels", "spawn_labels", "predator_frames", "descendant_frames", "late_frames"):
                if self.manifest["coverage"][split].get(event, 0) == 0:
                    raise ValueError(f"{split} corpus has no {event}; collect coverage before qualification.")
        self.metadata = self.manifest["episodes"]
        expected = set(train) | set(validation)
        if len(self.metadata) != len(expected) or {row["seed"] for row in self.metadata} != expected:
            raise ValueError("Corpus manifest does not cover exactly its split seeds.")
        self.model_config = model_config
        self.cache = {}

    def episode(self, index):
        if index not in self.cache:
            self.cache[index] = cache_frames(read_episode(self.path, self.metadata[index]), self.model_config)
        return self.cache[index]

    def indices(self, split):
        return [index for index, metadata in enumerate(self.metadata) if metadata["split"] == split]


class WindowSampler:
    def __init__(self, metadata, sequence_length, seed, state=None):
        self.windows = [
            (index, start, min(start + sequence_length, row["frames"]))
            for index, row in enumerate(metadata) if row["split"] == "train"
            for start in range(0, row["frames"], sequence_length)
        ]
        if not self.windows:
            raise ValueError("There are no training sequences.")
        self.rng = random.Random(seed)
        self.order = list(range(len(self.windows)))
        self.rng.shuffle(self.order)
        self.cursor = self.epochs = self.exposures = 0
        if state is not None:
            if (not isinstance(state.get("order"), list)
                    or any(type(index) is not int for index in state["order"])
                    or sorted(state["order"]) != list(range(len(self.windows)))):
                raise ValueError("Resume sampler windows changed.")
            if (any(type(state.get(key)) is not int or state[key] < 0 for key in ("cursor", "epochs", "exposures"))
                    or state["cursor"] > len(self.windows)):
                raise ValueError("Resume sampler counters are invalid.")
            expected_exposures = (
                state["epochs"] * sum(end - start for _, start, end in self.windows)
                + sum(self.windows[index][2] - self.windows[index][1] for index in state["order"][:state["cursor"]])
            )
            if state["exposures"] != expected_exposures:
                raise ValueError("Resume sampler exposure count does not match its order/cursor/epoch.")
            self.order, self.cursor = list(state["order"]), state["cursor"]
            self.epochs, self.exposures = state["epochs"], state["exposures"]
            self.rng.setstate(state["rng"])

    def take(self, count):
        if self.cursor == len(self.order):
            self.cursor = 0
            self.epochs += 1
            self.rng.shuffle(self.order)
        indices = self.order[self.cursor:self.cursor + count]
        self.cursor += len(indices)
        batch = [self.windows[index] for index in indices]
        self.exposures += sum(end - start for _, start, end in batch)
        return batch

    def state_dict(self):
        return {"order": list(self.order), "cursor": self.cursor, "epochs": self.epochs,
                "exposures": self.exposures, "rng": self.rng.getstate()}


def aggregate_metrics(parts):
    labels = sum(row["agent_labels"] for row in parts)
    frames = sum(row["frames"] for row in parts)
    if not labels or not frames:
        raise ValueError("No validation labels were evaluated.")
    idle = sum(row["stop_labels"] for row in parts)
    moving = sum(row["moving_labels"] for row in parts)
    total = lambda key: sum(row[key] for row in parts)
    weighted = lambda key, weight, denominator: (
        sum(row[key] * row[weight] for row in parts if row[key] is not None) / denominator if denominator else None
    )
    tp, fp, fn = (total(key) for key in ("spawn_tp", "spawn_fp", "spawn_fn"))
    samples = np.concatenate([row["_stop_distances"] for row in parts])
    return {
        "loss": weighted("loss", "frames", frames), "frames": frames, "agent_labels": labels,
        "distance_mae": weighted("distance_mae", "agent_labels", labels),
        "travel_mae_radians": weighted("travel_mae_radians", "moving_labels", moving),
        "turn_mae_radians": weighted("turn_mae_radians", "agent_labels", labels),
        "stop_labels": idle,
        "stop_distance_mean": weighted("stop_distance_mean", "stop_labels", idle),
        "stop_distance_p95": float(np.quantile(samples, 0.95)) if len(samples) else None,
        "stop_energy_mean_per_tick": weighted("stop_energy_mean_per_tick", "stop_labels", idle),
        "spawn_positive_labels": total("spawn_positive_labels"), "spawn_tp": tp, "spawn_fp": fp, "spawn_fn": fn,
        "spawn_precision": tp / (tp + fp) if tp + fp else None,
        "spawn_recall": tp / (tp + fn) if tp + fn else None,
    }


def evaluate_split(network, store, split, device, chunk_frames=256):
    """Whole expert histories from reset, with state carried across memory-sized chunks."""
    network.eval()
    parts = []
    with torch.no_grad():
        for index in store.indices(split):
            frames = store.episode(index)
            hidden, previous_ids = None, ()
            for start in range(0, len(frames), chunk_frames):
                window = frames[start:start + chunk_frames]
                batch = pack_sequence(window, network.config, device, previous_ids=previous_ids)
                prediction = forward_sequence(network, batch, hidden)
                parts.append(copying_metrics(prediction, batch, include_samples=True))
                previous_ids = window[-1].features.agent_ids
                hidden = prediction.next_hidden[-len(previous_ids):].detach()
    return aggregate_metrics(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Weights-only start; optimizer/sampler counters start at zero.")
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--resume", type=Path, help="Exact continuation with unchanged sources and protocol.")
    continuation.add_argument("--fork", type=Path, help="Controlled continuation with fresh source/evaluation provenance.")
    continuation.add_argument("--data-fork", type=Path,
                              help="Matched data comparison: preserve weights/optimizer, reset sampler/scheduler and hold LR fixed.")
    parser.add_argument("--peer-context", action="store_true", help="Enable the public peer-v3 input migration.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--burn-in", type=int, default=256)
    parser.add_argument("--batch-sequences", type=int, default=4)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-vram-mb", type=int, default=6144)
    parser.add_argument("--max-seconds", type=float, default=1000)
    args = parser.parse_args(argv)
    checkpoint = args.resume or args.fork or args.data_fork or args.checkpoint
    if args.checkpoint and args.data_fork:
        parser.error("Choose a weights-only checkpoint or an explicit data-fork, not both.")
    if checkpoint is None:
        parser.error("Provide --checkpoint, --resume, --fork or --data-fork.")
    if min(args.steps, args.sequence_length, args.batch_sequences, args.validate_every, args.max_vram_mb) < 1 or args.burn_in < 0:
        parser.error("Offline fitting counts must be positive and burn-in nonnegative.")
    started = time.perf_counter()
    torch.set_num_threads(1)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no silent fallback.")
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(args.max_vram_mb * 1024**2 / total)
        torch.cuda.reset_peak_memory_stats()
    loaded = load_checkpoint(checkpoint, device=args.device)
    network = loaded.network
    migrate = args.peer_context and not network.config.peer_context
    if migrate:
        if args.resume or args.data_fork:
            raise ValueError("Model-input changes require a peer anchor: use --fork, not --resume or --data-fork.")
        network = migrate_peer_context(network)
    store = EpisodeStore(args.corpus, network.config)
    if loaded.config.teacher is None or store.manifest["teacher"]["sha256"] != loaded.config.teacher.sha256:
        raise ValueError("Corpus and checkpoint teacher differ.")
    definition = {
        "dataset_id": store.manifest["dataset_id"], "dataset_manifest_sha256": store.manifest_sha256,
        "model": network.config.model_dump(mode="json"),
        "sequence_length": args.sequence_length, "burn_in": args.burn_in,
        "batch_sequences": args.batch_sequences, "validate_every": args.validate_every,
        "initial_lr": loaded.config.optimizer.learning_rate, "loss": "unchanged-bc-v1",
        "seed": loaded.config.seed,
    }
    definition_id = content_hash(definition)
    source = source_fingerprint()
    continuing = args.resume is not None or args.fork is not None or args.data_fork is not None
    state = loaded.training_state if continuing else {}
    parent_definition = _verify_data_fork(
        checkpoint, loaded, definition, source, store,
    ) if args.data_fork else _verify_continuation(
        checkpoint, loaded, definition, source, fork=args.fork is not None,
    ) if continuing else None
    sampler = WindowSampler(store.metadata, args.sequence_length, loaded.config.seed,
                            None if args.data_fork else state.get("sampler"))
    optimizer = torch.optim.Adam(network.parameters(), lr=loaded.config.optimizer.learning_rate)
    if continuing:
        optimizer.load_state_dict(copy.deepcopy(loaded.optimizer_state))
        if migrate:
            reset_peer_context_optimizer(network, optimizer)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, threshold=0.001, min_lr=1e-6,
    )
    if continuing and not args.data_fork:
        scheduler.load_state_dict(state["scheduler"])
    data_fork = ({
        "parent_checkpoint_sha256": loaded.sha256,
        "parent_definition_id": parent_definition["definition_id"],
        "parent_dataset_manifest_sha256": parent_definition["dataset_manifest_sha256"],
        "dataset_manifest_sha256": store.manifest_sha256,
        "sampler_reset": True, "scheduler_reset": True, "constant_learning_rate": True,
        "arm": "bc-dagger" if store.manifest.get("kind") == "expert-plus-dagger" else "bc-control",
        "prior_data_exposures": state["sampler"]["exposures"],
    } if args.data_fork else state.get("data_fork"))
    first = state.get("optimizer_steps", 0)
    prior_native = state.get("prior_native_ticks", 0) if continuing else (
        loaded.training_state.get("native_ticks_total", 0) + loaded.training_state.get("prior_native_ticks", 0)
    )
    training_native = sum(row["native_ticks"] for row in store.metadata if row["split"] == "train")
    validation_native = sum(row["native_ticks"] for row in store.metadata if row["split"] == "validation")
    if first >= args.steps:
        raise ValueError("Increase the target optimizer-step count for a continuation.")
    lineage = None
    if args.fork or args.data_fork:
        lineage = {
            "kind": "fork", "exact_resume": False, "parent_run": str(checkpoint.resolve().parent),
            "parent_checkpoint": str(checkpoint.resolve()), "parent_checkpoint_sha256": loaded.sha256,
            "parent_definition_id": parent_definition["definition_id"],
            "parent_definition_sha256": file_hash(checkpoint.parent / "offline-definition.json"),
            "parent_source_sha256": parent_definition["source"]["sha256"], "source_sha256": source["sha256"],
            "dataset_id": definition["dataset_id"],
            "dataset_manifest_sha256": definition["dataset_manifest_sha256"],
            "start_optimizer_step": first,
            "allowed_model_input_change": {"peer_context": {"from": False, "to": True}} if migrate else {},
            "parent_feature_version": feature_version(loaded.config.model.public_context, loaded.config.model.peer_context),
            "feature_version": feature_version(network.config.public_context, network.config.peer_context),
            "optimizer_migration": "zero Agent first-Linear moment columns 8/9 only" if migrate else "none",
            "data_comparison": data_fork,
        }
    elif args.resume:
        lineage = {"kind": "resume", "parent_run": str(checkpoint.resolve().parent), "resume_update": first}
    fork_origin = lineage if args.fork or args.data_fork else state.get("fork")
    config = loaded.config.model_copy(update={
        "mode": "imitation", "updates": args.steps, "checkpoint": str(checkpoint), "model": network.config,
        "imitation": loaded.config.imitation.model_copy(update={"dagger_rounds": 0}),
        "evaluation": EvaluationConfig(enabled=True, every_updates=args.validate_every,
                                       max_pending=4, max_snapshot_mb=8,
                                       max_storage_mb=max(1024, (args.steps // args.validate_every + 2) * 8),
                                       episode_timeout_seconds=900),
        "budget": RunBudget(kind="diagnostic", max_updates=args.steps,
                            max_native_ticks=store.manifest["native_ticks"],
                            max_evaluation_episodes=(args.steps // args.validate_every + 3) * 3,
                            host_memory_mb=65536),
    })
    run = TrainingRun(args.output, config)
    definition_path = run.path / "offline-definition.json"
    write_json(definition_path, {
        **definition, "definition_id": definition_id, "source": source,
        "feature_version": feature_version(network.config.public_context, network.config.peer_context),
        "start_optimizer_step": first, "fork": fork_origin,
    })
    definition_sha256 = file_hash(definition_path)
    schedule = initialize_schedule(
        run.path, config, run.manifest, lineage=lineage,
    )
    best = math.inf if args.fork or args.data_fork else state.get("best_validation_loss", math.inf)
    reductions = 0 if args.data_fork else state.get("lr_reductions", 0)
    previous_epoch = -1 if args.data_fork else state.get("scheduler_epoch", -1)
    records = []
    if args.resume and (args.resume.parent / "copying-metrics.jsonl").is_file():
        records = [json.loads(line) for line in (args.resume.parent / "copying-metrics.jsonl").read_text().splitlines()]
        (run.path / "copying-metrics.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in records), encoding="utf-8",
        )
    completed = first
    result_status = "budget_exhausted"
    last_validation = None

    def save(step):
        metadata = {
            "next_update": step, "optimizer_steps": step, "native_ticks_total": training_native,
            "prior_native_ticks": prior_native, "offline_definition_id": definition_id,
            "offline_definition_sha256": definition_sha256, "offline_source_sha256": source["sha256"],
            "sampler": sampler.state_dict(), "scheduler": scheduler.state_dict(),
            "scheduler_epoch": previous_epoch, "lr_reductions": reductions,
            "best_validation_loss": best, "fork": fork_origin,
            "data_fork": data_fork,
        }
        publish_snapshot(run.path, network, config, metadata, schedule)
        digest = save_checkpoint(run.path / "checkpoint.pt", network, config, optimizer, metadata)
        write_json(run.path / "policy.json", RuntimeConfig(
            policy="neural", checkpoint=str(run.path / "checkpoint.pt"), checkpoint_sha256=digest,
        ).model_dump(mode="json"))
        return metadata

    try:
        if continuing:
            restore_rng(loaded.rng_state)
        for step in range(first, args.steps + 1):
            timed_out = time.perf_counter() - started >= args.max_seconds
            if step == first or step % args.validate_every == 0 or step == args.steps:
                validation = evaluate_split(network, store, "validation", args.device)
                last_validation = validation
                epoch = sampler.epochs + (sampler.cursor == len(sampler.order))
                if epoch > previous_epoch and not (data_fork and data_fork["constant_learning_rate"]):
                    old_lr = optimizer.param_groups[0]["lr"]
                    scheduler.step(validation["loss"])
                    reductions += int(optimizer.param_groups[0]["lr"] < old_lr)
                    previous_epoch = epoch
                row = {"optimizer_steps": step, "split": "validation", "data_passes": sampler.exposures /
                       sum(item["frames"] for item in store.metadata if item["split"] == "train"),
                       "learning_rate": optimizer.param_groups[0]["lr"],
                       "elapsed_seconds": time.perf_counter() - started, **validation}
                records.append(row)
                with (run.path / "copying-metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                improved = validation["loss"] < best
                best = min(best, validation["loss"])
                metadata = save(step)
                if improved:
                    digest = save_checkpoint(run.path / "best.pt", network, config, optimizer, metadata)
                    write_json(run.path / "best-policy.json", RuntimeConfig(
                        policy="neural", checkpoint=str(run.path / "best.pt"), checkpoint_sha256=digest,
                    ).model_dump(mode="json"))
                render_curve(records, run.path)
                print(json.dumps(row), flush=True)
                timed_out = time.perf_counter() - started >= args.max_seconds
                if step < args.steps:
                    check_capacity(run.path, config)
                if reductions > 3:
                    result_status = "plateau_needs_diagnosis"
                    break
            if timed_out:
                save(step)
                result_status = "paused_resource_budget"
                break
            if step == args.steps:
                break
            network.train()
            windows = sampler.take(args.batch_sequences)
            frame_count = sum(end - start for _, start, end in windows)
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            batch_metrics = []
            for episode, start, end in windows:
                beginning = max(0, start - args.burn_in)
                window = store.episode(episode)[beginning:end]
                batch = pack_sequence(window, network.config, args.device, start - beginning)
                prediction = forward_sequence(network, batch)
                loss = sequence_loss(prediction, batch) * (end - start) / frame_count
                if (step + 1) % 50 == 0:
                    with torch.no_grad():
                        batch_metrics.append(copying_metrics(prediction, batch, include_samples=True))
                loss.backward()
                loss_value += float(loss.detach())
            norm = checked_optimizer_step(network, optimizer, loaded.config.optimizer.max_grad_norm)
            completed = step + 1
            if completed % 50 == 0:
                row = {"split": "train-minibatch", "optimizer_steps": completed, **aggregate_metrics(batch_metrics),
                       "gradient_norm": norm, "examples_seen": sampler.exposures,
                       "elapsed_seconds": time.perf_counter() - started}
                run.emit(row)
                records.append(row)
                with (run.path / "copying-metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row), flush=True)
        result = {
            "status": result_status, "optimizer_steps": completed,
            "best_validation_loss": best, "validation": last_validation,
            "data_exposures": sampler.exposures, "native_ticks_collected_once": store.manifest["native_ticks"],
            "data_fork": data_fork,
            "training_native_ticks": training_native, "validation_native_ticks": validation_native,
            "teacher_level_qualified": False, "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2 if args.device == "cuda" else None,
        }
        run.finish("paused" if result_status.startswith("paused") else "complete", result)
    except EvaluationBackpressure as error:
        run.finish("paused", {"reason": str(error), "optimizer_steps": completed})
        print(str(error), flush=True)
        return 3
    except BaseException as error:
        run.finish("failed", {"error": str(error), "optimizer_steps": completed})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
