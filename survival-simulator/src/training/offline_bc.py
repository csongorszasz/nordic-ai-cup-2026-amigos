"""Offline recurrent BC with real optimizer-step counters and copying diagnostics."""

import argparse
import copy
import gzip
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.benchmarking.config import PROJECT_ROOT, content_hash, read_json
from src.policies.actions import imitation_components, imitation_targets
from src.policies.config import RuntimeConfig, load_experiment
from src.policies.features import ENTITY_TYPES, FeatureBatch, encode_step, feature_version
from src.policies.geometry import movement_limit
from src.policies.networks import PolicyNetwork
from src.training.artifacts import file_hash, load_checkpoint, save_checkpoint, write_json
from src.training.demonstrations import legacy_replay, read_episode
from src.training.imitation import checked_optimizer_step
from src.training.preflight import source_fingerprint


@dataclass
class PackedSequence:
    features: FeatureBatch
    offsets: list[int]
    previous: list[torch.Tensor]
    groups: torch.Tensor
    targets: torch.Tensor
    eligible: torch.Tensor
    limits: torch.Tensor
    speeds: torch.Tensor
    energies: torch.Tensor
    burn_in: int

    @property
    def frames(self):
        return len(self.offsets) - 1


@dataclass
class CachedFrame:
    step: object
    actions: tuple
    history: dict
    features: FeatureBatch
    targets: list
    eligible: list


def cache_frames(frames, model_config):
    result = []
    for step, actions, history in frames:
        targets, eligible = imitation_targets(step, actions)
        result.append(CachedFrame(step, actions, history, encode_step(
            step, history, public_context=model_config.public_context, peer_context=model_config.peer_context,
        ), targets, eligible))
    return result


def pack_sequence(frames, model_config, device="cpu", burn_in=0, previous_ids=()) -> PackedSequence:
    if not frames or not 0 <= burn_in < len(frames):
        raise ValueError("A sequence needs scored frames after its burn-in.")
    scalar_rows, entities, types, owners, groups = [], [], [], [], []
    schema = feature_version(model_config.public_context, model_config.peer_context)
    offsets = [0]
    indices, targets, eligible, limits, speeds, energies = [], [], [], [], [], []
    for index, item in enumerate(frames):
        if isinstance(item, CachedFrame):
            step, batch = item.step, item.features
            target, mask = item.targets, item.eligible
        else:
            step, actions, history = item
            batch = encode_step(step, history, public_context=model_config.public_context,
                                peer_context=model_config.peer_context)
            target, mask = imitation_targets(step, actions)
        if batch.feature_version != schema:
            raise ValueError("Cached demonstration feature schema does not match the model.")
        if not batch.agent_ids:
            raise ValueError("Demonstrations cannot contain a terminal empty action frame.")
        mapping = {identity: position for position, identity in enumerate(previous_ids)}
        indices.append(torch.tensor([mapping.get(identity, len(previous_ids)) for identity in batch.agent_ids],
                                    dtype=torch.long, device=device))
        previous_ids = batch.agent_ids
        scalar_rows.append(batch.scalars)
        entities.append(batch.entities)
        types.append(batch.entity_types)
        owners.append(batch.entity_owners + offsets[-1])
        groups.extend([index] * len(batch.agent_ids))
        offsets.append(offsets[-1] + len(batch.agent_ids))
        targets.extend(target)
        eligible.extend(mask)
        limits.extend(movement_limit(agent) for agent in step.agent_status)
        speeds.extend(agent.speed for agent in step.agent_status)
        energies.extend(agent.energy for agent in step.agent_status)
    features = FeatureBatch(
        tuple(range(offsets[-1])), np.concatenate(scalar_rows), np.concatenate(entities),
        np.concatenate(types), np.concatenate(owners), feature_version=schema,
    )
    tensor = lambda values: torch.tensor(values, dtype=torch.float32, device=device)
    return PackedSequence(
        features, offsets, indices, torch.tensor(groups, dtype=torch.long, device=device),
        tensor(targets), torch.tensor(eligible, dtype=torch.bool, device=device),
        tensor(limits), tensor(speeds), tensor(energies), burn_in,
    )


def forward_sequence(network: PolicyNetwork, batch: PackedSequence, initial_hidden=None):
    features, _ = network.encode_agents(batch.features)
    hidden = features.new_zeros((0, network.config.hidden_size)) if initial_hidden is None else initial_hidden
    states = []
    grad = torch.is_grad_enabled()
    for index, (start, end) in enumerate(zip(batch.offsets[:-1], batch.offsets[1:])):
        with torch.set_grad_enabled(grad and index >= batch.burn_in):
            padded = torch.cat((hidden, hidden.new_zeros((1, hidden.shape[1]))))
            hidden = padded.index_select(0, batch.previous[index])
            hidden = network.recurrent(features[start:end], hidden) if network.recurrent else features[start:end]
        states.append(hidden)
    return network.decode_agents(torch.cat(states), batch.groups, batch.frames)


def sequence_loss(output, batch: PackedSequence):
    components = imitation_components(output, batch.targets, batch.eligible)
    counts = torch.bincount(batch.groups, minlength=batch.frames).to(components.dtype)
    weights = counts.index_select(0, batch.groups).reciprocal()
    weights = weights * (batch.groups >= batch.burn_in) / (batch.frames - batch.burn_in)
    loss = (components.sum(1) * weights).sum()
    if not torch.isfinite(loss).item():
        raise FloatingPointError("Offline BC loss is non-finite.")
    return loss


def copying_metrics(output, batch: PackedSequence, *, include_samples=False) -> dict:
    selected = batch.groups >= batch.burn_in
    target = batch.targets[selected]
    mean = output.mean[selected]
    logits = output.spawn_logits[selected]
    limits, speeds, energies = batch.limits[selected], batch.speeds[selected], batch.energies[selected]
    distance = limits * mean[:, 0].sigmoid()
    teacher_distance = limits * target[:, 0]
    angles = math.pi * mean[:, 1:].tanh()
    angle_error = torch.remainder(angles - target[:, 1:3] + math.pi, math.tau) - math.pi
    movement_cost = torch.minimum(distance, speeds) * 0.05 + (distance - speeds).clamp_min(0) * 0.5
    can_spawn = energies - movement_cost - angles[:, 1].abs().clamp_max(math.pi) / math.tau > 100
    predicted_spawn = (logits >= 0) & can_spawn
    desired_spawn = target[:, 3].bool()
    idle = target[:, 4] == 0
    moving = ~idle
    tp = int((predicted_spawn & desired_spawn).sum())
    fp = int((predicted_spawn & ~desired_spawn).sum())
    fn = int((~predicted_spawn & desired_spawn).sum())
    components = imitation_components(output, batch.targets, batch.eligible)[selected]
    result = {
        "loss": float(sequence_loss(output, batch)), "agent_labels": len(target),
        "frames": batch.frames - batch.burn_in, "moving_labels": int(moving.sum()),
        "distance_mae": float((distance - teacher_distance).abs().mean()),
        "travel_mae_radians": float(angle_error[moving, 0].abs().mean()) if moving.any() else None,
        "turn_mae_radians": float(angle_error[:, 1].abs().mean()),
        "stop_labels": int(idle.sum()),
        "stop_distance_mean": float(distance[idle].mean()) if idle.any() else None,
        "stop_distance_p95": float(torch.quantile(distance[idle], 0.95)) if idle.any() else None,
        "stop_energy_mean_per_tick": float(movement_cost[idle].mean()) if idle.any() else None,
        "spawn_positive_labels": int(desired_spawn.sum()), "spawn_tp": tp, "spawn_fp": fp, "spawn_fn": fn,
        "spawn_precision": tp / (tp + fp) if tp + fp else None,
        "spawn_recall": tp / (tp + fn) if tp + fn else None,
        "components_per_agent": dict(zip(("distance", "travel", "turn", "spawn"), components.mean(0).cpu().tolist())),
    }
    if include_samples:
        result["_stop_distances"] = distance[idle].cpu().numpy()
    return result


def evaluate_sequence(network, packed):
    was_training = network.training
    network.eval()
    with torch.no_grad():
        result = copying_metrics(forward_sequence(network, packed), packed)
    network.train(was_training)
    return result


def render_curve(records: list[dict], directory: Path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11, 7), dpi=120)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    for axis, key, title in zip(axes.flat,
                                ("loss", "stop_distance_mean", "distance_mae", "travel_mae_radians"),
                                ("BC loss", "Distance when teacher stops", "Movement distance MAE", "Moving-target angle MAE (radians)")):
        for split in ("train", "train-minibatch", "validation"):
            rows = [row for row in records if row["split"] == split and row.get(key) is not None]
            if rows:
                axis.plot([row["optimizer_steps"] for row in rows],
                          [row.get(key) if row.get(key) is not None else math.nan for row in rows],
                          marker=".", label=split)
        axis.set(xlabel="Actual optimizer steps", title=title)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Offline BC copying fidelity (not full-game score)")
    figure.tight_layout()
    temporary = directory / "copying-curve.pending.png"
    figure.savefig(temporary)
    temporary.replace(directory / "copying-curve.png")
    figure.clear()


def migrate_peer_context(network: PolicyNetwork) -> PolicyNetwork:
    """Copy a public-v2 model; zero only the formerly unused Agent inputs 8/9."""
    if not network.config.public_context or network.config.peer_context:
        raise ValueError("Peer migration requires public_context=True and peer_context=False.")
    migrated = copy.deepcopy(network)
    migrated.config = network.config.model_copy(update={"peer_context": True})
    with torch.no_grad():
        migrated.entity_encoders[ENTITY_TYPES.index("Agent")][0].weight[:, 8:10].zero_()
    return migrated


def reset_peer_context_optimizer(network: PolicyNetwork, optimizer: torch.optim.Optimizer) -> None:
    """Zero Adam moments only for the newly activated input columns; keep step/LR."""
    if not network.config.peer_context:
        raise ValueError("Peer optimizer migration requires a peer-context model.")
    weight = network.entity_encoders[ENTITY_TYPES.index("Agent")][0].weight
    if not any(weight is parameter for group in optimizer.param_groups for parameter in group["params"]):
        raise ValueError("Optimizer is not bound to the migrated model.")
    state = optimizer.state.get(weight, {})
    moments = []
    for name, value in state.items():
        if name == "step":
            continue
        if (name not in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
                or not isinstance(value, torch.Tensor) or value.shape != weight.shape):
            raise ValueError(f"Unsupported peer optimizer moment: {name}.")
        moments.append(value)
    with torch.no_grad():
        for moment in moments:
            moment[:, 8:10].zero_()


def migrate_angle_head(network, mode, seed):
    if mode is None or mode == network.config.angle_head:
        return network
    if network.config.angle_head != "bounded" or mode != "vector_bc":
        raise ValueError("Only explicit bounded-to-vector BC migration is supported.")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        migrated = PolicyNetwork(network.config.model_copy(update={"angle_head": mode}))
    reference = next(network.parameters())
    migrated.to(device=reference.device, dtype=reference.dtype)
    missing, unexpected = migrated.load_state_dict(network.state_dict(), strict=False)
    if set(missing) != {"angle_vectors.weight", "angle_vectors.bias"} or unexpected:
        raise ValueError("Unexpected architecture difference during angle-head migration.")
    return migrated


def fit_control(args):
    device = args.device
    torch.set_num_threads(1)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no silent CPU fallback.")
    if device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(args.max_vram_mb * 1024**2 / total)
        torch.cuda.reset_peak_memory_stats()
    episodes, loaded = legacy_replay(args.checkpoint)
    if len(episodes) != 1:
        raise ValueError("The first convergence control requires one intact recorded episode prefix.")
    frames = episodes[0]
    if args.frames:
        frames = frames[:args.frames]
    if len(frames) != loaded.training_state["collector"]["frame_count"] and not args.frames:
        raise ValueError("The original fixed demonstration prefix is incomplete.")
    if args.resume and args.warm_start:
        raise ValueError("Choose exact resume or a warm-start correction, not both.")
    continuation = load_checkpoint(args.resume or args.warm_start) if args.resume or args.warm_start else loaded
    if args.resume:
        old_definition = continuation.training_state.get("offline_definition", {})
        if old_definition.get("source_sha256") != loaded.sha256 or old_definition.get("frames") != len(frames):
            raise ValueError("Resume requires the identical fixed demonstration source and frame subset.")
    if args.resume and args.angle_head and args.angle_head != continuation.config.model.angle_head:
        raise ValueError("Architecture changes are warm starts, not exact resumes.")
    network = migrate_angle_head(continuation.network, args.angle_head, loaded.config.seed).to(device)
    experiment = loaded.config.model_copy(update={"model": network.config})
    optimizer = torch.optim.Adam(network.parameters(), lr=loaded.config.optimizer.learning_rate)
    if continuation.optimizer_state is not None and not args.warm_start:
        optimizer.load_state_dict(continuation.optimizer_state)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, threshold=0.001, min_lr=1e-6,
    )
    if args.resume and "offline_scheduler" in continuation.training_state:
        scheduler.load_state_dict(continuation.training_state["offline_scheduler"])
    first_step = continuation.training_state.get("optimizer_steps", 0) if args.resume else 0
    source_optimizer_steps = (
        continuation.training_state.get("offline_definition", {}).get("initial_source_optimizer_steps", 0)
        if args.resume else
        continuation.training_state.get("optimizer_steps", 0)
        + continuation.training_state.get("offline_definition", {}).get("initial_source_optimizer_steps", 0)
    )
    if args.steps <= first_step:
        raise ValueError("Target optimizer steps must exceed the resumed counter.")
    packed = pack_sequence(frames, network.config, device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    definition = {
        "kind": "fixed-data-convergence-control", "source_checkpoint": str(args.checkpoint),
        "source_sha256": loaded.sha256, "source": source_fingerprint(),
        "frames": len(frames), "agent_labels": packed.offsets[-1], "device": device,
        "architecture": network.config.model_dump(mode="json"), "teacher": loaded.config.teacher.model_dump(),
        "learning_rate": loaded.config.optimizer.learning_rate,
        "gradient_bound": loaded.config.optimizer.max_grad_norm,
        "target_optimizer_steps": args.steps, "start_optimizer_step": first_step,
        "initial_source_optimizer_steps": source_optimizer_steps,
        "warm_start": str(args.warm_start) if args.warm_start else None,
        "scheduler": {"factor": 0.5, "patience_validation_checks": 3, "relative_improvement": 0.001, "min_lr": 1e-6},
        "recurrence": "full current-weight prefix from zero, no cached collection-time hidden states",
        "claim": "training-data fit only; missing predator/reproduction labels remain unassessed",
    }
    write_json(output / "manifest.json", {**definition, "status": "running"})
    records = []
    milestones = {0, 10, 50, 100, 250, 500, 1000, args.steps}
    started = time.perf_counter()
    best_loss = math.inf
    completed = first_step
    fit_checks = 0
    fit_met = False
    try:
        for step in range(first_step, args.steps + 1):
            if step == first_step or step in milestones or step % args.evaluate_every == 0:
                metrics = evaluate_sequence(network, packed)
                scheduler.step(metrics["loss"])
                row = {"split": "train", "optimizer_steps": step, "data_passes": step,
                       "lineage_optimizer_steps": step + source_optimizer_steps,
                       "learning_rate": optimizer.param_groups[0]["lr"],
                       "elapsed_seconds": time.perf_counter() - started, **metrics}
                records.append(row)
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                state = {"next_update": step, "optimizer_steps": step,
                         "native_ticks_total": 0, "prior_native_ticks": loaded.training_state["native_ticks_total"],
                         "offline_definition": definition, "offline_scheduler": scheduler.state_dict()}
                digest = save_checkpoint(output / "checkpoint.pt", network, experiment, optimizer, state)
                write_json(output / "policy.json", RuntimeConfig(
                    policy="neural", checkpoint=str(output / "checkpoint.pt"), checkpoint_sha256=digest,
                ).model_dump(mode="json"))
                if metrics["loss"] < best_loss:
                    best_loss = metrics["loss"]
                    best_digest = save_checkpoint(output / "best.pt", network, experiment, optimizer, state)
                    write_json(output / "best-policy.json", RuntimeConfig(
                        policy="neural", checkpoint=str(output / "best.pt"), checkpoint_sha256=best_digest,
                    ).model_dump(mode="json"))
                render_curve(records, output)
                print(json.dumps(row), flush=True)
                fit_met = (
                    metrics["stop_energy_mean_per_tick"] is not None
                    and metrics["stop_energy_mean_per_tick"] <= 0.001
                    and metrics["distance_mae"] <= 0.1
                    and (metrics["travel_mae_radians"] is None or metrics["travel_mae_radians"] <= 0.01)
                    and metrics["turn_mae_radians"] <= 0.01
                    and metrics["spawn_fp"] == 0 and metrics["spawn_fn"] == 0
                )
                fit_checks = fit_checks + 1 if fit_met else 0
                if fit_checks >= 2:
                    break
            if step == args.steps:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = sequence_loss(forward_sequence(network, packed), packed)
            loss.backward()
            checked_optimizer_step(network, optimizer, loaded.config.optimizer.max_grad_norm)
            completed += 1
        metrics = evaluate_sequence(network, packed)
        result = {
            "status": "training_fit" if fit_checks >= 2 else "budget_exhausted",
            "optimizer_steps": completed, "metrics": metrics,
            "best_loss": best_loss, "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2 if device == "cuda" else None,
            "teacher_level_qualified": False,
        }
        write_json(output / "summary.json", result)
        write_json(output / "manifest.json", {**definition, "status": "complete", "result": result})
    except BaseException as error:
        write_json(output / "failure.json", {"type": type(error).__name__, "message": str(error), "optimizer_steps": completed})
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--angle-head", choices=("bounded", "vector_bc"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--evaluate-every", type=int, default=50)
    parser.add_argument("--max-vram-mb", type=int, default=6144)
    args = parser.parse_args()
    if args.steps < 1 or args.evaluate_every < 1 or (args.frames is not None and args.frames < 1):
        parser.error("Step, frame and evaluation limits must be positive.")
    print(json.dumps(fit_control(args), indent=2))


if __name__ == "__main__":
    main()
