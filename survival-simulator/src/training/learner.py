"""Learning orchestration; the caller owns devices, optimizer loading and artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable

import torch

from src.policies.config import ExperimentConfig
from src.policies.networks import PolicyNetwork
from src.training.imitation import ImitationDataset, dagger_schedule, train_imitation
from src.training.ppo import ACTOR_DIVISOR, RunningReturnStats, train_ppo
from src.training.rollout import RolloutCollector


TRAINING_STATE_VERSION = 1
MAX_EVENT_BYTES = 16384


def _validate_event(event: dict) -> None:
    encoded = json.dumps(event, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_EVENT_BYTES:
        raise ValueError("Learning metrics exceeded the bounded event size; do not emit rollout/dataset objects.")


def run_learning(
    config: ExperimentConfig,
    network: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    *,
    emit: Callable[[dict], None],
    checkpoint: Callable[[dict], None],
    start_update: int = 0,
    training_state: dict | None = None,
    dataset_callback: Callable[[dict], None] | None = None,
) -> dict:
    """Run to TARGET total updates, checkpointing after each successful update.

    Checkpoints contain consumed world-seed counters, independent policy/mixture
    RNGs, raw-return scale statistics, and bounded imitation replay (when used).
    Replay's state_dict is a weights-only-safe dataset artifact, never a metrics
    payload. Environments/history explicitly restart on resume; no exact engine
    continuation is claimed. The caller must restore network and optimizer.

    Optional dataset_callback receives ImitationDataset.json_snapshot() after
    each successful imitation checkpoint, augmented with next_update, the JSON
    experiment_config, collection_seed_index, and current DAgger mixture. The
    caller may atomically write gzip JSON. Errors propagate; regenerate a failed
    artifact by calling dataset.load_state_dict(state["imitation_dataset"]), then
    dataset.json_snapshot(). PPO never invokes this callback, and JSON replay is
    not materialized when it is omitted.
    """
    if config.mode not in ("imitation", "ppo"):
        raise ValueError("run_learning requires mode='imitation' or mode='ppo'.")
    if type(start_update) is not int or not 0 <= start_update < config.updates:
        raise ValueError("start_update must be in [0, updates); updates is the target TOTAL count.")
    if network.config != config.model:
        raise ValueError("The supplied policy architecture does not match config.model.")
    if start_update and training_state is None:
        raise ValueError("Resuming an update index requires the corresponding saved training_state.")
    collector_state = None
    optimizer_steps = 0
    dagger_round = 0
    normalizer = RunningReturnStats()
    dataset = ImitationDataset(config.imitation.max_frames) if config.mode == "imitation" else None
    if training_state is not None:
        if (
            training_state.get("version") != TRAINING_STATE_VERSION
            or training_state.get("mode") != config.mode
            or type(training_state.get("next_update")) is not int
            or training_state.get("next_update") != start_update
        ):
            raise ValueError("Saved learning mode/version/next_update does not match this resume.")
        collector_state = training_state.get("collector")
        optimizer_steps = training_state.get("optimizer_steps")
        dagger_round = training_state.get("dagger_round")
        if (
            not isinstance(collector_state, dict)
            or type(optimizer_steps) is not int or optimizer_steps < 0
            or type(dagger_round) is not int or dagger_round < 0
        ):
            raise ValueError("Saved learning counters or collector state are invalid.")
        normalizer.load_state_dict(training_state["value_normalizer"])
        if dataset is not None:
            dataset.load_state_dict(training_state["imitation_dataset"])
    if dataset is not None:
        dagger_schedule(config, start_update, dagger_round)
    initial_optimizer_steps = optimizer_steps

    def emit_checked(event: dict) -> None:
        _validate_event(event)
        emit(event)

    emit_checked({
        "event": "learning_semantics", "mode": config.mode,
        "start_update": start_update, "target_total_updates": config.updates,
        "raw_native_team_rewards": True, "actor_divisor": ACTOR_DIVISOR,
        "entropy": "per_agent_mean_analytic_latent_surrogate_then_mean_team_ticks",
        "value_normalization": "scale_squared_raw_errors_only_std_floor_1",
        "recurrence": "detached_collected_boundary_then_identity_aligned_truncated_BPTT",
        "sequence_microbatches": "gradient_accumulation_equal_weight_per_team_tick",
        "action_repeat": config.resources.action_repeat,
        "dataset_checkpointed": dataset is not None,
        "dataset_json_callback": dataset is not None and dataset_callback is not None,
    })
    last_metrics = {}
    with RolloutCollector(config, network, emit=emit_checked, state=collector_state) as collector:
        for update in range(start_update, config.updates):
            teacher_probability = 0.0
            if dataset is not None:
                teacher_probability, dagger_round = dagger_schedule(config, update, dagger_round)
            rollout = collector.collect(
                config.rollout_steps, policy_version=update, teacher_probability=teacher_probability,
            )
            if not rollout.frame_count:
                raise ValueError("Collection produced no actionable frames; the update cannot succeed.")
            if dataset is not None:
                dataset.extend(rollout)
                metrics = train_imitation(network, optimizer, dataset, config)
            else:
                metrics = train_ppo(
                    network, optimizer, rollout, config, update=update, value_normalizer=normalizer,
                )
            successful_steps = metrics["optimizer_steps"]
            if type(successful_steps) is not int or successful_steps <= 0:
                raise RuntimeError("The learner did not perform a successful optimizer step.")
            optimizer_steps += successful_steps
            last_metrics = {
                **rollout.metrics, **metrics,
                "teacher_probability": teacher_probability, "dagger_round": dagger_round,
            }
            event = {
                "event": "learning_update", "update": update + 1, "policy_version": update,
                "target_total_updates": config.updates,
                "cumulative_optimizer_steps": optimizer_steps, **last_metrics,
            }
            _validate_event(event)
            state = {
                "version": TRAINING_STATE_VERSION, "mode": config.mode,
                "next_update": update + 1, "optimizer_steps": optimizer_steps,
                "dagger_round": dagger_round, "collector": collector.state_dict(),
                "value_normalizer": normalizer.state_dict(),
            }
            if dataset is not None:
                state["imitation_dataset"] = dataset.state_dict()
            checkpoint(state)
            if dataset is not None and dataset_callback is not None:
                snapshot = {
                    **dataset.json_snapshot(), "next_update": update + 1,
                    "experiment_config": config.model_dump(mode="json"),
                    "collection_seed_index": collector.seed_index,
                    "dagger_round": dagger_round, "teacher_probability": teacher_probability,
                }
                dataset_callback(snapshot)
            emit_checked(event)
        result = {
            "mode": config.mode, "next_update": config.updates,
            "updates_completed_this_run": config.updates - start_update,
            "optimizer_steps": optimizer_steps,
            "optimizer_steps_this_run": optimizer_steps - initial_optimizer_steps,
            "collection_frames": collector.frame_count, "seed_index": collector.seed_index,
            "completed_episodes": collector.completed_episodes,
            "environments_reset_on_resume": training_state is not None,
            "last_update": last_metrics,
        }
    _validate_event(result)
    return result
