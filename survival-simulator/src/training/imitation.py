"""Bounded expert-labelled replay and truncated recurrent behavior cloning."""

from __future__ import annotations

from collections import defaultdict, deque
import math

import torch

from src.policies.actions import ACTION_VERSION, imitation_loss
from src.policies.config import ExperimentConfig
from src.policies.features import (
    ENTITY_DIM, FEATURE_VERSION, FEATURE_VERSIONS, SCALAR_DIM, PUBLIC_SCALAR_DIM,
    FeatureBatch, feature_version,
)
from src.policies.networks import PolicyNetwork
from src.training.rollout import Rollout, RolloutFrame, require_finite, sequence_chunks, unroll_sequence
from src.utils.DTOs import ActionRequest, StepResponse


def dagger_schedule(
    config: ExperimentConfig, update: int, previous_round: int = 0,
) -> tuple[float, int]:
    """Spread baseline + DAgger stages over target updates, ending at learner-only."""
    rounds = config.imitation.dagger_rounds
    if rounds >= config.updates:
        raise ValueError("DAgger requires updates >= imitation.dagger_rounds + 1.")
    if not 0 <= update < config.updates or not 0 <= previous_round <= rounds:
        raise ValueError("Invalid update or saved DAgger round.")
    round_index = max(previous_round, min(rounds, update * (rounds + 1) // config.updates))
    probability = config.imitation.teacher_probability
    if rounds:
        probability *= 1.0 - round_index / rounds
    return probability, round_index


def checked_optimizer_step(
    network: PolicyNetwork, optimizer: torch.optim.Optimizer, max_grad_norm: float,
) -> float:
    """Clip an already accumulated gradient; invalid updates must fail, not be skipped."""
    parameters = [parameter for parameter in network.parameters() if parameter.requires_grad]
    if not any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("The training objective produced no parameter gradients.")
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    for parameter in parameters:
        require_finite(parameter, "Updated model parameter")
    return float(norm.detach().cpu().item())


_TENSOR_FIELDS = ("initial_hidden", "latent", "spawn", "eligible", "old_log_prob", "teacher_mask")
_SCALAR_FIELDS = (
    "value", "reward", "terminated", "episode_start", "env_index",
    "world_seed", "episode_step", "collection_index",
)
_JSON_DTYPES = {
    str(dtype).removeprefix("torch."): dtype for dtype in (
        torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        torch.float16, torch.bfloat16, torch.float32, torch.float64,
    )
}


def _tensor_state(tensor: torch.Tensor, json_tensors: bool) -> torch.Tensor | dict:
    tensor = tensor.detach().cpu()
    if not json_tensors:
        return tensor.clone()
    dtype = str(tensor.dtype).removeprefix("torch.")
    if dtype not in _JSON_DTYPES:
        raise ValueError(f"Unsupported imitation snapshot tensor dtype: {dtype}.")
    require_finite(tensor, "Imitation JSON snapshot tensor")
    return {"dtype": dtype, "shape": list(tensor.shape), "data": tensor.tolist()}


def _tensor_from_json(value: dict) -> torch.Tensor:
    if not isinstance(value, dict) or value.get("dtype") not in _JSON_DTYPES:
        raise ValueError("Invalid imitation JSON tensor dtype.")
    shape = value.get("shape")
    if (
        not isinstance(shape, list) or len(shape) > 2
        or any(type(dimension) is not int or dimension < 0 for dimension in shape)
    ):
        raise ValueError("Invalid imitation JSON tensor shape.")
    tensor = torch.tensor(value["data"], dtype=_JSON_DTYPES[value["dtype"]], device="cpu")
    require_finite(tensor, "Restored imitation JSON tensor")
    return tensor.reshape(shape)


def _frame_state(frame: RolloutFrame, *, json_tensors: bool = False) -> dict:
    features = frame.features
    return {
        **{name: _tensor_state(getattr(frame, name), json_tensors) for name in _TENSOR_FIELDS},
        **{name: getattr(frame, name) for name in _SCALAR_FIELDS},
        "features": {
            "agent_ids": list(features.agent_ids),
            "feature_version": features.feature_version,
            **{
                name: _tensor_state(torch.from_numpy(getattr(features, name)), json_tensors)
                for name in ("scalars", "entities", "entity_types", "entity_owners")
            },
        },
        "step": frame.step.model_dump(mode="json"),
        "actions": [action.model_dump(mode="json") for action in frame.actions],
        "teacher_actions": [action.model_dump(mode="json") for action in frame.teacher_actions],
    }


def _restore_frame(state: dict, expected_features: str | None = None) -> RolloutFrame:
    for name in ("terminated", "episode_start"):
        if type(state.get(name)) is not bool:
            raise ValueError(f"Saved imitation {name} must be boolean.")
    for name in ("env_index", "world_seed", "episode_step", "collection_index"):
        value = state.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f"Saved imitation {name} must be a nonnegative integer.")
    if state["world_seed"] > 2**32 - 1:
        raise ValueError("Saved imitation world_seed must be a uint32.")
    for name in ("value", "reward"):
        value = state.get(name)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Saved imitation {name} must be a finite number.")
    data = state["features"]
    agent_ids = tuple(data["agent_ids"])
    if any(type(agent_id) is not int or agent_id < 0 for agent_id in agent_ids):
        raise ValueError("Saved imitation agent IDs must be nonnegative integers.")
    arrays = {}
    for name, dtype in (
        ("scalars", torch.float32), ("entities", torch.float32),
        ("entity_types", torch.int64), ("entity_owners", torch.int64),
    ):
        tensor = data[name]
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype:
            raise ValueError(f"Invalid saved imitation feature: {name}.")
        require_finite(tensor, f"Saved imitation {name}")
        arrays[name] = tensor.detach().cpu().numpy().copy()
    count, entities = len(agent_ids), len(arrays["entities"])
    if (
        not count or len(set(agent_ids)) != count
        or arrays["scalars"].shape not in ((count, SCALAR_DIM), (count, PUBLIC_SCALAR_DIM))
        or arrays["entities"].shape != (entities, ENTITY_DIM)
        or arrays["entity_types"].shape != (entities,)
        or arrays["entity_owners"].shape != (entities,)
        or (entities and (
            arrays["entity_owners"].min() < 0 or arrays["entity_owners"].max() >= count
        ))
    ):
        raise ValueError("Saved imitation features have inconsistent ragged dimensions.")
    schema = data.get("feature_version", expected_features or feature_version(
        arrays["scalars"].shape[1] == PUBLIC_SCALAR_DIM,
    ))
    if (
        schema not in FEATURE_VERSIONS
        or (expected_features is not None and schema != expected_features)
        or arrays["scalars"].shape[1] != (SCALAR_DIM if schema == FEATURE_VERSION else PUBLIC_SCALAR_DIM)
    ):
        raise ValueError("Saved imitation feature schema does not match its arrays or snapshot.")
    tensors = {}
    for name in _TENSOR_FIELDS:
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Invalid saved imitation tensor: {name}.")
        require_finite(tensor, f"Saved imitation {name}")
        tensors[name] = tensor.detach().cpu().clone()
    if (
        tensors["initial_hidden"].ndim != 2 or tensors["initial_hidden"].shape[0] != count
        or tensors["latent"].shape != (count, 3)
        or any(tensors[name].shape != (count,) for name in _TENSOR_FIELDS[2:])
        or tensors["eligible"].dtype != torch.bool or tensors["teacher_mask"].dtype != torch.bool
    ):
        raise ValueError("Saved imitation action/hidden tensors have inconsistent dimensions.")
    step = StepResponse.model_validate(state["step"])
    actions = tuple(ActionRequest.model_validate(action) for action in state["actions"])
    teachers = tuple(ActionRequest.model_validate(action) for action in state["teacher_actions"])
    if (
        tuple(agent.agent_id for agent in step.agent_status) != agent_ids
        or tuple(action.agent_id for action in actions) != agent_ids
        or tuple(action.agent_id for action in teachers) != agent_ids
    ):
        raise ValueError("Saved imitation labels do not match the observed agent identities.")
    return RolloutFrame(
        features=FeatureBatch(agent_ids=agent_ids, feature_version=schema, **arrays), step=step,
        actions=actions, teacher_actions=teachers, **tensors,
        **{name: state[name] for name in _SCALAR_FIELDS},
    )


class ImitationDataset:
    """FIFO by collected team tick, including learner-visited recovery observations.

    A saved boundary may have been produced by an older policy. It initializes
    truncated BPTT; labels and previous-action features always describe the actual
    trajectory, never a counterfactual all-teacher history.
    """

    def __init__(self, max_frames: int):
        if type(max_frames) is not int or max_frames <= 0:
            raise ValueError("An imitation dataset needs a positive max_frames.")
        self.max_frames = max_frames
        self._frames: deque[RolloutFrame] = deque()
        self.total_seen = 0
        self.evicted_frames = 0

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def frames(self) -> tuple[RolloutFrame, ...]:
        return tuple(self._frames)

    def extend(self, rollout: Rollout) -> None:
        if not rollout.frame_count:
            raise ValueError("Cannot add an empty rollout to the imitation dataset.")
        # Time-major insertion makes FIFO eviction fair to synchronous environments.
        for tick in range(max(map(len, rollout.trajectories))):
            for trajectory in rollout.trajectories:
                if tick < len(trajectory):
                    if len(self._frames) == self.max_frames:
                        self._frames.popleft()
                        self.evicted_frames += 1
                    self._frames.append(trajectory[tick])
                    self.total_seen += 1

    def trajectories(self) -> list[tuple[RolloutFrame, ...]]:
        by_environment: dict[int, list[RolloutFrame]] = defaultdict(list)
        for frame in self._frames:
            by_environment[frame.env_index].append(frame)
        return [tuple(by_environment[index]) for index in sorted(by_environment)]

    def summary(self) -> dict:
        return {
            "dataset_frames": len(self), "max_frames": self.max_frames,
            "total_seen": self.total_seen, "evicted_frames": self.evicted_frames,
            "recovery_frames": sum(not frame.teacher_mask.all().item() for frame in self._frames),
            "agent_labels": sum(len(frame.features.agent_ids) for frame in self._frames),
            "dataset_tokens": sum(
                len(frame.features.agent_ids) + len(frame.features.entities) for frame in self._frames
            ),
            "eviction_policy": "oldest_collected_team_tick_first",
        }

    def state_dict(self) -> dict:
        """Bounded checkpoint/artifact payload accepted by torch.load(weights_only=True)."""
        return {
            "version": 1, "max_frames": self.max_frames, "total_seen": self.total_seen,
            "feature_version": self._feature_version(),
            "evicted_frames": self.evicted_frames,
            "frames": [_frame_state(frame) for frame in self._frames],
        }

    def _feature_version(self) -> str:
        schemas = {frame.features.feature_version for frame in self._frames}
        if len(schemas) > 1 or not schemas.issubset(FEATURE_VERSIONS):
            raise ValueError("Imitation replay cannot mix feature schemas.")
        return next(iter(schemas), FEATURE_VERSION)

    def json_snapshot(self) -> dict:
        """Independent JSON-only replay artifact, bounded by max_frames.

        Arrays use {dtype, shape, data} records, including empty ragged arrays.
        Observations, actual/expert actions, previous-action features, cached
        hidden boundaries, proposal samples/masks, raw rewards and world/step
        counters are retained. This can rebuild replay, not a live engine state.
        No dataclasses, NumPy objects or torch tensors appear in the result.
        """
        return {
            "format": "imitation-dataset-json", "version": 1,
            "feature_version": self._feature_version(), "action_version": ACTION_VERSION,
            "max_frames": self.max_frames, "total_seen": self.total_seen,
            "evicted_frames": self.evicted_frames,
            "eviction_policy": "oldest_collected_team_tick_first",
            "frames": [_frame_state(frame, json_tensors=True) for frame in self._frames],
        }

    def load_json_snapshot(self, snapshot: dict) -> None:
        """Rebuild bounded replay from json.loads of a compatible artifact."""
        if (
            snapshot.get("format") != "imitation-dataset-json" or snapshot.get("version") != 1
            or snapshot.get("feature_version") not in FEATURE_VERSIONS
            or snapshot.get("action_version") != ACTION_VERSION
            or snapshot.get("max_frames") != self.max_frames
        ):
            raise ValueError("Incompatible imitation JSON snapshot format/features/actions/max_frames.")
        frames = snapshot.get("frames")
        if not isinstance(frames, list) or len(frames) > self.max_frames:
            raise ValueError("Imitation JSON snapshot exceeds its bounded frame count.")
        restored = []
        for frame in frames:
            features = frame["features"]
            restored.append({
                **frame,
                **{name: _tensor_from_json(frame[name]) for name in _TENSOR_FIELDS},
                "features": {
                    **features,
                    **{
                        name: _tensor_from_json(features[name])
                        for name in ("scalars", "entities", "entity_types", "entity_owners")
                    },
                },
            })
        self.load_state_dict({**snapshot, "frames": restored})

    def load_state_dict(self, state: dict) -> None:
        if state.get("version") != 1 or state.get("max_frames") != self.max_frames:
            raise ValueError("Resume requires the same imitation dataset version and max_frames.")
        schema = state.get("feature_version")
        if schema is not None and schema not in FEATURE_VERSIONS:
            raise ValueError("Incompatible saved imitation feature schema.")
        frames = state.get("frames")
        seen, evicted = state.get("total_seen"), state.get("evicted_frames")
        if (
            not isinstance(frames, list) or len(frames) > self.max_frames
            or type(seen) is not int or type(evicted) is not int
            or evicted < 0 or seen != evicted + len(frames)
        ):
            raise ValueError("Saved imitation dataset counts are inconsistent.")
        restored = [_restore_frame(frame, schema) for frame in frames]
        if len({frame.features.feature_version for frame in restored}) > 1:
            raise ValueError("Imitation replay cannot mix feature schemas.")
        self._frames = deque(restored)
        self.total_seen, self.evicted_frames = seen, evicted


def train_imitation(
    network: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    dataset: ImitationDataset,
    config: ExperimentConfig,
) -> dict:
    """Accumulate token-bounded sequences into an equal-team-tick epoch objective."""
    chunks = sequence_chunks(
        dataset.trajectories(), config.ppo.sequence_length, config.resources.max_tokens,
    )
    frame_count = len(dataset)
    was_training = network.training
    network.train()
    losses, norms = [], []
    try:
        for _ in range(config.imitation.epochs):
            optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            for chunk in chunks:
                terms = []
                for frame, output in unroll_sequence(network, chunk):
                    loss = imitation_loss(output, frame.step, list(frame.teacher_actions))
                    if loss.numel() != 1:
                        raise ValueError("imitation_loss must produce one scalar per team tick.")
                    require_finite(loss, "Imitation loss")
                    terms.append(loss.reshape(()))
                sequence_loss = torch.stack(terms).sum() / frame_count
                require_finite(sequence_loss, "Imitation sequence loss")
                sequence_loss.backward()
                epoch_loss += float(sequence_loss.detach().cpu().item())
            norms.append(checked_optimizer_step(network, optimizer, config.optimizer.max_grad_norm))
            losses.append(epoch_loss)
    finally:
        network.train(was_training)
    return {
        "imitation_loss": losses[-1], "initial_imitation_loss": losses[0],
        "gradient_norm": norms[-1], "optimizer_steps": config.imitation.epochs,
        "sequence_chunks": len(chunks), **dataset.summary(),
    }
