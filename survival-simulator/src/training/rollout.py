"""Synchronous, observation-only collection and bounded recurrent unrolling.

Rollouts live on CPU. Cached initial hidden states are detached truncation
boundaries, not full-history BPTT. Within a sequence, surviving IDs retain the
autograd graph; births and episode resets do not inherit another agent's state.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass

import torch

from src.benchmarking.policies import validate_actions
from src.policies.actions import sample_actions
from src.policies.config import ExperimentConfig
from src.policies.features import FeatureBatch, encode_step
from src.policies.heuristic import build_policy
from src.policies.memory import PolicyMemory
from src.policies.networks import PolicyNetwork, PolicyOutput
from src.training.env import EnvironmentAdapter
from src.training.seeds import training_seeds
from src.training.workers import EnvironmentPool
from src.utils.DTOs import ActionRequest, StepResponse


@dataclass(frozen=True)
class RolloutFrame:
    features: FeatureBatch
    step: StepResponse
    initial_hidden: torch.Tensor
    latent: torch.Tensor
    spawn: torch.Tensor
    eligible: torch.Tensor
    old_log_prob: torch.Tensor
    value: float
    actions: tuple[ActionRequest, ...]
    teacher_actions: tuple[ActionRequest, ...]
    teacher_mask: torch.Tensor
    reward: float
    terminated: bool
    episode_start: bool
    env_index: int
    world_seed: int
    episode_step: int
    collection_index: int

    # The latent/log-prob fields describe the learner proposal. When teacher_mask
    # is true, actions instead contains the executed expert action; PPO rejects it.


@dataclass(frozen=True)
class Rollout:
    trajectories: tuple[tuple[RolloutFrame, ...], ...]
    last_values: tuple[float, ...]
    policy_version: int
    metrics: dict[str, int | float]

    @property
    def frame_count(self) -> int:
        return sum(len(trajectory) for trajectory in self.trajectories)


def require_finite(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"{name} contains non-finite values.")


def raw_team_value(output: PolicyOutput) -> float:
    if output.value.numel() != 1:
        raise ValueError("The critic must produce exactly one raw team value per tick.")
    require_finite(output.value, "Team value")
    return float(output.value.detach().cpu().item())


def feature_tokens(features: FeatureBatch, max_tokens: int) -> int:
    count = len(features.agent_ids) + len(features.entities)
    if not features.agent_ids:
        raise ValueError("A learned frame requires at least one living-agent observation.")
    if count > max_tokens:
        raise ValueError(
            f"A single frame needs {count} tokens, exceeding resources.max_tokens="
            f"{max_tokens}. Increase the budget or use a smaller world; agents/entities "
            "will not be truncated."
        )
    return count


def align_hidden(
    previous_ids: Sequence[int],
    agent_ids: Sequence[int],
    hidden: torch.Tensor | None,
    *,
    hidden_size: int,
    device: torch.device | str,
    reset: bool = False,
) -> torch.Tensor:
    """Gather surviving rows without detaching; zero new IDs or an episode reset."""
    if len(set(previous_ids)) != len(previous_ids) or len(set(agent_ids)) != len(agent_ids):
        raise ValueError("Recurrent identity alignment requires unique agent IDs.")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive.")
    if hidden is not None and hidden.shape != (len(previous_ids), hidden_size):
        raise ValueError("Hidden-state rows do not match the preceding agent identities.")
    if hidden is None:
        return torch.zeros((len(agent_ids), hidden_size), device=device)
    if hidden.device != torch.device(device):
        raise ValueError("Differentiable hidden alignment must stay on the policy device.")
    if reset or not agent_ids:
        return hidden.new_zeros((len(agent_ids), hidden_size))
    positions = {agent_id: index for index, agent_id in enumerate(previous_ids)}
    return torch.stack([
        hidden[positions[agent_id]] if agent_id in positions else hidden.new_zeros(hidden_size)
        for agent_id in agent_ids
    ])


def sequence_chunks(
    trajectories: Sequence[Sequence[RolloutFrame]],
    sequence_length: int,
    max_tokens: int,
) -> list[tuple[RolloutFrame, ...]]:
    """Split per environment/episode and token budget, retaining every whole frame."""
    if sequence_length <= 0 or max_tokens <= 0:
        raise ValueError("Sequence length and token budget must be positive.")
    chunks = []
    for trajectory in trajectories:
        current: list[RolloutFrame] = []
        tokens = 0
        for frame in trajectory:
            frame_tokens = feature_tokens(frame.features, max_tokens)
            previous = current[-1] if current else None
            discontinuity = previous is not None and (
                frame.episode_start or previous.terminated
                or frame.env_index != previous.env_index
                or frame.world_seed != previous.world_seed
                or frame.episode_step != previous.episode_step + 1
            )
            if current and (
                discontinuity or len(current) >= sequence_length
                or tokens + frame_tokens > max_tokens
            ):
                chunks.append(tuple(current))
                current, tokens = [], 0
            current.append(frame)
            tokens += frame_tokens
        if current:
            chunks.append(tuple(current))
    if not chunks:
        raise ValueError("Cannot train on an empty rollout or imitation dataset.")
    return chunks


def unroll_sequence(
    network: PolicyNetwork, frames: Sequence[RolloutFrame],
) -> Iterator[tuple[RolloutFrame, PolicyOutput]]:
    """Use a cached detached boundary, then differentiable identity-aligned recurrence."""
    if not frames:
        raise ValueError("A recurrent sequence cannot be empty.")
    device = next(network.parameters()).device
    recurrent = network.config.memory == "gru"
    first = frames[0]
    hidden = first.initial_hidden.detach().to(device) if recurrent else None
    previous_ids = first.features.agent_ids
    previous = None
    for frame in frames:
        if previous is not None and not frame.episode_start and (
            previous.terminated or previous.env_index != frame.env_index
            or previous.world_seed != frame.world_seed
            or previous.episode_step + 1 != frame.episode_step
        ):
            raise ValueError("A recurrent sequence crossed a discontinuity without a reset.")
        if recurrent:
            hidden = align_hidden(
                previous_ids, frame.features.agent_ids, hidden,
                hidden_size=network.config.hidden_size, device=device,
                reset=frame.episode_start,
            )
        output = network(frame.features, hidden)
        yield frame, output
        hidden = output.next_hidden if recurrent else None
        previous_ids = frame.features.agent_ids
        previous = frame


class RolloutCollector:
    """One frozen policy version per synchronous rollout, with independent env state."""

    def __init__(
        self,
        config: ExperimentConfig,
        network: PolicyNetwork,
        *,
        emit: Callable[[dict], None],
        state: dict | None = None,
        environment_factory: Callable[[], EnvironmentAdapter] | None = None,
        pool_factory: Callable[..., EnvironmentPool] | None = None,
        teacher_factory: Callable[..., object] | None = None,
    ):
        self.config = config
        self.network = network
        self.emit = emit
        self.device = next(network.parameters()).device
        self.workers = config.resources.workers
        self._environment_factory = environment_factory or EnvironmentAdapter
        self._pool_factory = pool_factory or EnvironmentPool
        self._teacher_factory = teacher_factory or build_policy
        self._stack: ExitStack | None = None
        self.pool = None
        self.environment = None
        self.memories = [PolicyMemory() for _ in range(self.workers)]
        self.policy_generators = [
            torch.Generator(device=self.device).manual_seed(config.seed + 104729 * (index + 1))
            for index in range(self.workers)
        ]
        self.teacher_generators = [
            torch.Generator(device="cpu").manual_seed(config.seed + 1000003 * (index + 1))
            for index in range(self.workers)
        ]
        self.seed_index = self.frame_count = self.rollout_count = self.completed_episodes = 0
        self._resuming = state is not None
        if state is not None:
            self._restore_state(state)
        self._seeds = training_seeds(config.seed, max(64, self.seed_index * 2))
        self.observations: list[StepResponse] = []
        self.world_seeds = [0] * self.workers
        self.episode_returns = [0.0] * self.workers
        self.episode_steps = [0] * self.workers
        self.episode_starts = [True] * self.workers
        self.done = [True] * self.workers
        self.teachers: list[object] = []

    def _restore_state(self, state: dict) -> None:
        if state.get("seed") != self.config.seed or state.get("workers") != self.workers:
            raise ValueError("Resume requires the same collection seed and worker count.")
        if state.get("policy_rng_device") != str(self.device):
            raise ValueError("Resume requires the same policy RNG device; use a warm start to change it.")
        for name in ("seed_index", "frame_count", "rollout_count", "completed_episodes"):
            value = state.get(name)
            if type(value) is not int or value < 0:
                raise ValueError(f"Invalid collector counter: {name}.")
            setattr(self, name, value)
        for name, generators in (
            ("policy_rng_states", self.policy_generators),
            ("teacher_rng_states", self.teacher_generators),
        ):
            states = state.get(name)
            if not isinstance(states, list) or len(states) != self.workers:
                raise ValueError(f"Resume requires one {name} entry per environment.")
            for generator, rng_state in zip(generators, states, strict=True):
                if not isinstance(rng_state, torch.Tensor):
                    raise ValueError(f"{name} entries must be torch RNG tensors.")
                generator.set_state(rng_state.cpu())

    def state_dict(self) -> dict:
        return {
            "seed": self.config.seed, "workers": self.workers,
            "seed_index": self.seed_index, "frame_count": self.frame_count,
            "rollout_count": self.rollout_count, "completed_episodes": self.completed_episodes,
            "policy_rng_device": str(self.device),
            "policy_rng_states": [rng.get_state().cpu().clone() for rng in self.policy_generators],
            "teacher_rng_states": [rng.get_state().cpu().clone() for rng in self.teacher_generators],
        }

    def _next_seed(self) -> int:
        if self.seed_index == len(self._seeds):
            self._seeds = training_seeds(self.config.seed, max(64, len(self._seeds) * 2))
        seed = int(self._seeds[self.seed_index])
        self.seed_index += 1
        return seed

    def _record_reset(self, index: int, observation: StepResponse, seed: int) -> None:
        if not math.isfinite(observation.score):
            raise FloatingPointError("The bootstrap episode score is not finite.")
        self.emit({
            "event": "episode_start", "env_index": index, "world_seed": seed,
            "initial_episode_return": float(observation.score),
            "bootstrap_is_learned_frame": False,
        })
        if observation.game_status != "ok" or not observation.agent_status:
            raise ValueError(
                f"World {seed} ended during its empty bootstrap tick; no learned frame exists."
            )
        self.memories[index].reset()
        self.world_seeds[index] = seed
        self.episode_returns[index] = float(observation.score)
        self.episode_steps[index] = 0
        self.episode_starts[index] = True
        self.done[index] = False
        teacher = self._teacher_factory(seed, self.config.heuristic)
        if index == len(self.teachers):
            self.teachers.append(teacher)
        else:
            self.teachers[index] = teacher

    def _release_environments(self) -> None:
        """Drop native cores after pool cleanup, without process-global Pygame shutdown."""
        self.environment = None
        self.pool = None

    def __enter__(self) -> RolloutCollector:
        if self._stack is not None:
            raise RuntimeError("The rollout collector is already open.")
        with ExitStack() as stack:
            stack.callback(self._release_environments)
            if self.workers == 1:
                self.environment = (
                    self._environment_factory()
                    if self.config.resources.action_repeat == 1
                    else self._environment_factory(
                        action_repeat=self.config.resources.action_repeat,
                    )
                )
            else:
                pool_kwargs = {
                    "timeout_seconds": self.config.resources.worker_timeout_seconds,
                }
                if self.config.resources.action_repeat != 1:
                    pool_kwargs["action_repeat"] = self.config.resources.action_repeat
                self.pool = stack.enter_context(self._pool_factory(self.workers, **pool_kwargs))
            seeds = [self._next_seed() for _ in range(self.workers)]
            self.observations = (
                self.pool.reset(seeds) if self.pool is not None
                else [self.environment.reset(seeds[0])]
            )
            if len(self.observations) != self.workers:
                raise ValueError("Environment reset returned the wrong number of observations.")
            for index, (observation, seed) in enumerate(zip(self.observations, seeds, strict=True)):
                self._record_reset(index, observation, seed)
            if self._resuming:
                self.emit({
                    "event": "environments_reset_on_resume", "workers": self.workers,
                    "seed_index": self.seed_index, "exact_engine_resume": False,
                    "recurrent_state_reset": True, "policy_and_teacher_rng_restored": True,
                })
            self._stack = stack.pop_all()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._stack is not None:
            stack, self._stack = self._stack, None
            return stack.__exit__(exc_type, exc, traceback)
        return False

    def _reset_finished(self) -> None:
        for index, done in enumerate(self.done):
            if done:
                seed = self._next_seed()
                observation = (
                    self.pool.reset_at(index, seed) if self.pool is not None
                    else self.environment.reset(seed)
                )
                self.observations[index] = observation
                self._record_reset(index, observation, seed)

    def collect(
        self, steps: int, *, policy_version: int, teacher_probability: float = 0.0,
    ) -> Rollout:
        if self._stack is None:
            raise RuntimeError("Open the collector as a context manager before collection.")
        if type(steps) is not int or steps <= 0:
            raise ValueError("A rollout must contain a positive number of environment ticks.")
        if type(policy_version) is not int or policy_version < 0:
            raise ValueError("The frozen rollout policy version must be a nonnegative update index.")
        if not math.isfinite(teacher_probability) or not 0 <= teacher_probability <= 1:
            raise ValueError("teacher_probability must lie in [0, 1].")
        started = time.perf_counter()
        trajectories: list[list[RolloutFrame]] = [[] for _ in range(self.workers)]
        peak_tokens = agent_actions = 0
        completed_before = self.completed_episodes
        was_training = self.network.training
        self.network.eval()
        try:
            with torch.no_grad():
                for _ in range(steps):
                    self._reset_finished()
                    pending = []
                    batch_actions = []
                    for index, observation in enumerate(self.observations):
                        memory = self.memories[index]
                        features = encode_step(observation, memory.previous_actions)
                        peak_tokens = max(
                            peak_tokens, feature_tokens(features, self.config.resources.max_tokens),
                        )
                        hidden = memory.prepare(
                            features.agent_ids, self.network.config.hidden_size, self.device,
                        )
                        initial_hidden = hidden.detach().cpu().clone()
                        output = self.network(
                            features, hidden if self.network.config.memory == "gru" else None,
                        )
                        value = raw_team_value(output)
                        sample = sample_actions(
                            output, observation, generator=self.policy_generators[index],
                        )
                        for name in ("latent", "spawn", "log_prob"):
                            require_finite(getattr(sample, name), f"Collected {name}")
                        teacher_by_id = {
                            action.agent_id: action for action in validate_actions(
                                self.teachers[index].act(observation), features.agent_ids,
                            )
                        }
                        teacher_actions = [teacher_by_id[agent_id] for agent_id in features.agent_ids]
                        if tuple(action.agent_id for action in sample.actions) != features.agent_ids:
                            raise ValueError("Sampled actions must cover every living ID in observation order.")
                        teacher_mask = torch.rand(
                            len(features.agent_ids), generator=self.teacher_generators[index],
                        ) < teacher_probability
                        actions = [
                            teacher_actions[row] if teacher_mask[row].item() else sample.actions[row]
                            for row in range(len(features.agent_ids))
                        ]
                        pending.append((
                            features, observation.model_copy(deep=True), initial_hidden,
                            sample, value, tuple(action.model_copy(deep=True) for action in actions),
                            tuple(action.model_copy(deep=True) for action in teacher_actions),
                            teacher_mask, output.next_hidden,
                        ))
                        batch_actions.append(actions)
                    transitions = (
                        self.pool.step(batch_actions) if self.pool is not None
                        else [self.environment.step(batch_actions[0])]
                    )
                    if len(transitions) != self.workers:
                        raise ValueError("Environment step returned the wrong number of transitions.")
                    for index, (transition, record) in enumerate(zip(transitions, pending, strict=True)):
                        features, step, hidden, sample, value, actions, teachers, mask, next_hidden = record
                        reward = float(transition.reward)
                        if not math.isfinite(reward):
                            raise FloatingPointError("Native team reward is not finite.")
                        trajectories[index].append(RolloutFrame(
                            features, step, hidden, sample.latent.detach().cpu().clone(),
                            sample.spawn.detach().cpu().clone(), sample.eligible.detach().cpu().clone(),
                            sample.log_prob.detach().cpu().clone(), value, actions, teachers, mask,
                            reward, bool(transition.terminated), self.episode_starts[index],
                            index, self.world_seeds[index], self.episode_steps[index], self.frame_count,
                        ))
                        self.memories[index].commit(features.agent_ids, next_hidden, list(actions))
                        self.frame_count += 1
                        agent_actions += len(features.agent_ids)
                        self.episode_returns[index] += reward
                        self.episode_steps[index] += 1
                        self.episode_starts[index] = False
                        self.done[index] = bool(transition.terminated)
                        self.observations[index] = transition.observation
                        if transition.terminated:
                            self.completed_episodes += 1
                            self.emit({
                                "event": "episode_end", "env_index": index,
                                "world_seed": self.world_seeds[index],
                                "episode_return": self.episode_returns[index],
                                "learned_ticks": self.episode_steps[index],
                            })
                    del pending
                last_values = []
                for index, observation in enumerate(self.observations):
                    if self.done[index]:
                        last_values.append(0.0)
                        continue
                    memory = self.memories[index]
                    features = encode_step(observation, memory.previous_actions)
                    feature_tokens(features, self.config.resources.max_tokens)
                    hidden = memory.prepare(features.agent_ids, self.network.config.hidden_size, self.device)
                    output = self.network(
                        features, hidden if self.network.config.memory == "gru" else None,
                    )
                    last_values.append(raw_team_value(output))
                    # This lookahead is not an action: do not commit its hidden state.
        finally:
            self.network.train(was_training)
        self.rollout_count += 1
        return Rollout(
            tuple(tuple(trajectory) for trajectory in trajectories), tuple(last_values), policy_version,
            {
                "frames": steps * self.workers, "agent_actions": agent_actions,
                "action_repeat": self.config.resources.action_repeat,
                "native_steps": steps * self.workers * self.config.resources.action_repeat,
                "peak_frame_tokens": peak_tokens,
                "completed_episodes": self.completed_episodes - completed_before,
                "collection_seconds": time.perf_counter() - started,
            },
        )
