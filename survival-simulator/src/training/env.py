"""An in-process adapter around the unchanged, rendering-inclusive native core."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real

from src.benchmarking.config import SimulationSettings
from src.benchmarking.policies import validate_actions
from src.core import SimulationCore
from src.training.seeds import _validate_seed
from src.utils.DTOs import ActionRequest, ObservationResponse, StepResponse


@dataclass(frozen=True)
class EnvTransition:
    observation: StepResponse
    reward: float
    terminated: bool


def _detached_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Real):
        if not math.isfinite(value):
            raise ValueError("Simulator returned a non-finite perception value.")
        return value
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Simulator perception dictionary keys must be strings.")
        return {key: _detached_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detached_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detached_value(item) for item in value)
    raise ValueError(f"Unsupported simulator perception value: {type(value).__name__}.")


def _observation(state: dict, time_limit: float) -> StepResponse:
    response = StepResponse.model_validate(
        {
            "game_status": "ok",
            "score": state["score"],
            "sim_time": state["sim_time"],
            "n_agents": state["num_agents"],
            "agent_status": [
                ObservationResponse.model_validate(agent, strict=True)
                for agent in state["observations"]
            ],
        },
        strict=True,
    )
    ids = [agent.agent_id for agent in response.agent_status]
    if (
        response.n_agents < 0
        or response.n_agents != len(ids)
        or len(set(ids)) != len(ids)
        or any(agent_id < 0 for agent_id in ids)
    ):
        raise ValueError("Simulator agent count or observation IDs are inconsistent.")
    if not math.isfinite(response.score) or not math.isfinite(response.sim_time):
        raise ValueError("Simulator returned non-finite score or time.")
    if response.sim_time < 0:
        raise ValueError("Simulator returned negative simulated time.")

    for agent in response.agent_status:
        if not all(math.isfinite(value) for value in (
            agent.energy, agent.age, agent.speed, agent.sprint_speed,
            agent.hearing_radius, agent.vision_angle, agent.vision_range, agent.max_energy,
        )):
            raise ValueError(f"Simulator returned non-finite data for agent {agent.agent_id}.")
        # Pydantic owns the outer list/dicts, but Dict values (notably edge coords)
        # are untyped and can still alias the engine's cached perceptions.
        for perception in agent.observations:
            for key, value in perception.items():
                if not isinstance(key, str):
                    raise ValueError("Simulator perception dictionary keys must be strings.")
                perception[key] = _detached_value(value)

    if response.n_agents == 0:
        response.game_status = "game_over"
    elif response.sim_time > time_limit:
        response.game_status = "game_over"
    return response


class EnvironmentAdapter:
    """Expose DTOs to collection code, never the core to a policy actor.

    Reset constructs a fresh native world, including its graphics RNG draws.
    Live cores are neither copied nor serialized; resuming training resets worlds.
    Timing counters describe the current reset/episode and include its bootstrap.
    """

    def __init__(
        self,
        settings: SimulationSettings | None = None,
        *,
        simulation_factory: Callable[..., SimulationCore] = SimulationCore,
    ):
        if settings is not None and not isinstance(settings, SimulationSettings):
            raise TypeError("Settings must be SimulationSettings or None.")
        self.settings = (
            SimulationSettings() if settings is None
            else SimulationSettings.model_validate(settings.model_dump(), strict=True)
        )
        if not callable(simulation_factory):
            raise TypeError("Simulation factory must be callable.")
        self._simulation_factory = simulation_factory
        self._simulation: SimulationCore | None = None
        self._observation: StepResponse | None = None
        self._seed: int | None = None
        self._done = False
        self._needs_reset = True
        self._native_score = 0.0
        self._expected_ids: tuple[int, ...] = ()
        self._initialization_seconds = 0.0
        self._simulation_seconds = 0.0

    @property
    def observation(self) -> StepResponse:
        if self._observation is None:
            raise RuntimeError("Reset the environment before requesting an observation.")
        return self._observation

    @property
    def done(self) -> bool:
        return self._done

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def initialization_seconds(self) -> float:
        return self._initialization_seconds

    @property
    def simulation_seconds(self) -> float:
        return self._simulation_seconds

    def reset(self, seed: int) -> StepResponse:
        _validate_seed(seed)
        self._simulation = None
        self._observation = None
        self._seed = seed
        self._done = False
        self._needs_reset = True
        self._expected_ids = ()
        self._native_score = 0.0
        self._initialization_seconds = 0.0
        self._simulation_seconds = 0.0
        started = time.perf_counter()
        try:
            self._simulation = self._simulation_factory(
                seed=seed, **self.settings.core_kwargs(),
            )
        finally:
            self._initialization_seconds = time.perf_counter() - started
        response = self._tick([])
        self._remember(response)
        return response

    def step(self, actions: Sequence[ActionRequest]) -> EnvTransition:
        if self._simulation is None or self._needs_reset:
            raise RuntimeError("Reset the environment before stepping it.")
        if self._done:
            raise RuntimeError("Cannot step a terminal environment; reset it first.")
        validated = validate_actions(actions, self._expected_ids)
        response = self._tick([(action.agent_id, action) for action in validated])
        reward = response.score - self._native_score
        if not math.isfinite(reward):
            raise ValueError("Simulator score delta is non-finite.")
        self._remember(response)
        return EnvTransition(response, reward, self._done)

    def _tick(self, actions: list[tuple[int, ActionRequest]]) -> StepResponse:
        if self._simulation is None:
            raise RuntimeError("Reset the environment before stepping it.")
        # A failed native tick may already have mutated the world. It cannot be
        # retried using the previous observation; only a fresh reset is safe.
        self._needs_reset = True
        started = time.perf_counter()
        try:
            state = self._simulation.step(actions)
        finally:
            self._simulation_seconds += time.perf_counter() - started
        return _observation(state, self.settings.time_limit)

    def _remember(self, response: StepResponse) -> None:
        self._observation = response
        self._native_score = response.score
        self._expected_ids = tuple(agent.agent_id for agent in response.agent_status)
        self._done = response.game_status == "game_over"
        self._needs_reset = False
