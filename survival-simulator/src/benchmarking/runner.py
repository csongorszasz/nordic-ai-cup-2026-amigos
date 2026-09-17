import copy
import math
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager

import numpy as np
from pydantic import JsonValue

from src.benchmarking.config import (
    EpisodeCase, EpisodeResult, FailureInfo, SimulationSettings, Timings,
)
from src.benchmarking.policies import Policy, PolicyFactory, validate_actions
from src.core import SimulationCore
from src.utils.DTOs import ObservationResponse, StepResponse


class EpisodeExecutionError(RuntimeError):
    def __init__(self, result: EpisodeResult):
        self.result = result
        failure = result.failure
        super().__init__(f"{result.case.case_id}: {failure.stage}: {failure.message}")


class EpisodeInterrupted(KeyboardInterrupt):
    def __init__(self, result: EpisodeResult):
        self.result = result
        super().__init__("Episode interrupted.")


@contextmanager
def _timed(totals: dict[str, float], key: str, samples: list[float] | None = None):
    started = time.perf_counter_ns()
    try:
        yield
    finally:
        duration = (time.perf_counter_ns() - started) / 1e9
        totals[key] += duration
        if samples is not None:
            samples.append(duration * 1000)


def _observation(state: dict) -> StepResponse:
    response = StepResponse(
        game_status="ok",
        score=state["score"],
        sim_time=state["sim_time"],
        n_agents=state["num_agents"],
        agent_status=[
            ObservationResponse.model_validate(copy.deepcopy(observation))
            for observation in state["observations"]
        ],
    )
    ids = [agent.agent_id for agent in response.agent_status]
    if response.n_agents != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Simulator agent count or observation IDs are inconsistent.")
    if not math.isfinite(response.score) or not math.isfinite(response.sim_time):
        raise ValueError("Simulator returned non-finite score or time.")
    if response.sim_time < 0:
        raise ValueError("Simulator returned negative simulated time.")
    return response


def run_episode(
    case: EpisodeCase,
    factory: PolicyFactory,
    config: dict[str, JsonValue],
    settings: SimulationSettings | None = None,
    max_steps: int | None = None,
    *,
    simulation_factory: Callable[..., SimulationCore] = SimulationCore,
) -> EpisodeResult:
    settings = settings or SimulationSettings()
    if max_steps is not None and (
        isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1
    ):
        raise ValueError("The diagnostic step cap must be a positive integer.")
    started = time.perf_counter_ns()
    totals = {
        "initialization_seconds": 0.0,
        "policy_construction_seconds": 0.0,
        "simulation_seconds": 0.0,
        "policy_seconds": 0.0,
    }
    latencies: list[float] = []
    initial_agents = None
    peak_agents = None
    last_step = None
    ticks = 0
    population_sum = 0
    stage = "simulation_initialization"

    def result(status, termination, failure=None):
        mean_ms = float(np.mean(latencies)) if latencies else None
        p50_ms, p95_ms = (
            (float(value) for value in np.percentile(latencies, [50, 95]))
            if latencies else (None, None)
        )
        return EpisodeResult(
            case=case,
            status=status,
            termination=termination,
            score=last_step.score if last_step and status in ("ok", "truncated") else None,
            sim_time=last_step.sim_time if last_step else None,
            survival_seconds=min(last_step.sim_time, settings.time_limit) if last_step else None,
            ticks=ticks,
            completed=termination == "time_limit",
            initial_agents=initial_agents,
            final_agents=last_step.n_agents if last_step else None,
            peak_agents=peak_agents,
            mean_population=population_sum / ticks if ticks else None,
            timings=Timings(
                **totals,
                episode_seconds=(time.perf_counter_ns() - started) / 1e9,
                policy_calls=len(latencies),
                batch_mean_ms=mean_ms,
                batch_p50_ms=p50_ms,
                batch_p95_ms=p95_ms,
                batch_max_ms=max(latencies) if latencies else None,
            ),
            failure=failure,
        )

    try:
        with _timed(totals, "initialization_seconds"):
            sim = simulation_factory(seed=case.world_seed, **settings.core_kwargs())
        initial_agents = len(sim.env.agents)
        peak_agents = initial_agents
        stage = "policy_construction"
        with _timed(totals, "policy_construction_seconds"):
            policy = factory(seed=case.policy_seed, config=copy.deepcopy(config))
            if not isinstance(policy, Policy) or not callable(policy.act):
                raise TypeError("Policy factory must return an object with an act(step) method.")
        actions = []
        while True:
            stage = "simulation_step"
            with _timed(totals, "simulation_seconds"):
                state = sim.step(actions)
            stage = "observations"
            last_step = _observation(state)
            ticks += 1
            population_sum += last_step.n_agents
            peak_agents = max(peak_agents, last_step.n_agents)
            if last_step.n_agents == 0:
                return result("ok", "extinction")
            if last_step.sim_time > settings.time_limit:
                return result("ok", "time_limit")
            if max_steps is not None and ticks >= max_steps:
                return result("truncated", "step_limit")

            expected_ids = tuple(agent.agent_id for agent in last_step.agent_status)
            policy_input = last_step.model_copy(deep=True)
            stage = "policy_decision"
            with _timed(totals, "policy_seconds", latencies):
                proposed = policy.act(policy_input)
            stage = "action_validation"
            validated = validate_actions(proposed, expected_ids)
            actions = [(action.agent_id, action) for action in validated]
    except KeyboardInterrupt as exc:
        failure = FailureInfo(stage=stage, error_type=type(exc).__name__, message="Interrupted by user.")
        raise EpisodeInterrupted(result("interrupted", "interrupted", failure)) from exc
    except (Exception, SystemExit) as exc:
        # Preserve arbitrary plugin/engine failures at the episode boundary, then propagate them.
        failure = FailureInfo(
            stage=stage, error_type=type(exc).__name__, message=str(exc),
            traceback=traceback.format_exc(),
        )
        raise EpisodeExecutionError(result("failed", "error", failure)) from exc
