import importlib
import inspect
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from src.benchmarking.config import BASELINE_NAME, PolicySpec
from src.utils.DTOs import ActionRequest, StepResponse
from src.utils.controllers.dummy_agent_policy import action_decision


@runtime_checkable
class Policy(Protocol):
    def act(self, step: StepResponse) -> Sequence[ActionRequest]: ...


class PolicyFactory(Protocol):
    def __call__(self, seed: int, config: dict[str, JsonValue]) -> Policy: ...


class RandomPolicy:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def act(self, step: StepResponse) -> list[ActionRequest]:
        return [action_decision(agent.model_dump(), self.rng) for agent in step.agent_status]


def create_random_policy(seed: int, config: dict[str, JsonValue]) -> RandomPolicy:
    if config:
        raise ValueError("The canonical random baseline has no configurable parameters.")
    return RandomPolicy(seed)


@dataclass(frozen=True)
class LoadedPolicy:
    spec: PolicySpec
    factory: PolicyFactory
    source_files: tuple[Path, ...]


def load_policy(
    reference: str, *, label: str | None = None, config: dict[str, JsonValue] | None = None,
) -> LoadedPolicy:
    options = {} if config is None else config
    if reference in ("random", BASELINE_NAME):
        if options:
            raise ValueError("The canonical random baseline has no configurable parameters.")
        source = inspect.getsourcefile(action_decision)
        factory = create_random_policy
        name = BASELINE_NAME
        reference = BASELINE_NAME
    else:
        module_name, separator, attribute = reference.partition(":")
        if not separator or not attribute.isidentifier() or not all(
            part.isidentifier() for part in module_name.split(".")
        ):
            raise ValueError("Policy must be 'random' or an importable 'module:factory'.")
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute, None)
        if not callable(factory):
            raise ValueError(f"Policy factory {reference!r} does not exist or is not callable.")
        source = inspect.getsourcefile(factory)
        name = reference
    if source is None:
        raise ValueError(f"Cannot identify the Python source for policy {reference!r}.")
    return LoadedPolicy(
        spec=PolicySpec(reference=reference, label=label or name, config=options),
        factory=factory,
        source_files=(Path(source).resolve(),),
    )


def validate_actions(
    proposed: Sequence[ActionRequest], expected_ids: Sequence[int],
) -> list[ActionRequest]:
    if not isinstance(proposed, Sequence) or isinstance(proposed, (str, bytes)):
        raise ValueError("Policy.act must return a sequence of ActionRequest objects.")
    actions = []
    for action in proposed:
        if not isinstance(action, ActionRequest):
            raise ValueError("Each policy action must be an ActionRequest object.")
        validated = ActionRequest.model_validate(action.model_dump(), strict=True)
        if not all(math.isfinite(value) for value in (
            validated.move_distance, validated.move_direction, validated.turn_angle,
        )):
            raise ValueError(f"Agent {validated.agent_id} returned a non-finite action.")
        actions.append(validated)
    actual_ids = [action.agent_id for action in actions]
    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError("Policy returned duplicate agent IDs.")
    if set(actual_ids) != set(expected_ids):
        raise ValueError(
            f"Policy actions must cover exactly the living agents: "
            f"expected {sorted(expected_ids)}, received {sorted(actual_ids)}."
        )
    return actions
