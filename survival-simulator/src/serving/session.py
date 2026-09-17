"""Serialize recurrent policy state only under an explicit single-game contract."""

from collections.abc import Callable
from threading import Lock

from src.benchmarking.config import canonical_json
from src.benchmarking.policies import Policy, validate_actions
from src.policies.features import is_bootstrap, validate_step
from src.utils.DTOs import ActionRequest, StepResponse


class SessionConflict(ValueError):
    """A request cannot be assigned safely to the current recurrent game."""


class PolicySession:
    def __init__(
        self, factory: Callable[[], Policy], *, stateful: bool = False, single_stream: bool = False,
    ):
        if stateful and not single_stream:
            raise ValueError("Recurrent serving requires an explicit single-stream agreement.")
        self.policy = factory()
        self.stateful = stateful
        self._reset = getattr(self.policy, "reset", None)
        if stateful and not callable(self._reset):
            raise TypeError("A stateful policy must provide reset().")
        self._lock = Lock()
        self._last_time: float | None = None
        self._last_request: str | None = None
        self._last_actions: list[ActionRequest] = []

    def predict(self, step: StepResponse) -> list[ActionRequest]:
        validate_step(step)
        with self._lock:
            if not self.stateful:
                if step.game_status == "game_over":
                    return []
                return validate_actions(self.policy.act(step), [a.agent_id for a in step.agent_status])
            signature = canonical_json(step.model_dump(mode="json"))
            if signature == self._last_request:
                return [action.model_copy(deep=True) for action in self._last_actions]
            bootstrap = is_bootstrap(step)
            new_first_tick = (
                self._last_time is not None and step.sim_time < self._last_time
                and 0 < step.sim_time <= 0.100001
            )
            if bootstrap or new_first_tick:
                self._reset()
                self._last_time = None
            elif self._last_time is not None and step.sim_time <= self._last_time:
                raise SessionConflict("Conflicting or out-of-order request; recurrent games cannot interleave.")
            if step.game_status == "game_over":
                self._reset()
                actions = []
            elif bootstrap:
                actions = []
            else:
                actions = validate_actions(self.policy.act(step), [a.agent_id for a in step.agent_status])
            self._last_time = step.sim_time
            self._last_request = signature
            self._last_actions = [action.model_copy(deep=True) for action in actions]
            return actions
