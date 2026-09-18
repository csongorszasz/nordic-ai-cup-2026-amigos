"""Serialize policy state and make retries idempotent across verifier payload variants."""

from collections.abc import Callable
from threading import Lock

from src.benchmarking.policies import Policy, validate_actions
from src.policies.features import is_bootstrap, validate_step
from src.utils.DTOs import ActionRequest, StepResponse


class SessionConflict(ValueError):
    """A request cannot be assigned safely to the current recurrent game."""


class PolicySession:
    def __init__(
        self, factory: Callable[[], Policy], *, stateful: bool | None = None,
        single_stream: bool = False,
    ):
        self.policy = factory()
        inferred = bool(getattr(self.policy, "stateful", False))
        self.stateful = inferred if stateful is None else stateful
        self.single_stream = single_stream
        self._reset = getattr(self.policy, "reset", None)
        if self.stateful and not callable(self._reset):
            raise TypeError("A stateful policy must provide reset().")
        self._lock = Lock()
        self._last_time: float | None = None
        self._last_time_explicit = False
        self._inferred_time = 0.0
        self._last_request: str | bytes | None = None
        self._last_actions: list[ActionRequest] = []

    @staticmethod
    def _first_observed_tick(step: StepResponse) -> bool:
        return (
            step.game_status == "ok"
            and bool(step.agent_status)
            and max(agent.age for agent in step.agent_status) <= 0.100001
            and step.score <= 1.0
            and sorted(agent.agent_id for agent in step.agent_status)
            == list(range(len(step.agent_status)))
        )

    def _act(self, step: StepResponse) -> list[ActionRequest]:
        method = getattr(self.policy, "act_validated", None)
        proposed = method(step) if callable(method) else self.policy.act(step)
        return validate_actions(
            proposed, [agent.agent_id for agent in step.agent_status], revalidate=False,
        )

    def predict(
        self, step: StepResponse, *, validated: bool = False,
        signature: str | bytes | None = None,
    ) -> list[ActionRequest]:
        if not validated:
            validate_step(step)
        with self._lock:
            if not self.stateful:
                if step.game_status == "game_over":
                    return []
                return self._act(step)
            signature = step.model_dump_json() if signature is None else signature
            if signature == self._last_request:
                return [action.model_copy() for action in self._last_actions]
            bootstrap = is_bootstrap(step)
            explicit_time = "sim_time" in step.model_fields_set
            new_first_tick = (
                self._last_request is not None and (
                    (
                        explicit_time and self._last_time_explicit
                        and self._last_time is not None and step.sim_time < self._last_time
                        and 0 < step.sim_time <= 0.100001
                    )
                    or (not explicit_time and self._first_observed_tick(step))
                )
            )
            if bootstrap or new_first_tick:
                self._reset()
                self._last_time = None
                self._last_time_explicit = False
                self._inferred_time = 0.0
            elif (
                self.single_stream and explicit_time and self._last_time_explicit
                and self._last_time is not None and step.sim_time <= self._last_time
            ):
                raise SessionConflict("Conflicting or out-of-order request; recurrent games cannot interleave.")
            if step.game_status == "game_over":
                self._reset()
                actions = []
            elif bootstrap:
                actions = []
            else:
                policy_step = step
                if not explicit_time:
                    self._inferred_time += 0.1
                    policy_step = step.model_copy(update={"sim_time": self._inferred_time})
                actions = self._act(policy_step)
            self._last_time = step.sim_time if explicit_time else None
            self._last_time_explicit = explicit_time
            self._last_request = signature
            self._last_actions = [action.model_copy() for action in actions]
            return actions
