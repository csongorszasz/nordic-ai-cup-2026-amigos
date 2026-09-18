"""Benchmark an actual HTTP submission endpoint with evaluator-equivalent budgets."""

from __future__ import annotations

import time
from collections.abc import Callable

import requests
from pydantic import BaseModel, ConfigDict, Field
from requests.exceptions import JSONDecodeError

from src.benchmarking.policies import validate_actions
from src.utils.DTOs import ActionRequest, StepResponse


class HTTPPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    url: str = Field(min_length=1)
    per_request_timeout_seconds: float = Field(default=10.0, gt=0, le=10.0)
    cumulative_timeout_seconds: float = Field(default=600.0, gt=0, le=600.0)
    verify_tls: bool = True
    headers: dict[str, str] = Field(default_factory=dict)


class HTTPPolicy:
    def __init__(
        self, config: HTTPPolicyConfig, *,
        session: requests.Session | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.config = config
        self.session = session or requests.Session()
        self.clock = clock
        self.calls = 0
        self.total_wait_seconds = 0.0

    def act(self, step: StepResponse) -> list[ActionRequest]:
        body = step.model_dump_json()
        started = self.clock()
        try:
            response = self.session.post(
                self.config.url, data=body,
                headers={"Content-Type": "application/json", **self.config.headers},
                timeout=self.config.per_request_timeout_seconds,
                verify=self.config.verify_tls,
            )
        finally:
            elapsed = self.clock() - started
            self.calls += 1
            self.total_wait_seconds += elapsed
        if elapsed > self.config.per_request_timeout_seconds:
            raise TimeoutError(
                f"HTTP request {self.calls} took {elapsed:.3f}s, exceeding "
                f"{self.config.per_request_timeout_seconds:.3f}s.",
            )
        if self.total_wait_seconds > self.config.cumulative_timeout_seconds:
            raise TimeoutError(
                f"HTTP endpoint accumulated {self.total_wait_seconds:.3f}s after "
                f"{self.calls} calls, exceeding {self.config.cumulative_timeout_seconds:.3f}s.",
            )
        if response.status_code != 200:
            raise RuntimeError(
                f"HTTP endpoint returned {response.status_code} on call {self.calls}: "
                f"{response.text[:500]}",
            )
        try:
            payload = response.json()
        except JSONDecodeError as exc:
            raise ValueError("HTTP endpoint returned invalid JSON.") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("actions"), list):
            raise ValueError("HTTP endpoint response must contain an actions list.")
        actions = [ActionRequest.model_validate(value, strict=True) for value in payload["actions"]]
        return validate_actions(
            actions, [agent.agent_id for agent in step.agent_status], revalidate=False,
        )


def create_policy(seed: int, config: dict) -> HTTPPolicy:
    del seed
    return HTTPPolicy(HTTPPolicyConfig.model_validate(config))
