"""Opt-in evaluator diagnostics. Hidden engine data never enters policy inputs."""

from __future__ import annotations

import math
import time
from collections import Counter, deque
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from src.benchmarking.artifacts import _atomic_text
from src.benchmarking.config import EpisodeCase, EpisodeResult, canonical_json

if TYPE_CHECKING:
    from src.elements.environment import Environment
    from src.utils.DTOs import ActionRequest, StepResponse


TELEMETRY_VERSION = 2
_COSTS = ("move", "turn", "living", "senescence", "birth")


class EpisodeTelemetry:
    def __init__(
        self, root: Path, case: EpisodeCase, *, time_limit: float = 3000.0,
        max_steps: int | None = None, sample_every: int = 10, max_trace_mb: int = 64,
        refresh: Callable[[], None] | None = None, refresh_seconds: float = 10.0,
        candidate: int | None = None,
    ):
        if type(sample_every) is not int or sample_every < 1:
            raise ValueError("Telemetry sample interval must be a positive integer.")
        if type(max_trace_mb) is not int or max_trace_mb < 1:
            raise ValueError("Telemetry storage budget must be a positive integer.")
        if not math.isfinite(refresh_seconds) or refresh_seconds <= 0:
            raise ValueError("Progress refresh interval must be positive and finite.")
        prefix = Path("telemetry") if candidate is None else Path("telemetry") / f"candidate-{candidate:05d}"
        self.path = Path(root) / prefix / case.case_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.sample_every = sample_every
        self.max_bytes = max_trace_mb * 1024**2
        self.bytes_written = 0
        self.refresh = refresh
        self.refresh_seconds = refresh_seconds
        self.last_refresh = 0.0
        self.last_publish = 0.0
        self.started = time.perf_counter()
        self.counts: Counter = Counter({name: 0 for name in (
            "birth", "death", "fruit_spawn", "fruit_eaten", "fruit_removed",
            "tree_spawn", "tree_removed", "predator_spawn", "spawn_requests",
            "deaths_while_senescent",
        )})
        self.totals = {f"cost_{kind}": 0.0 for kind in _COSTS}
        self.totals.update(
            food_energy=0.0, food_retained=0.0, newborn_energy=0.0,
            removed_energy=0.0, predation_energy=0.0,
        )
        self.death_causes: Counter = Counter()
        self.latencies: deque[float] = deque(maxlen=512)
        self.policy_data: dict = {}
        self.http_wait: float | None = None
        self.initial_energy = 0.0
        self.tick = 0
        self.last_sample_tick = -1
        self.latest: dict | None = None
        self.recent: deque[dict] = deque(maxlen=120)
        self.last_decision: tuple[int, StepResponse, list[ActionRequest]] | None = None
        self.env: Environment | None = None
        self.finished = False
        self.metadata = {
            "schema_version": TELEMETRY_VERSION,
            "case": case.model_dump(mode="json"), "candidate": candidate,
            "time_limit": time_limit, "max_steps": max_steps,
            "sample_every": sample_every, "max_trace_mb": max_trace_mb,
            "scope": "diagnostic" if max_steps is not None else (
                "native" if time_limit == 3000.0 else "extended_or_custom_horizon"
            ),
            "visibility": "engine_truth_for_evaluation_only",
            "live_state": "last_complete_record_in_samples_jsonl",
        }
        self._json("case.json", self.metadata)
        self.streams = ExitStack()
        try:
            self.events = self.streams.enter_context((self.path / "events.jsonl").open("x", encoding="utf-8", newline="\n"))
            self.samples = self.streams.enter_context((self.path / "samples.jsonl").open("x", encoding="utf-8", newline="\n"))
            self.decisions = self.streams.enter_context((self.path / "decisions.jsonl").open("x", encoding="utf-8", newline="\n"))
        except BaseException:
            self.streams.close()
            raise

    def _json(self, name: str, value: object) -> None:
        _atomic_text(self.path / name, canonical_json(value) + "\n")

    def _line(self, stream, value: object) -> None:
        text = canonical_json(value) + "\n"
        size = len(text.encode("utf-8"))
        if self.bytes_written + size > self.max_bytes:
            raise OSError(f"Telemetry storage budget exceeded for {self.path.name}.")
        stream.write(text)
        self.bytes_written += size

    def attach(self, env: Environment) -> None:
        if self.env is not None or self.finished:
            raise ValueError("Telemetry must attach exactly once to one episode.")
        if getattr(env, "event_sink", None) is not None:
            raise ValueError("The environment already has an event sink.")
        self.env = env
        self.initial_energy = math.fsum(agent.energy for agent in env.agents)
        self._json("initial.json", {
            "population": len(env.agents), "energy": self.initial_energy,
            "trees": len(env.trees), "fruit": len(env.fruits), "predators": len(env.predators),
            "agents": [
                {"id": a.agent_id, "x": float(a.x), "y": float(a.y), "max_age": a.max_age}
                for a in env.agents
            ],
        })
        env.event_sink = self.record_event

    def record_event(self, event: dict) -> None:
        kind = event["kind"]
        if kind == "energy_cost":
            reason = event["reason"]
            if reason not in _COSTS:
                raise ValueError(f"Unknown energy cost category: {reason}.")
            self.totals[f"cost_{reason}"] += event["amount"]
            return
        self.counts[kind] += 1
        if kind == "birth":
            self.totals["newborn_energy"] += event["energy"]
        elif kind == "fruit_eaten":
            self.totals["food_energy"] += event["energy"]
            self.totals["food_retained"] += event["retained"]
        elif kind == "death":
            self.totals["removed_energy"] += event["energy"]
            self.death_causes[event["reason"]] += 1
            if event["reason"] == "predation":
                self.totals["predation_energy"] += event["energy"]
            if event["senescent"]:
                self.counts["deaths_while_senescent"] += 1
        self._line(self.events, {"tick": self.tick + 1, **event})

    def decision(self, step: StepResponse, actions: list[ActionRequest], policy, latency_ms: float) -> None:
        self.latencies.append(latency_ms)
        self.counts["spawn_requests"] += sum(action.spawn_agent for action in actions)
        explain = getattr(policy, "diagnostics", None)
        if callable(explain):
            data = explain()
            if not isinstance(data, dict):
                raise TypeError("Policy diagnostics must be a JSON object.")
            canonical_json(data)
            self.policy_data = data
        self.http_wait = getattr(policy, "total_wait_seconds", None)
        self.last_decision = self.tick, step, actions
        record_history = self.tick == 1 or self.tick % max(1000, self.sample_every) == 0
        if record_history:
            record = {
                "tick": self.tick, "step": step.model_dump(mode="json"),
                "actions": [action.model_dump(mode="json") for action in actions],
            }
            self._line(self.decisions, record)
            self.decisions.flush()

    def observe(self, step: StepResponse, tick: int, *, force: bool = False) -> None:
        self.tick = tick
        if not force and tick != 1 and tick % self.sample_every:
            return
        if self.env is None:
            raise RuntimeError("Telemetry has not attached to an environment.")
        agents = self.env.agents
        energies = [a.energy for a in agents]
        ages = [a.age for a in agents]
        total_energy = math.fsum(energies)
        expected = (
            self.initial_energy + self.totals["food_retained"] + self.totals["newborn_energy"]
            - self.totals["removed_energy"]
            - math.fsum(self.totals[f"cost_{kind}"] for kind in _COSTS)
        )
        self.latest = {
            "tick": tick, "sim_time": step.sim_time, "population": len(agents),
            "score": step.score, "energy": total_energy,
            "energy_p10": float(np.percentile(energies, 10)) if energies else None,
            "young": sum(a.age < 60.0 for a in agents),
            "reproductive": sum(a.energy > 100.0 and a.max_energy > 100.0 for a in agents),
            "young_reproductive": sum(
                a.age < 60.0 and a.energy > 100.0 and a.max_energy > 100.0 for a in agents
            ),
            "elders": sum(a.age > a.max_age for a in agents),
            "ages": ages, "trees": len(self.env.trees), "fruit": len(self.env.fruits),
            "predators": len(self.env.predators), "counts": dict(self.counts),
            "totals": dict(self.totals), "death_causes": dict(self.death_causes),
            "energy_balance_residual": total_energy - expected,
            "batch_p95_ms": float(np.percentile(self.latencies, 95)) if self.latencies else None,
            "http_wait_seconds": self.http_wait,
            "wall_seconds": time.perf_counter() - self.started,
            "policy": self.policy_data,
            "positions": [[a.agent_id, float(a.x), float(a.y)] for a in agents],
        }
        if tick != self.last_sample_tick:
            self._line(self.samples, self.latest)
            self.recent.append(self.latest)
            self.last_sample_tick = tick
        now = time.perf_counter()
        if force or now - self.last_publish >= 1.0:
            self.events.flush()
            self.samples.flush()
            self.last_publish = now
        if self.refresh is not None and now - self.last_refresh >= self.refresh_seconds:
            self.refresh()
            self.last_refresh = time.perf_counter()

    def finish(self, result: EpisodeResult) -> None:
        if self.finished:
            raise ValueError("Telemetry has already finished.")
        self.finished = True
        try:
            if self.last_decision is not None:
                tick, step, actions = self.last_decision
                self._json("last-request.json", {
                    "tick": tick,
                    "step": step.model_dump(mode="json"),
                    "actions": [action.model_dump(mode="json") for action in actions],
                })
            self._json("summary.json", {
                **self.metadata, "state": result.status,
                "result": result.model_dump(mode="json"),
                "counts": dict(self.counts), "totals": dict(self.totals),
                "death_causes": dict(self.death_causes), "last_sample": self.latest,
                "trace_bytes": self.bytes_written,
                "diagnosis": self._diagnose(result),
            })
        finally:
            if self.env is not None:
                self.env.event_sink = None
            self.streams.close()
        if self.refresh is not None and result.status in ("ok", "truncated"):
            self.refresh()

    def _diagnose(self, result: EpisodeResult) -> dict:
        hypotheses = []
        if result.termination == "extinction" and self.latest is not None:
            last = self.latest
            start = next((sample for sample in self.recent
                          if sample["sim_time"] >= last["sim_time"] - 30.0), self.recent[0])
            energy_deaths = self.death_causes["energy_depletion"]
            if last["sim_time"] < 60 and energy_deaths and not self.counts["birth"]:
                hypotheses.append("opening_energy_depletion_without_replacement")
            if energy_deaths and self.counts["deaths_while_senescent"] >= max(1, self.counts["death"] / 2):
                hypotheses.append("cohort_senescence")
            spending = sum(last["totals"][f"cost_{kind}"] - start["totals"][f"cost_{kind}"] for kind in _COSTS)
            intake = last["totals"]["food_retained"] - start["totals"]["food_retained"]
            newborn = last["totals"]["newborn_energy"] - start["totals"]["newborn_energy"]
            if energy_deaths and intake + newborn < spending:
                hypotheses.append("recent_resource_budget_deficit")
            if start["policy"].get("known_patches", 0) > last["policy"].get("known_patches", 0):
                hypotheses.append("loss_of_known_patches")
            if self.death_causes["predation"] and last["policy"].get("colonies") == 1:
                hypotheses.append("predation_with_single_assigned_refuge")
            if not hypotheses:
                hypotheses.append("unclassified")
        return {
            "version": 1, "outcome": result.termination,
            "confirmed_death_causes": dict(self.death_causes), "hypotheses": hypotheses,
            "qualification": "Rule-based collapse hypotheses, not proof of causation or an unwinnable seed.",
        }
