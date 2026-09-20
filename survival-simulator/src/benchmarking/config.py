import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 2
PROTOCOL_VERSION = "local-benchmark-v2"
SEED_VERSION = "sha256-policy-seed-v1"
BASELINE_NAME = "random-local-v1"
SUITE_NAMES = ("quick", "standard", "holdout")

NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
Seed = Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]
Duration = Annotated[float, Field(ge=0)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
RunStatus = Literal["running", "complete", "failed", "interrupted", "truncated"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _invalid_constant(value: str):
    raise ValueError(f"Non-finite JSON number: {value}")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=_invalid_constant)


class Suite(Record):
    version: Literal[1] = 1
    name: str = Field(min_length=1)
    seeds: list[Seed] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_seeds(self):
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Suite seeds must be unique.")
        return self


def load_suite(name: str, root: Path = PROJECT_ROOT) -> Suite:
    if name not in SUITE_NAMES:
        raise ValueError(f"Unknown suite {name!r}; choose from {', '.join(SUITE_NAMES)}.")
    suites = {
        key: Suite.model_validate(read_json(root / "benchmarks" / "suites" / f"{key}.json"))
        for key in SUITE_NAMES
    }
    for key, size in (("quick", 3), ("standard", 20), ("holdout", 20)):
        if suites[key].name != key or len(suites[key].seeds) != size:
            raise ValueError(f"The {key} suite must contain {size} seeds and use its own name.")
    if suites["quick"].seeds != suites["standard"].seeds[:3]:
        raise ValueError("Quick seeds must be the first three standard seeds.")
    if set(suites["standard"].seeds) & set(suites["holdout"].seeds):
        raise ValueError("Standard and holdout seeds must be disjoint.")
    return suites[name]


def policy_seed(world_seed: int, repeat_index: int) -> int:
    if repeat_index == 0:
        return world_seed
    payload = f"{SEED_VERSION}:{world_seed}:{repeat_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


class EpisodeCase(Record):
    case_id: str
    world_seed: Seed
    repeat_index: NonnegativeInt
    policy_seed: Seed
    policy_seed_mode: Literal["derived", "fixed"] = "derived"

    @model_validator(mode="after")
    def valid_identity(self):
        if self.case_id != f"world-{self.world_seed}-repeat-{self.repeat_index}":
            raise ValueError("Case ID does not match its world seed and repeat index.")
        if (
            self.policy_seed_mode == "derived"
            and self.policy_seed != policy_seed(self.world_seed, self.repeat_index)
        ):
            raise ValueError("Policy seed does not match the recorded derivation protocol.")
        return self


def make_cases(
    suite: Suite, repeats: int = 1, *, fixed_policy_seed: int | None = None,
) -> list[EpisodeCase]:
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("Repeats must be a positive integer.")
    if fixed_policy_seed is not None and (
        isinstance(fixed_policy_seed, bool) or not isinstance(fixed_policy_seed, int)
        or not 0 <= fixed_policy_seed <= 2**32 - 1
    ):
        raise ValueError("Fixed policy seed must be a uint32.")
    return [
        EpisodeCase(
            case_id=f"world-{seed}-repeat-{repeat}",
            world_seed=seed,
            repeat_index=repeat,
            policy_seed=(
                fixed_policy_seed if fixed_policy_seed is not None
                else policy_seed(seed, repeat)
            ),
            policy_seed_mode="fixed" if fixed_policy_seed is not None else "derived",
        )
        for seed in suite.seeds
        for repeat in range(repeats)
    ]


class SimulationSettings(Record):
    env_width: PositiveInt = 1600
    env_height: PositiveInt = 1200
    chunk_size: PositiveInt = 400
    starting_agents: PositiveInt = 5
    starting_predators: NonnegativeInt = 0
    starting_fruits: PositiveInt = 32
    starting_trees: NonnegativeInt = 50
    dt: float = Field(default=0.1, gt=0)
    time_limit: float = Field(default=3000.0, gt=0)

    def core_kwargs(self) -> dict[str, int | float]:
        return self.model_dump(exclude={"time_limit"})


class PolicySpec(Record):
    reference: str = Field(min_length=1)
    label: str = Field(min_length=1)
    config: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def finite_config(self):
        canonical_json(self.config)
        return self


class Fingerprint(Record):
    sha256: Digest
    files: dict[str, Digest]


class GitInfo(Record):
    revision: str | None
    dirty: bool | None
    status: str


class RuntimeInfo(Record):
    python_version: str
    implementation: str
    os: str
    os_release: str
    architecture: str
    dependencies: dict[str, str]
    native_libraries: dict[str, str]


class Provenance(Record):
    git: GitInfo
    engine: Fingerprint
    runner: Fingerprint
    policy: Fingerprint
    reporting: Fingerprint
    artifacts: Fingerprint
    runtime: RuntimeInfo
    timing_context: dict[str, JsonValue]


class FailureInfo(Record):
    stage: str
    error_type: str
    message: str
    traceback: str | None = None


class Timings(Record):
    initialization_seconds: Duration = 0.0
    policy_construction_seconds: Duration = 0.0
    simulation_seconds: Duration = 0.0
    policy_seconds: Duration = 0.0
    episode_seconds: Duration = 0.0
    policy_calls: NonnegativeInt = 0
    batch_mean_ms: Duration | None = None
    batch_p50_ms: Duration | None = None
    batch_p95_ms: Duration | None = None
    batch_max_ms: Duration | None = None


class EpisodeResult(Record):
    case: EpisodeCase
    status: Literal["ok", "failed", "interrupted", "truncated"]
    termination: Literal["extinction", "time_limit", "step_limit", "error", "interrupted"]
    score: float | None = None
    sim_time: Duration | None = None
    survival_seconds: Duration | None = None
    ticks: NonnegativeInt = 0
    completed: bool = False
    initial_agents: NonnegativeInt | None = None
    final_agents: NonnegativeInt | None = None
    peak_agents: NonnegativeInt | None = None
    mean_population: Duration | None = None
    timings: Timings = Field(default_factory=Timings)
    failure: FailureInfo | None = None

    @model_validator(mode="after")
    def consistent_outcome(self):
        if self.status in ("ok", "truncated"):
            required = (
                self.score, self.sim_time, self.survival_seconds, self.initial_agents,
                self.final_agents, self.peak_agents, self.mean_population,
            )
            if any(value is None for value in required) or self.ticks == 0:
                raise ValueError("Successful or truncated episodes require complete measurements.")
            if self.failure is not None:
                raise ValueError("Successful or truncated episodes cannot contain an error.")
        elif self.score is not None or self.failure is None:
            raise ValueError("Failed/interrupted episodes require error details and no final score.")
        allowed = {
            "ok": ("extinction", "time_limit"),
            "truncated": ("step_limit",),
            "failed": ("error",),
            "interrupted": ("interrupted",),
        }
        if self.termination not in allowed[self.status]:
            raise ValueError("Episode status and termination reason disagree.")
        if self.completed != (self.termination == "time_limit"):
            raise ValueError("Completion means surviving to the time limit.")
        if self.termination == "extinction" and self.final_agents != 0:
            raise ValueError("Extinction requires no surviving agents.")
        if self.termination == "time_limit" and not self.final_agents:
            raise ValueError("Time-limit completion requires surviving agents.")
        return self


class RunManifest(Record):
    schema_version: Literal[1, 2] = SCHEMA_VERSION
    protocol_version: Literal["local-benchmark-v1", "local-benchmark-v2"] = PROTOCOL_VERSION
    seed_version: Literal["sha256-policy-seed-v1"] = SEED_VERSION
    run_id: str
    created_at: str
    suite: Suite
    suite_sha256: Digest
    repeats: PositiveInt = 1
    fixed_policy_seed: Seed | None = None
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    max_steps: PositiveInt | None = None
    cases: list[EpisodeCase] = Field(min_length=1)
    policy: PolicySpec
    provenance: Provenance
    execution: Literal["sequential", "processes"] = "sequential"
    status: RunStatus = "running"
    failure: FailureInfo | None = None

    @model_validator(mode="after")
    def complete_case_definition(self):
        expected_protocol = (
            "local-benchmark-v1" if self.schema_version == 1 else PROTOCOL_VERSION
        )
        if self.protocol_version != expected_protocol:
            raise ValueError("Manifest schema and runner protocol versions disagree.")
        if self.suite_sha256 != content_hash(self.suite.model_dump()):
            raise ValueError("Suite hash does not match the seed manifest.")
        actual = {case.case_id: case for case in self.cases}
        expected = {
            case.case_id: case for case in make_cases(
                self.suite, self.repeats, fixed_policy_seed=self.fixed_policy_seed,
            )
        }
        if len(actual) != len(self.cases) or actual != expected:
            raise ValueError("Manifest cases must exactly cover the suite and repeat count.")
        return self
