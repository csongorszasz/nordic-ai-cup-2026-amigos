"""Versioned experiment settings; configuration errors are never ignored."""

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.benchmarking.config import read_json


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


PositiveInt = Annotated[int, Field(strict=True, gt=0)]
Seed = Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]
PositiveFloat = Annotated[float, Field(gt=0)]
Probability = Annotated[float, Field(ge=0, le=1)]


class HeuristicConfig(Settings):
    backend: Literal["scalar", "vectorized", "hierarchical"] = "scalar"
    food_weight: float = Field(default=2.0, ge=0)
    danger_weight: float = Field(default=5.0, ge=0)
    wall_weight: float = Field(default=4.0, ge=0)
    energy_weight: float = Field(default=0.15, ge=0)
    exploration_weight: float = Field(default=0.3, ge=0)
    threat_distance: PositiveFloat = 100.0
    sprint_distance: PositiveFloat = 45.0
    wall_margin: PositiveFloat = 12.0
    breeding_energy: PositiveFloat = 240.0
    breeding_age: PositiveFloat = 60.0
    breeding_reserve: float = Field(default=65.0, ge=0)
    max_turn: float = Field(default=0.45, gt=0, le=3.141592653589793)
    scan_turn: float = Field(default=0.12, ge=0, le=3.141592653589793)
    directions: int = Field(default=16, ge=8, le=64, strict=True)
    fruit_contact_radius: float = Field(default=10.0, ge=5.0, le=20.0)
    patch_radius: PositiveFloat = 28.0
    patch_tolerance: PositiveFloat = 8.0
    patch_capacity: int = Field(default=2, ge=1, le=8, strict=True)
    patch_move_fraction: float = Field(default=0.65, gt=0, le=1)
    exploration_move_fraction: float = Field(default=0.35, ge=0, le=1)
    scout_fraction: Probability = 0.2
    breeder_fraction: Probability = 0.25
    min_population: int = Field(default=7, ge=1, le=128, strict=True)
    target_population: int = Field(default=12, ge=1, le=128, strict=True)
    max_population: int = Field(default=16, ge=1, le=128, strict=True)
    low_population_breeding_energy: PositiveFloat = 185.0
    target_population_breeding_energy: PositiveFloat = 255.0
    high_population_breeding_energy: PositiveFloat = 330.0
    predator_face_turn: float = Field(default=1.2, gt=0, le=3.141592653589793)
    emergency_spawn_distance: PositiveFloat = 35.0

    @model_validator(mode="after")
    def hierarchical_ranges(self):
        if not self.min_population <= self.target_population <= self.max_population:
            raise ValueError("Population settings must satisfy min <= target <= max.")
        if not (
            self.low_population_breeding_energy
            <= self.target_population_breeding_energy
            <= self.high_population_breeding_energy
        ):
            raise ValueError("Breeding energy settings must be nondecreasing.")
        if self.patch_tolerance >= self.patch_radius:
            raise ValueError("patch_tolerance must be smaller than patch_radius.")
        return self


class ModelConfig(Settings):
    encoder: Literal["pool", "attention"] = "pool"
    memory: Literal["none", "gru"] = "gru"
    hidden_size: int = Field(default=128, ge=16, le=512, strict=True)
    entity_size: int = Field(default=32, ge=8, le=128, strict=True)
    team_context: bool = True
    critic: Literal["team", "local"] = "team"
    entity_chunk_size: PositiveInt = 4096


class OptimizerConfig(Settings):
    learning_rate: float = Field(default=0.0003, gt=0, le=1)
    max_grad_norm: PositiveFloat = 0.5


class PPOConfig(Settings):
    gamma: float = Field(default=1.0, gt=0, le=1)
    gae_lambda: Probability = 0.95
    clip_ratio: float = Field(default=0.2, gt=0, lt=1)
    value_weight: float = Field(default=0.5, ge=0)
    entropy_weight: float = Field(default=0.01, ge=0)
    imitation_weight: float = Field(default=0.1, ge=0)
    target_kl: PositiveFloat = 0.03
    epochs: PositiveInt = 4
    sequence_length: PositiveInt = 32
    normalize_value: bool = True


class ImitationConfig(Settings):
    epochs: PositiveInt = 4
    teacher_probability: Probability = 1.0
    dagger_rounds: Annotated[int, Field(strict=True, ge=0)] = 0
    max_frames: PositiveInt = 8192


class SearchConfig(Settings):
    method: Literal["random", "cma"] = "random"
    candidates: PositiveInt = 8
    worlds: PositiveInt = 3
    sigma: float = Field(default=0.2, gt=0, le=1)
    lower_tail_fraction: float = Field(default=0.25, gt=0, le=1)
    lower_tail_weight: Probability = 0.0
    parameters: dict[str, tuple[float, float]] = Field(default_factory=lambda: {
        "breeding_energy": (180.0, 320.0),
        "breeding_age": (40.0, 85.0),
        "breeding_reserve": (35.0, 100.0),
        "threat_distance": (60.0, 130.0),
        "danger_weight": (2.0, 9.0),
    })

    @model_validator(mode="after")
    def valid_parameters(self):
        if not self.parameters:
            raise ValueError("Search requires at least one bounded parameter.")
        for name, (lower, upper) in self.parameters.items():
            if name not in HeuristicConfig.model_fields or name in ("backend", "directions"):
                raise ValueError(f"Unsupported continuous heuristic parameter: {name}")
            if lower >= upper:
                raise ValueError(f"Search bounds must be increasing: {name}")
            for value in (lower, upper):
                HeuristicConfig.model_validate({name: value})
        return self


class ResourceConfig(Settings):
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    workers: int = Field(default=1, ge=1, le=32, strict=True)
    torch_threads: PositiveInt = 1
    max_vram_mb: int = Field(default=6144, ge=256, le=8192, strict=True)
    max_tokens: PositiveInt = 32768
    worker_timeout_seconds: PositiveFloat = 180.0
    action_repeat: int = Field(default=1, ge=1, le=10, strict=True)


class ExperimentConfig(Settings):
    version: Literal[1] = 1
    mode: Literal["search", "imitation", "ppo", "profile"] = "ppo"
    seed: Seed = 12345
    updates: PositiveInt = 20
    rollout_steps: PositiveInt = 128
    policy: Literal["heuristic", "neural"] = "neural"
    heuristic: HeuristicConfig = Field(default_factory=HeuristicConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    imitation: ImitationConfig = Field(default_factory=ImitationConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    checkpoint: str | None = None

    @model_validator(mode="after")
    def compatible(self):
        if self.mode in ("imitation", "ppo") and self.policy != "neural":
            raise ValueError("Learning modes require policy='neural'.")
        if self.mode == "search" and self.policy != "heuristic":
            raise ValueError("Search mode requires policy='heuristic'.")
        if self.mode == "profile" and self.policy != "heuristic":
            raise ValueError("The reference-collection profiler requires policy='heuristic'.")
        if self.ppo.sequence_length > self.rollout_steps and self.mode == "ppo":
            raise ValueError("PPO sequence_length cannot exceed rollout_steps.")
        return self


class RuntimeConfig(Settings):
    version: Literal[1] = 1
    policy: Literal["heuristic", "neural"] = "heuristic"
    heuristic: HeuristicConfig = Field(default_factory=HeuristicConfig)
    checkpoint: str | None = None
    checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    device: Literal["cpu", "cuda"] = "cpu"
    torch_threads: PositiveInt = 1

    @model_validator(mode="after")
    def checkpoint_requirement(self):
        if self.policy == "neural" and not self.checkpoint:
            raise ValueError("A neural policy requires a checkpoint; random weights are not a fallback.")
        if self.policy == "heuristic" and (self.checkpoint or self.checkpoint_sha256):
            raise ValueError("Heuristic policies do not load neural checkpoints.")
        return self


def apply_overrides(config: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    values = config.model_dump(mode="json")
    for override in overrides:
        path, separator, raw = override.partition("=")
        if not separator or not path:
            raise ValueError(f"Expected a setting.path=value override: {override!r}")
        parts = path.split(".")
        parent = values
        for part in parts[:-1]:
            child = parent.get(part)
            if not isinstance(child, dict):
                raise ValueError(f"Unknown configuration path: {path}")
            parent = child
        if parts[-1] not in parent:
            raise ValueError(f"Unknown configuration path: {path}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        parent[parts[-1]] = value
    return ExperimentConfig.model_validate(values)


def load_experiment(path: Path, overrides: list[str] | None = None) -> ExperimentConfig:
    return apply_overrides(ExperimentConfig.model_validate(read_json(path)), overrides or [])
