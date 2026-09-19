"""Versioned, ragged features built exclusively from the public observations."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import numpy as np

from src.policies.geometry import BIOME_MOVEMENT, Segment, closest_point
from src.utils.DTOs import ActionRequest, ObservationResponse, StepResponse


FEATURE_VERSION = "structured-v1"
PUBLIC_FEATURE_VERSION = "structured-public-v2"
PEER_FEATURE_VERSION = "structured-public-peer-v3"
FEATURE_VERSIONS = (FEATURE_VERSION, PUBLIC_FEATURE_VERSION, PEER_FEATURE_VERSION)
ENTITY_TYPES = ("Fruit", "Tree", "Predator", "Agent", "Edge")
BIOMES = tuple(BIOME_MOVEMENT)
SCALAR_DIM = 22
PUBLIC_SCALAR_DIM = SCALAR_DIM + 3
ENTITY_DIM = 10
PEER_ENERGY_UNAVAILABLE = -1.0


def feature_version(public_context: bool, peer_context: bool = False) -> str:
    if peer_context:
        if not public_context:
            raise ValueError("peer_context requires public_context=True.")
        return PEER_FEATURE_VERSION
    return PUBLIC_FEATURE_VERSION if public_context else FEATURE_VERSION


@dataclass(frozen=True)
class Entity:
    kind: str
    x: float
    y: float
    distance: float
    angle: float
    rel_dir: float | None = None
    agent_id: int | None = None
    segment: Segment | None = None


@dataclass(frozen=True)
class FeatureBatch:
    agent_ids: tuple[int, ...]
    scalars: np.ndarray
    entities: np.ndarray
    entity_types: np.ndarray
    entity_owners: np.ndarray
    feature_version: str | None = None

    def __post_init__(self):
        # Untagged legacy callers can distinguish v1/v2 by width, never peer-v3.
        if self.feature_version is None:
            public = (
                isinstance(self.scalars, np.ndarray) and self.scalars.ndim == 2
                and self.scalars.shape[1] == PUBLIC_SCALAR_DIM
            )
            object.__setattr__(self, "feature_version", feature_version(public))


def finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number.")
    return float(value)


def is_bootstrap(step: StepResponse) -> bool:
    # The supplied HTTP client announces population before its first observations exist.
    return (
        step.game_status == "ok" and step.sim_time == 0 and step.score == 0
        and step.n_agents >= 0 and not step.agent_status
    )


def validate_step(step: StepResponse) -> None:
    if step.game_status not in ("ok", "game_over"):
        raise ValueError(f"Unknown game status: {step.game_status!r}")
    if isinstance(step.n_agents, bool) or not isinstance(step.n_agents, int) or step.n_agents < 0:
        raise ValueError("n_agents must be a nonnegative integer.")
    if step.n_agents != len(step.agent_status) and not is_bootstrap(step):
        raise ValueError("n_agents does not match agent_status.")
    ids = [agent.agent_id for agent in step.agent_status]
    if len(set(ids)) != len(ids) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ids
    ):
        raise ValueError("Living agent IDs must be unique and nonnegative.")
    finite_number(step.score, "score")
    if finite_number(step.sim_time, "sim_time") < 0:
        raise ValueError("sim_time must be nonnegative.")
    for agent in step.agent_status:
        for field in ("energy", "age", "speed", "sprint_speed", "hearing_radius",
                      "vision_range", "vision_angle", "max_energy"):
            value = finite_number(getattr(agent, field), field)
            if field != "energy" and value < 0:
                raise ValueError(f"Agent {agent.agent_id}: {field} must be nonnegative.")
        if agent.max_energy <= 0 or agent.biome not in BIOMES:
            raise ValueError(f"Agent {agent.agent_id}: invalid energy capacity or biome.")


def parse_entities(agent: ObservationResponse) -> tuple[Entity, ...]:
    entities = []
    seen_edges: set[Segment] = set()
    for observation in agent.observations:
        kind = observation.get("type")
        if kind not in ENTITY_TYPES:
            raise ValueError(f"Agent {agent.agent_id}: unknown observation type {kind!r}.")
        if kind == "Edge":
            coords = observation.get("coords")
            if not isinstance(coords, (list, tuple)) or len(coords) != 2:
                raise ValueError("Edge coords must contain two endpoints.")
            points = []
            for point in coords:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ValueError("Edge endpoint must contain x and y.")
                points.append((finite_number(point[0], "edge.x"), finite_number(point[1], "edge.y")))
            segment = (min(points), max(points))
            if segment in seen_edges:
                continue
            seen_edges.add(segment)
            x, y = closest_point((0.0, 0.0), segment)
            entities.append(Entity(kind, x, y, math.hypot(x, y), math.atan2(y, x), segment=segment))
            continue
        distance = finite_number(observation.get("distance"), "entity.distance")
        angle = finite_number(observation.get("angle"), "entity.angle")
        if distance < 0:
            raise ValueError("Entity distance must be nonnegative.")
        rel_dir = None
        if kind in ("Agent", "Predator"):
            rel_dir = finite_number(observation.get("rel_dir"), "entity.rel_dir")
        agent_id = observation.get("id") if kind == "Agent" else None
        if kind == "Agent" and (
            isinstance(agent_id, bool) or not isinstance(agent_id, int) or agent_id < 0
        ):
            raise ValueError("Observed teammate id must be a nonnegative integer.")
        entities.append(Entity(
            kind, distance * math.cos(angle), distance * math.sin(angle), distance,
            angle, rel_dir, agent_id,
        ))
    return tuple(sorted(entities, key=lambda entity: (
        ENTITY_TYPES.index(entity.kind), entity.x, entity.y,
        entity.rel_dir if entity.rel_dir is not None else 0.0, entity.segment or (),
        entity.agent_id if entity.agent_id is not None else -1,
    )))


def encode_step(
    step: StepResponse, previous_actions: Mapping[int, ActionRequest] | None = None, *,
    validate: bool = True, public_context: bool = False, peer_context: bool = False,
) -> FeatureBatch:
    """Encode public DTOs without changing body-frame geometry or row coverage.

    Peer-v3 uses Agent columns 8/9 for energy/max_energy and log1p(observed ID).
    Energy is -1 when that ID is absent from this response (stale/dead/unknown),
    or its current energy is nonpositive. The observed identity is always kept;
    unavailable status is never replaced by a healthy default or cached status.
    """
    schema = feature_version(public_context, peer_context)
    if validate:
        validate_step(step)
    previous_actions = previous_actions or {}
    peers = {agent.agent_id: agent for agent in step.agent_status} if peer_context else {}
    ranks = {agent_id: index for index, agent_id in enumerate(sorted(a.agent_id for a in step.agent_status))}
    scalars, features, types, owners = [], [], [], []
    for index, agent in enumerate(step.agent_status):
        previous = previous_actions.get(agent.agent_id)
        history = (
            [previous.move_distance / 40.0, math.sin(previous.move_direction),
             math.cos(previous.move_direction), math.sin(previous.turn_angle),
             math.cos(previous.turn_angle), float(previous.spawn_agent)]
            if previous is not None else [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        )
        public = [
            math.log1p(agent.agent_id), step.score / 3000.0,
            ranks[agent.agent_id] / max(1, len(ranks) - 1),
        ] if public_context else []
        scalars.append([
            agent.energy / 500.0, agent.energy / agent.max_energy, agent.age / 120.0,
            agent.speed / 20.0, agent.sprint_speed / 40.0, agent.hearing_radius / 100.0,
            agent.vision_range / 400.0, agent.vision_angle / math.pi,
            agent.max_energy / 1000.0, step.sim_time / 3000.0, math.log1p(step.n_agents) / 5.0,
            *(float(agent.biome == biome) for biome in BIOMES), *public, *history,
        ])
        for entity in parse_entities(agent):
            if entity.segment is not None:
                (sx, sy), (ex, ey) = entity.segment
                row = [sx / 400.0, sy / 400.0, ex / 400.0, ey / 400.0,
                       entity.x / 400.0, entity.y / 400.0, entity.distance / 400.0, 0, 0, 0]
            else:
                row = [entity.x / 400.0, entity.y / 400.0, entity.distance / 400.0,
                       math.sin(entity.angle), math.cos(entity.angle),
                       math.sin(entity.rel_dir) if entity.rel_dir is not None else 0.0,
                       math.cos(entity.rel_dir) if entity.rel_dir is not None else 0.0,
                       float(entity.rel_dir is not None), 0, 0]
                if peer_context and entity.kind == "Agent" and entity.agent_id is not None:
                    peer = peers.get(entity.agent_id)
                    row[8:] = [
                        peer.energy / peer.max_energy
                        if peer is not None and peer.energy > 0 else PEER_ENERGY_UNAVAILABLE,
                        math.log1p(entity.agent_id),
                    ]
            features.append(row)
            types.append(ENTITY_TYPES.index(entity.kind))
            owners.append(index)
    scalar_array = np.asarray(scalars, dtype=np.float32).reshape(
        -1, PUBLIC_SCALAR_DIM if public_context else SCALAR_DIM,
    )
    entity_array = np.asarray(features, dtype=np.float32).reshape(-1, ENTITY_DIM)
    if not np.isfinite(scalar_array).all() or not np.isfinite(entity_array).all():
        raise ValueError("Encoded features contain non-finite or overflowing values.")
    return FeatureBatch(
        tuple(agent.agent_id for agent in step.agent_status), scalar_array, entity_array,
        np.asarray(types, dtype=np.int64), np.asarray(owners, dtype=np.int64),
        feature_version=schema,
    )
