"""Float64 candidate scoring; scalar adjudication preserves ties and branch boundaries."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.policies.config import HeuristicConfig
from src.policies.features import Entity
from src.policies.geometry import (
    BIOME_MOVEMENT, Segment, movement_cost, movement_limit, point_segment_distance,
)
from src.policies.heuristic import (
    AGENT_RADIUS, COLLECTION_RADIUS, FOOD_MIN_RANGE, PREDATOR_RADIUS, PREDATOR_TRAVEL,
    SCORE_DECIMALS, THREAT_CAPTURE_WEIGHT, THREAT_EXPOSURE_WEIGHT, TURN_ENERGY_FRACTION,
    WALL_SWEEP_WEIGHT, _EPSILON, _ORIGIN,
)
from src.utils.DTOs import ObservationResponse


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
_ROUNDING_TOLERANCE = 10.0 ** -SCORE_DECIMALS
_FLOAT_TOLERANCE = 64.0 * np.finfo(np.float64).eps


@dataclass(frozen=True)
class WallScores:
    blocked: BoolArray
    scores: FloatArray
    swept_clearance: FloatArray | None
    endpoint_clearance: FloatArray | None
    uncertain: BoolArray


@dataclass(frozen=True)
class CandidateScores:
    """Rows follow input candidates; blocked rows have -inf scores and no scored terms."""

    endpoints: FloatArray
    turns: FloatArray
    costs: FloatArray
    terms: FloatArray
    scores: FloatArray
    blocked: BoolArray
    uncertain: BoolArray
    score_errors: FloatArray

    def contenders(self) -> BoolArray:
        valid = ~self.blocked
        if not np.any(valid):
            return self.uncertain.copy()
        best_lower_bound = np.max(self.scores[valid] - self.score_errors[valid])
        return self.uncertain | (
            valid & (self.scores + self.score_errors >= best_lower_bound - _ROUNDING_TOLERANCE)
        )


def _near(first: FloatArray, second: FloatArray | float) -> BoolArray:
    scale = np.maximum(1.0, np.maximum(np.abs(first), np.abs(second)))
    return np.abs(first - second) <= _FLOAT_TOLERANCE * scale


def _wrap(angles: FloatArray) -> FloatArray:
    return (angles + math.pi) % math.tau - math.pi


def _ordered_sum(values: FloatArray) -> FloatArray:
    # Match the scalar left-to-right accumulation rather than NumPy's pairwise sum.
    return np.cumsum(values, axis=1)[:, -1]


def point_segment_distances(points: FloatArray, segments: FloatArray) -> FloatArray:
    """Return a points x segments distance matrix, including zero-length segments."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    segments = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)
    starts, ends = segments[:, 0], segments[:, 1]
    dx, dy = ends[:, 0] - starts[:, 0], ends[:, 1] - starts[:, 1]
    length_squared = dx * dx + dy * dy
    numerator = (
        (points[:, None, 0] - starts[:, 0]) * dx
        + (points[:, None, 1] - starts[:, 1]) * dy
    )
    fraction = np.divide(
        numerator, length_squared, out=np.zeros_like(numerator), where=length_squared != 0.0,
    )
    np.clip(fraction, 0.0, 1.0, out=fraction)
    return np.hypot(
        points[:, None, 0] - (starts[:, 0] + fraction * dx),
        points[:, None, 1] - (starts[:, 1] + fraction * dy),
    )


def wall_scores(endpoints: FloatArray, walls: tuple[Segment, ...], margin: float) -> WallScores:
    """Vectorized swept clearance and outward recovery, without truncating walls."""
    count = len(endpoints)
    if not walls:
        return WallScores(np.zeros(count, dtype=bool), np.zeros(count), None, None,
                          np.zeros(count, dtype=bool))
    segments = np.asarray(walls, dtype=np.float64)
    starts, ends = segments[:, 0], segments[:, 1]
    paths = np.stack((np.zeros_like(endpoints), endpoints), axis=1)
    initial = np.asarray([point_segment_distance(_ORIGIN, wall) for wall in walls])
    final = point_segment_distances(endpoints, segments)
    swept = np.minimum(initial, final)
    np.minimum(swept, point_segment_distances(starts, paths).T, out=swept)
    np.minimum(swept, point_segment_distances(ends, paths).T, out=swept)
    ex, ey = endpoints[:, None, 0], endpoints[:, None, 1]
    sx, sy = starts[:, 0], starts[:, 1]
    dx, dy = ends[:, 0] - sx, ends[:, 1] - sy
    crossing = (
        (ex * sy - ey * sx) * (ex * ends[:, 1] - ey * ends[:, 0]) < 0.0
    ) & ((dx * -sy - dy * -sx) * (dx * (ey - sy) - dy * (ex - sx)) < 0.0)
    swept[crossing] = 0.0
    if not (np.isfinite(initial).all() and np.isfinite(final).all() and np.isfinite(swept).all()):
        raise ValueError("Wall geometry produced non-finite distances.")
    travel = np.hypot(endpoints[:, 0], endpoints[:, 1])
    moving = travel[:, None] > _EPSILON
    overlap = initial < AGENT_RADIUS + _EPSILON
    blocked = moving & np.where(
        overlap,
        (swept < initial - _EPSILON) | (final <= initial + _EPSILON),
        swept < AGENT_RADIUS + _EPSILON,
    )
    uncertain = moving & np.where(
        overlap,
        _near(swept, initial - _EPSILON) | _near(final, initial + _EPSILON),
        _near(swept, AGENT_RADIUS + _EPSILON),
    )
    start_room = np.clip((initial - AGENT_RADIUS) / margin, 0.0, 1.0)
    end_room = np.clip((final - AGENT_RADIUS) / margin, 0.0, 1.0)
    sweep_room = np.clip((swept - AGENT_RADIUS) / margin, 0.0, 1.0)
    scores = _ordered_sum(end_room - start_room - WALL_SWEEP_WEIGHT * (1.0 - sweep_room) ** 2)
    return WallScores(
        blocked.any(axis=1), scores, swept.min(axis=1) - AGENT_RADIUS,
        final.min(axis=1) - AGENT_RADIUS,
        uncertain.any(axis=1) | _near(travel, _EPSILON),
    )


def _entity_arrays(
    entities: tuple[Entity, ...], endpoints: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    positions = np.asarray([(entity.x, entity.y) for entity in entities], dtype=np.float64)
    distances = np.asarray([entity.distance for entity in entities], dtype=np.float64)
    offsets = positions[None, :, :] - endpoints[:, None, :]
    remaining = np.hypot(offsets[:, :, 0], offsets[:, :, 1])
    return positions, distances, offsets, remaining


def _nearest(positions: FloatArray, remaining: FloatArray) -> tuple[NDArray[np.intp], BoolArray]:
    order = np.lexsort((positions[:, 1], positions[:, 0]))
    nearest = order[np.argmin(remaining[:, order], axis=1)]
    minimum = remaining[np.arange(len(remaining)), nearest]
    uncertain = _near(remaining, minimum[:, None]).sum(axis=1) > 1
    return nearest, uncertain


def score_candidates(
    agent: ObservationResponse, config: HeuristicConfig,
    candidates: list[tuple[float, float]], walls: tuple[Segment, ...],
    fruits: tuple[Entity, ...], predators: tuple[Entity, ...], *,
    preferred: float, scan: float, tree_target: bool, walking: float,
) -> CandidateScores:
    """Score the existing finite candidate set; do not enumerate actions or change targets."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _score_candidates(
                agent, config, candidates, walls, fruits, predators,
                preferred=preferred, scan=scan, tree_target=tree_target, walking=walking,
            )
    except FloatingPointError as error:
        raise ValueError(f"Agent {agent.agent_id}: vectorized scoring produced a non-finite value.") from error


def _score_candidates(
    agent: ObservationResponse, config: HeuristicConfig,
    candidates: list[tuple[float, float]], walls: tuple[Segment, ...],
    fruits: tuple[Entity, ...], predators: tuple[Entity, ...], *,
    preferred: float, scan: float, tree_target: bool, walking: float,
) -> CandidateScores:
    count = len(candidates)
    modifier = BIOME_MOVEMENT[agent.biome]
    # Keep the oracle's trigonometry/order for travel endpoints and near-wall comparisons.
    endpoints = np.asarray([
        (distance * modifier * math.cos(direction), distance * modifier * math.sin(direction))
        for distance, direction in candidates
    ], dtype=np.float64).reshape(-1, 2)
    walls_result = wall_scores(endpoints, walls, config.wall_margin)
    active = np.flatnonzero(~walls_result.blocked)
    turns, costs = np.zeros(count), np.zeros(count)
    terms = np.zeros((count, 5))
    scores, errors = np.full(count, -np.inf), np.zeros(count)
    uncertain = walls_result.uncertain.copy()
    if not len(active):
        return CandidateScores(endpoints, turns, costs, terms, scores,
                               walls_result.blocked, uncertain, errors)
    values = np.asarray(candidates, dtype=np.float64)[active]
    positions = endpoints[active]
    distances, directions = values[:, 0], values[:, 1]
    unique_distances, inverse = np.unique(distances, return_inverse=True)
    movement = np.asarray([movement_cost(agent, float(value)) for value in unique_distances])[inverse]
    desired = scan + (directions if tree_target else _wrap(directions - preferred))
    row = np.arange(len(active))
    food = np.zeros(len(active))
    if fruits:
        fruit_positions, fruit_distances, offsets, remaining = _entity_arrays(fruits, positions)
        nearest, tied = _nearest(fruit_positions, remaining)
        contact = remaining <= COLLECTION_RADIUS + _EPSILON
        desired = np.where(contact[row, nearest], 0.0,
                           np.arctan2(offsets[row, nearest, 1], offsets[row, nearest, 0]))
        stride = max(1.0, min(agent.speed, movement_limit(agent)) * modifier)
        hunger = 1.0 + max(0.0, 1.0 - agent.energy / agent.max_energy)
        proximity = 1.0 / (1.0 + fruit_distances / max(agent.hearing_radius, FOOD_MIN_RANGE))
        progress = (
            np.maximum(0.0, fruit_distances - COLLECTION_RADIUS)
            - np.maximum(0.0, remaining - COLLECTION_RADIUS)
        ) / stride
        food = np.max(hunger * proximity * (np.clip(progress, -1.0, 1.0) + contact), axis=1)
        uncertain[active] |= tied | _near(remaining, COLLECTION_RADIUS + _EPSILON).any(axis=1)
    if predators:
        predator_positions, predator_distances, offsets, remaining = _entity_arrays(predators, positions)
        nearest, tied = _nearest(predator_positions, remaining)
        facing_distance = np.minimum(predator_distances[nearest], remaining[row, nearest])
        facing = facing_distance <= config.threat_distance + PREDATOR_TRAVEL
        desired = np.where(
            facing, np.arctan2(offsets[row, nearest, 1], offsets[row, nearest, 0]), desired,
        )
        uncertain[active] |= tied | _near(facing_distance, config.threat_distance + PREDATOR_TRAVEL)
    desired = _wrap(desired)
    uncertain[active] |= _near(np.abs(desired), math.pi)
    turn_budget = np.maximum(0.0, agent.energy - movement) * TURN_ENERGY_FRACTION
    cap = np.minimum(config.max_turn, turn_budget * math.tau)
    turning = np.clip(desired, -cap, cap)
    expenditure = movement + np.minimum(math.pi, np.abs(turning)) / math.tau
    danger = np.zeros(len(active))
    if predators:
        paths = np.stack((np.zeros_like(positions), positions), axis=1)
        closest = point_segment_distances(predator_positions, paths).T
        scale = max(AGENT_RADIUS, movement_limit(agent) * modifier)
        contact_margin = AGENT_RADIUS + PREDATOR_RADIUS + PREDATOR_TRAVEL
        urgency = np.maximum(0.0, 1.0 - predator_distances / (config.threat_distance + contact_margin))
        proximity = np.maximum(0.0, 1.0 - (remaining - contact_margin) / config.threat_distance)
        capture = np.maximum(0.0, 1.0 - (remaining - contact_margin) / PREDATOR_TRAVEL)
        approach = np.maximum(0.0, (predator_distances - closest) / PREDATOR_TRAVEL)
        bearing = np.arctan2(offsets[:, :, 1], offsets[:, :, 0])
        exposure = THREAT_EXPOSURE_WEIGHT * urgency * (1.0 - np.cos(_wrap(bearing - turning[:, None])))
        danger = _ordered_sum(
            urgency * (remaining - predator_distances) / scale - proximity ** 2
            - THREAT_CAPTURE_WEIGHT * capture ** 2 - urgency * approach ** 2 - exposure,
        )
    exploration = np.zeros(len(active))
    if not fruits:
        progress = positions[:, 0] * math.cos(preferred) + positions[:, 1] * math.sin(preferred)
        exploration = np.clip(progress / max(1.0, walking * modifier), -1.0, 1.0)
    turns[active], costs[active] = turning, expenditure
    terms[active] = np.column_stack((food, -expenditure, walls_result.scores[active], danger, exploration))
    scores[active] = (
        config.food_weight * food + config.energy_weight * -expenditure
        + config.wall_weight * walls_result.scores[active] + config.danger_weight * danger
        + config.exploration_weight * exploration
    )
    weights = np.asarray([config.food_weight, config.energy_weight, config.wall_weight,
                          config.danger_weight, config.exploration_weight])
    errors[active] = _FLOAT_TOLERANCE * np.maximum(1.0, (np.abs(terms[active]) * weights).sum(axis=1))
    if not (np.isfinite(scores[active]).all() and np.isfinite(terms[active]).all()):
        raise ValueError(f"Agent {agent.agent_id}: vectorized scoring produced a non-finite value.")
    return CandidateScores(endpoints, turns, costs, terms, scores,
                           walls_result.blocked, uncertain, errors)
