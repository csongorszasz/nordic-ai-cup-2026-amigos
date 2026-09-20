"""Stateless, observation-only controller with independently inspectable scores."""

import hashlib
import heapq
import math
from collections.abc import Iterator
from dataclasses import dataclass

from src.policies.config import HeuristicConfig
from src.policies.features import Entity, parse_entities, validate_step
from src.policies.geometry import (
    BIOME_MOVEMENT, Point, Segment, can_spawn, movement_cost, movement_limit,
    point_segment_distance, segment_distance, turn_cost, wrap_angle,
)
from src.utils.DTOs import ActionRequest, ObservationResponse, StepResponse


AGENT_RADIUS = 5.0
COLLECTION_RADIUS = 4.0
PREDATOR_RADIUS = 10.0
PREDATOR_TRAVEL = 15.0
MAX_FOOD_TARGETS = 4
MAX_EXPLANATIONS = 128
FOOD_MIN_RANGE = 10.0
WALL_SWEEP_WEIGHT = 0.5
THREAT_CAPTURE_WEIGHT = 3.0
THREAT_EXPOSURE_WEIGHT = 0.15
TURN_ENERGY_FRACTION = 0.5
SCORE_DECIMALS = 12
_EPSILON = 1e-9
_ORIGIN = (0.0, 0.0)


@dataclass(frozen=True)
class ScoreTerms:
    food: float
    energy: float
    wall: float
    threat: float
    exploration: float

    def weighted(self, config: HeuristicConfig) -> float:
        return (
            config.food_weight * self.food + config.energy_weight * self.energy
            + config.wall_weight * self.wall + config.danger_weight * self.threat
            + config.exploration_weight * self.exploration
        )


@dataclass(frozen=True)
class WallAssessment:
    blocked: bool
    score: float
    swept_clearance: float | None
    endpoint_clearance: float | None


@dataclass(frozen=True)
class BreedingDecision:
    spawn: bool
    reason: str
    post_action_energy: float


@dataclass(frozen=True)
class Decision:
    action: ActionRequest
    reason: str
    terms: ScoreTerms
    score: float
    breeding: BreedingDecision
    candidates: int


def _distance(entity: Entity, endpoint: Point) -> float:
    return math.hypot(entity.x - endpoint[0], entity.y - endpoint[1])


def food_score(
    agent: ObservationResponse, endpoint: Point, fruits: tuple[Entity, ...],
) -> float:
    """Reward reachable progress and contact, not unobserved fruit energy."""
    stride = max(1.0, min(agent.speed, movement_limit(agent)) * BIOME_MOVEMENT[agent.biome])
    hunger = 1.0 + max(0.0, 1.0 - agent.energy / agent.max_energy)
    values = []
    for fruit in fruits:
        remaining = _distance(fruit, endpoint)
        progress = (
            max(0.0, fruit.distance - COLLECTION_RADIUS)
            - max(0.0, remaining - COLLECTION_RADIUS)
        ) / stride
        contact = float(remaining <= COLLECTION_RADIUS + _EPSILON)
        proximity = 1.0 / (1.0 + fruit.distance / max(agent.hearing_radius, FOOD_MIN_RANGE))
        values.append(hunger * proximity * (max(-1.0, min(1.0, progress)) + contact))
    return max(values) if values else 0.0


def energy_expenditure(agent: ObservationResponse, distance: float, turn: float) -> float:
    """Requested travel costs energy before collision; terrain only changes displacement."""
    return movement_cost(agent, distance) + turn_cost(turn)


def assess_walls(endpoint: Point, walls: tuple[Segment, ...], margin: float) -> WallAssessment:
    """Check the swept body, allowing outward recovery from an existing overlap."""
    path = (_ORIGIN, endpoint)
    moving = math.hypot(*endpoint) > _EPSILON
    blocked, score = False, 0.0
    swept_clearance = endpoint_clearance = None
    for wall in walls:
        initial = point_segment_distance(_ORIGIN, wall)
        swept = segment_distance(path, wall)
        final = point_segment_distance(endpoint, wall)
        if not all(math.isfinite(value) for value in (initial, swept, final)):
            raise ValueError("Wall geometry produced non-finite distances.")
        if moving:
            if initial < AGENT_RADIUS + _EPSILON:
                blocked |= swept < initial - _EPSILON or final <= initial + _EPSILON
            else:
                blocked |= swept < AGENT_RADIUS + _EPSILON
        start_room = max(0.0, min(1.0, (initial - AGENT_RADIUS) / margin))
        end_room = max(0.0, min(1.0, (final - AGENT_RADIUS) / margin))
        sweep_room = max(0.0, min(1.0, (swept - AGENT_RADIUS) / margin))
        score += end_room - start_room - WALL_SWEEP_WEIGHT * (1.0 - sweep_room) ** 2
        swept_clearance = (
            swept - AGENT_RADIUS if swept_clearance is None
            else min(swept_clearance, swept - AGENT_RADIUS)
        )
        endpoint_clearance = (
            final - AGENT_RADIUS if endpoint_clearance is None
            else min(endpoint_clearance, final - AGENT_RADIUS)
        )
    return WallAssessment(blocked, score, swept_clearance, endpoint_clearance)


def threat_score(
    agent: ObservationResponse, endpoint: Point, turn: float,
    predators: tuple[Entity, ...], config: HeuristicConfig,
) -> float:
    """All visible predators contribute, including approach along the swept path."""
    score = 0.0
    scale = max(AGENT_RADIUS, movement_limit(agent) * BIOME_MOVEMENT[agent.biome])
    # Cached observations can precede one predator move. Its next move is additional.
    contact_margin = AGENT_RADIUS + PREDATOR_RADIUS + PREDATOR_TRAVEL
    horizon = config.threat_distance + contact_margin
    for predator in predators:
        final = _distance(predator, endpoint)
        closest = point_segment_distance((predator.x, predator.y), (_ORIGIN, endpoint))
        urgency = max(0.0, 1.0 - predator.distance / horizon)
        proximity = max(0.0, 1.0 - (final - contact_margin) / config.threat_distance)
        capture = max(0.0, 1.0 - (final - contact_margin) / PREDATOR_TRAVEL)
        approach = max(0.0, (predator.distance - closest) / PREDATOR_TRAVEL)
        bearing = math.atan2(predator.y - endpoint[1], predator.x - endpoint[0])
        exposure = THREAT_EXPOSURE_WEIGHT * urgency * (1.0 - math.cos(wrap_angle(bearing - turn)))
        score += (
            urgency * (final - predator.distance) / scale - proximity ** 2
            - THREAT_CAPTURE_WEIGHT * capture ** 2 - urgency * approach ** 2 - exposure
        )
    return score


def exploration_score(endpoint: Point, bearing: float, stride: float) -> float:
    progress = endpoint[0] * math.cos(bearing) + endpoint[1] * math.sin(bearing)
    return max(-1.0, min(1.0, progress / max(1.0, stride)))


def candidate_walls(
    walls: tuple[Segment, ...], maximum_displacement: float, margin: float,
) -> tuple[Segment, ...]:
    # Beyond this bound, every candidate has full clearance and exactly zero wall score.
    radius = maximum_displacement + AGENT_RADIUS + margin + _EPSILON
    return tuple(wall for wall in walls if point_segment_distance(_ORIGIN, wall) <= radius)


def breeding_decision(
    agent: ObservationResponse, distance: float, turn: float, endpoint: Point,
    fruits: tuple[Entity, ...], predators: tuple[Entity, ...],
    walls: tuple[Segment, ...], config: HeuristicConfig,
) -> BreedingDecision:
    post = agent.energy - energy_expenditure(agent, distance, turn)
    if not can_spawn(agent, distance, turn):
        return BreedingDecision(False, "engine_threshold", post)
    possible_age_drain = 0.01 * agent.age if agent.age >= 60.0 else 0.0
    if post - 100.0 < config.breeding_reserve + possible_age_drain:
        return BreedingDecision(False, "reserve", post)
    if any(
        min(predator.distance, _distance(predator, endpoint))
        <= config.threat_distance + PREDATOR_TRAVEL
        for predator in predators
    ):
        return BreedingDecision(False, "threat", post)
    if any(point_segment_distance(endpoint, wall) < 2.0 * AGENT_RADIUS for wall in walls):
        return BreedingDecision(False, "wall", post)
    food_range = max(agent.hearing_radius, 2.0 * AGENT_RADIUS)
    nearby_food = sum(_distance(fruit, endpoint) <= food_range for fruit in fruits)
    rich = post >= config.breeding_energy
    old = agent.age >= config.breeding_age
    # Renew old lives with ample stored energy even when no fruit is currently visible.
    # Younger parents need a patch, not a single cached fruit they will themselves eat.
    if old and (nearby_food >= 1 or rich):
        return BreedingDecision(True, "renew_age", post)
    if rich and nearby_food >= 2:
        return BreedingDecision(True, "food_surplus", post)
    return BreedingDecision(False, "food_or_age", post)


def _affordable_limit(agent: ObservationResponse) -> float:
    budget = max(0.0, agent.energy) * 0.95
    walking_cost = agent.speed * 0.05
    distance = (
        budget / 0.05 if budget <= walking_cost
        else agent.speed + (budget - walking_cost) / 0.5
    )
    return min(movement_limit(agent), distance)


def _food_targets(
    entities: tuple[Entity, ...], walls: tuple[Segment, ...], margin: float,
) -> tuple[Entity, ...]:
    nearest = heapq.nsmallest(
        MAX_FOOD_TARGETS, (entity for entity in entities if entity.kind == "Fruit"),
        key=lambda fruit: (fruit.distance, fruit.x, fruit.y),
    )
    reachable = []
    for fruit in nearest:
        travel = max(0.0, fruit.distance - COLLECTION_RADIUS)
        endpoint = (travel * math.cos(fruit.angle), travel * math.sin(fruit.angle))
        if not assess_walls(endpoint, walls, margin).blocked:
            reachable.append(fruit)
    return tuple(reachable)


@dataclass(frozen=True)
class HeuristicPolicy:
    seed: int
    config: HeuristicConfig

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("Heuristic seed must be an integer.")
        if not isinstance(self.config, HeuristicConfig):
            raise TypeError("HeuristicPolicy requires a validated HeuristicConfig.")

    def act(self, step: StepResponse) -> list[ActionRequest]:
        return [decision.action for decision in self._decisions(step)]

    def act_validated(self, step: StepResponse) -> list[ActionRequest]:
        return [decision.action for decision in self._decisions(step, validate=False)]

    def explain(self, step: StepResponse, limit: int = 32) -> tuple[Decision, ...]:
        """Return at most 128 decisions; retain no history and still validate the full step."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= MAX_EXPLANATIONS:
            raise ValueError(f"Explanation limit must be an integer in [0, {MAX_EXPLANATIONS}].")
        result = []
        for decision in self._decisions(step):
            if len(result) < limit:
                result.append(decision)
        return tuple(result)

    def _decisions(self, step: StepResponse, *, validate: bool = True) -> Iterator[Decision]:
        if validate:
            validate_step(step)
        for agent in step.agent_status:
            yield self._decide(agent, parse_entities(agent))

    def _decide(self, agent: ObservationResponse, entities: tuple[Entity, ...]) -> Decision:
        config = self.config
        walls = tuple(entity.segment for entity in entities if entity.segment is not None)
        predators = tuple(entity for entity in entities if entity.kind == "Predator")
        fruits = _food_targets(entities, walls, config.wall_margin)
        digest = hashlib.blake2b(
            f"{self.seed}:{agent.agent_id}".encode("ascii"), digest_size=8,
        ).digest()
        phase = math.tau * int.from_bytes(digest, "big") / 2**64
        preferred = 0.2 * math.sin(phase)
        scan = config.scan_turn * math.sin(4.0 * agent.age + phase)
        trees = (entity for entity in entities if entity.kind == "Tree")
        tree = min(trees, key=lambda value: (value.distance, value.x, value.y), default=None)
        tree_target = tree is not None and tree.distance > max(agent.hearing_radius, 10.0)
        if tree is not None and tree_target and not fruits:
            preferred = tree.angle
        limit = _affordable_limit(agent)
        walking = min(agent.speed, limit)
        if walking == 0.0 and limit > 0.0:
            walking = min(AGENT_RADIUS, limit)
        modifier = BIOME_MOVEMENT[agent.biome]
        nearby_walls = candidate_walls(walls, limit * modifier, config.wall_margin)
        nearest_threat = min(
            predators, key=lambda value: (value.distance, value.x, value.y), default=None,
        )
        sprinting = (
            nearest_threat is not None
            and nearest_threat.distance <= config.sprint_distance + AGENT_RADIUS + PREDATOR_RADIUS
        )
        speeds = {walking * 0.5, walking}
        if sprinting:
            speeds.update((walking + (limit - walking) * 0.5, limit))
        angles = {wrap_angle(math.tau * index / config.directions) for index in range(config.directions)}
        angles.add(preferred)
        angles.update(wrap_angle(fruit.angle) for fruit in fruits)
        if nearest_threat is not None:
            angles.update(wrap_angle(nearest_threat.angle + offset) for offset in (
                math.pi, math.pi / 2.0, -math.pi / 2.0,
            ))
        if walls:
            nearest_wall = min(walls, key=lambda wall: (point_segment_distance(_ORIGIN, wall), wall))
            (sx, sy), (ex, ey) = nearest_wall
            tangent = math.atan2(ey - sy, ex - sx)
            angles.update(wrap_angle(tangent + offset) for offset in (0.0, math.pi, math.pi / 2, -math.pi / 2))
        candidates = {(0.0, 0.0)}
        candidates.update(
            (distance, direction) for direction in angles for distance in speeds if distance > 0.0
        )
        for fruit in fruits:
            distance = min(
                limit if sprinting else walking,
                max(0.0, fruit.distance - COLLECTION_RADIUS) / modifier,
            )
            candidates.add((distance, wrap_angle(fruit.angle) if distance else 0.0))
        scoring_candidates = sorted(candidates)
        if config.backend == "vectorized":
            from src.policies.vectorized import score_candidates

            scores = score_candidates(
                agent, config, scoring_candidates, nearby_walls, fruits, predators,
                preferred=preferred, scan=scan, tree_target=bool(tree_target and not fruits),
                walking=walking,
            )
            # Preserve scalar tie-breaking and exact diagnostics at numerical boundaries.
            scoring_candidates = [
                candidate for candidate, keep in zip(scoring_candidates, scores.contenders(), strict=True)
                if keep
            ]
        best = None
        best_key = None
        for distance, direction in scoring_candidates:
            endpoint = (distance * modifier * math.cos(direction), distance * modifier * math.sin(direction))
            wall = assess_walls(endpoint, nearby_walls, config.wall_margin)
            if wall.blocked:
                continue
            turn = self._turn(agent, endpoint, distance, direction, preferred, scan,
                              bool(tree_target and not fruits), fruits, predators)
            cost = energy_expenditure(agent, distance, turn)
            terms = ScoreTerms(
                food_score(agent, endpoint, fruits), -cost, wall.score,
                threat_score(agent, endpoint, turn, predators, config),
                0.0 if fruits else exploration_score(endpoint, preferred, walking * modifier),
            )
            score = terms.weighted(config)
            if not math.isfinite(score) or not math.isfinite(cost):
                raise ValueError(f"Agent {agent.agent_id}: heuristic scoring produced a non-finite value.")
            key = (round(score, SCORE_DECIMALS), -cost, -distance, -abs(turn),
                   -abs(wrap_angle(direction - preferred)), -direction)
            if best_key is None or key > best_key:
                best_key = key
                best = (distance, direction, endpoint, turn, terms, score)
        if best is None:
            raise ValueError(f"Agent {agent.agent_id}: no valid movement candidate.")
        distance, direction, endpoint, turn, terms, score = best
        breeding = breeding_decision(
            agent, distance, turn, endpoint, fruits, predators, walls, config,
        )
        if nearest_threat is not None and nearest_threat.distance <= config.threat_distance + PREDATOR_TRAVEL:
            reason = "escape" if sprinting else "evade"
        elif distance == 0.0 and fruits:
            reason = "collect" if any(fruit.distance <= COLLECTION_RADIUS for fruit in fruits) else "conserve"
        elif fruits and terms.food > 0.0:
            reason = "forage"
        elif any(
            point_segment_distance(_ORIGIN, segment) < AGENT_RADIUS + config.wall_margin
            for segment in walls
        ):
            reason = "clear_wall" if distance else "conserve"
        else:
            reason = "explore" if distance else "conserve"
        return Decision(
            ActionRequest(agent_id=agent.agent_id, move_distance=distance,
                          move_direction=direction, turn_angle=turn, spawn_agent=breeding.spawn),
            reason, terms, score, breeding, len(candidates),
        )

    def _turn(
        self, agent: ObservationResponse, endpoint: Point, distance: float, direction: float,
        preferred: float, scan: float, tree_target: bool, fruits: tuple[Entity, ...],
        predators: tuple[Entity, ...],
    ) -> float:
        threat = min(
            predators, key=lambda entity: (_distance(entity, endpoint), entity.x, entity.y),
            default=None,
        )
        if threat is not None and min(threat.distance, _distance(threat, endpoint)) <= self.config.threat_distance + PREDATOR_TRAVEL:
            desired = math.atan2(threat.y - endpoint[1], threat.x - endpoint[0])
        elif fruits:
            fruit = min(fruits, key=lambda entity: (_distance(entity, endpoint), entity.x, entity.y))
            desired = (
                0.0 if _distance(fruit, endpoint) <= COLLECTION_RADIUS + _EPSILON
                else math.atan2(fruit.y - endpoint[1], fruit.x - endpoint[0])
            )
        else:
            desired = scan + (direction if tree_target else wrap_angle(direction - preferred))
        turn_budget = max(0.0, agent.energy - movement_cost(agent, distance)) * TURN_ENERGY_FRACTION
        cap = min(self.config.max_turn, turn_budget * math.tau)
        return max(-cap, min(cap, wrap_angle(desired)))


def build_policy(seed: int, config: HeuristicConfig):
    if config.backend == "turnaway":
        from src.policies.turnaway import TurnawayPolicy

        return TurnawayPolicy(seed)
    if config.backend == "hierarchical":
        from src.policies.hierarchical import HierarchicalPolicy

        return HierarchicalPolicy(seed, config)
    return HeuristicPolicy(seed, config)


def create_policy(seed: int, config: dict):
    return build_policy(seed, HeuristicConfig.model_validate(config))
