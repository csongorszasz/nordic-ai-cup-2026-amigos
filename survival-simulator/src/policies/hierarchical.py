"""Latency-first centralized controller with lightweight team state."""

from __future__ import annotations

import math
from collections import deque
from numbers import Real
from dataclasses import dataclass, field

from src.policies.config import HeuristicConfig
from src.policies.features import ENTITY_TYPES, Entity, validate_step
from src.policies.geometry import (
    AGENT_RADIUS, BIOME_MOVEMENT, Point, Segment, can_spawn, movement_limit, relative_heading,
    segment_distance, wrap_angle,
)
from src.utils.DTOs import ActionRequest, ObservationResponse, StepResponse


PREDATOR_TRAVEL_PER_TICK = 15.0
PREDATOR_LOOKAHEAD_TICKS = 2.0
FRUIT_CLUSTER_RADIUS = 8.0
TREE_CLUSTER_RADIUS = 18.0
_EPSILON = 1e-9


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    heading: float

    def local_point(self, x: float, y: float) -> Point:
        cosine, sine = math.cos(self.heading), math.sin(self.heading)
        return self.x + cosine * x - sine * y, self.y + sine * x + cosine * y

    def world_point(self, x: float, y: float) -> Point:
        dx, dy = x - self.x, y - self.y
        cosine, sine = math.cos(self.heading), math.sin(self.heading)
        return cosine * dx + sine * dy, -sine * dx + cosine * dy


@dataclass(frozen=True)
class AgentContext:
    agent: ObservationResponse
    entities: tuple[Entity, ...]
    pose: Pose
    component: int
    walls: tuple[Segment, ...]
    predators: tuple[Entity, ...]


@dataclass
class SceneTarget:
    kind: str
    component: int
    x: float
    y: float
    sightings: int = 1
    observers: set[int] = field(default_factory=set)

    def merge(self, point: Point, observer: int) -> None:
        total = self.sightings + 1
        self.x = (self.x * self.sightings + point[0]) / total
        self.y = (self.y * self.sightings + point[1]) / total
        self.sightings = total
        self.observers.add(observer)


@dataclass(frozen=True)
class Assignment:
    kind: str
    x: float
    y: float


def _trait_score(agent: ObservationResponse) -> float:
    return (
        5.0 * agent.speed / 20.0
        + 4.0 * agent.sprint_speed / 40.0
        + 2.0 * agent.max_energy / 1000.0
        + agent.hearing_radius / 100.0
        + agent.vision_range / 400.0
        + agent.vision_angle / (math.pi / 2.0)
    )


def _clamp_turn(value: float, maximum: float) -> float:
    return max(-maximum, min(maximum, wrap_angle(value)))


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number.")
    return float(value)


def _parse_entities(agent: ObservationResponse) -> tuple[Entity, ...]:
    entities = []
    seen_edges = set()
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
                points.append((
                    _number(point[0], "edge.x"),
                    _number(point[1], "edge.y"),
                ))
            segment = min(points), max(points)
            if segment in seen_edges:
                continue
            seen_edges.add(segment)
            (sx, sy), (ex, ey) = segment
            dx, dy = ex - sx, ey - sy
            length = dx * dx + dy * dy
            fraction = 0.0 if length == 0.0 else max(
                0.0, min(1.0, -(sx * dx + sy * dy) / length),
            )
            x, y = sx + fraction * dx, sy + fraction * dy
            entities.append(Entity(
                kind, x, y, math.hypot(x, y), math.atan2(y, x), segment=segment,
            ))
            continue
        distance = _number(observation.get("distance"), "entity.distance")
        angle = _number(observation.get("angle"), "entity.angle")
        if distance < 0:
            raise ValueError("Entity distance must be nonnegative.")
        rel_dir = (
            _number(observation.get("rel_dir"), "entity.rel_dir")
            if kind in ("Agent", "Predator") else None
        )
        agent_id = observation.get("id") if kind == "Agent" else None
        if kind == "Agent" and (
            isinstance(agent_id, bool) or not isinstance(agent_id, int) or agent_id < 0
        ):
            raise ValueError("Observed teammate id must be a nonnegative integer.")
        entities.append(Entity(
            kind, distance * math.cos(angle), distance * math.sin(angle),
            distance, angle, rel_dir, agent_id,
        ))
    order = {kind: index for index, kind in enumerate(ENTITY_TYPES)}
    entities.sort(key=lambda entity: (
        order[entity.kind], entity.x, entity.y,
        entity.agent_id if entity.agent_id is not None else -1,
    ))
    return tuple(entities)


def _build_contexts(step: StepResponse) -> dict[int, AgentContext]:
    agents = {agent.agent_id: agent for agent in step.agent_status}
    entities = {agent_id: _parse_entities(agent) for agent_id, agent in agents.items()}
    adjacency: dict[int, list[tuple[int, float, float, float]]] = {
        agent_id: [] for agent_id in agents
    }
    for observer_id, observations in entities.items():
        for entity in observations:
            teammate_id = entity.agent_id
            if entity.kind != "Agent" or teammate_id not in agents:
                continue
            heading = relative_heading(entity.angle, entity.rel_dir or 0.0)
            adjacency[observer_id].append((teammate_id, entity.x, entity.y, heading))
            cosine, sine = math.cos(heading), math.sin(heading)
            inverse_x = -(cosine * entity.x + sine * entity.y)
            inverse_y = sine * entity.x - cosine * entity.y
            adjacency[teammate_id].append((observer_id, inverse_x, inverse_y, -heading))
    poses: dict[int, Pose] = {}
    components: dict[int, int] = {}
    component = 0
    for root in sorted(agents):
        if root in poses:
            continue
        poses[root] = Pose(0.0, 0.0, 0.0)
        components[root] = component
        queue = deque([root])
        while queue:
            observer_id = queue.popleft()
            observer_pose = poses[observer_id]
            for teammate_id, local_x, local_y, local_heading in adjacency[observer_id]:
                if teammate_id in poses:
                    continue
                x, y = observer_pose.local_point(local_x, local_y)
                heading = wrap_angle(observer_pose.heading + local_heading)
                poses[teammate_id] = Pose(x, y, heading)
                components[teammate_id] = component
                queue.append(teammate_id)
        component += 1
    return {
        agent_id: AgentContext(
            agents[agent_id], entities[agent_id], poses[agent_id], components[agent_id],
            tuple(entity.segment for entity in entities[agent_id] if entity.segment is not None),
            tuple(entity for entity in entities[agent_id] if entity.kind == "Predator"),
        )
        for agent_id in agents
    }


def _cluster_targets(
    contexts: dict[int, AgentContext], *, fruit_contact_radius: float,
) -> dict[str, list[SceneTarget]]:
    clusters = {"Fruit": [], "Tree": []}
    radii = {
        "Fruit": FRUIT_CLUSTER_RADIUS,
        "Tree": TREE_CLUSTER_RADIUS,
    }
    for agent_id in sorted(contexts):
        context = contexts[agent_id]
        for entity in context.entities:
            if entity.kind not in clusters or entity.segment is not None:
                continue
            if entity.kind == "Fruit" and entity.distance <= fruit_contact_radius + _EPSILON:
                # Observations are captured before fruit collection. Anything already in
                # guaranteed contact has been removed before this response is returned.
                continue
            point = context.pose.local_point(entity.x, entity.y)
            best = None
            best_distance = math.inf
            for candidate in clusters[entity.kind]:
                if candidate.component != context.component:
                    continue
                distance = math.hypot(candidate.x - point[0], candidate.y - point[1])
                if distance <= radii[entity.kind] and distance < best_distance:
                    best, best_distance = candidate, distance
            if best is None:
                clusters[entity.kind].append(SceneTarget(
                    entity.kind, context.component, point[0], point[1],
                    observers={agent_id},
                ))
            else:
                best.merge(point, agent_id)
    return clusters


def _greedy_assign(
    contexts: dict[int, AgentContext], agent_ids: set[int],
    targets: list[SceneTarget], *, capacity: int, patch_radius: float = 0.0,
) -> dict[int, Assignment]:
    slots: list[tuple[SceneTarget, int]] = [
        (target, slot) for target in targets for slot in range(capacity)
    ]
    pairs = []
    for agent_id in agent_ids:
        context = contexts[agent_id]
        hunger = max(0.0, 1.0 - context.agent.energy / context.agent.max_energy)
        for slot_index, (target, slot) in enumerate(slots):
            if target.component != context.component:
                continue
            angle = math.tau * slot / max(1, capacity) + 0.73 * (slot_index + 1)
            x = target.x + patch_radius * math.cos(angle)
            y = target.y + patch_radius * math.sin(angle)
            distance = math.hypot(context.pose.x - x, context.pose.y - y)
            priority = distance / (1.0 + 1.5 * hunger)
            pairs.append((priority, distance, agent_id, slot_index, x, y, target.kind))
    assignments: dict[int, Assignment] = {}
    used_slots: set[int] = set()
    for _, _, agent_id, slot_index, x, y, kind in sorted(pairs):
        if agent_id in assignments or slot_index in used_slots:
            continue
        assignments[agent_id] = Assignment(kind, x, y)
        used_slots.add(slot_index)
    return assignments


def _path_blocked(
    distance: float, direction: float, biome: str, walls: tuple[Segment, ...],
) -> bool:
    if distance <= _EPSILON or not walls:
        return False
    displacement = distance * BIOME_MOVEMENT[biome]
    endpoint = displacement * math.cos(direction), displacement * math.sin(direction)
    path = ((0.0, 0.0), endpoint)
    return any(segment_distance(path, wall) < AGENT_RADIUS + _EPSILON for wall in walls)


def _clear_direction(
    distance: float, direction: float, biome: str, walls: tuple[Segment, ...],
) -> tuple[float, float]:
    if not _path_blocked(distance, direction, biome, walls):
        return distance, wrap_angle(direction)
    for offset in (
        math.pi / 3.0, -math.pi / 3.0, math.pi / 2.0, -math.pi / 2.0, math.pi,
    ):
        candidate = wrap_angle(direction + offset)
        if not _path_blocked(distance, candidate, biome, walls):
            return distance, candidate
    return 0.0, 0.0


class HierarchicalPolicy:
    """Shared-target controller designed for the submission latency budget."""

    stateful = True

    def __init__(self, seed: int, config: HeuristicConfig):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Hierarchical policy seed must be an integer.")
        if not isinstance(config, HeuristicConfig):
            raise TypeError("HierarchicalPolicy requires a validated HeuristicConfig.")
        self.seed = seed
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._last_score: float | None = None
        self._last_sim_time: float | None = None
        self._food_rate = 0.0
        self._capacity_hint = float(self.config.min_population)
        self._tick = 0
        self.previous_actions: dict[int, ActionRequest] = {}

    def act(self, step: StepResponse) -> list[ActionRequest]:
        validate_step(step)
        return self.act_validated(step)

    def act_validated(self, step: StepResponse) -> list[ActionRequest]:
        if not step.agent_status:
            self.reset()
            return []
        self._update_score_state(step.score, step.sim_time)
        contexts = _build_contexts(step)
        targets = _cluster_targets(
            contexts,
            fruit_contact_radius=self.config.fruit_contact_radius,
        )
        fruits, trees = targets["Fruit"], targets["Tree"]
        threatened = {
            agent_id for agent_id, context in contexts.items()
            if context.predators and min(value.distance for value in context.predators)
            <= self.config.threat_distance + PREDATOR_TRAVEL_PER_TICK
        }
        available = set(contexts) - threatened
        assignments = _greedy_assign(contexts, available, fruits, capacity=1)
        remaining = available - set(assignments)
        assignments.update(_greedy_assign(
            contexts, remaining, trees, capacity=self.config.patch_capacity,
            patch_radius=self.config.patch_radius,
        ))
        breeders, scouts = self._roles(contexts)
        actions = {
            agent_id: self._movement(
                context, assignments.get(agent_id), agent_id in scouts,
            )
            for agent_id, context in contexts.items()
        }
        desired_population = self._desired_population(len(contexts), fruits, trees)
        spawn_ids = self._select_spawns(
            contexts, actions, breeders, desired_population,
            observed_food=bool(fruits or trees) or self._food_rate > 0.002,
        )
        result = []
        for agent in step.agent_status:
            action = actions[agent.agent_id]
            if agent.agent_id in spawn_ids:
                action = action.model_copy(update={"spawn_agent": True})
            result.append(action)
        self.previous_actions = {action.agent_id: action for action in result}
        self._tick += 1
        return result

    def _update_score_state(self, score: float, sim_time: float) -> None:
        if self._last_score is not None:
            elapsed = (
                sim_time - self._last_sim_time
                if self._last_sim_time is not None and sim_time > self._last_sim_time
                else 0.1
            )
            excess = max(0.0, score - self._last_score - elapsed)
            self._food_rate = 0.98 * self._food_rate + 0.02 * excess
        self._last_score = score
        self._last_sim_time = sim_time

    def _roles(
        self, contexts: dict[int, AgentContext],
    ) -> tuple[set[int], set[int]]:
        count = len(contexts)
        breeder_count = max(1, min(4, math.ceil(count * self.config.breeder_fraction)))
        scout_count = max(1, math.ceil(count * self.config.scout_fraction))
        breeders = {
            agent_id for agent_id, _ in sorted(
                contexts.items(), key=lambda item: (-_trait_score(item[1].agent), item[0]),
            )[:breeder_count]
        }
        scouts = {
            agent_id for agent_id, _ in sorted(
                contexts.items(),
                key=lambda item: (
                    -(item[1].agent.speed + 0.5 * item[1].agent.sprint_speed
                      + 0.03 * item[1].agent.vision_range
                      + 0.05 * item[1].agent.hearing_radius),
                    item[0],
                ),
            )[:scout_count]
        }
        return breeders, scouts

    def _scan_turn(self, agent: ObservationResponse) -> float:
        phase = 2.399963229728653 * (agent.agent_id + 1)
        return self.config.scan_turn * math.sin(0.7 * agent.age + phase)

    def _movement(
        self, context: AgentContext, assignment: Assignment | None, scout: bool,
    ) -> ActionRequest:
        agent = context.agent
        if context.predators:
            nearest = min(context.predators, key=lambda value: value.distance)
            if nearest.distance <= self.config.threat_distance + PREDATOR_TRAVEL_PER_TICK:
                return self._escape(context, nearest)
        distance = direction = 0.0
        turn = self._scan_turn(agent)
        if assignment is not None:
            local_x, local_y = context.pose.world_point(assignment.x, assignment.y)
            target_distance = math.hypot(local_x, local_y)
            target_angle = math.atan2(local_y, local_x)
            modifier = BIOME_MOVEMENT[agent.biome]
            if assignment.kind == "Fruit":
                remaining = max(0.0, target_distance - self.config.fruit_contact_radius)
                distance = min(max(0.0, agent.speed), remaining / modifier)
            else:
                remaining = max(0.0, target_distance - self.config.patch_tolerance)
                distance = min(
                    max(0.0, agent.speed) * self.config.patch_move_fraction,
                    remaining / modifier,
                )
            if distance > _EPSILON:
                direction = target_angle
                turn = _clamp_turn(target_angle, self.config.max_turn)
        elif scout and agent.energy > max(25.0, 0.15 * agent.max_energy):
            distance = max(0.0, agent.speed) * self.config.exploration_move_fraction
            direction = 0.45 * math.sin(
                0.13 * agent.age + 1.618033988749895 * (agent.agent_id + 1),
            )
        distance = min(distance, movement_limit(agent))
        distance, direction = _clear_direction(distance, direction, agent.biome, context.walls)
        return ActionRequest(
            agent_id=agent.agent_id, move_distance=distance, move_direction=direction,
            turn_angle=turn, spawn_agent=False,
        )

    def _escape(self, context: AgentContext, predator: Entity) -> ActionRequest:
        if self.config.escape_strategy == "direct":
            return self._direct_escape(context, predator, wall_aware=False)
        if self.config.escape_strategy == "direct_wall_aware":
            return self._direct_escape(context, predator, wall_aware=True)
        return self._predictive_wall_aware_escape(context, predator)

    def _escape_distance(self, agent: ObservationResponse, predator: Entity) -> float:
        sprint = (
            predator.distance
            <= self.config.sprint_distance + AGENT_RADIUS + PREDATOR_TRAVEL_PER_TICK
        )
        return movement_limit(agent) if sprint else min(agent.speed, movement_limit(agent))

    def _direct_escape(
        self, context: AgentContext, predator: Entity, *, wall_aware: bool,
    ) -> ActionRequest:
        agent = context.agent
        distance = self._escape_distance(agent, predator)
        away = wrap_angle(predator.angle + math.pi)
        if wall_aware:
            distance, direction = _clear_direction(
                distance, away, agent.biome, context.walls,
            )
        else:
            direction = away
        return ActionRequest(
            agent_id=agent.agent_id, move_distance=distance,
            move_direction=direction,
            turn_angle=_clamp_turn(direction, self.config.predator_face_turn),
            spawn_agent=False,
        )

    def _predictive_wall_aware_escape(
        self, context: AgentContext, predator: Entity,
    ) -> ActionRequest:
        agent = context.agent
        distance = self._escape_distance(agent, predator)
        heading = relative_heading(predator.angle, predator.rel_dir or 0.0)
        lookahead = PREDATOR_TRAVEL_PER_TICK * PREDATOR_LOOKAHEAD_TICKS
        predicted = (
            predator.x + lookahead * math.cos(heading),
            predator.y + lookahead * math.sin(heading),
        )
        away = math.atan2(-predicted[1], -predicted[0])
        candidates = (
            away, away + math.pi / 3.0, away - math.pi / 3.0,
            away + math.pi / 2.0, away - math.pi / 2.0,
        )
        best_direction = away
        best_score = -math.inf
        modifier = BIOME_MOVEMENT[agent.biome]
        for candidate in candidates:
            candidate = wrap_angle(candidate)
            endpoint = (
                distance * modifier * math.cos(candidate),
                distance * modifier * math.sin(candidate),
            )
            blocked = _path_blocked(distance, candidate, agent.biome, context.walls)
            separation = math.hypot(predicted[0] - endpoint[0], predicted[1] - endpoint[1])
            covered = any(
                segment_distance((endpoint, predicted), wall) <= _EPSILON
                for wall in context.walls
            )
            score = separation + 25.0 * covered - 1000.0 * blocked
            if score > best_score:
                best_score, best_direction = score, candidate
        endpoint = (
            distance * modifier * math.cos(best_direction),
            distance * modifier * math.sin(best_direction),
        )
        desired_facing = math.atan2(
            predicted[1] - endpoint[1], predicted[0] - endpoint[0],
        )
        return ActionRequest(
            agent_id=agent.agent_id, move_distance=distance,
            move_direction=best_direction,
            turn_angle=_clamp_turn(desired_facing, self.config.predator_face_turn),
            spawn_agent=False,
        )

    def _desired_population(
        self, population: int, fruits: list[SceneTarget], trees: list[SceneTarget],
    ) -> int:
        observed_capacity = (
            self.config.min_population
            + min(4, len(fruits))
            + min(4, len(trees) * self.config.patch_capacity // 2)
            + int(self._food_rate >= 0.005)
        )
        self._capacity_hint = 0.95 * self._capacity_hint + 0.05 * observed_capacity
        desired = max(
            self.config.min_population,
            min(self.config.target_population, round(self._capacity_hint)),
        )
        if population < self.config.min_population:
            desired = self.config.min_population
        return min(desired, self.config.max_population)

    def _select_spawns(
        self, contexts: dict[int, AgentContext], actions: dict[int, ActionRequest],
        breeders: set[int], desired_population: int, *, observed_food: bool,
    ) -> set[int]:
        population = len(contexts)
        if population >= self.config.max_population:
            return set()
        regular_slots = max(0, desired_population - population)
        if population < self.config.min_population:
            threshold = self.config.low_population_breeding_energy
        elif population < self.config.target_population:
            threshold = self.config.target_population_breeding_energy
        else:
            threshold = self.config.high_population_breeding_energy
        candidates = []
        for agent_id, context in contexts.items():
            agent = context.agent
            action = actions[agent_id]
            nearest_threat = min(
                (predator.distance for predator in context.predators), default=math.inf,
            )
            emergency = (
                nearest_threat <= self.config.emergency_spawn_distance
                and population < self.config.max_population
            )
            if not can_spawn(agent, action.move_distance, action.turn_angle):
                continue
            post_action_energy = agent.energy
            if action.move_distance <= agent.speed:
                post_action_energy -= action.move_distance * 0.05
            else:
                post_action_energy -= (
                    agent.speed * 0.05 + (action.move_distance - agent.speed) * 0.5
                )
            post_action_energy -= min(math.pi, abs(action.turn_angle)) / math.tau
            regular = (
                agent_id in breeders
                and regular_slots > 0
                and nearest_threat > self.config.threat_distance
                and post_action_energy >= threshold
                and (observed_food or agent.age >= self.config.breeding_age)
            )
            if regular or (emergency and post_action_energy >= 135.0):
                candidates.append((
                    not emergency, -_trait_score(agent), -post_action_energy, agent_id,
                ))
        selected = set()
        for regular_only, _, _, agent_id in sorted(candidates):
            emergency = not regular_only
            if len(selected) + population >= self.config.max_population:
                break
            if emergency or len(selected) < regular_slots:
                selected.add(agent_id)
        return selected


def create_policy(seed: int, config: dict) -> HierarchicalPolicy:
    settings = HeuristicConfig.model_validate(config)
    if settings.backend != "hierarchical":
        raise ValueError("Hierarchical policy factory requires backend='hierarchical'.")
    return HierarchicalPolicy(seed, settings)
