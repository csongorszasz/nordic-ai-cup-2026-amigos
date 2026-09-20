"""Ordered survival rules without tunable behavioral thresholds."""

import math
import random

from src.policies.features import Entity, parse_entities, validate_step
from src.policies.geometry import (
    AGENT_RADIUS, BIOME_MOVEMENT, Point, Segment, can_spawn, closest_point,
    movement_limit, wrap_angle,
)
from src.utils.DTOs import ActionRequest, ObservationResponse, StepResponse


def _clear_fraction(displacement: Point, walls: tuple[Segment, ...]) -> float:
    """Sweep against wall rectangles expanded by the engine's square body buffer."""
    fraction = 1.0
    for start, end in walls:
        edge_x, edge_y = end[0] - start[0], end[1] - start[1]
        length = math.hypot(edge_x, edge_y)
        tangent_x, tangent_y = (edge_x / length, edge_y / length) if length else (1.0, 0.0)
        radius = AGENT_RADIUS if length else math.hypot(AGENT_RADIUS, AGENT_RADIUS)
        along = -start[0] * tangent_x - start[1] * tangent_y
        across = start[0] * tangent_y - start[1] * tangent_x
        travel_along = displacement[0] * tangent_x + displacement[1] * tangent_y
        travel_across = -displacement[0] * tangent_y + displacement[1] * tangent_x
        entry, exit = -math.inf, math.inf
        for position, travel, lower, upper in (
            (along, travel_along, -radius, length + radius),
            (across, travel_across, -radius, radius),
        ):
            if travel == 0.0:
                if not lower < position < upper:
                    break
            else:
                first, last = (lower - position) / travel, (upper - position) / travel
                entry = max(entry, min(first, last))
                exit = min(exit, max(first, last))
        else:
            if max(entry, 0.0) < min(exit, 1.0):
                fraction = min(fraction, max(0.0, math.nextafter(entry, 0.0)))
    return fraction


def _avoid_walls(
    distance: float, direction: float, biome: str, walls: tuple[Segment, ...],
) -> tuple[float, float]:
    direction = wrap_angle(direction)
    displacement = distance * BIOME_MOVEMENT[biome]

    def clear_fraction(candidate: float) -> float:
        return _clear_fraction((
            displacement * math.cos(candidate), displacement * math.sin(candidate),
        ), walls)

    fraction = clear_fraction(direction)
    if fraction == 1.0 or distance == 0.0:
        return distance, direction
    candidates = set()
    for start, end in walls:
        tangent = math.atan2(end[1] - start[1], end[0] - start[0])
        nearest_x, nearest_y = closest_point((0.0, 0.0), (start, end))
        candidates.update((
            wrap_angle(tangent), wrap_angle(tangent + math.pi),
            math.atan2(-nearest_y, -nearest_x),
        ))
    for candidate in sorted(candidates, key=lambda angle: (abs(wrap_angle(angle - direction)), angle)):
        if abs(wrap_angle(candidate - direction)) <= math.pi / 2 and clear_fraction(candidate) == 1.0:
            return distance, candidate
    return distance * fraction, direction


class TurnawayPolicy:
    stateful = True

    def __init__(self, seed: int):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Turnaway policy seed must be an integer.")
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self._rng = random.Random(self.seed)

    def act(self, step: StepResponse) -> list[ActionRequest]:
        validate_step(step)
        return self.act_validated(step)

    def act_validated(self, step: StepResponse) -> list[ActionRequest]:
        actions = {
            agent.agent_id: self._decide(agent)
            for agent in sorted(step.agent_status, key=lambda value: value.agent_id)
        }
        return [actions[agent.agent_id] for agent in step.agent_status]

    def _decide(self, agent: ObservationResponse) -> ActionRequest:
        entities = parse_entities(agent)
        walls = tuple(entity.segment for entity in entities if entity.segment is not None)
        if can_spawn(agent, 0.0, 0.0):
            return ActionRequest(
                agent_id=agent.agent_id, move_distance=0.0, move_direction=0.0,
                turn_angle=0.0, spawn_agent=True,
            )
        predators = [entity for entity in entities if entity.kind == "Predator"]
        if predators:
            predator = min(predators, key=lambda entity: entity.distance)
            return self._travel(
                agent, movement_limit(agent), predator.angle + math.pi, walls, facing=predator,
            )
        distance = min(agent.speed, movement_limit(agent))
        modifier = BIOME_MOVEMENT[agent.biome]
        fruits = [entity for entity in entities if entity.kind == "Fruit"]
        if fruits:
            fruit = min(fruits, key=lambda entity: entity.distance)
            return self._travel(agent, min(distance, fruit.distance / modifier), fruit.angle, walls)
        direction = self._rng.uniform(-math.pi, math.pi)
        trees = [entity for entity in entities if entity.kind == "Tree"]
        if trees:
            tree = min(trees, key=lambda entity: entity.distance)
            radius = distance * modifier * math.sqrt(self._rng.random())
            target_x = tree.x + radius * math.cos(direction)
            target_y = tree.y + radius * math.sin(direction)
            direction = math.atan2(target_y, target_x)
            distance = min(distance, math.hypot(target_x, target_y) / modifier)
        return self._travel(agent, distance, direction, walls)

    @staticmethod
    def _travel(
        agent: ObservationResponse, distance: float, direction: float,
        walls: tuple[Segment, ...], *, facing: Entity | None = None,
    ) -> ActionRequest:
        distance, direction = _avoid_walls(distance, direction, agent.biome, walls)
        turn = direction
        if facing is not None:
            displacement = distance * BIOME_MOVEMENT[agent.biome]
            turn = math.atan2(
                facing.y - displacement * math.sin(direction),
                facing.x - displacement * math.cos(direction),
            )
        return ActionRequest(
            agent_id=agent.agent_id, move_distance=distance, move_direction=direction,
            turn_angle=turn, spawn_agent=False,
        )