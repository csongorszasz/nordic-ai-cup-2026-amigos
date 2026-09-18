"""Body-frame geometry and reference-engine movement costs."""

import math

from src.utils.DTOs import ObservationResponse


Point = tuple[float, float]
Segment = tuple[Point, Point]
AGENT_RADIUS = 5.0
BIOME_MOVEMENT = {"forest": 1.0, "grassland": 1.0, "swamp": 0.5, "desert": 0.8, "river": 0.3}


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % math.tau - math.pi


def closest_point(point: Point, segment: Segment) -> Point:
    (sx, sy), (ex, ey) = segment
    dx, dy = ex - sx, ey - sy
    length_squared = dx * dx + dy * dy
    t = 0.0 if length_squared == 0 else max(
        0.0, min(1.0, ((point[0] - sx) * dx + (point[1] - sy) * dy) / length_squared),
    )
    return sx + t * dx, sy + t * dy


def point_segment_distance(point: Point, segment: Segment) -> float:
    x, y = closest_point(point, segment)
    return math.hypot(point[0] - x, point[1] - y)


def segment_distance(first: Segment, second: Segment) -> float:
    def cross(a: Point, b: Point, c: Point) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    a, b = first
    c, d = second
    if cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0:
        return 0.0
    return min(point_segment_distance(a, second), point_segment_distance(b, second),
               point_segment_distance(c, first), point_segment_distance(d, first))


def movement_limit(agent: ObservationResponse) -> float:
    limit = max(0.0, agent.sprint_speed)
    if agent.energy < agent.max_energy / 5:
        limit = min(limit, max(0.0, agent.speed))
    return limit


def movement_cost(agent: ObservationResponse, distance: float) -> float:
    distance = max(0.0, min(distance, max(0.0, agent.sprint_speed)))
    if agent.energy < agent.max_energy / 5 and distance > agent.speed:
        distance = agent.speed
    return min(distance, agent.speed) * 0.05 + max(0.0, distance - agent.speed) * 0.5


def turn_cost(angle: float) -> float:
    return min(math.pi, abs(angle)) / math.tau


def can_spawn(agent: ObservationResponse, distance: float, angle: float) -> bool:
    return agent.energy - movement_cost(agent, distance) - turn_cost(angle) > 100.0


def relative_heading(bearing: float, rel_dir: float) -> float:
    """Convert the engine's object-to-observer bearing into an object heading."""
    return wrap_angle(bearing + math.pi - rel_dir)
