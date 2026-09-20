"""Public-observation population renewal, resource budgets, and patch coordination."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from statistics import fmean, median

from src.policies.features import Entity, validate_step
from src.policies.geometry import (
    BIOME_MOVEMENT, can_spawn, movement_cost, movement_limit, relative_heading,
    turn_cost, wrap_angle,
)
from src.policies.hierarchical import (
    AgentContext, HierarchicalPolicy, Pose, _build_contexts, _clamp_turn,
    _cluster_targets, _trait_score,
)
from src.policies.turnaway import _avoid_walls
from src.utils.DTOs import ActionRequest, ObservationResponse, StepResponse


_FRUIT_RATE = {"forest": 0.1, "grassland": 0.1, "swamp": 0.08, "desert": 0.05}
_TRACK_LIMIT = 2048


@dataclass
class Life:
    status: ObservationResponse
    pose: Pose
    frame: int
    exploration_heading: float
    uncertainty: float = 0.0
    elder: bool = False
    aging_evidence: int = 0
    income: float = 0.0
    burn: float = 1.0
    children: set[int] = field(default_factory=set)
    home: int | None = None
    stuck: int = 0


@dataclass
class Resource:
    id: int
    kind: str
    frame: int
    x: float
    y: float
    first_seen: float
    last_seen: float
    sightings: int = 1
    missing: int = 0
    exposure: float = 0.0
    income: float = 0.0
    biome: str | None = None
    productive: bool = False
    heading: float = 0.0


def _transform_pose(value: Pose, transform: Pose) -> Pose:
    x, y = transform.local_point(value.x, value.y)
    return Pose(x, y, wrap_angle(value.heading + transform.heading))


def _pose_transform(source: Pose, target: Pose) -> Pose:
    rotation = wrap_angle(target.heading - source.heading)
    cosine, sine = math.cos(rotation), math.sin(rotation)
    return Pose(
        target.x - cosine * source.x + sine * source.y,
        target.y - sine * source.x - cosine * source.y, rotation,
    )


def _observed_heading(entity: Entity) -> float:
    if entity.distance == 0.0:
        # Both zero-vector bearings use atan2(0, 0); there is no opposite-bearing pi term.
        return wrap_angle(entity.angle - (entity.rel_dir or 0.0))
    return relative_heading(entity.angle, entity.rel_dir or 0.0)


class PopulationPolicy(HierarchicalPolicy):
    stateful = True

    def reset(self) -> None:
        super().reset()
        self._rng = random.Random(self.seed)
        self.lives: dict[int, Life] = {}
        self.resources: dict[int, Resource] = {}
        self._next_frame = 0
        self._next_resource = 0
        self._time = 0.0
        self._public_time: float | None = None
        self._last_birth = -math.inf
        self._target = 1
        self._target_evidence = 0.0
        self._target_proposal = 1
        self._replacement_demand = 0
        self._intake: dict[int, float] = {}
        self._confirmed_births = 0
        self._migrations = 0
        self._evictions = 0
        self._localization_resets = 0
        self._data: dict = {}
        self._decoys: set[int] = set()
        self._stale_observers: set[int] = set()
        self._pose_uncertainties: dict[int, float] = {}

    def diagnostics(self) -> dict:
        return dict(self._data)

    def act(self, step: StepResponse) -> list[ActionRequest]:
        validate_step(step)
        return self.act_validated(step)

    def _clock(self, step: StepResponse) -> float:
        if self._public_time is not None and 0 < step.sim_time < self._public_time:
            self.reset()
        elapsed = (
            step.sim_time - self._public_time
            if self._public_time is not None and step.sim_time > self._public_time else 0.1
        )
        if elapsed > 1.0:
            raise ValueError("Population control requires observations at the native decision cadence.")
        self._public_time = step.sim_time
        self._time += elapsed
        return elapsed

    def _merge_frame(self, source: int, target: int, transform: Pose, predicted, resolved) -> None:
        for agent_id, (frame, pose, uncertainty) in list(predicted.items()):
            if frame == source:
                predicted[agent_id] = (target, _transform_pose(pose, transform), uncertainty)
        for life in self.lives.values():
            if life.frame == source:
                life.pose = _transform_pose(life.pose, transform)
                life.frame = target
                life.exploration_heading = wrap_angle(life.exploration_heading + transform.heading)
        for resource in self.resources.values():
            if resource.frame == source:
                resource.x, resource.y = transform.local_point(resource.x, resource.y)
                resource.heading = wrap_angle(resource.heading + transform.heading)
                resource.frame = target
        for agent_id, context in list(resolved.items()):
            if context.component == source:
                resolved[agent_id] = replace(
                    context, component=target, pose=_transform_pose(context.pose, transform),
                )

    def _contexts(self, step: StepResponse) -> dict[int, AgentContext]:
        canonical = step.model_copy(update={"agent_status": [
            agent.model_copy(update={"observations": []})
            if agent.agent_id in self._stale_observers else agent
            for agent in sorted(step.agent_status, key=lambda agent: agent.agent_id)
        ]})
        fresh = _build_contexts(canonical, zero_distance_aware=True)
        fresh = {
            agent_id: replace(context, predators=tuple(sorted(
                context.predators, key=lambda predator: (
                    predator.distance, predator.angle, predator.rel_dir,
                ),
            )))
            for agent_id, context in fresh.items()
        }
        predicted = {}
        for agent_id, life in self.lives.items():
            action = self.previous_actions.get(agent_id)
            pose, uncertainty = life.pose, life.uncertainty
            if action is not None:
                distance = min(max(0.0, action.move_distance), movement_limit(life.status))
                displacement = distance * BIOME_MOVEMENT[life.status.biome]
                heading = pose.heading + action.move_direction
                pose = Pose(
                    pose.x + displacement * math.cos(heading),
                    pose.y + displacement * math.sin(heading),
                    wrap_angle(pose.heading + action.turn_angle),
                )
                visible_path = (
                    abs(wrap_angle(action.move_direction)) < life.status.vision_angle / 2
                    and displacement < life.status.vision_range
                )
                uncertainty = min(150.0, uncertainty + displacement * (0.05 if visible_path else 2.0))
            predicted[agent_id] = (life.frame, pose, uncertainty)
        for agent_id, context in fresh.items():
            if agent_id in predicted:
                continue
            anchors = [
                entity for entity in context.entities
                if entity.kind == "Agent" and entity.agent_id in predicted and entity.agent_id not in fresh
            ]
            if anchors:
                anchor = min(anchors, key=lambda entity: (predicted[entity.agent_id][2], entity.agent_id))
                frame, parent_pose, uncertainty = predicted[anchor.agent_id]
                heading = wrap_angle(parent_pose.heading - _observed_heading(anchor))
                cosine, sine = math.cos(heading), math.sin(heading)
                predicted[agent_id] = (
                    frame, Pose(
                        parent_pose.x - cosine * anchor.x + sine * anchor.y,
                        parent_pose.y - sine * anchor.x - cosine * anchor.y, heading,
                    ), uncertainty,
                )
        groups: dict[int, list[int]] = defaultdict(list)
        for agent_id, context in fresh.items():
            groups[context.component].append(agent_id)
        resolved: dict[int, AgentContext] = {}
        for ids in groups.values():
            anchors = [agent_id for agent_id in ids if agent_id in predicted]
            if anchors:
                anchor = min(anchors, key=lambda agent_id: (predicted[agent_id][2], agent_id))
                frame, pose, uncertainty = predicted[anchor]
                transform = _pose_transform(fresh[anchor].pose, pose)
                for other in anchors:
                    other_frame, other_pose, _ = predicted[other]
                    if other_frame != frame:
                        target_pose = _transform_pose(fresh[other].pose, transform)
                        self._merge_frame(
                            other_frame, frame, _pose_transform(other_pose, target_pose),
                            predicted, resolved,
                        )
            else:
                frame = self._next_frame
                self._next_frame += 1
                transform = Pose(0.0, 0.0, 0.0)
                uncertainty = 0.0
            if uncertainty > min(50.0, max(10.0, fresh[ids[0]].agent.hearing_radius / 2)):
                frame = self._next_frame
                self._next_frame += 1
                self._localization_resets += 1
                for agent_id in ids:
                    if agent_id in self.lives:
                        self.lives[agent_id].exploration_heading = wrap_angle(
                            self.lives[agent_id].exploration_heading - transform.heading,
                        )
                transform, uncertainty = Pose(0.0, 0.0, 0.0), 0.0
            for agent_id in ids:
                context = fresh[agent_id]
                resolved[agent_id] = replace(
                    context, component=frame, pose=_transform_pose(context.pose, transform),
                )
                old_frame, old_pose, _ = predicted.get(agent_id, (frame, resolved[agent_id].pose, uncertainty))
                predicted[agent_id] = (old_frame, old_pose, uncertainty)
        for frame in sorted({context.component for context in resolved.values()}):
            matches = []
            for context in resolved.values():
                if context.component != frame:
                    continue
                for entity in context.entities:
                    if entity.kind != "Tree":
                        continue
                    x, y = context.pose.local_point(entity.x, entity.y)
                    candidates = sorted(
                        (math.hypot(r.x - x, r.y - y), r.id)
                        for r in self.resources.values()
                        if r.kind == "Tree" and r.frame == frame
                        and self._time - r.last_seen < 3.0 and r.sightings >= 2
                    )
                    if candidates and candidates[0][0] < 12.0 and (
                        len(candidates) == 1 or candidates[1][0] - candidates[0][0] > 8.0
                    ):
                        resource = self.resources[candidates[0][1]]
                        matches.append((resource.x - x, resource.y - y))
            if matches:
                dx, dy = median(p[0] for p in matches), median(p[1] for p in matches)
                if max(math.hypot(x - dx, y - dy) for x, y in matches) <= 4.0:
                    for agent_id, context in list(resolved.items()):
                        if context.component == frame:
                            resolved[agent_id] = replace(context, pose=Pose(
                                context.pose.x + dx, context.pose.y + dy, context.pose.heading,
                            ))
                            if agent_id in predicted:
                                old_frame, old_pose, _ = predicted[agent_id]
                                predicted[agent_id] = (old_frame, old_pose, 1.0)
        for agent_id, context in resolved.items():
            if agent_id in self.lives:
                self.lives[agent_id].uncertainty = predicted[agent_id][2]
        self._pose_uncertainties = {agent_id: predicted[agent_id][2] for agent_id in resolved}
        return resolved

    def _update_lives(self, contexts: dict[int, AgentContext], elapsed: float) -> None:
        live_ids = set(contexts)
        new_ids = live_ids - set(self.lives)
        pending = [
            agent_id for agent_id, action in self.previous_actions.items()
            if action.spawn_agent and agent_id in self.lives
            and can_spawn(self.lives[agent_id].status, action.move_distance, action.turn_angle)
        ]
        if self._tick and len(pending) == 1 and len(new_ids) == 1:
            child = next(iter(new_ids))
            if contexts[child].agent.age <= 0.100001:
                self.lives[pending[0]].children.add(child)
                self._confirmed_births += 1
        alpha = -math.expm1(-elapsed / 8.0)
        self._intake = {}
        for agent_id in sorted(contexts):
            context, old = contexts[agent_id], self.lives.get(agent_id)
            status = context.agent
            if old is None:
                self.lives[agent_id] = Life(
                    status.model_copy(update={"observations": []}), context.pose,
                    context.component, self._rng.uniform(-math.pi, math.pi),
                    uncertainty=self._pose_uncertainties[agent_id],
                    elder=status.age > 120.0,
                )
                continue
            action = self.previous_actions.get(agent_id)
            cost = (
                movement_cost(old.status, action.move_distance) + turn_cost(action.turn_angle)
                if action is not None else 0.0
            )
            birth_cost = 100.0 if agent_id in pending else 0.0
            age_delta = max(0.0, status.age - old.status.age)
            age_cost = 0.01 * status.age if age_delta > 0 else 0.0
            residual = status.energy - old.status.energy + cost + birth_cost + age_delta
            if status.age >= 60.0 and age_delta > 0 and abs(residual + age_cost) < 0.06:
                old.aging_evidence = min(3, old.aging_evidence + 1)
            else:
                old.aging_evidence = max(0, old.aging_evidence - 1)
            old.elder = old.elder or old.aging_evidence >= 2 or status.age > 120.0
            intake = max(0.0, residual + (age_cost if old.elder else 0.0))
            self._intake[agent_id] = intake
            old.income += alpha * (intake / elapsed - old.income)
            burn = (cost + age_delta + (age_cost if old.elder else 0.0)) / elapsed
            old.burn += alpha * (burn - old.burn)
            if action is not None and action.move_distance > 1.0 and old.frame == context.component:
                moved = math.hypot(context.pose.x - old.pose.x, context.pose.y - old.pose.y)
                old.stuck = old.stuck + 1 if moved < 0.2 else 0
            old.status = status.model_copy(update={"observations": []})
            old.pose, old.frame = context.pose, context.component
            old.children.intersection_update(live_ids)
        self.lives = {agent_id: self.lives[agent_id] for agent_id in sorted(live_ids)}

    def _track(self, kind, frame, x, y, *, biome=None, heading=0.0, exclude=None) -> Resource:
        radius = {"Tree": 18.0, "Fruit": 8.0, "Predator": 30.0}[kind]
        candidates = [
            (math.hypot(r.x - x, r.y - y), r.id)
            for r in self.resources.values()
            if r.kind == kind and r.frame == frame and (exclude is None or r.id not in exclude)
            and not (
                kind == "Predator" and r.last_seen == self._time
                and (math.hypot(r.x - x, r.y - y) > 1e-4 or abs(wrap_angle(r.heading - heading)) > 1e-4)
            )
        ]
        nearest = min(candidates, default=(math.inf, -1))
        if nearest[0] <= radius:
            resource = self.resources[nearest[1]]
            resource.x, resource.y = x, y
            resource.last_seen = self._time
            resource.sightings += 1
            resource.missing = 0
            resource.heading = heading
            if biome in _FRUIT_RATE:
                resource.biome = biome
        else:
            resource = Resource(
                self._next_resource, kind, frame, x, y, self._time, self._time,
                biome=biome if biome in _FRUIT_RATE else None, heading=heading,
            )
            self.resources[resource.id] = resource
            self._next_resource += 1
        return resource

    def _update_resources(self, contexts, elapsed):
        if not self.config.patch_memory:
            self.resources.clear()
        targets = _cluster_targets(contexts, fruit_contact_radius=self.config.fruit_contact_radius)
        visible = set()
        for kind in ("Tree", "Fruit"):
            for target in targets[kind]:
                observer = min(target.observers)
                context = contexts[observer]
                resource = self._track(
                    kind, target.component, target.x, target.y,
                    biome=context.agent.biome, exclude=visible,
                )
                visible.add(resource.id)
        for context in contexts.values():
            seen_predators = set()
            for predator in context.predators:
                x, y = context.pose.local_point(predator.x, predator.y)
                tracked = self._track(
                    "Predator", context.component, x, y,
                    heading=wrap_angle(context.pose.heading + _observed_heading(predator)),
                    exclude=seen_predators,
                )
                seen_predators.add(tracked.id)
        alpha = -math.expm1(-elapsed / 12.0)
        patches = [r for r in self.resources.values() if r.kind == "Tree"]
        for patch in patches:
            if patch.id in visible:
                patch.exposure += elapsed
                patch.income *= 1 - alpha
                patch.productive = patch.productive or any(
                    fruit.component == patch.frame
                    and math.hypot(fruit.x - patch.x, fruit.y - patch.y) <= 80.0
                    for fruit in targets["Fruit"]
                )
            else:
                for agent_id, context in contexts.items():
                    if context.component != patch.frame or agent_id in self._stale_observers:
                        continue
                    distance = math.hypot(context.pose.x - patch.x, context.pose.y - patch.y)
                    if distance + self.lives[agent_id].uncertainty + 2 < context.agent.hearing_radius:
                        patch.missing += 1
                        break
        for agent_id, intake in self._intake.items():
            if intake <= 0:
                continue
            context = contexts[agent_id]
            nearby = sorted(
                (math.hypot(context.pose.x - p.x, context.pose.y - p.y), p.id)
                for p in patches if p.frame == context.component and p.id in visible
            )
            if nearby and nearby[0][0] < 90.0:
                patch = self.resources[nearby[0][1]]
                patch.income += alpha * intake / elapsed
                patch.productive = True
        active_frames = {context.component for context in contexts.values()}
        expiry = {"Tree": self.config.patch_memory_seconds, "Fruit": 5.0, "Predator": 0.3}
        self.resources = {
            key: resource for key, resource in self.resources.items()
            if resource.frame in active_frames and resource.missing < 2
            and self._time - resource.last_seen <= expiry[resource.kind]
        }
        if len(self.resources) > _TRACK_LIMIT:
            keep = sorted(self.resources.values(), key=lambda r: (-r.last_seen, r.id))[:_TRACK_LIMIT]
            self._evictions += len(self.resources) - len(keep)
            self.resources = {r.id: r for r in keep}
        for agent_id, context in list(contexts.items()):
            predators = []
            for threat in self.resources.values():
                if threat.kind != "Predator" or threat.frame != context.component:
                    continue
                x, y = context.pose.world_point(threat.x, threat.y)
                bearing = math.atan2(y, x)
                predators.append(Entity(
                    "Predator", x, y, math.hypot(x, y), bearing,
                    wrap_angle(bearing + math.pi - (threat.heading - context.pose.heading)),
                ))
            contexts[agent_id] = replace(context, predators=tuple(predators))
        return targets["Fruit"]

    def _patch_supply(self, patch: Resource) -> float:
        prior = 20.0 * _FRUIT_RATE.get(patch.biome, 0.05)
        if not patch.productive:
            prior *= 0.5
        weight = patch.exposure / (patch.exposure + 20.0)
        freshness = math.exp(-(self._time - patch.last_seen) / self.config.patch_memory_seconds)
        return freshness * ((1 - weight) * prior + weight * patch.income)

    def _capacity(self, elapsed: float) -> tuple[int, float, float]:
        supply = math.fsum(
            self._patch_supply(resource) for resource in self.resources.values() if resource.kind == "Tree"
        )
        young_burn = [life.burn for life in self.lives.values() if not life.elder]
        burn = max(1.0, fmean(young_burn) if young_burn else 1.5)
        replacement_tax = 25.0 / min(60.0, self.config.breeding_age)
        estimate = supply * self.config.population_safety / (burn + replacement_tax)
        proposal = max(1, min(self.config.target_population, int(estimate)))
        if not self.config.capacity_feedback:
            proposal = self.config.target_population
        if proposal == self._target_proposal:
            self._target_evidence += elapsed
        else:
            self._target_proposal, self._target_evidence = proposal, elapsed
        gap = abs(proposal - self._target)
        if gap and self._target_evidence >= self.config.birth_spacing and (
            gap >= max(1.0, self._target * self.config.population_hysteresis)
        ):
            self._target = proposal
        renewable = sum(
            not life.elder and life.status.age < self.config.breeding_age
            and life.status.max_energy > 100.0 for life in self.lives.values()
        )
        self._replacement_demand = max(0, self._target - renewable)
        return self._target, supply, burn

    def _retired(self, life: Life) -> bool:
        return (life.elder or life.status.age >= self.config.breeding_age) and any(
            child in self.lives and self.lives[child].status.max_energy > 100.0
            for child in life.children
        )

    def _assign(self, contexts, fruits):
        assignments = {}
        pairs = []
        for agent_id, context in contexts.items():
            life = self.lives[agent_id]
            if context.predators and min(p.distance for p in context.predators) < self.config.threat_distance:
                continue
            for index, fruit in enumerate(fruits):
                if fruit.component != context.component:
                    continue
                distance = math.hypot(context.pose.x - fruit.x, context.pose.y - fruit.y)
                hunger = max(0.0, 1.0 - context.agent.energy / context.agent.max_energy)
                renewal = float(context.agent.age >= self.config.breeding_age - 10 and not life.children)
                priority = distance / (1 + 2 * hunger + renewal)
                if self._retired(life):
                    priority += 1000.0
                pairs.append((priority, agent_id, index, fruit.x, fruit.y))
        used = set()
        for _, agent_id, index, x, y in sorted(pairs):
            if agent_id not in assignments and index not in used and not self._retired(self.lives[agent_id]):
                assignments[agent_id] = ("Fruit", x, y)
                used.add(index)
        occupied: Counter = Counter()
        patches = [resource for resource in self.resources.values() if resource.kind == "Tree"]
        for agent_id in sorted(contexts, key=lambda key: (self._retired(self.lives[key]), key)):
            context, life = contexts[agent_id], self.lives[agent_id]
            if self._retired(life):
                life.home = None
                continue
            candidates = []
            for patch in patches:
                if patch.frame != context.component:
                    continue
                danger = min((
                    math.hypot(threat.x - patch.x, threat.y - patch.y)
                    for threat in self.resources.values()
                    if threat.kind == "Predator" and threat.frame == patch.frame
                ), default=math.inf)
                distance = math.hypot(context.pose.x - patch.x, context.pose.y - patch.y)
                capacity = max(1, min(self.config.patch_capacity, math.ceil(self._patch_supply(patch) / 1.5)))
                crowding = 150.0 * max(0, occupied[patch.id] - capacity + 1) if self.config.dispersion else 0.0
                priority = (
                    distance + crowding + max(0, self.config.threat_distance - danger) * 4
                    + 2 * (self._time - patch.last_seen) - 25 * (life.home == patch.id)
                )
                candidates.append((priority, patch.id))
            if not candidates:
                if life.home is not None:
                    self._migrations += 1
                life.home = None
                continue
            _, home = min(candidates)
            if life.home is not None and home != life.home:
                self._migrations += 1
            life.home = home
            occupied[home] += 1
            if agent_id not in assignments:
                patch = self.resources[home]
                angle = 2.399963229728653 * (agent_id + 1)
                assignments[agent_id] = (
                    "Tree", patch.x + self.config.patch_radius * math.cos(angle),
                    patch.y + self.config.patch_radius * math.sin(angle),
                )
        return assignments, occupied

    def _move(self, context: AgentContext, assignment) -> ActionRequest:
        agent, life = context.agent, self.lives[context.agent.agent_id]
        nearest = min(context.predators, key=lambda p: p.distance, default=None)
        if nearest is not None and nearest.distance <= self.config.threat_distance + 15.0:
            if self.config.elder_decoys and len(context.predators) == 1 and life.elder and agent.energy < 75 and self._retired(life):
                children = [
                    child for child in life.children if child in self.lives
                    and self.lives[child].frame == life.frame
                    and self.lives[child].status.energy > 20
                ]
                if children and nearest.distance > 30:
                    refuge_x = fmean(self.lives[child].pose.x for child in children)
                    refuge_y = fmean(self.lives[child].pose.y for child in children)
                    predator_x, predator_y = context.pose.local_point(nearest.x, nearest.y)
                    if all(
                        math.hypot(
                            predator_x - self.lives[child].pose.x,
                            predator_y - self.lives[child].pose.y,
                        ) > nearest.distance + 15 for child in children
                    ):
                        away = wrap_angle(math.atan2(
                            context.pose.y - refuge_y, context.pose.x - refuge_x,
                        ) - context.pose.heading)
                        distance, direction = _avoid_walls(
                            min(agent.speed, movement_limit(agent)), away, agent.biome, context.walls,
                        )
                        displacement = distance * BIOME_MOVEMENT[agent.biome]
                        separation = math.hypot(
                            nearest.x - displacement * math.cos(direction),
                            nearest.y - displacement * math.sin(direction),
                        )
                        if distance > 0 and separation > nearest.distance and abs(wrap_angle(direction - away)) < math.pi / 3:
                            self._decoys.add(agent.agent_id)
                            return ActionRequest(
                                agent_id=agent.agent_id, move_distance=distance,
                                move_direction=direction,
                                turn_angle=_clamp_turn(nearest.angle + 0.08, self.config.predator_face_turn),
                                spawn_agent=False,
                            )
            escaped = self._escape(context, nearest)
            distance, direction = _avoid_walls(
                escaped.move_distance, escaped.move_direction, agent.biome, context.walls,
            )
            if len(context.predators) > 1:
                closest = sorted(context.predators, key=lambda p: (p.distance, p.angle, p.rel_dir))[:3]
                headings = [
                    direction, direction + math.pi / 3, direction - math.pi / 3,
                    direction + math.pi / 2, direction - math.pi / 2,
                    *(p.angle + math.pi for p in closest),
                    math.atan2(
                        -sum(p.y / max(1.0, p.distance**2) for p in context.predators),
                        -sum(p.x / max(1.0, p.distance**2) for p in context.predators),
                    ),
                ]
                candidates = []
                for heading in headings:
                    travel, bearing = _avoid_walls(
                        escaped.move_distance, heading, agent.biome, context.walls,
                    )
                    dx = travel * BIOME_MOVEMENT[agent.biome] * math.cos(bearing)
                    dy = travel * BIOME_MOVEMENT[agent.biome] * math.sin(bearing)
                    clearance = min(math.hypot(p.x - dx, p.y - dy) for p in context.predators)
                    candidates.append((clearance, -abs(wrap_angle(bearing - direction)), travel, bearing))
                _, _, distance, direction = max(candidates)
            displacement = distance * BIOME_MOVEMENT[agent.biome]
            face = math.atan2(
                nearest.y - displacement * math.sin(direction),
                nearest.x - displacement * math.cos(direction),
            )
            return escaped.model_copy(update={
                "move_distance": distance, "move_direction": direction,
                "turn_angle": _clamp_turn(face + 0.08, self.config.predator_face_turn),
            })
        if agent.agent_id in self._stale_observers:
            return ActionRequest(
                agent_id=agent.agent_id, move_distance=0.0, move_direction=0.0,
                turn_angle=0.0, spawn_agent=False,
            )
        distance = direction = turn = 0.0
        if assignment is not None:
            kind, x, y = assignment
            local_x, local_y = context.pose.world_point(x, y)
            remaining = math.hypot(local_x, local_y)
            tolerance = self.config.fruit_contact_radius - 0.1 if kind == "Fruit" else self.config.patch_tolerance
            distance = min(
                agent.speed, max(0.0, remaining - tolerance) / BIOME_MOVEMENT[agent.biome],
            )
            direction = math.atan2(local_y, local_x)
            if distance > 0:
                turn = _clamp_turn(direction, self.config.max_turn)
        else:
            if life.stuck >= 3 or (self._tick + agent.agent_id * 7) % 120 == 0:
                life.exploration_heading = wrap_angle(life.exploration_heading + 2.399963229728653)
                life.stuck = 0
            direction = wrap_angle(life.exploration_heading - context.pose.heading)
            floor = 15.0 if self._retired(life) else 5.0
            if agent.energy > floor:
                distance = agent.speed * self.config.exploration_move_fraction
            turn = _clamp_turn(direction, self.config.max_turn)
        distance = min(distance, movement_limit(agent))
        distance, direction = _avoid_walls(distance, direction, agent.biome, context.walls)
        if distance <= 1e-9:
            turn = (
                math.pi / 2 if (self._tick + agent.agent_id * 7) % self.config.population_scan_ticks == 0 else 0.0
            )
        return ActionRequest(
            agent_id=agent.agent_id, move_distance=distance, move_direction=direction,
            turn_angle=turn, spawn_agent=False,
        )

    def _spawns(self, contexts, actions) -> tuple[set[int], Counter]:
        blocked: Counter = Counter()
        if len(contexts) >= self.config.max_population:
            blocked["population_safety_cap"] += 1
            return set(), blocked
        spacing_blocked = self._time - self._last_birth < self.config.birth_spacing
        expansion = max(0, self._target - len(contexts))
        eligible = [
            agent_id for agent_id, context in contexts.items()
            if can_spawn(context.agent, 0.0, 0.0) and context.agent.max_energy > 100.0
        ]
        if not eligible and (expansion or self._replacement_demand):
            reason = (
                "no_reproductive_trait_capacity"
                if all(context.agent.max_energy <= 100 for context in contexts.values())
                else "no_energy_eligible_parent"
            )
            blocked[reason] += 1
        candidates = []
        for agent_id in eligible:
            context, life = contexts[agent_id], self.lives[agent_id]
            agent = context.agent
            distance = min((p.distance for p in context.predators), default=math.inf)
            due = (life.elder or agent.age >= self.config.breeding_age) and not any(
                child in self.lives and self.lives[child].status.max_energy > 100
                for child in life.children
            )
            last_chance = len(eligible) == 1 and (
                due or agent.energy <= 100 + self.config.population_reserve + max(1.0, life.burn)
            )
            if spacing_blocked and not last_chance:
                blocked["cohort_spacing"] += 1
                continue
            if not (expansion or self._replacement_demand or last_chance):
                blocked["no_replacement_demand"] += 1
                continue
            if distance <= max(90.0, self.config.emergency_spawn_distance + 45.0):
                blocked["unsafe_newborn_geometry"] += 1
                continue
            if distance <= self.config.threat_distance and not last_chance:
                blocked["predator_alarm"] += 1
                continue
            reserve = max(self.config.population_reserve, min(50.0, life.burn * 10.0))
            reserve = min(reserve, max(0.0, (agent.max_energy - 100.0) / 2))
            if last_chance:
                reserve = min(reserve, 5.0)
            if agent.energy <= 100 + reserve:
                blocked["parent_reserve"] += 1
                continue
            local_supply = math.fsum(
                self._patch_supply(resource) for resource in self.resources.values()
                if resource.kind == "Tree" and resource.frame == context.component
            )
            if not (life.home is not None and local_supply > 0 or due or last_chance):
                blocked["unknown_food_supply"] += 1
                continue
            candidates.append((
                not last_chance, not due, -agent.age if due else 0.0,
                -_trait_score(agent), -agent.energy, agent_id,
            ))
        if not candidates:
            return set(), blocked
        agent_id = min(candidates)[-1]
        actions[agent_id] = ActionRequest(
            agent_id=agent_id, move_distance=0.0, move_direction=0.0,
            turn_angle=0.0, spawn_agent=True,
        )
        self._last_birth = self._time
        return {agent_id}, blocked

    def act_validated(self, step: StepResponse) -> list[ActionRequest]:
        if not step.agent_status or step.game_status == "game_over":
            self.reset()
            return []
        elapsed = self._clock(step)
        self._stale_observers = {
            agent.agent_id for agent in step.agent_status
            if agent.agent_id in self.lives and agent.age <= self.lives[agent.agent_id].status.age
        }
        contexts = self._contexts(step)
        self._update_lives(contexts, elapsed)
        fruits = self._update_resources(contexts, elapsed)
        target, supply, burn = self._capacity(elapsed)
        assignments, occupied = self._assign(contexts, fruits)
        self._decoys.clear()
        actions = {
            agent_id: self._move(context, assignments.get(agent_id))
            for agent_id, context in contexts.items()
        }
        _, blocked = self._spawns(contexts, actions)
        self.previous_actions = actions
        self._data = {
            "estimated_at": self._time, "target_population": target,
            "replacement_demand": self._replacement_demand,
            "estimated_income": supply, "estimated_young_burn": burn,
            "known_patches": sum(r.kind == "Tree" for r in self.resources.values()),
            "colonies": len(occupied), "migrations": self._migrations,
            "confirmed_successors": self._confirmed_births,
            "inferred_elders": sum(life.elder for life in self.lives.values()),
            "decoys": len(self._decoys),
            "stale_observers": len(self._stale_observers),
            "blocked_births": dict(blocked), "blocked_births_count": sum(blocked.values()),
            "map_evictions": self._evictions,
            "localization_resets": self._localization_resets,
            "unaligned_frames": len({context.component for context in contexts.values()}),
        }
        self._tick += 1
        return [actions[agent_id] for agent_id in sorted(actions)]
