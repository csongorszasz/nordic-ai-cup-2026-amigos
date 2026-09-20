import math
import unittest
from types import SimpleNamespace

from src.benchmarking.config import PROJECT_ROOT, read_json
from src.policies.geometry import AGENT_RADIUS, BIOME_MOVEMENT, movement_limit
from src.policies.runtime import create_policy
from src.policies.turnaway import TurnawayPolicy, _clear_fraction
from test_policy_hierarchical import agent, frame
from test_policy_semantics import agent as native_agent, environment


def predator(distance=30.0, angle=0.0, rel_dir=0.0):
    return {"type": "Predator", "distance": distance, "angle": angle, "rel_dir": rel_dir}


class TurnawayPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = TurnawayPolicy(7)

    def test_runtime_preset_loads_the_rule_only_policy(self):
        options = read_json(PROJECT_ROOT / "configs" / "controller-turnaway-rules.json")
        policy = create_policy(7, options)
        self.assertIsInstance(policy, TurnawayPolicy)
        state = frame(agent(0, energy=50.0))
        self.assertEqual(policy.act(state), self.policy.act(state))

    def test_spawn_has_priority_over_every_movement_rule(self):
        value = agent(0, energy=math.nextafter(100.0, math.inf), age=0.0, observations=[
            predator(distance=1.0),
            {"type": "Fruit", "distance": 20.0, "angle": 0.0},
            {"type": "Tree", "distance": 30.0, "angle": 0.0},
        ])
        action = self.policy.act(frame(value))[0]
        self.assertTrue(action.spawn_agent)
        self.assertEqual((action.move_distance, action.turn_angle), (0.0, 0.0))

    def test_every_eligible_agent_spawns_without_age_or_population_gates(self):
        values = [agent(index, energy=101.0, age=0.0) for index in range(20)]
        self.assertTrue(all(action.spawn_agent for action in self.policy.act(frame(*values))))

    def test_exact_spawn_threshold_uses_next_rule(self):
        value = agent(0, energy=100.0, observations=[predator()])
        action = self.policy.act(frame(value))[0]
        self.assertFalse(action.spawn_agent)
        self.assertEqual(action.move_distance, value.sprint_speed)

    def test_any_sensed_predator_takes_priority_over_fruit_and_tree(self):
        value = agent(0, energy=100.0, observations=[
            {"type": "Fruit", "distance": 12.0, "angle": 0.0},
            {"type": "Tree", "distance": 10.0, "angle": 0.0},
            predator(distance=150.0, angle=math.pi / 2),
        ])
        action = self.policy.act(frame(value))[0]
        self.assertAlmostEqual(action.move_direction, -math.pi / 2)
        self.assertAlmostEqual(action.turn_angle, math.pi / 2)

    def test_nearest_predator_is_selected_by_bearing_not_its_heading(self):
        value = agent(0, energy=100.0, observations=[
            predator(distance=80.0, angle=math.pi),
            predator(distance=30.0, angle=0.0, rel_dir=math.pi / 2),
        ])
        action = self.policy.act(frame(value))[0]
        self.assertAlmostEqual(abs(action.move_direction), math.pi)
        self.assertAlmostEqual(action.turn_angle, 0.0)

    def test_escape_respects_energy_and_mutated_movement_limits(self):
        for energy, speed, sprint in ((50.0, 10.0, 20.0), (100.0, 20.0, 8.0)):
            with self.subTest(energy=energy, speed=speed, sprint=sprint):
                value = agent(0, energy=energy, speed=speed, sprint_speed=sprint, observations=[predator()])
                action = self.policy.act(frame(value))[0]
                self.assertEqual(action.move_distance, movement_limit(value))

    def test_nearest_fruit_beats_tree_and_stops_at_its_position(self):
        value = agent(0, energy=50.0, observations=[
            {"type": "Tree", "distance": 3.0, "angle": math.pi},
            {"type": "Fruit", "distance": 40.0, "angle": 0.0},
            {"type": "Fruit", "distance": 3.0, "angle": math.pi / 2},
        ])
        action = self.policy.act(frame(value))[0]
        self.assertAlmostEqual(action.move_distance, 3.0)
        self.assertAlmostEqual(action.move_direction, math.pi / 2)

    def test_fruit_approach_accounts_for_terrain(self):
        value = agent(0, energy=50.0, biome="river", observations=[
            {"type": "Fruit", "distance": 2.0, "angle": 0.0},
        ])
        action = self.policy.act(frame(value))[0]
        self.assertAlmostEqual(action.move_distance * BIOME_MOVEMENT[value.biome], 2.0)

    def test_tree_exploration_uses_a_walking_step_neighborhood(self):
        for biome in BIOME_MOVEMENT:
            with self.subTest(biome=biome):
                value = agent(0, energy=50.0, biome=biome, observations=[
                    {"type": "Tree", "distance": 0.0, "angle": 0.0},
                ])
                actions = [self.policy.act(frame(value))[0] for _ in range(4)]
                self.assertTrue(all(0.0 < action.move_distance <= value.speed for action in actions))
                self.assertGreater(len({action.move_direction for action in actions}), 1)

    def test_distant_tree_is_approached(self):
        value = agent(0, energy=50.0, observations=[
            {"type": "Tree", "distance": 100.0, "angle": math.pi / 2},
        ])
        action = self.policy.act(frame(value))[0]
        self.assertEqual(action.move_distance, value.speed)
        self.assertGreater(math.sin(action.move_direction), 0.0)

    def test_random_exploration_is_seeded_and_resettable(self):
        state = frame(agent(0, energy=50.0))
        actions = [self.policy.act(state)[0] for _ in range(4)]
        self.assertTrue(all(action.move_distance == 10.0 for action in actions))
        self.assertGreater(len({action.move_direction for action in actions}), 1)
        self.policy.reset()
        self.assertEqual(actions, [self.policy.act(state)[0] for _ in range(4)])

    def test_random_choices_do_not_depend_on_input_agent_order(self):
        values = [agent(index, energy=50.0) for index in (8, 2)]
        expected = {action.agent_id: action for action in self.policy.act(frame(*values))}
        self.policy.reset()
        actual = {action.agent_id: action for action in self.policy.act(frame(*reversed(values)))}
        self.assertEqual(actual, expected)

    def test_empty_team_returns_no_actions(self):
        self.assertEqual(self.policy.act(frame()), [])

    def test_invalid_observations_and_seeds_are_rejected(self):
        with self.assertRaises(ValueError):
            TurnawayPolicy(True)
        with self.assertRaises(ValueError):
            self.policy.act(frame(agent(0, observations=[{"type": "Unknown"}])))


class TurnawayWallTests(unittest.TestCase):
    def test_shared_body_radius_matches_reference_agents(self):
        self.assertEqual(AGENT_RADIUS, native_agent().size)

    def test_wall_detour_preserves_facing_from_the_new_position(self):
        value = agent(0, energy=100.0, observations=[
            predator(), {"type": "Edge", "coords": ((-8.0, -40.0), (-8.0, 40.0))},
        ])
        action = TurnawayPolicy(1).act(frame(value))[0]
        endpoint_x = action.move_distance * math.cos(action.move_direction)
        endpoint_y = action.move_distance * math.sin(action.move_direction)
        self.assertGreater(action.move_distance, 0.0)
        self.assertAlmostEqual(endpoint_x, 0.0)
        self.assertAlmostEqual(action.turn_angle, math.atan2(-endpoint_y, 30.0 - endpoint_x))
        native = native_agent(energy=value.energy)
        env = environment(native)
        obstacle = SimpleNamespace(x=80.0, y=40.0, width=12.0, height=120.0)
        env.update_entity_position(native, action.move_distance, action.move_direction, [obstacle])
        self.assertAlmostEqual(native.x, 100.0 + endpoint_x)
        self.assertAlmostEqual(native.y, 100.0 + endpoint_y)
        self.assertFalse(env._in_obstacle((native.x, native.y), native.size, [obstacle]))

    def test_boxed_agent_shortens_movement_instead_of_crossing_walls(self):
        walls = (
            ((-8.0, -8.0), (-8.0, 8.0)), ((8.0, -8.0), (8.0, 8.0)),
            ((-8.0, -8.0), (8.0, -8.0)), ((-8.0, 8.0), (8.0, 8.0)),
        )
        value = agent(0, energy=100.0, observations=[predator()] + [
            {"type": "Edge", "coords": wall} for wall in walls
        ])
        action = TurnawayPolicy(1).act(frame(value))[0]
        self.assertGreater(action.move_distance, 0.0)
        self.assertLessEqual(action.move_distance, 8.0 - AGENT_RADIUS)
        displacement = (
            action.move_distance * math.cos(action.move_direction),
            action.move_distance * math.sin(action.move_direction),
        )
        self.assertEqual(_clear_fraction(displacement, walls), 1.0)

    def test_rotated_wall_observations_match_native_collision_geometry(self):
        obstacle = SimpleNamespace(x=80.0, y=40.0, width=12.0, height=120.0)
        for heading in (math.pi / 3, -math.pi / 4, math.pi):
            with self.subTest(heading=heading):
                cosine, sine = math.cos(heading), math.sin(heading)
                wall = tuple(
                    (cosine * world_x + sine * world_y, -sine * world_x + cosine * world_y)
                    for world_x, world_y in ((-8.0, -60.0), (-8.0, 60.0))
                )
                value = agent(0, energy=100.0, observations=[
                    predator(angle=-heading), {"type": "Edge", "coords": wall},
                ])
                action = TurnawayPolicy(1).act(frame(value))[0]
                native = native_agent(energy=value.energy)
                native.direction = heading
                env = environment(native)
                env.update_entity_position(native, action.move_distance, action.move_direction, [obstacle])
                self.assertGreater(action.move_distance, 0.0)
                self.assertAlmostEqual(
                    native.x, 100.0 + action.move_distance * math.cos(heading + action.move_direction),
                )
                self.assertAlmostEqual(
                    native.y, 100.0 + action.move_distance * math.sin(heading + action.move_direction),
                )
                self.assertFalse(env._in_obstacle((native.x, native.y), native.size, [obstacle]))

    def test_square_corner_clearance_matches_engine_not_just_a_circle(self):
        wall = (((-10.0, -10.0), (-10.0, 0.0)),)
        self.assertLess(_clear_fraction((-6.0, 4.0), wall), 1.0)
        self.assertEqual(_clear_fraction((0.0, 10.0), wall), 1.0)

    def test_swept_check_prevents_tunneling_through_thin_walls(self):
        wall = (((-8.0, -40.0), (-8.0, 40.0)),)
        self.assertLess(_clear_fraction((-30.0, 0.0), wall), 1.0)

    def test_parallel_touching_and_degenerate_walls_are_handled(self):
        wall = (((-AGENT_RADIUS, -40.0), (-AGENT_RADIUS, 40.0)),)
        self.assertEqual(_clear_fraction((0.0, 10.0), wall), 1.0)
        self.assertEqual(_clear_fraction((10.0, 0.0), wall), 1.0)
        self.assertEqual(_clear_fraction((-10.0, 0.0), wall), 0.0)
        self.assertLess(_clear_fraction((-10.0, 0.0), (((-10.0, 0.0), (-10.0, 0.0)),)), 1.0)

    def test_swamp_displacement_can_be_clear_when_full_speed_would_collide(self):
        value = agent(0, energy=100.0, biome="swamp", observations=[
            predator(), {"type": "Edge", "coords": ((-16.0, -40.0), (-16.0, 40.0))},
        ])
        action = TurnawayPolicy(1).act(frame(value))[0]
        self.assertEqual(action.move_distance, value.sprint_speed)
        self.assertAlmostEqual(abs(action.move_direction), math.pi)

    def test_fruit_approach_also_avoids_walls(self):
        value = agent(0, energy=50.0, observations=[
            {"type": "Fruit", "distance": 20.0, "angle": 0.0},
            {"type": "Edge", "coords": ((8.0, -40.0), (8.0, 40.0))},
        ])
        action = TurnawayPolicy(1).act(frame(value))[0]
        self.assertNotAlmostEqual(action.move_direction, 0.0)


if __name__ == "__main__":
    unittest.main()