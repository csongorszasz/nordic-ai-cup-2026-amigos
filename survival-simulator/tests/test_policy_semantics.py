import math
import random
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.benchmarking.config import SimulationSettings, Suite, make_cases
from src.benchmarking.runner import run_episode
from src.core import SimulationCore
from src.elements.agent import Agent
from src.elements.creature import Creature
from src.elements.environment import Environment
from src.elements.fruit import Fruit
from src.training.env import EnvironmentAdapter
from src.utils.DTOs import ActionRequest
from src.utils.simulation import step_environment


def agent(agent_id=0, **changes):
    values = {"x": 100.0, "y": 100.0, "energy": 150.0, "rng": random.Random(10)}
    values.update(changes)
    value = Agent(**values)
    value.agent_id = agent_id
    value.direction = 0.0
    return value


def environment(*agents, move_penalty=1.0):
    """Keep native interaction/grid methods, without constructing maps or surfaces."""
    env = Environment.__new__(Environment)
    env.width = env.height = env.chunk_size = 256
    env.rng = random.Random(1)
    env.agents = list(agents)
    env.agents_dict = {value.agent_id: value for value in agents}
    env._next_agent_id = max(env.agents_dict, default=-1) + 1
    env.fruits = []
    env.fruits_dict = {}
    env._next_fruit_id = 0
    env.trees = []
    env.obstacles = []
    env.predators = []
    env.edges = [
        ((0.0, 0.0), (256.0, 0.0)), ((0.0, 0.0), (0.0, 256.0)),
        ((256.0, 0.0), (256.0, 256.0)), ((0.0, 256.0), (256.0, 256.0)),
    ]
    env.agent_observations = {}
    env.time = env.score = 0.0
    biome = SimpleNamespace(type="grassland", move_penalty=move_penalty, energy_drain_rate=1.0)
    env.biome_map = np.full((256, 256), biome, dtype=object)
    env._update_spatial_grid()
    env.spawn_tree = Mock(return_value=None)
    env.spawn_predator = Mock(return_value=None)
    return env


def core(env, dt=0.1):
    sim = SimulationCore.__new__(SimulationCore)
    sim.env = env
    sim.dt = dt
    return sim


class MovementSemanticsTests(unittest.TestCase):
    def test_relative_movement_uses_heading_before_the_turn(self):
        value = agent()
        value.direction = math.pi / 2
        env = environment(value)
        env.agent_step(0, 10, math.pi / 2, -math.pi / 2)
        self.assertAlmostEqual(value.x, 90.0)
        self.assertAlmostEqual(value.y, 100.0)
        self.assertAlmostEqual(value.direction, 0.0)
        self.assertAlmostEqual(value.energy, 149.25)
        self.assertEqual(value.age, 0.0)

    def test_turn_cost_is_capped_but_heading_is_not_wrapped(self):
        for turn, cost in (
            (math.pi / 2, 0.25), (-math.pi, 0.5), (8 * math.pi, 0.5),
        ):
            with self.subTest(turn=turn):
                value = agent()
                value.direction = 0.3
                environment(value).update_entity_direction(value, turn)
                self.assertAlmostEqual(value.energy, 150.0 - cost)
                self.assertAlmostEqual(value.direction, 0.3 + turn)

    def test_cost_is_charged_before_terrain_reduces_displacement(self):
        value = agent()
        environment(value, move_penalty=0.5).agent_step(0, 20, 0, 0)
        self.assertAlmostEqual(value.x, 110.0)
        self.assertAlmostEqual(value.energy, 144.5)

    def test_fully_blocked_movement_still_pays_full_cost(self):
        value = agent()
        env = environment(value, move_penalty=0.5)
        obstacle = SimpleNamespace(x=50, y=50, width=100, height=100)
        env.update_entity_position(value, 20, 0, [obstacle])
        self.assertEqual((value.x, value.y), (100.0, 100.0))
        self.assertAlmostEqual(value.energy, 144.5)

    def test_low_energy_limit_is_strict_and_uses_energy_before_movement(self):
        for energy, x, remaining in ((99.0, 110.0, 98.5), (100.0, 120.0, 94.5)):
            with self.subTest(energy=energy):
                value = agent(energy=energy)
                environment(value).agent_step(0, 20, 0, 0)
                self.assertAlmostEqual(value.x, x)
                self.assertAlmostEqual(value.energy, remaining)

    def test_mutated_sprint_below_walk_caps_at_sprint_and_pays_walking_cost(self):
        for energy, remaining in ((150.0, 149.5), (99.0, 98.5)):
            with self.subTest(energy=energy):
                value = agent(energy=energy, speed=20, sprint_speed=10)
                environment(value).agent_step(0, 1000, 0, 0)
                self.assertAlmostEqual(value.x, 110.0)
                self.assertAlmostEqual(value.energy, remaining)

    def test_negative_movement_is_zero_not_backwards(self):
        value = agent()
        environment(value).agent_step(0, -20, math.pi, 0)
        self.assertEqual((value.x, value.y, value.energy), (100.0, 100.0, 150.0))


class ReproductionSemanticsTests(unittest.TestCase):
    def test_strict_threshold_is_checked_after_both_movement_and_turn_costs(self):
        for energy, distance, turn, births, remaining in (
            (100.0, 0, 0, 0, 100.0),
            (100.5, 10, 0, 0, 100.0),
            (100.25, 0, math.pi / 2, 0, 100.0),
            (100.75, 10, math.pi / 2, 0, 100.0),
            (100.76, 10, math.pi / 2, 1, 0.01),
        ):
            with self.subTest(energy=energy, distance=distance, turn=turn):
                value = agent(energy=energy)
                env = environment(value)
                env.agent_step(0, distance, 0, turn, spawn_agent=True)
                self.assertEqual(len(env.agents), 1 + births)
                self.assertAlmostEqual(value.energy, remaining)

    def test_child_starts_at_75_and_participates_in_the_birth_tick(self):
        parent = agent(speed=20, sprint_speed=10)
        env = environment(parent)
        env.rng = Mock()
        env.rng.random.return_value = 0.9
        env.rng.uniform.side_effect = lambda lower, upper: (lower + upper) / 2
        env.agent_step(0, 0, 0, 0, spawn_agent=True)
        child = env.agents[1]
        self.assertEqual(child.agent_id, 1)
        self.assertIs(env.agents_dict[1], child)
        self.assertEqual((parent.energy, child.energy, child.age), (50.0, 75.0, 0.0))
        self.assertEqual((child.speed, child.sprint_speed), (20, 10))
        self.assertAlmostEqual(child.x, 80.0)
        self.assertAlmostEqual(child.y, 100.0)
        self.assertAlmostEqual(child.direction, math.pi)
        self.assertEqual(child.max_age, 90.0)
        # Position, temporary base Agent, and final Agent each consume two uniforms.
        self.assertEqual(env.rng.uniform.call_count, 6)
        self.assertEqual(env.rng.random.call_count, 6)
        state = step_environment(env, [], dt=0.1)
        self.assertEqual([obs["agent_id"] for obs in state["observations"]], [0, 1])
        self.assertAlmostEqual(parent.energy, 49.9)
        self.assertAlmostEqual(child.energy, 74.9)
        self.assertAlmostEqual(child.age, 0.1)
        self.assertAlmostEqual(state["score"], 0.1)


class ObservationSemanticsTests(unittest.TestCase):
    def test_relative_facing_points_from_target_to_observer_not_between_headings(self):
        observer = agent()
        observer.direction = math.pi / 2
        target = agent(1, x=110.0, y=100.0)
        env = environment(observer, target)
        for direction, expected in ((math.pi, 0.0), (0.0, -math.pi)):
            with self.subTest(target_direction=direction):
                target.direction = direction
                observation = observer.observe(agents=[target], edges=env.edges)[0]
                self.assertEqual(observation["type"], "Agent")
                self.assertEqual(observation["id"], 1)
                self.assertAlmostEqual(observation["distance"], 10.0)
                self.assertAlmostEqual(observation["angle"], -math.pi / 2)
                self.assertAlmostEqual(observation["rel_dir"], expected)

    def test_distance_and_bearing_are_observer_relative(self):
        observer = agent()
        observer.direction = math.pi / 2
        distances, angles = observer.relative_distance_angle([103.0], [104.0])
        self.assertAlmostEqual(distances[0], 5.0)
        self.assertAlmostEqual(angles[0], -0.6435011087932844)

    def test_native_perception_order_is_not_distance_sorted(self):
        observer = agent()
        first = Fruit(120, 100)
        second = Fruit(110, 100)
        target = agent(1, x=105, y=100)
        observations = observer.observe(
            fruits=[first, second], agents=[target], edges=environment(observer).edges,
        )
        self.assertEqual([obs["type"] for obs in observations[:3]], ["Fruit", "Fruit", "Agent"])
        self.assertEqual([obs["distance"] for obs in observations[:3]], [20.0, 10.0, 5.0])

    def test_fruit_observation_is_cached_before_eating_but_energy_is_current(self):
        value = agent()
        env = environment(value)
        fruit = env.spawn_fruit(x=100.0, y=100.5)
        fruit.energy = 40.0
        state = step_environment(env, [], dt=0.1)
        self.assertEqual(env.fruits, [])
        self.assertAlmostEqual(state["score"], 0.14)
        status = state["observations"][0]
        self.assertAlmostEqual(status["energy"], 189.9)
        self.assertIs(status["observations"], env.agent_observations[0])
        perceived = status["observations"][0]
        self.assertEqual(perceived["type"], "Fruit")
        self.assertAlmostEqual(perceived["distance"], 0.5)
        self.assertAlmostEqual(perceived["angle"], math.pi / 2)
        self.assertFalse(any(
            obs["type"] == "Fruit"
            for obs in step_environment(env, [], dt=0.1)["observations"][0]["observations"]
        ))

    def test_predator_observation_precedes_its_native_movement(self):
        value = agent()
        env = environment(value)
        predator = Creature(130.0, 100.0, energy=150.0, rng=random.Random(8))
        predator.direction = math.pi
        predator.resting = False
        predator.step = Mock(return_value={"move": 10.0, "direction": 0.0})
        env.predators.append(predator)
        env._update_predator_grid()
        state = step_environment(env, [], dt=0.1)
        self.assertAlmostEqual(predator.x, 120.0)
        observed = next(
            obs for obs in state["observations"][0]["observations"]
            if obs["type"] == "Predator"
        )
        self.assertAlmostEqual(observed["distance"], 30.0)
        self.assertAlmostEqual(observed["rel_dir"], 0.0)

    def test_native_removal_skips_the_next_agent_in_the_live_iteration(self):
        first = agent(0, energy=0.05)
        second = agent(1, energy=0.05)
        state = step_environment(environment(first, second), [], dt=0.1)
        self.assertEqual(state["num_agents"], 1)
        status = state["observations"][0]
        self.assertEqual(status["agent_id"], 1)
        self.assertEqual((status["age"], status["energy"], status["observations"]), (0.0, 0.05, []))


class BootstrapSemanticsTests(unittest.TestCase):
    def test_reset_performs_one_empty_action_native_bootstrap(self):
        value = agent()
        env = environment(value)
        sim = core(env)
        sim.step = Mock(wraps=sim.step)
        env.agent_step = Mock(wraps=env.agent_step)
        adapter = EnvironmentAdapter(simulation_factory=lambda **kwargs: sim)
        response = adapter.reset(99)
        sim.step.assert_called_once_with([])
        env.agent_step.assert_not_called()
        self.assertEqual(response.game_status, "ok")
        self.assertAlmostEqual(response.sim_time, 0.1)
        self.assertAlmostEqual(response.score, 0.1)
        self.assertAlmostEqual(response.agent_status[0].energy, 149.9)
        self.assertAlmostEqual(response.agent_status[0].age, 0.1)

    def test_extinction_precedes_time_limit_in_the_reference_runner(self):
        env = environment(agent(energy=0.05))
        policy = SimpleNamespace(act=Mock(side_effect=AssertionError("terminal policy call")))
        case = make_cases(Suite(name="fixture", seeds=[99]))[0]
        result = run_episode(
            case, lambda seed, config: policy, {},
            settings=SimulationSettings(time_limit=0.05),
            simulation_factory=lambda **kwargs: core(env),
        )
        self.assertEqual(result.termination, "extinction")
        self.assertFalse(result.completed)
        self.assertEqual(result.ticks, 1)
        self.assertEqual(result.final_agents, 0)
        self.assertAlmostEqual(result.sim_time, 0.1)
        policy.act.assert_not_called()

    def test_action_order_is_the_callers_order_not_agent_id_order(self):
        first, second = agent(2), agent(7)
        env = environment(first, second)
        env.agent_step = Mock(wraps=env.agent_step)
        actions = [
            ActionRequest(agent_id=value, move_distance=0, move_direction=0,
                          turn_angle=0, spawn_agent=False)
            for value in (7, 2)
        ]
        step_environment(env, [(action.agent_id, action) for action in actions], dt=0.1)
        self.assertEqual([call.args[0] for call in env.agent_step.call_args_list], [7, 2])

    def test_graphics_rendering_draws_from_the_same_rng_once_per_pixel(self):
        env = Environment.__new__(Environment)
        env.width, env.height = 2, 2
        env.rng = Mock()
        palette = [(1, 2, 3), (4, 5, 6)]
        env.rng.choice.side_effect = [palette[1], palette[0], palette[0], palette[1]]
        env.biome_map = np.full((2, 2), SimpleNamespace(color_palette=palette), dtype=object)
        env.biome_surface = Mock()
        with patch("src.elements.environment.smooth_surface") as smooth:
            env._render_biome_surface()
        self.assertEqual(env.rng.choice.call_count, 4)
        self.assertEqual(
            [call.args for call in env.biome_surface.set_at.call_args_list],
            [((0, 0), palette[1]), ((0, 1), palette[0]),
             ((1, 0), palette[0]), ((1, 1), palette[1])],
        )
        smooth.assert_called_once_with(env.biome_surface, size=3)


if __name__ == "__main__":
    unittest.main()
