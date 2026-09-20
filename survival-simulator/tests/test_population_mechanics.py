import math
import random
import unittest
from unittest.mock import Mock

from src.elements.biome import (
    Desert_biome, Forest_biome, Grassland_biome, River_biome, Swamp_biome,
)
from src.elements.predator import Predator
from src.elements.tree import Tree
from test_policy_semantics import agent, environment


class PopulationMechanicsTests(unittest.TestCase):
    def test_senescence_is_a_per_update_surcharge_not_an_age_death(self):
        value = agent(max_age=60.0)
        value.age = 60.0
        env = environment(value)
        env.non_agent_step(0.1)
        self.assertIn(value, env.agents)
        self.assertAlmostEqual(value.energy, 150.0 - 0.1 - 0.601)
        env.non_agent_step(0.1)
        self.assertAlmostEqual(value.energy, 150.0 - 0.2 - 0.601 - 0.602)

    def test_food_can_rescue_energy_made_negative_by_senescence(self):
        value = agent(energy=0.2, max_age=60.0)
        value.age = 100.0
        env = environment(value)
        env.spawn_fruit(x=value.x, y=value.y)
        env.non_agent_step(0.1)
        self.assertEqual(len(env.agents), 1)
        self.assertAlmostEqual(value.energy, 0.2 - 0.1 - 1.001 + 20.0)

    def test_birth_loses_25_total_energy_before_living_costs(self):
        value = agent(energy=150.0)
        env = environment(value)
        env.agent_step(value.agent_id, 0.0, 0.0, 0.0, spawn_agent=True)
        self.assertEqual(len(env.agents), 2)
        self.assertAlmostEqual(sum(a.energy for a in env.agents), 125.0)

    def test_biomes_change_travel_and_production_not_passive_drain(self):
        for cls, movement, fruit_rate in (
            (Forest_biome, 1.0, 0.1), (Grassland_biome, 1.0, 0.1),
            (Swamp_biome, 0.5, 0.08), (Desert_biome, 0.8, 0.05),
            (River_biome, 0.3, 0.0),
        ):
            with self.subTest(biome=cls.__name__):
                biome = cls()
                self.assertEqual(biome.energy_drain_rate, 1.0)
                self.assertEqual(biome.move_penalty, movement)
                self.assertEqual(biome.fruit_spawn_rate, fruit_rate)

    def test_fruit_energy_and_rot_age_use_different_simulated_time_units(self):
        env = environment()
        fruit = env.spawn_fruit(x=100, y=100)
        self.assertEqual(fruit.energy, 20.0)
        for _ in range(200):
            fruit.grow(0.2)
        self.assertAlmostEqual(fruit.energy, 60.0)
        self.assertAlmostEqual(fruit.age, 40.0)
        fruit.age = 100.0
        env.non_agent_step(0.1)
        self.assertIn(fruit, env.fruits)
        env.non_agent_step(0.1)
        self.assertNotIn(fruit, env.fruits)

    def test_tree_replenishment_halves_after_300_simulated_seconds(self):
        for time, expected in ((3000.0, 1), (3300.0, 0)):
            with self.subTest(time=time):
                env = environment()
                env.time = time
                env.trees = [Tree(100, 100) for _ in range(20)]
                env.rng = Mock()
                env.rng.random.return_value = 0.0007
                env.non_agent_step(0.1)
                self.assertEqual(env.spawn_tree.call_count, expected)

    def test_mature_trees_are_not_permanent_resources(self):
        env = environment()
        tree = Tree(100, 100)
        tree.age = 101.0
        env.trees = [tree]
        env._update_tree_grid()
        env.non_agent_step(0.1)
        self.assertNotIn(tree, env.trees)

    def test_facing_does_not_force_circling_at_close_range_or_exact_alignment(self):
        predator = Predator(0.0, 0.0, rng=random.Random(1))
        for distance, facing, expected in (
            (89.0, 0.1, 0.0), (100.0, 0.0, 0.0),
            (100.0, 0.1, -math.pi / 4),
        ):
            with self.subTest(distance=distance, facing=facing):
                signals = predator.step([{
                    "type": "Agent", "distance": distance, "angle": 0.0,
                    "rel_dir": facing,
                }])
                self.assertAlmostEqual(signals["direction"], expected)
        self.assertLess(agent().speed, predator.sprint_speed / math.sqrt(2))

    def test_predators_rest_and_restore_energy_instead_of_dying(self):
        env = environment()
        predator = Predator(150.0, 150.0, energy=99.0, rng=random.Random(1))
        predator.step = Mock(return_value={})
        env.predators = [predator]
        env._update_predator_grid()
        env.non_agent_step(0.1)
        self.assertTrue(predator.resting)
        self.assertEqual(predator.energy, 102.0)
        env.non_agent_step(0.1)
        self.assertFalse(predator.resting)
        self.assertIn(predator, env.predators)

    def test_one_predator_can_kill_multiple_agents_in_one_tick(self):
        env = environment(agent(0), agent(1))
        predator = Predator(100.0, 100.0, energy=150.0, rng=random.Random(1))
        predator.resting = False
        predator.step = Mock(return_value={})
        env.predators = [predator]
        env._update_predator_grid()
        env.non_agent_step(0.1)
        self.assertEqual(env.agents, [])


if __name__ == "__main__":
    unittest.main()
