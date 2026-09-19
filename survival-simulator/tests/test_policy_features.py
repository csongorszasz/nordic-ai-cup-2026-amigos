import math
import unittest

import numpy as np

from src.policies.features import ENTITY_DIM, SCALAR_DIM, encode_step, parse_entities
from src.policies.geometry import (
    can_spawn, closest_point, movement_cost, movement_limit, relative_heading,
    segment_distance, turn_cost,
)
from src.utils.DTOs import ActionRequest, ObservationResponse, StepResponse


def agent(agent_id=0, observations=None, **changes):
    values = {
        "agent_id": agent_id, "energy": 150.0, "age": 1.0, "biome": "forest",
        "speed": 10.0, "sprint_speed": 20.0, "hearing_radius": 50.0,
        "vision_range": 200.0, "vision_angle": math.pi / 3, "max_energy": 500.0,
        "observations": observations or [],
    }
    values.update(changes)
    return ObservationResponse(**values)


def frame(agents):
    return StepResponse(
        game_status="ok", score=1.0, sim_time=1.0, n_agents=len(agents), agent_status=agents,
    )


class FeatureTests(unittest.TestCase):
    def test_bootstrap_announces_a_configurable_initial_population(self):
        for count in (0, 1, 5, 12):
            step = StepResponse(
                game_status="ok", score=0, sim_time=0, n_agents=count, agent_status=[],
            )
            self.assertEqual(encode_step(step).agent_ids, ())
        with self.assertRaises(ValueError):
            encode_step(StepResponse(
                game_status="ok", score=0, sim_time=0, n_agents=-1, agent_status=[],
            ))

    def test_empty_and_ragged_shapes(self):
        empty = encode_step(frame([]))
        self.assertEqual(empty.scalars.shape, (0, SCALAR_DIM))
        self.assertEqual(empty.entities.shape, (0, ENTITY_DIM))
        value = encode_step(frame([agent(), agent(7, [
            {"type": "Fruit", "distance": 10, "angle": 0},
        ])]))
        self.assertEqual(value.agent_ids, (0, 7))
        self.assertEqual(value.scalars.shape, (2, SCALAR_DIM))
        np.testing.assert_array_equal(value.entity_owners, [1])
        self.assertEqual(value.entities.shape, (1, ENTITY_DIM))

    def test_permutation_and_duplicate_edge_normalization(self):
        observations = [
            {"type": "Fruit", "distance": 10, "angle": 0.2},
            {"type": "Tree", "distance": 30, "angle": -0.2},
            {"type": "Predator", "distance": 80, "angle": 1, "rel_dir": 0.5},
            {"type": "Edge", "coords": [[20, -20], [20, 20]]},
            {"type": "Edge", "coords": [[20, 20], [20, -20]]},
        ]
        first = encode_step(frame([agent(observations=observations)]))
        second = encode_step(frame([agent(observations=list(reversed(observations)))]))
        self.assertEqual(first.entities.shape[0], 4)
        np.testing.assert_array_equal(first.entities, second.entities)
        np.testing.assert_array_equal(first.entity_types, second.entity_types)

    def test_actual_previous_action_is_encoded_by_identity(self):
        action = ActionRequest(agent_id=7, move_distance=20, move_direction=math.pi / 2,
                               turn_angle=math.pi, spawn_agent=True)
        features = encode_step(frame([agent(7), agent(8)]), {7: action})
        np.testing.assert_allclose(features.scalars[0, -6:], [0.5, 1, 0, 0, -1, 1], atol=1e-7)
        np.testing.assert_allclose(features.scalars[1, -6:], [0, 0, 1, 0, 1, 0])

    def test_invalid_data_is_not_silently_discarded(self):
        for observation in (
            {"type": "Unknown"}, {"type": "Fruit", "distance": -1, "angle": 0},
            {"type": "Fruit", "distance": 10, "angle": float("nan")},
            {"type": "Edge", "coords": [[1, 2]]},
            {"type": "Edge", "coords": [[True, 2], [3, 4]]},
            {"type": "Agent", "distance": 10, "angle": 0, "rel_dir": 0},
        ):
            with self.subTest(observation=observation), self.assertRaises(ValueError):
                parse_entities(agent(observations=[observation]))
        for agents in ([agent(0), agent(0)], [agent(biome="moon")],
                       [agent(energy=float("inf"))], [agent(max_energy=0)]):
            with self.subTest(agents=agents), self.assertRaises(ValueError):
                encode_step(frame(agents))


class GeometryTests(unittest.TestCase):
    def test_costs_match_reference_conventions(self):
        regular = agent()
        self.assertAlmostEqual(movement_cost(regular, 12), 1.5)
        self.assertAlmostEqual(movement_cost(regular, 100), 5.5)
        self.assertAlmostEqual(turn_cost(math.pi), 0.5)
        self.assertAlmostEqual(turn_cost(10 * math.pi), 0.5)
        self.assertEqual(movement_limit(agent(energy=99)), 10)
        self.assertEqual(movement_limit(agent(speed=20, sprint_speed=10)), 10)
        self.assertAlmostEqual(movement_cost(agent(energy=99), 20), 0.5)
        self.assertFalse(can_spawn(agent(energy=100.5), 10, 0))
        self.assertTrue(can_spawn(agent(energy=100.6), 10, 0))

    def test_relative_facing_is_not_a_heading_difference(self):
        self.assertAlmostEqual(relative_heading(0, 0), -math.pi)
        self.assertAlmostEqual(relative_heading(math.pi / 2, math.pi / 2), -math.pi)

    def test_segment_geometry_handles_crossing_degenerate_and_parallel_edges(self):
        self.assertEqual(closest_point((0, 0), ((2, 1), (2, 1))), (2, 1))
        self.assertEqual(segment_distance(((0, 0), (10, 0)), ((5, -1), (5, 1))), 0)
        self.assertEqual(segment_distance(((0, 0), (10, 0)), ((0, 2), (10, 2))), 2)
        self.assertEqual(segment_distance(((0, 0), (10, 0)), ((5, 0), (7, 0))), 0)


if __name__ == "__main__":
    unittest.main()
