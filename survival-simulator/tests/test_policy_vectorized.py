import math
import random
import statistics
import unittest
from time import perf_counter
from unittest.mock import patch

import numpy as np

from src.policies.config import HeuristicConfig
from src.policies.features import parse_entities
from src.policies.geometry import BIOME_MOVEMENT, point_segment_distance
from src.policies.heuristic import (
    AGENT_RADIUS, MAX_EXPLANATIONS, HeuristicPolicy, ScoreTerms, _EPSILON,
    assess_walls, candidate_walls, create_policy, energy_expenditure, exploration_score,
    food_score, threat_score,
)
from src.policies.vectorized import point_segment_distances, score_candidates, wall_scores
from tests.test_policy_heuristic import agent, edge, frame, fruit, predator


def synthetic_step(population, *, crowded=True, seed=20260917, wall_count=24):
    """Fixed observation-only scene, also used by the explicitly invoked timing probe."""
    rng = random.Random(seed)
    agents = []
    for agent_id in range(population):
        observations = []
        if crowded:
            for _ in range(wall_count):
                angle, radius, half_length = rng.uniform(-math.pi, math.pi), rng.uniform(18, 34), rng.uniform(3, 14)
                x, y = radius * math.cos(angle), radius * math.sin(angle)
                dx, dy = -half_length * math.sin(angle), half_length * math.cos(angle)
                observations.append({"type": "Edge", "coords": [[x - dx, y - dy], [x + dx, y + dy]]})
            observations += [fruit(rng.uniform(8, 120), rng.uniform(-math.pi, math.pi)) for _ in range(12)]
            observations += [predator(rng.uniform(25, 100), rng.uniform(-math.pi, math.pi)) for _ in range(3)]
            observations += [
                {"type": "Tree", "distance": rng.uniform(20, 150), "angle": rng.uniform(-math.pi, math.pi)}
                for _ in range(4)
            ]
            observations += [
                {"type": "Agent", "id": (agent_id + offset + 1) % population,
                 "distance": rng.uniform(5, 100), "angle": rng.uniform(-math.pi, math.pi),
                 "rel_dir": rng.uniform(-math.pi, math.pi)}
                for offset in range(min(8, population - 1))
            ]
        agents.append(agent(
            agent_id, observations, energy=rng.uniform(125, 450), age=rng.uniform(1, 110),
        ))
    return frame(*agents)


def benchmark_backends(*, repeats=5, warmups=2):
    """Opt-in fixed-scene timings; never a hardware-dependent unittest assertion."""
    if repeats < 1 or warmups < 0:
        raise ValueError("Benchmark needs positive repeats and nonnegative warmups.")
    results = []
    for crowded in (False, True):
        for population in (5, 50, 100):
            step = synthetic_step(population, crowded=crowded)
            policies = {
                backend: HeuristicPolicy(123, HeuristicConfig(backend=backend))
                for backend in ("scalar", "vectorized")
            }
            for _ in range(warmups):
                for policy in policies.values():
                    policy.act(step)
            samples = {backend: [] for backend in policies}
            for repeat in range(repeats):
                order = ("scalar", "vectorized") if repeat % 2 == 0 else ("vectorized", "scalar")
                for backend in order:
                    start = perf_counter()
                    policies[backend].act(step)
                    samples[backend].append((perf_counter() - start) * 1000.0)
            scalar = policies["scalar"].explain(step, limit=MAX_EXPLANATIONS)
            vector = policies["vectorized"].explain(step, limit=MAX_EXPLANATIONS)
            pairs = list(zip(scalar, vector, strict=True))
            medians = {backend: statistics.median(values) for backend, values in samples.items()}
            results.append({
                "scene": "wall-heavy" if crowded else "empty-observations",
                "agents": population, "seed": 20260917, "repeats": repeats, "warmups": warmups,
                "scalar_median_ms": medians["scalar"], "vectorized_median_ms": medians["vectorized"],
                "speedup": medians["scalar"] / medians["vectorized"],
                "scalar_range_ms": [min(samples["scalar"]), max(samples["scalar"])],
                "vectorized_range_ms": [min(samples["vectorized"]), max(samples["vectorized"])],
                "decision_mismatches": sum(left != right for left, right in pairs),
                "max_score_difference": max(abs(left.score - right.score) for left, right in pairs),
                "max_action_difference": max(
                    abs(getattr(left.action, field) - getattr(right.action, field))
                    for left, right in pairs for field in ("move_distance", "move_direction", "turn_angle")
                ),
            })
    return results


class VectorGeometryTests(unittest.TestCase):
    def test_pairwise_point_segment_distances_include_degenerate_segments(self):
        points = np.asarray([[0, 0], [1, 1], [-4, 3], [20, 0]], dtype=np.float64)
        segments = np.asarray([
            [[0, 0], [0, 0]], [[5, -10], [5, 10]], [[-10, 0], [10, 0]],
            [[2, 3], [-7, 13]],
        ], dtype=np.float64)
        expected = [
            [point_segment_distance(tuple(point.tolist()), tuple(map(tuple, segment.tolist())))
             for segment in segments]
            for point in points
        ]
        result = point_segment_distances(points, segments)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result, expected, rtol=2e-15, atol=2e-15)
        self.assertEqual(point_segment_distances(points, np.empty((0, 2, 2))).shape, (4, 0))
        self.assertEqual(point_segment_distances(np.empty((0, 2)), segments).shape, (0, 4))

    def test_crossing_collinear_overlapping_and_zero_length_walls(self):
        endpoints = np.asarray([
            [0, 0], [20, 0], [-20, 0], [0, 20], [20, 20], [1e-10, 0],
            [10, -10], [-4, -4],
        ], dtype=np.float64)
        for walls in (
            (), (((10.0, -20.0), (10.0, 20.0)),), (((4.0, -20.0), (4.0, 20.0)),),
            (((0.0, 0.0), (0.0, 0.0)),), (((-10.0, 0.0), (10.0, 0.0)),),
            (((8.0, 2.0), (8.0, 2.0)), ((-3.0, -3.0), (30.0, 30.0))),
        ):
            with self.subTest(walls=walls):
                actual = wall_scores(endpoints, walls, 12.0)
                expected = [assess_walls(tuple(point.tolist()), walls, 12.0) for point in endpoints]
                np.testing.assert_array_equal(actual.blocked, [result.blocked for result in expected])
                np.testing.assert_allclose(actual.scores, [result.score for result in expected], atol=2e-14)
                if walls:
                    np.testing.assert_allclose(actual.swept_clearance,
                                               [result.swept_clearance for result in expected], atol=2e-14)
                    np.testing.assert_allclose(actual.endpoint_clearance,
                                               [result.endpoint_clearance for result in expected], atol=2e-14)
                else:
                    self.assertIsNone(actual.swept_clearance)

    def test_random_swept_wall_scores_match_the_oracle(self):
        rng = random.Random(77)
        walls = tuple(
            ((rng.uniform(-50, 50), rng.uniform(-50, 50)),
             (rng.uniform(-50, 50), rng.uniform(-50, 50)))
            for _ in range(30)
        )
        endpoints = np.asarray([(rng.uniform(-20, 20), rng.uniform(-20, 20)) for _ in range(80)])
        actual = wall_scores(endpoints, walls, 12.0)
        expected = [assess_walls(tuple(point.tolist()), walls, 12.0) for point in endpoints]
        np.testing.assert_array_equal(actual.blocked, [result.blocked for result in expected])
        np.testing.assert_allclose(actual.scores, [result.score for result in expected], rtol=1e-13, atol=1e-13)

    def test_raw_candidate_components_match_scalar_computation(self):
        candidates = [(0.0, 0.0)] + [
            (distance, math.tau * index / 16 - math.pi)
            for distance in (2.0, 5.0, 10.0, 15.0, 20.0) for index in range(16)
        ]
        observations = [fruit(26, 0.43), fruit(33, -1.2), predator(45, -0.4), predator(85, 1.4),
                        edge(-24), edge(30), {"type": "Edge", "coords": [[0, 18], [30, 18]]}]
        for biome in BIOME_MOVEMENT:
            with self.subTest(biome=biome):
                value = agent(observations=observations, biome=biome)
                entities = parse_entities(value)
                walls = tuple(entity.segment for entity in entities if entity.segment is not None)
                fruits = tuple(entity for entity in entities if entity.kind == "Fruit")
                predators = tuple(entity for entity in entities if entity.kind == "Predator")
                config = HeuristicConfig()
                policy = HeuristicPolicy(1, config)
                actual = score_candidates(
                    value, config, candidates, walls, fruits, predators,
                    preferred=0.1, scan=0.07, tree_target=False, walking=10.0,
                )
                self.assertEqual(actual.terms.dtype, np.float64)
                self.assertEqual(actual.terms.shape, (len(candidates), 5))
                for index, (distance, direction) in enumerate(candidates):
                    point = tuple(actual.endpoints[index].tolist())
                    wall = assess_walls(point, walls, config.wall_margin)
                    self.assertEqual(actual.blocked[index], wall.blocked)
                    if wall.blocked or actual.uncertain[index]:
                        continue
                    turn = policy._turn(value, point, distance, direction, 0.1, 0.07, False, fruits, predators)
                    expected = ScoreTerms(
                        food_score(value, point, fruits), -energy_expenditure(value, distance, turn),
                        wall.score, threat_score(value, point, turn, predators, config),
                        0.0 if fruits else exploration_score(point, 0.1, 10 * BIOME_MOVEMENT[biome]),
                    )
                    np.testing.assert_allclose(
                        actual.terms[index], list(vars(expected).values()), rtol=2e-13, atol=2e-13,
                    )
                    self.assertAlmostEqual(actual.turns[index], turn, delta=2e-14)
                    self.assertAlmostEqual(actual.scores[index], expected.weighted(config), delta=2e-11)

    def test_empty_candidate_matrix_and_all_blocked_rows(self):
        result = score_candidates(
            agent(), HeuristicConfig(), [], (), (), (),
            preferred=0.0, scan=0.0, tree_target=False, walking=10.0,
        )
        self.assertEqual(result.terms.shape, (0, 5))
        self.assertEqual(result.contenders().shape, (0,))
        result = score_candidates(
            agent(), HeuristicConfig(), [(20.0, 0.0)], (((10.0, -20.0), (10.0, 20.0)),), (), (),
            preferred=0.0, scan=0.0, tree_target=False, walking=10.0,
        )
        self.assertFalse(result.contenders().any())


class VectorPolicyTests(unittest.TestCase):
    def assert_parity(self, step, config=None, seed=123):
        config = config or HeuristicConfig()
        scalar = HeuristicPolicy(seed, config.model_copy(update={"backend": "scalar"}))
        vector = HeuristicPolicy(seed, config.model_copy(update={"backend": "vectorized"}))
        expected = scalar.explain(step, limit=MAX_EXPLANATIONS)
        actual = vector.explain(step, limit=MAX_EXPLANATIONS)
        self.assertEqual(actual, expected)
        self.assertEqual([decision.action.agent_id for decision in actual],
                         [value.agent_id for value in step.agent_status])
        return actual

    def test_empty_and_crowded_five_fifty_and_hundred_agent_batches(self):
        for population in (0, 5, 50, 100):
            for crowded in (False, True):
                with self.subTest(population=population, crowded=crowded):
                    self.assert_parity(synthetic_step(population, crowded=crowded))

    def test_randomized_observations_traits_and_configuration(self):
        rng = random.Random(431)
        for index in range(60):
            step = synthetic_step(1, seed=index, wall_count=rng.randrange(25))
            value = step.agent_status[0]
            value.agent_id = index
            value.energy = rng.choice([0.001, 99.999, 100, 150, 400])
            value.speed, value.sprint_speed = rng.uniform(0, 20), rng.uniform(0, 40)
            value.biome = rng.choice(list(BIOME_MOVEMENT))
            value.age = rng.uniform(0, 160)
            if index % 4 == 0:
                value.observations = [item for item in value.observations if item["type"] != "Fruit"]
            if index % 3 == 0:
                value.observations = [item for item in value.observations if item["type"] != "Predator"]
            config = HeuristicConfig(
                directions=rng.choice([8, 16, 32, 64]), max_turn=rng.choice([0.001, 0.45, math.pi]),
                scan_turn=rng.uniform(0, math.pi), wall_margin=rng.uniform(0.5, 30),
                food_weight=rng.uniform(0, 5), danger_weight=rng.uniform(0, 9),
                energy_weight=rng.uniform(0, 1), wall_weight=rng.uniform(0, 8),
                exploration_weight=rng.uniform(0, 1), threat_distance=rng.uniform(10, 150),
            )
            with self.subTest(index=index):
                self.assert_parity(step, config, seed=index - 20)

    def test_zero_travel_mutated_caps_and_low_energy(self):
        for energy in (0, 0.001, 99.999, 100.0, 450.0):
            for speed, sprint in ((0.0, 0.0), (0.0, 8.0), (20.0, 7.0), (10.0, 20.0)):
                with self.subTest(energy=energy, speed=speed, sprint=sprint):
                    self.assert_parity(frame(agent(
                        energy=energy, speed=speed, sprint_speed=sprint,
                        observations=[fruit(6), predator(30, 0.2), edge(-15)],
                    )))

    def test_wall_recovery_and_exact_clearance_boundaries(self):
        for separation in (
            0.0, 4.0, AGENT_RADIUS, AGENT_RADIUS + _EPSILON,
            np.nextafter(AGENT_RADIUS + _EPSILON, math.inf),
            np.nextafter(AGENT_RADIUS + _EPSILON, -math.inf),
        ):
            observations = [edge(float(separation)), fruit(40, 0.4), predator(40, -0.4)]
            with self.subTest(separation=separation):
                self.assert_parity(frame(agent(observations=observations)))
        for coords in (
            [[0, 0], [0, 0]], [[-10, 0], [10, 0]], [[-4, -4], [30, 30]],
            [[10, -20], [10, 20]],
        ):
            with self.subTest(coords=coords):
                self.assert_parity(frame(agent(observations=[
                    fruit(30), {"type": "Edge", "coords": coords},
                ])))

    def test_tied_targets_duplicate_entities_and_zero_weight_scores(self):
        for observations in (
            [fruit(20, -0.5), fruit(20, 0.5)],
            [predator(40, -0.5), predator(40, 0.5)],
            [fruit(20), fruit(20), predator(40, math.pi), predator(40, math.pi)],
            [predator(115), fruit(4 + _EPSILON, math.pi), fruit(4 + _EPSILON)],
        ):
            with self.subTest(observations=observations):
                self.assert_parity(frame(agent(observations=observations)))
                self.assert_parity(frame(agent(observations=observations)), HeuristicConfig(
                    food_weight=0, danger_weight=0, energy_weight=0, wall_weight=0, exploration_weight=0,
                ))

    def test_breeding_and_post_movement_thresholds_are_unchanged(self):
        patch_food = [fruit(35), fruit(45)]
        decisions = self.assert_parity(frame(
            agent(0, patch_food, energy=240.2), agent(1, patch_food, energy=240.8),
            agent(2, age=65, energy=250), agent(3, [fruit(25)], age=100, energy=166),
            agent(4, patch_food + [predator(50)], age=65, energy=400),
        ))
        self.assertEqual([decision.action.spawn_agent for decision in decisions], [False, True, True, False, False])

    def test_parent_wall_broad_phase_is_used_and_far_food_routes_stay_checked(self):
        observations = [fruit(200), edge(100), edge(25), edge(-500)]
        step = frame(agent(observations=observations))
        with patch("src.policies.vectorized.score_candidates", wraps=score_candidates) as scoring:
            self.assert_parity(step)
        entities = parse_entities(step.agent_status[0])
        walls = tuple(entity.segment for entity in entities if entity.segment is not None)
        self.assertEqual(scoring.call_count, 1)
        self.assertEqual(scoring.call_args.args[3], candidate_walls(walls, 20.0, 12.0))
        self.assertEqual(len(scoring.call_args.args[3]), 1)
        self.assertEqual(scoring.call_args.args[4], ())

    def test_order_invariance_and_input_immutability(self):
        step = synthetic_step(5, seed=42)
        before = step.model_dump()
        expected = {decision.action.agent_id: decision for decision in self.assert_parity(step)}
        rng = random.Random(918)
        shuffled = step.model_copy(deep=True)
        rng.shuffle(shuffled.agent_status)
        for value in shuffled.agent_status:
            rng.shuffle(value.observations)
            for observation in value.observations:
                if observation["type"] == "Edge":
                    observation["coords"].reverse()
        actual = {decision.action.agent_id: decision for decision in self.assert_parity(shuffled)}
        self.assertEqual(actual, expected)
        self.assertEqual(step.model_dump(), before)

    def test_no_rng_or_numpy_error_mode_side_effects(self):
        step = synthetic_step(5)
        python_state, numpy_state, error_mode = random.getstate(), np.random.get_state(), np.geterr()
        policy = create_policy(99, {"backend": "vectorized"})
        expected = policy.act(step)
        policy.explain(frame(agent(100)))
        self.assertEqual(policy.act(step), expected)
        self.assertEqual(random.getstate(), python_state)
        actual = np.random.get_state()
        self.assertEqual(actual[0], numpy_state[0])
        np.testing.assert_array_equal(actual[1], numpy_state[1])
        self.assertEqual(actual[2:], numpy_state[2:])
        self.assertEqual(np.geterr(), error_mode)

    def test_bootstrap_bounded_explanations_and_invalid_data(self):
        policy = create_policy(1, {"backend": "vectorized"})
        self.assertEqual(policy.act(frame(n_agents=5, sim_time=0, score=0)), [])
        self.assertEqual(policy.act(frame(game_status="game_over")), [])
        step = synthetic_step(5)
        self.assertEqual(len(policy.explain(step, limit=2)), 2)
        for invalid in (fruit(1, float("nan")), {"type": "Unknown"}, edge(float("inf"))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                policy.act(frame(agent(observations=[invalid])))
        with self.assertRaises(ValueError):
            policy.act(frame(n_agents=5))
        with self.assertRaises(ValueError):
            policy.explain(frame(agent(), agent(1, [{"type": "Unknown"}])), limit=1)

    def test_nonfinite_scoring_is_not_silently_discarded(self):
        step = frame(agent(observations=[fruit(10)]))
        for backend in ("scalar", "vectorized"):
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                HeuristicPolicy(1, HeuristicConfig(backend=backend, food_weight=1e308)).act(step)


if __name__ == "__main__":
    unittest.main()
