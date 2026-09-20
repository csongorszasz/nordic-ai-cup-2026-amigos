import math
import random
import unittest

import numpy as np

from src.benchmarking.policies import validate_actions
from src.policies.config import HeuristicConfig
from src.policies.features import parse_entities
from src.policies.geometry import BIOME_MOVEMENT, movement_limit, segment_distance, turn_cost
from src.policies.heuristic import (
    AGENT_RADIUS, COLLECTION_RADIUS, MAX_EXPLANATIONS, HeuristicPolicy, assess_walls,
    breeding_decision, create_policy, energy_expenditure, exploration_score,
    food_score, threat_score,
)
from src.utils.DTOs import ObservationResponse, StepResponse


def agent(agent_id=0, observations=None, **changes):
    values = {
        "agent_id": agent_id, "energy": 150.0, "age": 1.0, "biome": "forest",
        "speed": 10.0, "sprint_speed": 20.0, "hearing_radius": 50.0,
        "vision_range": 200.0, "vision_angle": math.pi / 3, "max_energy": 500.0,
        "observations": [] if observations is None else observations,
    }
    return ObservationResponse(**(values | changes))


def frame(*agents, **changes):
    values = {
        "game_status": "ok", "score": 1.0, "sim_time": 1.0,
        "n_agents": len(agents), "agent_status": list(agents),
    }
    return StepResponse(**(values | changes))


def fruit(distance, angle=0.0):
    return {"type": "Fruit", "distance": distance, "angle": angle}


def predator(distance, angle=0.0):
    return {"type": "Predator", "distance": distance, "angle": angle, "rel_dir": 0.0}


def edge(x, lower=-100.0, upper=100.0):
    return {"type": "Edge", "coords": [[x, lower], [x, upper]]}


def endpoint(action, biome="forest"):
    distance = action.move_distance * BIOME_MOVEMENT[biome]
    return distance * math.cos(action.move_direction), distance * math.sin(action.move_direction)


class HeuristicPolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = HeuristicConfig()
        self.policy = HeuristicPolicy(123, self.config)

    def test_reaches_food_without_overshoot_or_dt_scaling(self):
        for angle in (0.0, 0.7, -2.0):
            with self.subTest(angle=angle):
                value = agent(observations=[fruit(12.0, angle)])
                action = self.policy.act(frame(value, sim_time=0.01))[0]
                self.assertAlmostEqual(action.move_distance, 8.0)
                self.assertAlmostEqual(action.move_direction, angle)
                x, y = endpoint(action)
                self.assertAlmostEqual(
                    math.hypot(12 * math.cos(angle) - x, 12 * math.sin(angle) - y),
                    COLLECTION_RADIUS,
                )
                self.assertEqual(action.turn_angle, 0.0)

    def test_terrain_changes_displacement_not_requested_cost(self):
        for biome, modifier in BIOME_MOVEMENT.items():
            with self.subTest(biome=biome):
                value = agent(biome=biome, observations=[fruit(6.0)])
                action = self.policy.act(frame(value))[0]
                self.assertAlmostEqual(action.move_distance, 2.0 / modifier)
                self.assertAlmostEqual(endpoint(action, biome)[0], 2.0)
                self.assertAlmostEqual(
                    energy_expenditure(value, action.move_distance, action.turn_angle),
                    0.05 * action.move_distance,
                )

    def test_stops_on_food_without_spending_energy_turning(self):
        for distance in (0.0, 1.5, COLLECTION_RADIUS):
            with self.subTest(distance=distance):
                decision = self.policy.explain(frame(agent(observations=[fruit(distance, 1.0)])))[0]
                self.assertEqual(decision.action.move_distance, 0.0)
                self.assertEqual(decision.action.turn_angle, 0.0)
                self.assertEqual(decision.reason, "collect")

    def test_distant_food_uses_a_walking_tick(self):
        action = self.policy.act(frame(agent(observations=[fruit(150.0)])))[0]
        self.assertEqual(action.move_distance, 10.0)
        self.assertEqual(action.move_direction, 0.0)

    def test_sprints_away_while_facing_a_close_visible_threat(self):
        value = agent(observations=[predator(35.0)])
        action = self.policy.act(frame(value))[0]
        x, y = endpoint(action)
        self.assertGreater(action.move_distance, value.speed)
        self.assertLess(x, -value.speed)
        self.assertGreater(math.hypot(35.0 - x, y), 35.0)
        self.assertGreater(math.cos(math.atan2(-y, 35.0 - x) - action.turn_angle), 0.99)
        self.assertAlmostEqual(action.turn_angle, 0.0)
        self.assertFalse(action.spawn_agent)

    def test_travel_is_body_relative_before_the_facing_turn(self):
        bearing = 1.0
        value = agent(observations=[predator(35.0, bearing)])
        action = self.policy.act(frame(value))[0]
        x, y = endpoint(action)
        self.assertGreater(math.hypot(35 * math.cos(bearing) - x, 35 * math.sin(bearing) - y), 45)
        self.assertLess(math.cos(action.move_direction - bearing), -0.9)
        self.assertAlmostEqual(action.turn_angle, self.config.max_turn)

    def test_threat_behind_turns_toward_it_without_retreating_toward_it(self):
        action = self.policy.act(frame(agent(observations=[predator(30.0, math.pi)])))[0]
        self.assertGreater(endpoint(action)[0], 10.0)
        self.assertAlmostEqual(abs(action.turn_angle), self.config.max_turn)
        self.assertLess(abs(math.pi - abs(action.turn_angle)), math.pi)

    def test_danger_overrides_a_fruit_underfoot(self):
        action = self.policy.act(frame(agent(observations=[fruit(0.0), predator(25.0)])))[0]
        self.assertGreater(action.move_distance, 0.0)
        self.assertLess(endpoint(action)[0], 0.0)

    def test_late_predator_is_not_dropped_after_many_observations(self):
        observations = [predator(1000.0 + index, math.pi) for index in range(80)]
        observations.append(predator(25.0))
        action = self.policy.act(frame(agent(observations=observations)))[0]
        self.assertLess(endpoint(action)[0], -10.0)

    def test_sprint_energy_boundary_and_mutated_speed_caps(self):
        for energy, speed, sprint, expected in (
            (99.999, 10.0, 20.0, 10.0), (100.0, 10.0, 20.0, 20.0),
            (150.0, 20.0, 7.0, 7.0), (50.0, 20.0, 7.0, 7.0),
            (150.0, 0.0, 0.0, 0.0), (150.0, 0.0, 8.0, 8.0),
        ):
            with self.subTest(energy=energy, speed=speed, sprint=sprint):
                value = agent(energy=energy, speed=speed, sprint_speed=sprint,
                              observations=[predator(25.0)])
                action = self.policy.act(frame(value))[0]
                self.assertGreaterEqual(action.move_distance, 0.0)
                self.assertLessEqual(action.move_distance, expected)
                self.assertLessEqual(action.move_distance, movement_limit(value))
                if energy == 100.0:
                    self.assertGreater(action.move_distance, speed)

    def test_nearly_exhausted_agent_does_not_spend_its_last_energy(self):
        for energy in (0.0, 0.001, 0.1, 0.5):
            with self.subTest(energy=energy):
                value = agent(energy=energy, observations=[fruit(20.0, 1.0)])
                action = self.policy.act(frame(value))[0]
                cost = energy_expenditure(value, action.move_distance, action.turn_angle)
                self.assertLessEqual(cost, energy)
                if energy:
                    self.assertLess(cost, energy)

    def test_avoids_a_wall_even_when_food_is_behind_it(self):
        value = agent(observations=[fruit(30.0), edge(8.0)])
        action = self.policy.act(frame(value))[0]
        path = ((0.0, 0.0), endpoint(action))
        self.assertGreaterEqual(segment_distance(path, ((8.0, -100.0), (8.0, 100.0))), AGENT_RADIUS)
        self.assertLessEqual(endpoint(action)[0], 3.0)

    def test_late_wall_is_not_dropped_after_many_edges(self):
        observations = [edge(-1000.0 - index) for index in range(80)] + [edge(8.0), fruit(25.0)]
        action = self.policy.act(frame(agent(observations=observations)))[0]
        self.assertGreaterEqual(
            segment_distance(((0.0, 0.0), endpoint(action)), ((8.0, -100.0), (8.0, 100.0))),
            AGENT_RADIUS,
        )

    def test_moves_out_of_an_existing_wall_overlap(self):
        action = self.policy.act(frame(agent(observations=[edge(4.0)])))[0]
        self.assertLess(endpoint(action)[0], 0.0)

    def test_empty_bootstrap_and_game_over(self):
        self.assertEqual(self.policy.act(frame()), [])
        self.assertEqual(self.policy.act(frame(game_status="game_over")), [])
        for population in (2, 5):
            with self.subTest(population=population):
                bootstrap = frame(score=0.0, sim_time=0.0, n_agents=population)
                self.assertEqual(self.policy.act(bootstrap), [])
                self.assertEqual(self.policy.explain(bootstrap), ())

    def test_empty_population_mismatch_is_only_allowed_at_bootstrap(self):
        bootstrap = frame(score=0.0, sim_time=0.0, n_agents=5)
        for changes in (
            {"sim_time": 0.1}, {"score": 0.1}, {"game_status": "game_over"},
        ):
            with self.subTest(changes=changes):
                step = bootstrap.model_copy(update=changes)
                with self.assertRaisesRegex(ValueError, "n_agents does not match agent_status"):
                    self.policy.act(step)
                with self.assertRaisesRegex(ValueError, "n_agents does not match agent_status"):
                    self.policy.explain(step)

    def test_returns_exactly_one_action_per_agent_in_incoming_order(self):
        ids = [12, 0, 9, 100]
        step = frame(*(agent(agent_id) for agent_id in ids))
        actions = self.policy.act(step)
        self.assertEqual([action.agent_id for action in actions], ids)
        self.assertEqual(validate_actions(actions, ids), actions)

    def test_agent_and_observation_permutations_do_not_change_decisions(self):
        observations = [
            fruit(25.0, 0.5), fruit(35.0, -0.5), predator(80.0, -1.0),
            {"type": "Tree", "distance": 90.0, "angle": 0.2},
            {"type": "Agent", "id": 9, "distance": 20.0, "angle": 0.8, "rel_dir": 0.1},
            edge(-20.0), {"type": "Edge", "coords": [[-20.0, 100.0], [-20.0, -100.0]]},
        ]
        original = frame(agent(8, observations), agent(1, list(reversed(observations))), agent(3))
        expected = {action.agent_id: action for action in self.policy.act(original)}
        rng = random.Random(81)
        for _ in range(6):
            shuffled = original.model_copy(deep=True)
            rng.shuffle(shuffled.agent_status)
            for value in shuffled.agent_status:
                rng.shuffle(value.observations)
            actual = {action.agent_id: action for action in self.policy.act(shuffled)}
            self.assertEqual(actual, expected)

    def test_diagnostics_and_interleaved_calls_have_no_state_or_input_effects(self):
        step = frame(agent(8, [fruit(20.0)]), agent(1, [predator(30.0)]))
        before = step.model_dump()
        expected = self.policy.act(step)
        self.policy.act(frame(agent(92, [edge(7.0)])))
        decisions = self.policy.explain(step)
        self.assertEqual([decision.action for decision in decisions], expected)
        self.assertEqual(self.policy.act(step), expected)
        self.assertEqual(step.model_dump(), before)
        expected[0].move_distance = -999
        self.assertGreaterEqual(self.policy.act(step)[0].move_distance, 0.0)
        self.assertFalse(hasattr(self.policy, "last_decisions"))

    def test_policy_does_not_touch_global_random_generators(self):
        python_state, numpy_state = random.getstate(), np.random.get_state()
        step = frame(agent(1), agent(20, [fruit(40.0), predator(45.0, -1.0)]))
        HeuristicPolicy(7, self.config).act(step)
        self.policy.explain(step)
        self.assertEqual(random.getstate(), python_state)
        after = np.random.get_state()
        self.assertEqual(after[0], numpy_state[0])
        np.testing.assert_array_equal(after[1], numpy_state[1])
        self.assertEqual(after[2:], numpy_state[2:])

    def test_diagnostics_have_a_hard_output_bound(self):
        step = frame(*(agent(index) for index in range(MAX_EXPLANATIONS + 1)))
        self.assertEqual(len(self.policy.explain(step)), 32)
        self.assertEqual(len(self.policy.explain(step, MAX_EXPLANATIONS)), MAX_EXPLANATIONS)
        self.assertEqual(self.policy.explain(step, 0), ())
        for invalid in (-1, MAX_EXPLANATIONS + 1, True, 1.5):
            with self.subTest(limit=invalid), self.assertRaises(ValueError):
                self.policy.explain(step, invalid)

    def test_candidate_count_is_bounded_despite_many_foods(self):
        observations = [fruit(20.0 + index, index / 10) for index in range(100)]
        observations += [predator(40.0), edge(-25.0)]
        decision = HeuristicPolicy(1, HeuristicConfig(directions=64)).explain(
            frame(agent(observations=observations)),
        )[0]
        self.assertLessEqual(decision.candidates, 330)

    def test_invalid_observations_and_frames_fail_loudly(self):
        invalid = [
            {"type": "Mystery"}, fruit(-1), fruit(2, float("nan")),
            {"type": "Predator", "distance": 20, "angle": 0},
            {"type": "Edge", "coords": [[0, 1], [2, float("inf")]]},
        ]
        for observation in invalid:
            with self.subTest(observation=observation), self.assertRaises(ValueError):
                self.policy.act(frame(agent(observations=[fruit(3), observation])))
        for step in (
            frame(agent(), n_agents=2), frame(agent(), agent()), frame(agent(biome="moon")),
            frame(agent(energy=float("nan"))), frame(game_status="unknown"),
        ):
            with self.subTest(step=step), self.assertRaises(ValueError):
                self.policy.act(step)
        with self.assertRaises(ValueError):
            self.policy.explain(frame(agent(), agent(1, [{"type": "Mystery"}])), limit=1)

    def test_factory_validates_unknown_configuration(self):
        policy = create_policy(5, {"directions": 32})
        self.assertIsInstance(policy, HeuristicPolicy)
        self.assertEqual(policy.config.directions, 32)
        with self.assertRaises(ValueError):
            create_policy(5, {"unrecognized": 1})
        with self.assertRaises(ValueError):
            HeuristicPolicy(True, self.config)

    def test_safe_exploration_prefers_walking_and_a_modest_turn(self):
        action = self.policy.act(frame(agent()))[0]
        self.assertEqual(action.move_distance, 10.0)
        self.assertLessEqual(abs(action.turn_angle), self.config.scan_turn + 0.2)
        self.assertFalse(action.spawn_agent)

    def test_young_breeding_threshold_is_applied_after_movement(self):
        observations = [fruit(35.0), fruit(45.0)]
        below = self.policy.act(frame(agent(energy=240.2, observations=observations)))[0]
        above = self.policy.act(frame(agent(energy=240.8, observations=observations)))[0]
        self.assertGreater(below.move_distance, 0.0)
        self.assertFalse(below.spawn_agent)
        self.assertTrue(above.spawn_agent)


class HeuristicScoringTests(unittest.TestCase):
    def test_food_progress_and_energy_terms_are_independently_measurable(self):
        value = agent(observations=[fruit(20)])
        fruits = parse_entities(value)
        self.assertGreater(food_score(value, (10.0, 0.0), fruits), food_score(value, (0.0, 0.0), fruits))
        self.assertLess(food_score(value, (-10.0, 0.0), fruits), 0.0)
        self.assertEqual(food_score(value, (0.0, 0.0), ()), 0.0)
        self.assertAlmostEqual(energy_expenditure(value, 12.0, math.pi), 2.0)
        self.assertGreater(energy_expenditure(value, 20.0, 0.0), energy_expenditure(value, 10.0, 0.0))

    def test_wall_term_checks_entire_sweep_and_recovery(self):
        walls = (((10.0, -20.0), (10.0, 20.0)),)
        self.assertTrue(assess_walls((20.0, 0.0), walls, 12.0).blocked)
        self.assertFalse(assess_walls((-10.0, 0.0), walls, 12.0).blocked)
        self.assertGreater(
            assess_walls((-10.0, 0.0), walls, 12.0).score,
            assess_walls((2.0, 0.0), walls, 12.0).score,
        )
        overlap = (((4.0, -20.0), (4.0, 20.0)),)
        self.assertFalse(assess_walls((-10.0, 0.0), overlap, 12.0).blocked)
        self.assertTrue(assess_walls((10.0, 0.0), overlap, 12.0).blocked)
        self.assertEqual(assess_walls((10.0, 0.0), (), 12.0).score, 0.0)

    def test_threat_term_rewards_escape_and_facing_not_a_distant_hazard(self):
        value = agent(observations=[predator(35.0)])
        threats, config = parse_entities(value), HeuristicConfig()
        retreat = threat_score(value, (-20.0, 0.0), 0.0, threats, config)
        self.assertGreater(retreat, threat_score(value, (10.0, 0.0), 0.0, threats, config))
        self.assertGreater(retreat, threat_score(value, (-20.0, 0.0), math.pi, threats, config))
        far = parse_entities(agent(observations=[predator(1000)]))
        self.assertEqual(threat_score(value, (10.0, 0.0), 0.0, far, config), 0.0)

    def test_exploration_does_not_reward_wasteful_extra_sprint_distance(self):
        self.assertEqual(exploration_score((10.0, 0.0), 0.0, 10.0), 1.0)
        self.assertEqual(exploration_score((20.0, 0.0), 0.0, 10.0), 1.0)
        self.assertEqual(exploration_score((-10.0, 0.0), 0.0, 10.0), -1.0)

    def test_breeding_engine_threshold_is_strict_and_uses_turn_cost(self):
        settings = HeuristicConfig(breeding_energy=100.0, breeding_reserve=0.0)
        fruits = parse_entities(agent(observations=[fruit(20), fruit(25)]))
        for energy, turn, expected in (
            (100.5, 0.0, False), (100.5001, 0.0, True),
            (100.55, 0.45, False), (100.5 + turn_cost(0.45) + 0.001, 0.45, True),
        ):
            with self.subTest(energy=energy, turn=turn):
                result = breeding_decision(
                    agent(energy=energy, age=0), 10.0, turn, (10.0, 0.0),
                    fruits, (), (), settings,
                )
                self.assertEqual(result.spawn, expected)
                self.assertAlmostEqual(result.post_action_energy, energy - 0.5 - turn_cost(turn))

    def test_breeding_requires_a_patch_or_sustainable_age_renewal(self):
        settings = HeuristicConfig()
        one = parse_entities(agent(observations=[fruit(25)]))
        two = parse_entities(agent(observations=[fruit(25), fruit(30)]))
        for age, energy, foods, expected, reason in (
            (1, 400, (), False, "food_or_age"), (1, 400, one, False, "food_or_age"),
            (1, 400, two, True, "food_surplus"), (65, 250, (), True, "renew_age"),
            (65, 200, (), False, "food_or_age"), (65, 200, one, True, "renew_age"),
            (100, 166, one, False, "reserve"),
        ):
            with self.subTest(age=age, energy=energy, food=len(foods)):
                result = breeding_decision(
                    agent(age=age, energy=energy), 10.0, 0.0, (10.0, 0.0),
                    foods, (), (), settings,
                )
                self.assertEqual(result.spawn, expected)
                self.assertEqual(result.reason, reason)

    def test_breeding_is_suppressed_near_threats_and_walls(self):
        value = agent(age=65, energy=400)
        threats = parse_entities(agent(observations=[predator(105)]))
        result = breeding_decision(value, 10, 0, (-10.0, 0.0), (), threats, (), HeuristicConfig())
        self.assertFalse(result.spawn)
        self.assertEqual(result.reason, "threat")
        result = breeding_decision(
            value, 0, 0, (0.0, 0.0), (), (), (((8.0, -10.0), (8.0, 10.0)),),
            HeuristicConfig(),
        )
        self.assertFalse(result.spawn)
        self.assertEqual(result.reason, "wall")


if __name__ == "__main__":
    unittest.main()
