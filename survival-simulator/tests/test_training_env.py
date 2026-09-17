import copy
import random
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import Mock, patch

from src.benchmarking.config import SimulationSettings, load_suite
from src.benchmarking.policies import validate_actions
from src.training.env import EnvironmentAdapter, EnvTransition
from src.training.seeds import training_seeds
from src.utils.DTOs import ActionRequest, StepResponse


def status(agent_id=0, **changes):
    values = {
        "agent_id": agent_id, "energy": 150.0, "age": 0.1, "biome": "forest",
        "speed": 10.0, "sprint_speed": 20.0, "hearing_radius": 50.0,
        "vision_angle": 1.0, "vision_range": 200.0, "max_energy": 500.0,
        "observations": [],
    }
    values.update(changes)
    return values


def frame(sim_time=0.1, score=0.1, ids=(0,)):
    return {
        "score": score, "sim_time": sim_time, "num_agents": len(ids),
        "observations": [status(agent_id) for agent_id in ids],
    }


def action(agent_id=0, **changes):
    values = {
        "agent_id": agent_id, "move_distance": 0.0, "move_direction": 0.0,
        "turn_angle": 0.0, "spawn_agent": False,
    }
    values.update(changes)
    return ActionRequest(**values)


class FakeSimulation:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.inputs = []

    def step(self, actions):
        self.inputs.append(actions)
        value = next(self.frames)
        if isinstance(value, BaseException):
            raise value
        return value


def adapter_for(*frames, settings=None):
    simulation = FakeSimulation(frames)
    adapter = EnvironmentAdapter(settings, simulation_factory=lambda **kwargs: simulation)
    return adapter, simulation


class EnvironmentAdapterTests(unittest.TestCase):
    def test_construction_is_deferred_and_reset_passes_native_settings_and_seed(self):
        settings = SimulationSettings(starting_agents=2, starting_trees=0, time_limit=0.3)
        simulation = FakeSimulation([frame(ids=(7, 2))])
        factory = Mock(return_value=simulation)
        adapter = EnvironmentAdapter(settings, simulation_factory=factory)
        factory.assert_not_called()
        self.assertIsNone(adapter.seed)
        self.assertFalse(adapter.done)
        with self.assertRaisesRegex(RuntimeError, "[Rr]eset"):
            _ = adapter.observation
        with self.assertRaisesRegex(RuntimeError, "[Rr]eset"):
            adapter.step([])
        observation = adapter.reset(2**32 - 1)
        factory.assert_called_once_with(seed=2**32 - 1, **settings.core_kwargs())
        self.assertEqual(simulation.inputs, [[]])
        self.assertIsInstance(observation, StepResponse)
        self.assertIs(adapter.observation, observation)
        self.assertEqual([agent.agent_id for agent in observation.agent_status], [7, 2])
        self.assertEqual(adapter.seed, 2**32 - 1)

    def test_reward_is_native_delta_once_per_team_even_when_population_changes(self):
        adapter, simulation = adapter_for(
            frame(score=10.0, ids=(0, 1)),
            frame(0.2, 10.3, ids=(0, 1, 2, 3)),
            frame(0.3, 8.05, ids=()),
        )
        adapter.reset(123)
        first = adapter.step([action(0), action(1)])
        self.assertIsInstance(first, EnvTransition)
        self.assertAlmostEqual(first.reward, 0.3)
        self.assertFalse(first.terminated)
        last = adapter.step([action(i) for i in range(4)])
        self.assertAlmostEqual(last.reward, -2.25)
        self.assertTrue(last.terminated)
        self.assertEqual(last.observation.game_status, "game_over")
        self.assertEqual([len(batch) for batch in simulation.inputs], [0, 2, 4])
        with self.assertRaises(FrozenInstanceError):
            last.reward = 100
        with self.assertRaisesRegex(RuntimeError, "terminal"):
            adapter.step([])

    def test_action_validation_is_reused_and_proposed_order_is_not_sorted(self):
        adapter, simulation = adapter_for(frame(ids=(2, 7)), frame(0.2, ids=(2, 7)))
        adapter.reset(1)
        actions = [action(7, move_distance=-20), action(2, turn_angle=999, move_distance=1000)]
        with patch("src.training.env.validate_actions", wraps=validate_actions) as validate:
            adapter.step(actions)
        validate.assert_called_once_with(actions, (2, 7))
        submitted = simulation.inputs[1]
        self.assertEqual([agent_id for agent_id, _ in submitted], [7, 2])
        self.assertEqual([value for _, value in submitted], actions)
        self.assertIsNot(submitted[0][1], actions[0])

    def test_bad_actions_do_not_advance_or_invalidate_the_environment(self):
        adapter, simulation = adapter_for(frame(), frame(0.2))
        adapter.reset(1)
        malformed = action()
        malformed.agent_id = True
        for proposed in (
            None, "actions", iter([action()]), {}, [], [action(7)],
            [action(), action()], [action().model_dump()], [malformed],
            [action(move_distance=float("nan"))],
            [action(move_direction=float("inf"))],
            [action(turn_angle=-float("inf"))],
        ):
            with self.subTest(proposed=proposed), self.assertRaises(ValueError):
                adapter.step(proposed)
            self.assertEqual(len(simulation.inputs), 1)
        adapter.step([action()])
        self.assertEqual(len(simulation.inputs), 2)

    def test_time_limit_uses_strict_greater_than(self):
        adapter, simulation = adapter_for(
            frame(0.1), frame(0.2), frame(0.3),
            settings=SimulationSettings(time_limit=0.2),
        )
        adapter.reset(1)
        at_limit = adapter.step([action()])
        self.assertFalse(at_limit.terminated)
        self.assertEqual(at_limit.observation.game_status, "ok")
        over_limit = adapter.step([action()])
        self.assertTrue(over_limit.terminated)
        self.assertTrue(adapter.done)
        self.assertEqual(over_limit.observation.game_status, "game_over")
        self.assertEqual(len(simulation.inputs), 3)

    def test_terminal_bootstrap_reports_game_over_without_an_extra_step(self):
        for first in (frame(ids=()), frame(0.3), frame(0.3, ids=())):
            with self.subTest(first=first):
                adapter, simulation = adapter_for(first, settings=SimulationSettings(time_limit=0.2))
                observation = adapter.reset(1)
                self.assertTrue(adapter.done)
                self.assertEqual(observation.game_status, "game_over")
                self.assertEqual(simulation.inputs, [[]])
                with self.assertRaisesRegex(RuntimeError, "terminal"):
                    adapter.step([])

    def test_actor_mutation_cannot_change_native_reward_or_expected_action_ids(self):
        source = frame(score=5.0)
        adapter, _ = adapter_for(source, frame(0.2, 6.5))
        observation = adapter.reset(1)
        observation.score = -999.0
        observation.sim_time = 99999.0
        observation.n_agents = 999
        observation.agent_status[0].agent_id = 99
        observation.agent_status.clear()
        transition = adapter.step([action()])
        self.assertEqual(transition.reward, 1.5)
        self.assertFalse(transition.terminated)
        self.assertEqual(source["observations"][0]["agent_id"], 0)
        self.assertEqual(source["score"], 5.0)

    def test_nested_dto_mutation_cannot_change_engine_cached_perceptions(self):
        source = frame()
        source["observations"][0]["observations"] = [
            {"type": "Edge", "coords": [[1.0, 2.0], [3.0, 4.0]]},
            {"type": "Fruit", "distance": 3.0, "angle": 0.1,
             "metadata": {"history": [1, {"value": 2}]}},
            {"type": "Edge", "coords": ((5.0, 6.0), (7.0, 8.0))},
        ]
        original = copy.deepcopy(source)
        adapter, _ = adapter_for(source)
        observation = adapter.reset(1)
        perceived = observation.agent_status[0].observations
        self.assertEqual([item["type"] for item in perceived], ["Edge", "Fruit", "Edge"])
        self.assertIsInstance(perceived[2]["coords"], tuple)
        perceived[0]["coords"][0][0] = -100
        perceived[1]["metadata"]["history"][1]["value"] = -100
        perceived[2]["type"] = "Fruit"
        perceived.pop()
        self.assertEqual(source, original)
        source["observations"][0]["observations"][0]["coords"][1][0] = 999
        self.assertEqual(perceived[0]["coords"][1][0], 3.0)

    def test_native_negative_energy_is_not_clipped_or_filtered(self):
        source = frame()
        source["observations"][0]["energy"] = -0.01
        adapter, _ = adapter_for(source)
        observation = adapter.reset(1)
        self.assertEqual(observation.agent_status[0].energy, -0.01)
        self.assertFalse(adapter.done)

    def test_invalid_counts_ids_and_top_level_data_are_rejected(self):
        frames = []
        for field, value in (
            ("num_agents", -1), ("num_agents", 2), ("num_agents", True),
            ("num_agents", 1.0), ("score", float("nan")),
            ("sim_time", float("inf")), ("sim_time", -0.1),
        ):
            source = frame()
            source[field] = value
            frames.append(source)
        frames.append(frame(ids=(0, 0)))
        frames.append(frame(ids=(-1,)))
        frames.append(frame(ids=(True,)))
        source = frame()
        source["observations"] = [None]
        frames.append(source)
        for source in frames:
            with self.subTest(source=source), self.assertRaises(ValueError):
                adapter_for(source)[0].reset(1)

    def test_every_numeric_agent_field_and_nested_perception_must_be_finite(self):
        for field in (
            "energy", "age", "speed", "sprint_speed", "hearing_radius",
            "vision_angle", "vision_range", "max_energy",
        ):
            source = frame()
            source["observations"][0][field] = float("nan")
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "non-finite"):
                adapter_for(source)[0].reset(1)
        for perception in (
            {"type": "Fruit", "distance": float("inf"), "angle": 0.0},
            {"type": "Edge", "coords": ((1.0, 2.0), (3.0, float("nan")))},
            {"type": "Fruit", "metadata": {"values": [float("-inf")]}},
            {"type": "Fruit", "mutable": object()},
            {"type": "Fruit", float("nan"): 1.0},
            {"type": "Fruit", "metadata": {float("inf"): 1.0}},
        ):
            source = frame()
            source["observations"][0]["observations"] = [perception]
            with self.subTest(perception=perception), self.assertRaises(ValueError):
                adapter_for(source)[0].reset(1)

    def test_engine_and_observation_errors_propagate_and_require_a_new_reset(self):
        failure = RuntimeError("native physics failed")
        adapter, simulation = adapter_for(frame(), failure)
        adapter.reset(1)
        with self.assertRaises(RuntimeError) as caught:
            adapter.step([action()])
        self.assertIs(caught.exception, failure)
        with self.assertRaisesRegex(RuntimeError, "[Rr]eset"):
            adapter.step([action()])
        self.assertEqual(len(simulation.inputs), 2)

        broken = frame(0.2)
        broken["num_agents"] = 2
        adapter, simulation = adapter_for(frame(), broken)
        adapter.reset(1)
        with self.assertRaisesRegex(ValueError, "count"):
            adapter.step([action()])
        with self.assertRaisesRegex(RuntimeError, "[Rr]eset"):
            adapter.step([action()])
        self.assertEqual(len(simulation.inputs), 2)

    def test_initialization_and_bootstrap_errors_are_not_replaced(self):
        for fails_in_constructor in (True, False):
            failure = RuntimeError("initialization failed")
            factory = (
                Mock(side_effect=failure) if fails_in_constructor
                else Mock(return_value=FakeSimulation([failure]))
            )
            adapter = EnvironmentAdapter(simulation_factory=factory)
            with self.subTest(constructor=fails_in_constructor):
                with self.assertRaises(RuntimeError) as caught:
                    adapter.reset(1)
                self.assertIs(caught.exception, failure)
                with self.assertRaisesRegex(RuntimeError, "[Rr]eset"):
                    adapter.step([action()])

    def test_finite_scores_with_overflowing_delta_fail_explicitly(self):
        adapter, _ = adapter_for(frame(score=-1e308), frame(0.2, 1e308))
        adapter.reset(1)
        with self.assertRaisesRegex(ValueError, "delta"):
            adapter.step([action()])
        with self.assertRaisesRegex(RuntimeError, "[Rr]eset"):
            adapter.step([action()])

    def test_reset_constructs_a_new_world_and_restarts_timing_and_score_baseline(self):
        first = FakeSimulation([frame(score=5.0), frame(0.2, 6.0)])
        second = FakeSimulation([frame(score=20.0)])
        factory = Mock(side_effect=[first, second])
        adapter = EnvironmentAdapter(simulation_factory=factory)
        with patch("src.training.env.time.perf_counter", side_effect=[
            1.0, 4.0, 10.0, 12.0, 20.0, 25.0, 30.0, 31.0, 40.0, 43.0,
        ]):
            adapter.reset(12)
            self.assertEqual(adapter.initialization_seconds, 3.0)
            self.assertEqual(adapter.simulation_seconds, 2.0)
            self.assertEqual(adapter.step([action()]).reward, 1.0)
            self.assertEqual(adapter.simulation_seconds, 7.0)
            adapter.reset(13)
        self.assertEqual(adapter.seed, 13)
        self.assertEqual(adapter.observation.score, 20.0)
        self.assertEqual(adapter.initialization_seconds, 1.0)
        self.assertEqual(adapter.simulation_seconds, 3.0)
        self.assertEqual(second.inputs, [[]])
        self.assertEqual(factory.call_count, 2)

    def test_invalid_seed_or_settings_fail_before_construction(self):
        factory = Mock(return_value=FakeSimulation([frame(), frame(0.2)]))
        adapter = EnvironmentAdapter(simulation_factory=factory)
        adapter.reset(0)
        for seed in (True, False, -1, 2**32, 1.5, "1", None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                adapter.reset(seed)
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(adapter.seed, 0)
        adapter.step([action()])
        with self.assertRaises(TypeError):
            EnvironmentAdapter({})
        with self.assertRaises(TypeError):
            EnvironmentAdapter(simulation_factory=None)
        with self.assertRaises(ValueError):
            EnvironmentAdapter(SimulationSettings().model_copy(update={"starting_agents": True}))

    def test_adapter_import_does_not_import_torch(self):
        code = "import sys; import src.training.env; assert 'torch' not in sys.modules"
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            check=True, capture_output=True, text=True, timeout=30,
        )


class TrainingSeedTests(unittest.TestCase):
    def test_deterministic_values_unique_prefix_and_global_rng_isolation(self):
        before = random.getstate()
        self.assertEqual(training_seeds(12345, 3), [1789368711, 3146859322, 43676229])
        values = training_seeds(88, 100)
        self.assertEqual(values, training_seeds(88, 100))
        self.assertEqual(values[:5], training_seeds(88, 5))
        self.assertEqual(len(set(values)), 100)
        self.assertEqual(random.getstate(), before)
        excluded = set().union(*(load_suite(name).seeds for name in ("quick", "standard", "holdout")))
        self.assertFalse(set(values) & excluded)
        self.assertTrue(all(0 <= value < 2**32 for value in values))

    def test_rejection_sampling_skips_evaluation_seeds_and_duplicate_training_worlds(self):
        rng = Mock()
        rng.getrandbits.side_effect = [1, 7, 42, 987654321, 987654321, 4_000_000_000]
        with patch("src.training.seeds.random.Random", return_value=rng) as factory:
            self.assertEqual(training_seeds(9, 2), [987654321, 4_000_000_000])
        factory.assert_called_once_with(9)
        self.assertEqual([call.args for call in rng.getrandbits.call_args_list], [(32,)] * 6)

    def test_invalid_seed_and_count_are_rejected(self):
        for seed in (True, False, -1, 2**32, 1.5, "1", None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                training_seeds(seed, 1)
        for count in (True, False, -1, 0, 2**32, 1.5, "1", None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                training_seeds(1, count)


if __name__ == "__main__":
    unittest.main()
