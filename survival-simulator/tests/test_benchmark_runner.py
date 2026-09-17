import copy
import os
import random
import unittest
from types import SimpleNamespace

from src.benchmarking.config import SimulationSettings, Suite, make_cases
from src.benchmarking.policies import (
    RandomPolicy, create_random_policy, load_policy, validate_actions,
)
from src.benchmarking.runner import EpisodeExecutionError, EpisodeInterrupted, run_episode
from src.utils.DTOs import ActionRequest, ObservationResponse, StepResponse
from src.utils.controllers.dummy_agent_policy import action_decision


def agent(agent_id=0):
    return {
        "agent_id": agent_id, "observations": [], "energy": 150.0,
        "biome": "forest", "age": 0.1, "speed": 10.0, "sprint_speed": 20.0,
        "hearing_radius": 50.0, "vision_angle": 1.0, "vision_range": 200.0,
        "max_energy": 500.0,
    }


def frame(sim_time, count=1, score=None):
    return {
        "score": sim_time if score is None else score, "sim_time": sim_time,
        "num_agents": count, "observations": [agent(i) for i in range(count)],
    }


def action(agent_id=0, **changes):
    fields = {
        "agent_id": agent_id, "move_distance": 0.0,
        "move_direction": 0.0, "turn_angle": 0.0, "spawn_agent": False,
    }
    fields.update(changes)
    return ActionRequest(**fields)


class FakeSimulation:
    def __init__(self, frames, initial_agents=1):
        self.env = SimpleNamespace(agents=list(range(initial_agents)))
        self.frames = iter(frames)
        self.inputs = []

    def step(self, actions):
        self.inputs.append(copy.deepcopy(actions))
        return next(self.frames)


class IdlePolicy:
    def __init__(self):
        self.calls = []

    def act(self, step):
        self.calls.append(step)
        return [action(status.agent_id) for status in step.agent_status]


class PolicyTests(unittest.TestCase):
    def test_random_matches_existing_function_across_ticks(self):
        rng = random.Random(42)
        policy = RandomPolicy(42)
        for ids in ([2, 0], [2, 0, 7], [7]):
            step = StepResponse(
                game_status="ok", score=1.0, sim_time=1.0, n_agents=len(ids),
                agent_status=[ObservationResponse(**agent(i)) for i in ids],
            )
            expected = [action_decision(status.model_dump(), rng) for status in step.agent_status]
            self.assertEqual(policy.act(step), expected)

    def test_loader_and_unconfigurable_baseline(self):
        loaded = load_policy("random")
        self.assertEqual(loaded.spec.reference, "random-local-v1")
        self.assertTrue(loaded.source_files[0].is_file())
        custom = load_policy("src.benchmarking.policies:create_random_policy", label="custom")
        self.assertIsInstance(custom.factory(seed=1, config={}), RandomPolicy)
        for reference in ("unknown", "src.benchmarking.policies:missing", "src.benchmarking.policies:BASELINE_NAME"):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                load_policy(reference)
        with self.assertRaises(ValueError):
            load_policy("random", config={"changed": True})

    def test_action_order_and_engine_clamping_are_preserved(self):
        actions = [action(2, move_distance=-1), action(0, move_distance=10000)]
        self.assertEqual(validate_actions(actions, [0, 2]), actions)

    def test_bad_actions_fail_instead_of_being_replaced(self):
        for proposed in (
            None, {"actions": []}, [action().model_dump()], [],
            [action(1)], [action(), action()],
            [action(move_distance=float("nan"))],
            [action(move_direction=float("inf"))],
            [action(turn_angle=-float("inf"))],
        ):
            with self.subTest(proposed=proposed), self.assertRaises(ValueError):
                validate_actions(proposed, [0])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.case = make_cases(Suite(name="fixture", seeds=[1]))[0]

    def run_fake(self, sim, policy=None, **kwargs):
        return run_episode(
            self.case, lambda seed, config: policy or IdlePolicy(), {},
            simulation_factory=lambda **settings: sim, **kwargs,
        )

    def test_bootstrap_terminal_suppression_and_population_metrics(self):
        sim = FakeSimulation([frame(0.1, 2), frame(0.2, 3), frame(0.3, 0)])
        policy = IdlePolicy()
        result = self.run_fake(sim, policy)
        self.assertEqual(sim.inputs[0], [])
        self.assertEqual([len(actions) for actions in sim.inputs], [0, 2, 3])
        self.assertEqual(len(policy.calls), 2)
        self.assertEqual(result.timings.policy_calls, 2)
        self.assertEqual(result.ticks, 3)
        self.assertEqual(result.termination, "extinction")
        self.assertFalse(result.completed)
        self.assertEqual(result.initial_agents, 1)
        self.assertEqual(result.peak_agents, 3)
        self.assertAlmostEqual(result.mean_population, 5 / 3)
        self.assertEqual(result.score, 0.3)
        self.assertIsNotNone(result.timings.batch_p95_ms)

    def test_strict_time_boundary(self):
        sim = FakeSimulation([frame(0.1), frame(0.2), frame(0.3)])
        policy = IdlePolicy()
        result = self.run_fake(sim, policy, settings=SimulationSettings(time_limit=0.2))
        self.assertEqual(result.ticks, 3)
        self.assertEqual(len(policy.calls), 2)
        self.assertEqual(result.sim_time, 0.3)
        self.assertEqual(result.survival_seconds, 0.2)
        self.assertTrue(result.completed)
        self.assertEqual(result.termination, "time_limit")

    def test_extinction_wins_when_both_terminal_conditions_apply(self):
        result = self.run_fake(
            FakeSimulation([frame(0.3, 0)]), settings=SimulationSettings(time_limit=0.2),
        )
        self.assertEqual(result.termination, "extinction")
        self.assertEqual(result.timings.policy_calls, 0)
        self.assertIsNone(result.timings.batch_mean_ms)

    def test_step_cap_counts_bootstrap_tick_and_never_claims_completion(self):
        result = self.run_fake(FakeSimulation([frame(0.1)]), max_steps=1)
        self.assertEqual(result.status, "truncated")
        self.assertEqual(result.termination, "step_limit")
        self.assertEqual(result.timings.policy_calls, 0)
        self.assertFalse(result.completed)
        for cap in (0, True, 0.5):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                self.run_fake(FakeSimulation([]), max_steps=cap)

    def test_fresh_factory_and_isolated_config_each_episode(self):
        instances = []
        configs = []
        options = {"nested": {"value": 1}}

        def factory(seed, config):
            configs.append(copy.deepcopy(config))
            config["nested"]["value"] = 2
            policy = IdlePolicy()
            instances.append(policy)
            return policy

        for _ in range(2):
            sim = FakeSimulation([frame(0.1), frame(0.2, 0)])
            run_episode(self.case, factory, options, simulation_factory=lambda **settings: sim)
        self.assertIsNot(instances[0], instances[1])
        self.assertEqual([len(policy.calls) for policy in instances], [1, 1])
        self.assertEqual(configs, [options, options])

    def test_policy_cannot_mutate_engine_observations_or_result_metrics(self):
        source = frame(0.1)
        source["observations"][0]["observations"] = [{"type": "Edge", "coords": [[1, 2], [3, 4]]}]
        untouched = copy.deepcopy(source)

        class MutatingPolicy:
            def act(self, step):
                step.score = -100
                step.agent_status[0].observations[0]["coords"][0][0] = -100
                step.agent_status.clear()
                return [action()]

        sim = FakeSimulation([source, frame(0.2, 0)])
        result = self.run_fake(sim, MutatingPolicy())
        self.assertEqual(source, untouched)
        self.assertEqual(result.score, 0.2)

    def test_invalid_actions_preserve_partial_metrics_and_error(self):
        class BadPolicy:
            def act(self, step):
                return []

        with self.assertRaises(EpisodeExecutionError) as caught:
            self.run_fake(FakeSimulation([frame(0.1)]), BadPolicy())
        result = caught.exception.result
        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.score)
        self.assertEqual(result.ticks, 1)
        self.assertEqual(result.failure.stage, "action_validation")
        self.assertIn("ValueError", result.failure.traceback)
        self.assertEqual(result.timings.policy_calls, 1)

    def test_initialization_failure_is_explicit(self):
        def failing_sim(**kwargs):
            raise RuntimeError("engine failed")

        with self.assertRaises(EpisodeExecutionError) as caught:
            run_episode(self.case, create_random_policy, {}, simulation_factory=failing_sim)
        result = caught.exception.result
        self.assertEqual(result.failure.stage, "simulation_initialization")
        self.assertEqual(result.ticks, 0)
        self.assertIsNone(result.score)
        self.assertIsNone(result.final_agents)

    def test_bad_factory_and_policy_exception(self):
        with self.assertRaises(EpisodeExecutionError) as caught:
            run_episode(
                self.case, lambda seed, config: object(), {},
                simulation_factory=lambda **kwargs: FakeSimulation([]),
            )
        self.assertEqual(caught.exception.result.failure.stage, "policy_construction")

        class RaisingPolicy:
            def act(self, step):
                raise RuntimeError("policy failed")

        with self.assertRaises(EpisodeExecutionError) as caught:
            self.run_fake(FakeSimulation([frame(0.1)]), RaisingPolicy())
        self.assertEqual(caught.exception.result.failure.stage, "policy_decision")
        self.assertEqual(caught.exception.result.failure.message, "policy failed")

    def test_keyboard_interrupt_is_not_a_zero_score(self):
        class InterruptedPolicy:
            def act(self, step):
                raise KeyboardInterrupt()

        with self.assertRaises(EpisodeInterrupted) as caught:
            self.run_fake(FakeSimulation([frame(0.1)]), InterruptedPolicy())
        self.assertEqual(caught.exception.result.status, "interrupted")
        self.assertIsNone(caught.exception.result.score)
        self.assertEqual(caught.exception.result.ticks, 1)

    @unittest.skipUnless(os.environ.get("BENCHMARK_INTEGRATION") == "1", "opt-in real-engine probe")
    def test_real_engine_headless_bootstrap_and_policy_call(self):
        result = run_episode(self.case, create_random_policy, {}, max_steps=2)
        self.assertEqual(result.status, "truncated")
        self.assertEqual(result.ticks, 2)
        self.assertEqual(result.timings.policy_calls, 1)
        self.assertEqual(result.initial_agents, 5)
        self.assertGreater(result.final_agents, 0)


if __name__ == "__main__":
    unittest.main()
