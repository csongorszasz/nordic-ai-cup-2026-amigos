import copy
import math
import unittest

import numpy as np
import torch

from src.policies.config import ModelConfig
from src.policies.features import encode_step
from src.policies.memory import PolicyMemory
from src.policies.neural import NeuralPolicy
from src.policies.networks import PolicyNetwork
from src.utils.DTOs import ActionRequest, ObservationResponse, StepResponse


def setUpModule():
    global _previous_threads
    _previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(_previous_threads)


def action(agent_id, distance=3.0, spawn=False):
    return ActionRequest(
        agent_id=agent_id, move_distance=distance, move_direction=0.2,
        turn_angle=-0.1, spawn_agent=spawn,
    )


def agent(agent_id=0, **changes):
    values = dict(
        agent_id=agent_id, energy=150.0, age=1.0, biome="forest", speed=10.0,
        sprint_speed=20.0, hearing_radius=50.0, vision_range=200.0,
        vision_angle=math.pi / 3, max_energy=500.0, observations=[],
    )
    values.update(changes)
    return ObservationResponse(**values)


def frame(agents):
    return StepResponse(
        game_status="ok", score=1.0, sim_time=1.0, n_agents=len(agents), agent_status=agents,
    )


class MemoryTests(unittest.TestCase):
    def test_identity_permutations_births_and_deaths_preserve_only_living_state(self):
        memory = PolicyMemory()
        initial = memory.prepare([3, 8], 4, "cpu")
        self.assertTrue(torch.equal(initial, torch.zeros(2, 4)))
        hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4).requires_grad_()
        memory.commit([3, 8], hidden, [action(8, spawn=True), action(3)])
        prepared = memory.prepare([8, 13, 3], 4, torch.device("cpu"))
        torch.testing.assert_close(prepared, torch.stack((hidden[1], torch.zeros(4), hidden[0])))
        self.assertEqual(set(memory.previous_actions), {3, 8})
        self.assertTrue(memory.previous_actions[8].spawn_agent)
        memory.prepare([13, 8], 4, "cpu")
        self.assertEqual(set(memory._hidden), {13, 8})
        self.assertEqual(set(memory.previous_actions), {8})
        reborn = memory.prepare([3, 13], 4, "cpu")
        self.assertTrue(torch.equal(reborn, torch.zeros(2, 4)))
        self.assertEqual(memory.previous_actions, {})
        self.assertEqual(set(memory._hidden), {3, 13})

    def test_snapshots_and_commits_are_detached_and_do_not_alias_callers(self):
        memory = PolicyMemory()
        initial = memory.prepare([7], 4, "cpu")
        initial.fill_(100)
        self.assertTrue(torch.equal(memory.prepare([7], 4, "cpu"), torch.zeros(1, 4)))
        source = (torch.arange(4, dtype=torch.float64, requires_grad=True) * 2).unsqueeze(0)
        actions = [action(7)]
        memory.commit([7], source, actions)
        saved = memory.prepare([7], 4, "cpu")
        self.assertEqual(saved.dtype, torch.float64)
        self.assertFalse(saved.requires_grad)
        self.assertIsNone(saved.grad_fn)
        self.assertIsNone(memory._hidden[7].grad_fn)
        self.assertIsNone(memory._hidden[7]._base)
        expected = saved.clone()
        with torch.no_grad():
            source.fill_(999)
        actions[0].move_distance = 999
        saved.fill_(-999)
        torch.testing.assert_close(memory.prepare([7], 4, "cpu"), expected)
        self.assertEqual(memory.previous_actions[7].move_distance, 3)
        memory.commit([7], torch.ones(1, 4, dtype=torch.float64), [action(7, 1)])
        torch.testing.assert_close(expected, torch.tensor([[0, 2, 4, 6]], dtype=torch.float64))

    def test_failed_validation_is_atomic_for_prepare_and_commit(self):
        memory = PolicyMemory()
        hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        memory.commit([3, 8], hidden, [action(3), action(8)])
        previous = copy.deepcopy(memory.previous_actions)
        cases = (
            ([3, 3], hidden, [action(3), action(8)]),
            ([3], hidden, [action(3)]),
            ([3, 8], torch.zeros(2, 5), [action(3), action(8)]),
            ([3, 8], torch.full((2, 4), float("nan")), [action(3), action(8)]),
            ([3, 8], torch.zeros(2, 4, dtype=torch.int64), [action(3), action(8)]),
            ([3, 8], hidden, [action(3), action(3)]),
            ([3, 8], hidden, [action(3)]),
            ([3, 8], hidden, [action(3), action(9)]),
            ([10, 11], hidden, [action(10), action(11, float("nan"))]),
        )
        for ids, state, actions in cases:
            with self.subTest(ids=ids, actions=actions), self.assertRaises(ValueError):
                memory.commit(ids, state, actions)
            self.assertEqual(set(memory._hidden), {3, 8})
            self.assertEqual(memory.previous_actions, previous)
            torch.testing.assert_close(memory.prepare([3, 8], 4, "cpu"), hidden)
        for ids, size in (([3, 3], 4), ([-1, 8], 4), ([True, 8], 4),
                          ([3, 8], 0), ([3, 8], 5), ([3, 8], True)):
            with self.subTest(ids=ids, size=size), self.assertRaises(ValueError):
                memory.prepare(ids, size, "cpu")
            self.assertEqual(memory.previous_actions, previous)
            torch.testing.assert_close(memory.prepare([3, 8], 4, "cpu"), hidden)

    def test_empty_population_and_reset_discard_all_state(self):
        memory = PolicyMemory()
        memory.commit([7], torch.ones(1, 4), [action(7)])
        empty = memory.prepare([], 4, "cpu")
        self.assertEqual(empty.shape, (0, 4))
        self.assertEqual(memory.previous_actions, {})
        self.assertEqual(memory._hidden, {})
        memory.commit([], torch.empty(0, 4), [])
        memory.commit([7], torch.ones(1, 4), [action(7)])
        memory.reset()
        memory.reset()
        self.assertEqual(memory.previous_actions, {})
        self.assertTrue(torch.equal(memory.prepare([7], 4, "cpu"), torch.zeros(1, 4)))

    def test_inference_state_can_be_prepared_for_training_without_retaining_graphs(self):
        memory = PolicyMemory()
        with torch.inference_mode():
            memory.commit([7], torch.ones(1, 4), [action(7)])
        prepared = memory.prepare([7], 4, "cpu")
        layer = torch.nn.Linear(4, 1)
        layer(prepared).sum().backward()
        self.assertTrue(torch.isfinite(layer.weight.grad).all().item())
        self.assertFalse(prepared.requires_grad)


class NeuralPolicyTests(unittest.TestCase):
    def setUp(self):
        self.rng_context = torch.random.fork_rng(devices=[])
        self.rng_context.__enter__()
        torch.manual_seed(619)
        self.network = PolicyNetwork(ModelConfig(
            encoder="attention", hidden_size=16, entity_size=8, entity_chunk_size=2,
        ))

    def tearDown(self):
        self.rng_context.__exit__(None, None, None)

    def test_deterministic_actions_are_identity_invariant_through_population_changes(self):
        first = NeuralPolicy(self.network, 4)
        second = NeuralPolicy(copy.deepcopy(self.network), 999)
        observations = [
            {"type": "Fruit", "distance": 10, "angle": 0.2},
            {"type": "Fruit", "distance": 20, "angle": -0.1},
            {"type": "Predator", "distance": 80, "angle": 1, "rel_dir": 0.5},
            {"type": "Edge", "coords": [[20, -20], [20, 20]]},
        ]
        for ids in ([19, 2, 77], [77, 5, 19], [5, 19]):
            agents = [agent(value, observations=observations if value == 19 else [],
                            energy=80 if value == 77 else 190) for value in ids]
            original = first.act(frame(agents))
            permuted_agents = [
                value.model_copy(update={"observations": list(reversed(value.observations))})
                for value in reversed(agents)
            ]
            permuted = second.act(frame(permuted_agents))
            self.assertEqual([value.agent_id for value in original], list(ids))
            self.assertEqual([value.agent_id for value in permuted], list(reversed(ids)))
            self.assertEqual({value.agent_id: value for value in original},
                             {value.agent_id: value for value in permuted})
            self.assertEqual(set(first.memory.previous_actions), set(ids))
            self.assertEqual(set(first.memory._hidden), set(ids))
            for value in ids:
                torch.testing.assert_close(first.memory._hidden[value], second.memory._hidden[value])

    def test_inference_uses_previous_executed_actions_and_explicit_initial_hidden(self):
        with torch.no_grad():
            self.network.mean_head.weight.zero_()
            self.network.mean_head.bias.zero_()
            self.network.spawn_head.weight.zero_()
            self.network.spawn_head.bias.fill_(100)
        policy = NeuralPolicy(self.network, 4)
        inputs, outputs = [], []

        def capture_input(module, args):
            inputs.append((args[0], args[1].clone(), torch.is_grad_enabled(),
                           torch.is_inference_mode_enabled(), module.training))

        def capture_output(module, args, output):
            outputs.append(output.next_hidden.clone())

        before = self.network.register_forward_pre_hook(capture_input)
        after = self.network.register_forward_hook(capture_output)
        try:
            actions = policy.act(frame([agent(7, energy=100.5)]))
            executed = actions[0].model_copy()
            self.assertFalse(executed.spawn_agent)
            actions[0].move_distance = 999
            next_step = frame([agent(7, energy=150)])
            policy.act(next_step)
        finally:
            before.remove()
            after.remove()
        np.testing.assert_array_equal(
            inputs[1][0].scalars, encode_step(next_step, {7: executed}).scalars,
        )
        self.assertEqual(inputs[1][0].scalars[0, -1], 0)
        self.assertTrue(torch.equal(inputs[0][1], torch.zeros(1, 16)))
        torch.testing.assert_close(inputs[1][1], outputs[0])
        for _, _, gradients, inference, training in inputs:
            self.assertFalse(gradients)
            self.assertTrue(inference)
            self.assertFalse(training)
        self.assertFalse(policy.memory._hidden[7].requires_grad)
        self.assertIsNone(policy.memory._hidden[7].grad_fn)

    def test_private_rng_is_reproducible_and_reset_restarts_the_session(self):
        first = NeuralPolicy(self.network, 17, deterministic=False)
        second = NeuralPolicy(copy.deepcopy(self.network), 17, deterministic=False)
        different = NeuralPolicy(copy.deepcopy(self.network), 18, deterministic=False)
        step = frame([agent(9), agent(2)])
        global_state = torch.random.get_rng_state().clone()
        first_actions = first.act(step)
        self.assertEqual(first_actions, second.act(step))
        self.assertNotEqual(first_actions, different.act(step))
        self.assertTrue(torch.equal(global_state, torch.random.get_rng_state()))
        first.act(step)
        first.reset()
        self.assertEqual(first.memory.previous_actions, {})
        self.assertEqual(first.act(step), first_actions)

    def test_deterministic_calls_do_not_advance_private_or_global_rng(self):
        policy = NeuralPolicy(self.network, 19)
        generator = policy._generator(torch.device("cpu"))
        private = generator.get_state().clone()
        global_state = torch.random.get_rng_state().clone()
        policy.act(frame([agent(7)]))
        self.assertTrue(torch.equal(private, generator.get_state()))
        self.assertTrue(torch.equal(global_state, torch.random.get_rng_state()))

    def test_bootstrap_and_extinction_reset_memory_but_invalid_frames_raise(self):
        policy = NeuralPolicy(self.network, 19)
        step = frame([agent(7)])
        first = policy.act(step)
        invalid = StepResponse(game_status="ok", score=1, sim_time=1, n_agents=2, agent_status=[])
        with self.assertRaises(ValueError):
            policy.act(invalid)
        self.assertEqual(set(policy.memory.previous_actions), {7})
        bootstrap = StepResponse(game_status="ok", score=0, sim_time=0, n_agents=7, agent_status=[])
        self.assertEqual(policy.act(bootstrap), [])
        self.assertEqual(policy.memory.previous_actions, {})
        self.assertEqual(policy.act(step), first)
        empty = StepResponse(game_status="game_over", score=4, sim_time=5, n_agents=0, agent_status=[])
        self.assertEqual(policy.act(empty), [])
        self.assertEqual(policy.memory._hidden, {})

    def test_all_agents_are_returned_without_a_population_cap(self):
        network = PolicyNetwork(ModelConfig(memory="none", hidden_size=16, entity_size=8, critic="local"))
        policy = NeuralPolicy(network, 7)
        ids = [10_000 + 7 * index for index in reversed(range(1025))]
        actions = policy.act(frame([agent(value) for value in ids]))
        self.assertEqual([value.agent_id for value in actions], ids)
        self.assertEqual(len(policy.memory.previous_actions), 1025)
        policy.act(frame([agent(ids[0]), agent(ids[-1])]))
        self.assertEqual(set(policy.memory.previous_actions), {ids[0], ids[-1]})
        self.assertEqual(set(policy.memory._hidden), {ids[0], ids[-1]})


if __name__ == "__main__":
    unittest.main()
