import math
import unittest
import weakref
from dataclasses import replace

import numpy as np
import torch
from torch import nn

from src.policies.actions import sample_actions
from src.policies.config import ExperimentConfig, ModelConfig
from src.policies.features import encode_step
from src.policies.networks import PolicyOutput
from src.training.env import EnvTransition
from src.training.rollout import (
    Rollout, RolloutCollector, RolloutFrame, align_hidden, sequence_chunks, unroll_sequence,
)
from src.utils.DTOs import ActionRequest, ObservationResponse, StepResponse


def make_step(ids=(0, 1), *, tick=1, score=10.0, terminal=False, entities=0, energy=150.0):
    return StepResponse(
        game_status="game_over" if terminal else "ok", score=score, sim_time=tick,
        n_agents=len(ids), agent_status=[
            ObservationResponse(
                agent_id=agent_id, energy=energy, age=20, biome="forest", speed=10,
                sprint_speed=20, hearing_radius=50, vision_range=200,
                vision_angle=math.pi / 3, max_energy=500,
                observations=[
                    {"type": "Fruit", "distance": 10 + index, "angle": 0.1 * index}
                    for index in range(entities)
                ],
            )
            for agent_id in ids
        ],
    )


def make_config(mode="ppo", *, updates=3, steps=3, workers=1, sequence_length=2, **changes):
    values = {
        "mode": mode, "updates": updates, "rollout_steps": steps,
        "model": {"hidden_size": 16, "entity_size": 8, "memory": "gru"},
        "ppo": {"sequence_length": min(sequence_length, steps), "epochs": 2, "target_kl": 1.0},
        "imitation": {"epochs": 2, "max_frames": 12},
        "resources": {"workers": workers, "max_tokens": 64},
    }
    values.update(changes)
    return ExperimentConfig.model_validate(values)


class TinyPolicy(nn.Module):
    """A differentiable actor/critic with identity-sensitive additive recurrence."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.action_bias = nn.Parameter(torch.tensor([0.4, 0.2, -0.2]))
        self.log_std = nn.Parameter(torch.full((3,), -0.5))
        self.spawn_bias = nn.Parameter(torch.tensor(0.2))
        self.value_bias = nn.Parameter(torch.tensor(2.0))
        self.recurrence_gain = nn.Parameter(torch.tensor(0.1))
        self.forward_modes = []

    def forward(self, batch, hidden=None):
        self.forward_modes.append(self.training)
        count = len(batch.agent_ids)
        if hidden is None:
            hidden = self.action_bias.new_zeros((count, self.config.hidden_size))
        previous_distance = torch.as_tensor(
            batch.scalars[:, -6], device=self.action_bias.device,
        )
        next_hidden = hidden + self.recurrence_gain * (1 + previous_distance[:, None])
        return PolicyOutput(
            self.action_bias.expand(count, 3) + 0.01 * next_hidden[:, :3],
            self.log_std.expand(count, 3), self.spawn_bias.expand(count),
            self.value_bias.reshape(()), next_hidden,
        )


class ConstantTeacher:
    def __init__(self, seed=0, config=None):
        self.seed = seed

    def act(self, step):
        return [
            ActionRequest(
                agent_id=agent.agent_id, move_distance=4, move_direction=0.3,
                turn_angle=-0.2, spawn_agent=False,
            )
            for agent in step.agent_status
        ]


class ScriptedEnvironment:
    def __init__(
        self, identities=((0, 1), (1, 2), (2,), ()), rewards=(-2.0, 3.0, -11.0),
        *, terminal_at_end=True, initial_score=10.0,
    ):
        self.identities = identities
        self.rewards = rewards
        self.terminal_at_end = terminal_at_end
        self.initial_score = initial_score
        self.reset_seeds = []
        self.executed = []
        self.position = 0
        self.done = False
        self.observation = None
        self.seed = None

    def reset(self, seed):
        self.seed = seed
        self.reset_seeds.append(seed)
        self.position = 0
        self.done = False
        self.observation = make_step(self.identities[0], score=self.initial_score)
        return self.observation

    def step(self, actions):
        if self.done:
            raise AssertionError("The collector attempted to step a terminal world.")
        if tuple(action.agent_id for action in actions) != tuple(
            agent.agent_id for agent in self.observation.agent_status
        ):
            raise AssertionError("Actions were not aligned to the current living IDs.")
        self.executed.append(tuple(action.model_copy(deep=True) for action in actions))
        reward = self.rewards[min(self.position, len(self.rewards) - 1)]
        self.position += 1
        self.done = self.terminal_at_end and self.position >= len(self.rewards)
        self.observation = make_step(
            self.identities[min(self.position, len(self.identities) - 1)],
            tick=self.position + 1, score=self.observation.score + reward, terminal=self.done,
        )
        return EnvTransition(self.observation, reward, self.done)


class FakePool:
    def __init__(self, workers, **kwargs):
        self.environments = [
            ScriptedEnvironment(
                ((index * 10, index * 10 + 1),) * 2, (1.0,), terminal_at_end=False,
            )
            for index in range(workers)
        ]
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def reset(self, seeds):
        return [environment.reset(seed) for environment, seed in zip(self.environments, seeds, strict=True)]

    def reset_at(self, index, seed):
        return self.environments[index].reset(seed)

    def step(self, batch_actions):
        return [
            environment.step(actions)
            for environment, actions in zip(self.environments, batch_actions, strict=True)
        ]


class EntityEnvironment(ScriptedEnvironment):
    def _add_entities(self):
        for agent in self.observation.agent_status:
            agent.observations = [
                {"type": "Fruit", "distance": 12.0, "angle": 0.1},
                {"type": "Fruit", "distance": 30.0, "angle": -0.2},
                {"type": "Edge", "coords": [[40.0, -20.0], [40.0, 20.0]]},
            ]

    def reset(self, seed):
        super().reset(seed)
        self._add_entities()
        return self.observation

    def step(self, actions):
        transition = super().step(actions)
        self._add_entities()
        return transition


def make_frame(
    network, step=None, *, reward=1.0, terminated=False, episode_step=0, env_index=0,
    world_seed=123, teacher=False, collection_index=None,
):
    step = step or make_step()
    features = encode_step(step)
    hidden = torch.zeros((len(features.agent_ids), network.config.hidden_size))
    with torch.no_grad():
        output = network(features, hidden if network.config.memory == "gru" else None)
        sample = sample_actions(output, step, deterministic=True)
    labels = tuple(ConstantTeacher().act(step))
    return RolloutFrame(
        features, step, hidden, sample.latent, sample.spawn, sample.eligible, sample.log_prob,
        float(output.value.detach()), labels if teacher else tuple(sample.actions), labels,
        torch.full((len(features.agent_ids),), teacher, dtype=torch.bool),
        reward, terminated, episode_step == 0, env_index, world_seed, episode_step,
        episode_step if collection_index is None else collection_index,
    )


class RolloutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = make_config()
        self.network = TinyPolicy(self.config.model)

    def collector(self, environment=None, **kwargs):
        return RolloutCollector(
            self.config, self.network, emit=kwargs.pop("emit", lambda event: None),
            environment_factory=lambda: environment or ScriptedEnvironment(),
            teacher_factory=ConstantTeacher, **kwargs,
        )

    def test_native_reward_is_once_per_team_and_bootstrap_is_not_a_frame(self):
        events = []
        environment = ScriptedEnvironment()
        with self.collector(environment, emit=events.append) as collector:
            rollout = collector.collect(3, policy_version=0, teacher_probability=1)
            frames = rollout.trajectories[0]
            self.assertEqual([frame.reward for frame in frames], [-2.0, 3.0, -11.0])
            self.assertEqual([len(frame.features.agent_ids) for frame in frames], [2, 2, 1])
            self.assertEqual([frame.terminated for frame in frames], [False, False, True])
            self.assertEqual(rollout.last_values, (0.0,))
            self.assertEqual(collector.frame_count, 3)
            self.assertEqual(len(environment.executed), 3)
            self.assertEqual(collector.episode_returns, [0.0])
            self.assertEqual(events[0]["initial_episode_return"], 10.0)
            self.assertFalse(events[0]["bootstrap_is_learned_frame"])
            self.assertEqual(events[-1]["episode_return"], 0.0)
            self.assertEqual(frames[0].step.sim_time, 1)
        self.assertTrue(self.network.training)
        self.assertTrue(all(not mode for mode in self.network.forward_modes))

    def test_actual_mixed_actions_drive_previous_features_and_births_start_empty(self):
        with self.collector() as collector:
            collector.teacher_generators[0].manual_seed(1)
            rollout = collector.collect(2, policy_version=0, teacher_probability=0.5)
            first, second = rollout.trajectories[0]
            actions = {action.agent_id: action for action in first.actions}
            self.assertEqual(second.features.agent_ids, (1, 2))
            self.assertAlmostEqual(second.features.scalars[0, -6], actions[1].move_distance / 40, places=6)
            np.testing.assert_allclose(second.features.scalars[1, -6:], [0, 0, 1, 0, 1, 0])
            self.assertTrue(torch.equal(second.initial_hidden[1], torch.zeros(16)))
            self.assertTrue(torch.all(second.initial_hidden[0] > 0))
            self.assertNotEqual(first.actions[0], first.teacher_actions[0])
            self.assertFalse(first.teacher_mask.all().item())
            for frame in (first, second):
                for tensor in (frame.initial_hidden, frame.latent, frame.old_log_prob):
                    self.assertEqual(tensor.device.type, "cpu")
                    self.assertFalse(tensor.requires_grad)

    def test_teacher_labels_are_matched_by_id_not_by_return_order(self):
        class ReversedTeacher(ConstantTeacher):
            def act(self, step):
                return list(reversed(super().act(step)))

        with RolloutCollector(
            self.config, self.network, emit=lambda event: None,
            environment_factory=ScriptedEnvironment, teacher_factory=ReversedTeacher,
        ) as collector:
            rollout = collector.collect(1, policy_version=0, teacher_probability=1)
            frame = rollout.trajectories[0][0]
            self.assertEqual(tuple(action.agent_id for action in frame.actions), frame.features.agent_ids)
            self.assertEqual(frame.actions, frame.teacher_actions)

    def test_nonterminal_boundary_bootstraps_without_committing_lookahead(self):
        environment = ScriptedEnvironment(((0,), (0,)), (1.0,), terminal_at_end=False)
        with self.collector(environment) as collector:
            first = collector.collect(1, policy_version=0, teacher_probability=1)
            second = collector.collect(1, policy_version=1, teacher_probability=1)
            self.assertEqual(first.last_values, (2.0,))
            self.assertFalse(first.trajectories[0][-1].terminated)
            self.assertFalse(second.trajectories[0][0].episode_start)
            torch.testing.assert_close(second.trajectories[0][0].initial_hidden, torch.full((1, 16), 0.1))
            self.assertEqual(collector.seed_index, 1)

    def test_only_team_terminal_resets_memory_and_consumes_a_new_world_seed(self):
        with self.collector() as collector:
            rollout = collector.collect(4, policy_version=0)
            frames = rollout.trajectories[0]
            self.assertEqual([frame.episode_start for frame in frames], [True, False, False, True])
            self.assertEqual(frames[0].world_seed, frames[2].world_seed)
            self.assertNotEqual(frames[0].world_seed, frames[3].world_seed)
            self.assertTrue(torch.equal(frames[3].initial_hidden, torch.zeros((2, 16))))
            self.assertEqual(frames[3].episode_step, 0)
            self.assertEqual(collector.seed_index, 2)

    def test_parallel_environments_have_independent_identity_memory_and_rng(self):
        config = make_config(workers=2)
        pool = FakePool(2)
        with RolloutCollector(
            config, self.network, emit=lambda event: None, pool_factory=lambda *args, **kwargs: pool,
            teacher_factory=ConstantTeacher,
        ) as collector:
            rollout = collector.collect(2, policy_version=0)
            left, right = rollout.trajectories
            self.assertEqual([len(trajectory) for trajectory in rollout.trajectories], [2, 2])
            self.assertNotEqual(left[0].world_seed, right[0].world_seed)
            self.assertFalse(torch.equal(left[0].latent, right[0].latent))
            self.assertEqual(set(collector.memories[0].previous_actions), {0, 1})
            self.assertEqual(set(collector.memories[1].previous_actions), {10, 11})
            self.assertEqual(sum(frame.reward for trajectory in rollout.trajectories for frame in trajectory), 4)
        self.assertTrue(pool.closed)
        self.assertIsNone(collector.pool)

    def test_teardown_releases_the_in_process_adapter_without_a_close_method(self):
        collector = self.collector()
        with collector:
            reference = weakref.ref(collector.environment)
            collector.collect(1, policy_version=0)
            self.assertIsNotNone(reference())
        self.assertIsNone(collector.environment)
        self.assertIsNone(reference())
        self.assertEqual(collector.state_dict()["frame_count"], 1)

    def test_failed_resume_reporting_closes_the_pool_before_enter_returns(self):
        config = make_config(workers=2)
        initial = RolloutCollector(config, self.network, emit=lambda event: None)
        pool = FakePool(2)

        def failing_emit(event):
            if event["event"] == "environments_reset_on_resume":
                raise OSError("Resume reporting failed.")

        collector = RolloutCollector(
            config, self.network, emit=failing_emit, state=initial.state_dict(),
            pool_factory=lambda *args, **kwargs: pool, teacher_factory=ConstantTeacher,
        )
        with self.assertRaisesRegex(OSError, "Resume reporting"):
            with collector:
                self.fail("Failed __enter__ must not leave an open collector.")
        self.assertTrue(pool.closed)
        self.assertIsNone(collector.pool)
        self.assertIsNone(collector.environment)
        with self.assertRaisesRegex(RuntimeError, "context manager"):
            collector.collect(1, policy_version=0)

    def test_resume_restores_rng_counters_but_explicitly_resets_environments(self):
        with self.collector() as collector:
            collector.collect(1, policy_version=0)
            state = collector.state_dict()
            seed = collector.world_seeds[0]
        events = []
        with self.collector(state=state, emit=events.append) as resumed:
            restored = resumed.state_dict()
            self.assertEqual(restored["frame_count"], 1)
            self.assertEqual(restored["seed_index"], state["seed_index"] + 1)
            self.assertNotEqual(resumed.world_seeds[0], seed)
            self.assertEqual(resumed.memories[0].previous_actions, {})
            for name in ("policy_rng_states", "teacher_rng_states"):
                torch.testing.assert_close(restored[name][0], state[name][0])
            reset_event = next(event for event in events if event["event"] == "environments_reset_on_resume")
            self.assertFalse(reset_event["exact_engine_resume"])
            rollout = resumed.collect(1, policy_version=1)
            self.assertTrue(rollout.trajectories[0][0].episode_start)
            self.assertEqual(rollout.trajectories[0][0].collection_index, 1)

    def test_empty_or_over_budget_frames_fail_instead_of_fabricating_samples(self):
        with self.collector() as collector:
            with self.assertRaisesRegex(ValueError, "positive"):
                collector.collect(0, policy_version=0)
        empty = ScriptedEnvironment(((), ()), (1.0,))
        with self.assertRaisesRegex(ValueError, "bootstrap"):
            with self.collector(empty):
                self.fail("An empty bootstrap cannot become a learned frame.")
        config = make_config(resources={"max_tokens": 1})
        with RolloutCollector(
            config, self.network, emit=lambda event: None,
            environment_factory=ScriptedEnvironment, teacher_factory=ConstantTeacher,
        ) as collector:
            with self.assertRaisesRegex(ValueError, "will not be truncated"):
                collector.collect(1, policy_version=0)
            self.assertEqual(collector.frame_count, 0)

    def test_nonfinite_native_rewards_are_not_skipped_or_replaced(self):
        environment = ScriptedEnvironment(((0,), (0,)), (float("nan"),), terminal_at_end=False)
        with self.collector(environment) as collector:
            with self.assertRaisesRegex(FloatingPointError, "Native team reward"):
                collector.collect(1, policy_version=0)
            self.assertEqual(collector.frame_count, 0)


class RecurrentSequenceTests(unittest.TestCase):
    def test_alignment_preserves_gradient_for_survivors_not_births_or_deaths(self):
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        aligned = align_hidden((10, 20), (20, 30, 10), hidden, hidden_size=2, device="cpu")
        torch.testing.assert_close(aligned, torch.tensor([[3.0, 4.0], [0.0, 0.0], [1.0, 2.0]]))
        aligned[0].sum().backward()
        torch.testing.assert_close(hidden.grad, torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
        reset = align_hidden((10, 20), (20, 10), hidden, hidden_size=2, device="cpu", reset=True)
        self.assertFalse(reset.requires_grad)
        self.assertTrue(torch.equal(reset, torch.zeros((2, 2))))
        with self.assertRaisesRegex(ValueError, "unique"):
            align_hidden((10, 20), (20, 20), hidden, hidden_size=2, device="cpu")

    def test_sequence_unroll_keeps_graph_across_permutation_and_zeros_newborns(self):
        network = TinyPolicy(make_config().model)
        frames = (
            make_frame(network, make_step((1, 2))),
            make_frame(network, make_step((2, 3)), episode_step=1),
        )
        outputs = list(unroll_sequence(network, frames))
        outputs[0][1].next_hidden.retain_grad()
        outputs[-1][1].next_hidden.sum().backward()
        gradient = outputs[0][1].next_hidden.grad
        torch.testing.assert_close(gradient[0], torch.zeros(16))
        torch.testing.assert_close(gradient[1], torch.ones(16))
        torch.testing.assert_close(outputs[1][1].next_hidden[1], torch.full((16,), 0.1))
        self.assertIsNotNone(network.recurrence_gain.grad)

    def test_unroll_resets_reused_ids_and_detaches_initial_segment_boundary(self):
        network = TinyPolicy(make_config().model)
        boundary = torch.ones((1, 16), requires_grad=True)
        first = replace(
            make_frame(network, make_step((1,)), episode_step=3),
            initial_hidden=boundary,
        )
        second = make_frame(network, make_step((1,)), world_seed=999)
        outputs = list(unroll_sequence(network, (first, second)))
        outputs[0][1].next_hidden.retain_grad()
        outputs[1][1].next_hidden.sum().backward()
        self.assertIsNone(boundary.grad)
        self.assertIsNone(outputs[0][1].next_hidden.grad)
        torch.testing.assert_close(outputs[1][1].next_hidden, torch.full((1, 16), 0.1))

    def test_chunking_preserves_all_frames_and_agents_under_both_limits(self):
        network = TinyPolicy(make_config().model)
        frames = tuple(
            make_frame(network, make_step((0, 1), entities=1), episode_step=index)
            for index in range(5)
        )
        chunks = sequence_chunks((frames,), sequence_length=3, max_tokens=8)
        self.assertEqual([len(chunk) for chunk in chunks], [2, 2, 1])
        self.assertEqual([id(frame) for chunk in chunks for frame in chunk], [id(frame) for frame in frames])
        self.assertTrue(all(frame.features.agent_ids == (0, 1) for chunk in chunks for frame in chunk))
        with self.assertRaisesRegex(ValueError, "single frame"):
            sequence_chunks((frames,), sequence_length=3, max_tokens=3)
        with self.assertRaisesRegex(ValueError, "empty"):
            sequence_chunks(((),), sequence_length=3, max_tokens=8)


if __name__ == "__main__":
    unittest.main()
