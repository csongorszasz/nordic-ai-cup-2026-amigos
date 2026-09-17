import copy
import io
import json
import math
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from src.policies.actions import evaluate_actions
from src.policies.config import ExperimentConfig
from src.policies.networks import PolicyNetwork, PolicyOutput
from src.training.imitation import ImitationDataset
from src.training.learner import MAX_EVENT_BYTES, run_learning
from src.training.ppo import (
    RunningReturnStats, clipped_policy_objective, generalized_advantage_estimate,
    scaled_value_loss, teacher_regularization_weight, train_ppo,
)
from src.training.rollout import Rollout, RolloutCollector
from tests.test_training_rollout import (
    ConstantTeacher, EntityEnvironment, ScriptedEnvironment, TinyPolicy, make_config,
)


class AdvantageTests(unittest.TestCase):
    def test_explicit_bootstrapped_and_terminal_monte_carlo_numbers(self):
        advantage, returns = generalized_advantage_estimate(
            [1, 2, 3], [10, 20, 30], [False, False, False], 40, gamma=1, lam=1,
        )
        torch.testing.assert_close(advantage, torch.tensor([36, 25, 13], dtype=torch.float64))
        torch.testing.assert_close(returns, torch.tensor([46, 45, 43], dtype=torch.float64))
        advantage, returns = generalized_advantage_estimate(
            [1, 2, 3], [10, 20, 30], [False, False, True], 999, gamma=1, lam=1,
        )
        torch.testing.assert_close(advantage, torch.tensor([-4, -15, -27], dtype=torch.float64))
        torch.testing.assert_close(returns, torch.tensor([6, 5, 3], dtype=torch.float64))

    def test_lambda_and_discount_are_not_ignored(self):
        advantage, returns = generalized_advantage_estimate(
            [1, 2, 3], [10, 20, 30], [False] * 3, 40, gamma=0.5, lam=0,
        )
        torch.testing.assert_close(advantage, torch.tensor([1, -3, -7], dtype=torch.float64))
        torch.testing.assert_close(returns, torch.tensor([11, 17, 23], dtype=torch.float64))
        advantage, _ = generalized_advantage_estimate(
            [1, 2, 3], [10, 20, 30], [False] * 3, 40, gamma=1, lam=0.5,
        )
        torch.testing.assert_close(advantage, torch.tensor([20.25, 18.5, 13], dtype=torch.float64))

    def test_terminal_prevents_cross_episode_leakage_and_negative_scores_are_valid(self):
        advantage, returns = generalized_advantage_estimate(
            [1, -10, 100, -2], [4, 5, 6, 7], [False, True, False, True], 999, gamma=1, lam=1,
        )
        torch.testing.assert_close(returns, torch.tensor([-9, -10, 98, -2], dtype=torch.float64))
        torch.testing.assert_close(advantage, torch.tensor([-13, -15, 92, -9], dtype=torch.float64))

    def test_invalid_empty_mismatched_nonfinite_or_nonboolean_inputs_fail(self):
        for rewards, values, terminals in (
            ([], [], []), ([1], [1, 2], [False]), ([float("nan")], [1], [True]),
            ([1], [float("inf")], [False]), ([1], [1], [0]),
        ):
            with self.subTest(rewards=rewards, values=values, terminals=terminals):
                with self.assertRaises((ValueError, FloatingPointError)):
                    generalized_advantage_estimate(rewards, values, terminals, 0)


class PolicyObjectiveTests(unittest.TestCase):
    def test_positive_and_negative_advantages_clip_the_correct_ratio_side(self):
        log_probability = torch.tensor([1.5, 0.5, 1.0]).log().requires_grad_()
        old = torch.zeros(3)
        loss, kl, fraction = clipped_policy_objective(log_probability, old, 2.0)
        self.assertAlmostEqual(float(loss.detach()), -1.08, places=6)
        self.assertAlmostEqual(float(fraction), 2 / 3, places=6)
        self.assertGreater(float(kl.detach()), 0)
        loss.backward()
        torch.testing.assert_close(log_probability.grad, torch.tensor([0, -0.2, -0.4]))
        negative = log_probability.detach().clone().requires_grad_()
        loss, _, _ = clipped_policy_objective(negative, old, -2.0)
        self.assertAlmostEqual(float(loss.detach()), 1.32, places=6)
        loss.backward()
        torch.testing.assert_close(negative.grad, torch.tensor([0.6, 0, 0.4]))

    def test_population_does_not_change_the_actor_divisor(self):
        single, _, _ = clipped_policy_objective(torch.zeros(1), torch.zeros(1), 1.0)
        team, _, _ = clipped_policy_objective(torch.zeros(5), torch.zeros(5), 1.0)
        self.assertAlmostEqual(float(team), 5 * float(single), places=6)
        self.assertAlmostEqual(float(team), -1.0)
        with self.assertRaisesRegex(FloatingPointError, "importance ratio"):
            clipped_policy_objective(torch.tensor([1000.0]), torch.zeros(1), -1)

    def test_hybrid_reproduction_mask_is_fixed_to_the_collected_action(self):
        latent, spawn = torch.zeros((2, 3)), torch.tensor([0.0, 1.0])
        eligible = torch.tensor([False, True])
        old = PolicyOutput(
            torch.zeros((2, 3)), torch.zeros((2, 3)), torch.zeros(2),
            torch.tensor(0.0), torch.zeros((2, 16)),
        )
        logits = torch.tensor([100.0, 2.0], requires_grad=True)
        new = PolicyOutput(old.mean, old.log_std, logits, old.value, old.next_hidden)
        old_logp, old_entropy = evaluate_actions(old, latent, spawn, eligible)
        logp, entropy = evaluate_actions(new, latent, spawn, eligible)
        self.assertEqual(float(logp[0].detach()), float(old_logp[0]))
        self.assertEqual(float(entropy[0].detach()), float(old_entropy[0]))
        self.assertGreater(float(logp[1].detach()), float(old_logp[1]))
        logp.sum().backward()
        self.assertEqual(float(logits.grad[0]), 0.0)
        self.assertGreater(float(logits.grad[1]), 0.0)
        with self.assertRaisesRegex(ValueError, "ineligible"):
            evaluate_actions(new, latent, torch.ones(2), eligible)

    def test_value_normalization_scales_raw_errors_without_recentering_or_clipping(self):
        stats = RunningReturnStats()
        raw_returns = torch.tensor([100.0, 200.0])
        stats.update(raw_returns)
        self.assertEqual(stats.mean, 150.0)
        self.assertEqual(stats.scale, 50.0)
        value = torch.tensor(130.0, requires_grad=True)
        loss = scaled_value_loss(value, 150.0, stats.scale)
        self.assertAlmostEqual(float(loss.detach()), 0.08, places=6)
        loss.backward()
        self.assertAlmostEqual(float(value.grad), -20 / 2500, places=7)
        stats.update([-100, -200])
        self.assertEqual(stats.mean, 0.0)
        self.assertAlmostEqual(stats.scale, math.sqrt(25000))
        self.assertEqual(float(value.detach()), 130.0)
        torch.testing.assert_close(raw_returns, torch.tensor([100.0, 200.0]))
        restored = RunningReturnStats()
        restored.load_state_dict(stats.state_dict())
        self.assertEqual(restored.state_dict(), stats.state_dict())
        floor = RunningReturnStats()
        floor.update([999, 999])
        self.assertEqual(floor.scale, 1.0)
        self.assertAlmostEqual(float(scaled_value_loss(torch.tensor(-100.0), -200.0, floor.scale)), 5000)

    def test_teacher_regularization_reaches_zero_at_target_update(self):
        config = make_config(updates=3)
        self.assertEqual([teacher_regularization_weight(config, update) for update in range(3)], [0.1, 0.05, 0])
        self.assertEqual(teacher_regularization_weight(make_config(updates=1), 0), 0)


class PPOUpdateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    @staticmethod
    def collect(config, network, *, teacher_probability=0, environment_factory=ScriptedEnvironment):
        with RolloutCollector(
            config, network, emit=lambda event: None, environment_factory=environment_factory,
            teacher_factory=ConstantTeacher,
        ) as collector:
            return collector.collect(config.rollout_steps, policy_version=0, teacher_probability=teacher_probability)

    def test_population_deaths_do_not_cut_team_gae_and_horizon_does_not_bootstrap(self):
        config = make_config()
        network = TinyPolicy(config.model)
        rollout = self.collect(config, network)
        frames = rollout.trajectories[0]
        _, returns = generalized_advantage_estimate(
            [frame.reward for frame in frames], [frame.value for frame in frames],
            [frame.terminated for frame in frames], rollout.last_values[0], gamma=1, lam=1,
        )
        torch.testing.assert_close(returns, torch.tensor([-10, -8, -11], dtype=torch.float64))
        horizon = make_config(steps=1, sequence_length=1)
        rollout = self.collect(
            horizon, network,
            environment_factory=lambda: ScriptedEnvironment(((0, 1), (0, 1)), (-3.0,)),
        )
        self.assertTrue(rollout.trajectories[0][0].terminated)
        self.assertEqual(rollout.last_values, (0.0,))
        self.assertEqual(len(rollout.trajectories[0][0].features.agent_ids), 2)

    def test_recurrent_ppo_update_is_finite_and_changes_actor_and_critic_weights(self):
        config = make_config()
        network = TinyPolicy(config.model)
        rollout = self.collect(config, network)
        weights = {name: parameter.detach().clone() for name, parameter in network.named_parameters()}
        optimizer = torch.optim.Adam(network.parameters(), lr=0.005)
        stats = RunningReturnStats()
        result = train_ppo(network, optimizer, rollout, config, update=0, value_normalizer=stats)
        self.assertEqual(result["optimizer_steps"], 2)
        self.assertEqual(stats.count, 3)
        self.assertFalse(torch.equal(weights["action_bias"], network.action_bias))
        self.assertFalse(torch.equal(weights["value_bias"], network.value_bias))
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in network.parameters()))
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["raw_reward_sum"], -10.0)

    def test_target_kl_stops_remaining_epochs_without_silent_invalid_batch_skips(self):
        config = make_config(
            updates=1, ppo={"sequence_length": 2, "epochs": 5, "target_kl": 1e-6},
        )
        network = TinyPolicy(config.model)
        rollout = self.collect(config, network)
        result = train_ppo(
            network, torch.optim.Adam(network.parameters(), lr=0.5), rollout, config,
            update=0, value_normalizer=RunningReturnStats(),
        )
        self.assertTrue(result["early_stopped"])
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertGreater(result["kl_at_stop"], config.ppo.target_kl)

    def test_teacher_mixed_rollout_is_rejected_before_any_ppo_step(self):
        config = make_config()
        network = TinyPolicy(config.model)
        rollout = self.collect(config, network, teacher_probability=1)
        optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
        stats = RunningReturnStats()
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with self.assertRaisesRegex(ValueError, "on-policy"):
                train_ppo(network, optimizer, rollout, config, update=0, value_normalizer=stats)
            step.assert_not_called()
        self.assertEqual(stats.count, 0)

    def test_missing_team_terminal_or_mixed_environment_order_cannot_leak_gae(self):
        config = make_config()
        network = TinyPolicy(config.model)
        rollout = self.collect(config, network)
        frames = rollout.trajectories[0]
        for changed in (replace(frames[1], episode_start=True), replace(frames[1], env_index=1)):
            invalid = replace(rollout, trajectories=((frames[0], changed, frames[2]),))
            with self.subTest(frame=changed), self.assertRaisesRegex(ValueError, "chronological"):
                train_ppo(
                    network, torch.optim.SGD(network.parameters(), lr=0.01), invalid,
                    config, update=0, value_normalizer=RunningReturnStats(),
                )

    def test_real_policy_variants_train_on_dynamic_entity_observations(self):
        for encoder, memory, critic in (("pool", "none", "local"), ("attention", "gru", "team")):
            with self.subTest(encoder=encoder, memory=memory, critic=critic):
                torch.manual_seed(51)
                config = make_config(
                    updates=1,
                    model={
                        "hidden_size": 16, "entity_size": 8, "encoder": encoder, "memory": memory,
                        "critic": critic, "entity_chunk_size": 2,
                    },
                    ppo={"sequence_length": 2, "epochs": 1, "target_kl": 1.0},
                )
                network = PolicyNetwork(config.model)
                rollout = self.collect(config, network, environment_factory=EntityEnvironment)
                before = network.mean_head.weight.detach().clone()
                result = train_ppo(
                    network, torch.optim.Adam(network.parameters(), lr=0.001), rollout,
                    config, update=0, value_normalizer=RunningReturnStats(),
                )
                self.assertEqual(result["optimizer_steps"], 1)
                self.assertFalse(torch.equal(before, network.mean_head.weight))
                self.assertTrue(any(
                    parameter.grad is not None and parameter.grad.abs().sum() > 0
                    for parameter in network.entity_encoders.parameters()
                ))
                json.dumps(result, allow_nan=False)

    def test_variable_token_chunks_preserve_equal_team_tick_gradient_weight(self):
        config = make_config(
            model={"hidden_size": 16, "entity_size": 8, "memory": "none"},
            ppo={"sequence_length": 3, "epochs": 1, "target_kl": 1.0},
        )
        network = TinyPolicy(config.model)
        clone = copy.deepcopy(network)
        rollout = self.collect(config, network)
        short = config.model_copy(update={"ppo": config.ppo.model_copy(update={"sequence_length": 2})})
        train_ppo(
            network, torch.optim.SGD(network.parameters(), lr=0.01), rollout, config,
            update=0, value_normalizer=RunningReturnStats(),
        )
        train_ppo(
            clone, torch.optim.SGD(clone.parameters(), lr=0.01), rollout, short,
            update=0, value_normalizer=RunningReturnStats(),
        )
        for full, split in zip(network.parameters(), clone.parameters(), strict=True):
            torch.testing.assert_close(full, split, atol=1e-7, rtol=1e-6)


class InterruptedAfterCheckpoint(Exception):
    pass


class LearningOrchestrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.environments = []

    def collector_factory(self, config, network, *, emit, state):
        environment = ScriptedEnvironment(
            ((0, 1), (1, 2)), (-2.0, 1.0), terminal_at_end=False,
        )
        self.environments.append(environment)
        return RolloutCollector(
            config, network, emit=emit, state=state, environment_factory=lambda: environment,
            teacher_factory=ConstantTeacher,
        )

    def test_both_modes_resume_target_total_counters_rng_and_replay_after_a_successful_update(self):
        for mode in ("ppo", "imitation"):
            with self.subTest(mode=mode):
                config = make_config(
                    mode, updates=2, steps=2,
                    ppo={"sequence_length": 2, "epochs": 1, "target_kl": 1.0},
                    imitation={"epochs": 1, "max_frames": 6, "dagger_rounds": 1},
                )
                network = TinyPolicy(config.model)
                optimizer = torch.optim.Adam(network.parameters(), lr=0.005)
                snapshots, events = [], []

                def interrupt(state):
                    snapshots.append((
                        state, copy.deepcopy(network.state_dict()), copy.deepcopy(optimizer.state_dict()),
                    ))
                    raise InterruptedAfterCheckpoint

                with patch("src.training.learner.RolloutCollector", side_effect=self.collector_factory):
                    with self.assertRaises(InterruptedAfterCheckpoint):
                        run_learning(config, network, optimizer, emit=events.append, checkpoint=interrupt)
                state, weights, optimizer_state = snapshots[0]
                self.assertEqual(state["next_update"], 1)
                self.assertEqual(state["collector"]["frame_count"], 2)
                first_seed = self.environments[-1].seed
                stream = io.BytesIO()
                torch.save(state, stream)
                stream.seek(0)
                saved_state = torch.load(stream, weights_only=True)
                resumed = TinyPolicy(config.model)
                resumed.load_state_dict(weights)
                resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=0.005)
                resumed_optimizer.load_state_dict(optimizer_state)
                checkpoints = []
                dataset_snapshots = []
                with patch("src.training.learner.RolloutCollector", side_effect=self.collector_factory):
                    result = run_learning(
                        config, resumed, resumed_optimizer, emit=events.append, checkpoint=checkpoints.append,
                        start_update=1, training_state=saved_state,
                        dataset_callback=dataset_snapshots.append,
                    )
                self.assertEqual(len(checkpoints), 1)
                self.assertEqual(checkpoints[0]["next_update"], 2)
                self.assertEqual(result["updates_completed_this_run"], 1)
                self.assertEqual(result["collection_frames"], 4)
                self.assertEqual(result["optimizer_steps"], 2)
                self.assertEqual(result["seed_index"], 2)
                self.assertNotEqual(self.environments[-1].seed, first_seed)
                self.assertTrue(result["environments_reset_on_resume"])
                self.assertTrue(any(event["event"] == "environments_reset_on_resume" for event in events))
                self.assertEqual([event["update"] for event in events if event["event"] == "learning_update"], [2])
                if mode == "imitation":
                    self.assertEqual(checkpoints[0]["imitation_dataset"]["total_seen"], 4)
                    self.assertEqual(result["last_update"]["teacher_probability"], 0)
                    self.assertGreater(result["last_update"]["recovery_frames"], 0)
                    self.assertEqual(len(dataset_snapshots), 1)
                    self.assertEqual(dataset_snapshots[0]["next_update"], 2)
                    self.assertEqual(dataset_snapshots[0]["total_seen"], 4)
                    json.dumps(dataset_snapshots[0], allow_nan=False)
                else:
                    self.assertEqual(checkpoints[0]["value_normalizer"]["count"], 4)
                    self.assertEqual(dataset_snapshots, [])
                self.assertNotIn("imitation_dataset", result)
                for event in events:
                    self.assertLessEqual(len(json.dumps(event, allow_nan=False).encode()), MAX_EVENT_BYTES)

    def test_json_dataset_callback_follows_each_checkpoint_without_polluting_metrics(self):
        config = make_config(
            "imitation", updates=2, steps=2,
            imitation={"epochs": 1, "max_frames": 3, "dagger_rounds": 1},
        )
        network = TinyPolicy(config.model)
        optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
        calls, snapshots, events = [], [], []

        def checkpoint(state):
            calls.append(("checkpoint", state["next_update"]))

        def snapshot(payload):
            calls.append(("dataset", payload["next_update"]))
            snapshots.append(json.loads(json.dumps(payload, allow_nan=False)))

        with patch("src.training.learner.RolloutCollector", side_effect=self.collector_factory):
            result = run_learning(
                config, network, optimizer, emit=events.append, checkpoint=checkpoint,
                dataset_callback=snapshot,
            )
        self.assertEqual(calls, [("checkpoint", 1), ("dataset", 1), ("checkpoint", 2), ("dataset", 2)])
        self.assertEqual([len(value["frames"]) for value in snapshots], [2, 3])
        self.assertEqual(snapshots[-1]["evicted_frames"], 1)
        self.assertEqual(snapshots[-1]["experiment_config"], config.model_dump(mode="json"))
        self.assertEqual(snapshots[-1]["collection_seed_index"], result["seed_index"])
        rebuilt = ImitationDataset(3)
        rebuilt.load_json_snapshot(snapshots[-1])
        self.assertEqual(rebuilt.total_seen, 4)
        self.assertGreater(rebuilt.summary()["recovery_frames"], 0)
        self.assertNotIn("frames", result)
        self.assertTrue(all("experiment_config" not in event for event in events))

    def test_omitting_dataset_callback_does_not_materialize_json_replay(self):
        config = make_config("imitation", updates=1, steps=1, imitation={"epochs": 1})
        network = TinyPolicy(config.model)
        with patch("src.training.learner.RolloutCollector", side_effect=self.collector_factory):
            with patch.object(ImitationDataset, "json_snapshot", side_effect=AssertionError("Unexpected JSON work")):
                result = run_learning(
                    config, network, torch.optim.SGD(network.parameters(), lr=0.01),
                    emit=lambda event: None, checkpoint=lambda state: None,
                )
        self.assertEqual(result["next_update"], 1)

    def test_dataset_writer_failure_propagates_and_checkpoint_can_regenerate_replay(self):
        config = make_config("imitation", updates=1, steps=1, imitation={"epochs": 1, "max_frames": 3})
        network = TinyPolicy(config.model)
        checkpoints = []

        def failing_writer(snapshot):
            raise OSError("Dataset output is unavailable.")

        with patch("src.training.learner.RolloutCollector", side_effect=self.collector_factory):
            with self.assertRaisesRegex(OSError, "Dataset output"):
                run_learning(
                    config, network, torch.optim.SGD(network.parameters(), lr=0.01),
                    emit=lambda event: None, checkpoint=checkpoints.append,
                    dataset_callback=failing_writer,
                )
        self.assertEqual(checkpoints[0]["next_update"], 1)
        dataset = ImitationDataset(3)
        dataset.load_state_dict(checkpoints[0]["imitation_dataset"])
        self.assertEqual(len(json.loads(json.dumps(dataset.json_snapshot()))["frames"]), 1)

    def test_mode_update_and_resume_mismatches_fail_before_opening_environments(self):
        config = make_config(updates=2)
        network = TinyPolicy(config.model)
        optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
        with patch("src.training.learner.RolloutCollector") as collector:
            for update, state in ((-1, None), (2, None), (True, None), (1, None), (1, {"next_update": 0})):
                with self.subTest(update=update, state=state), self.assertRaises(ValueError):
                    run_learning(
                        config, network, optimizer, emit=lambda event: None, checkpoint=lambda state: None,
                        start_update=update, training_state=state,
                    )
            with self.assertRaisesRegex(ValueError, "requires mode"):
                run_learning(
                    ExperimentConfig(mode="search", policy="heuristic"), network, optimizer,
                    emit=lambda event: None, checkpoint=lambda state: None,
                )
            collector.assert_not_called()

    def test_empty_rollout_cannot_checkpoint_a_successful_update(self):
        config = make_config()
        network = TinyPolicy(config.model)
        optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
        checkpoints = []
        with patch("src.training.learner.RolloutCollector", side_effect=self.collector_factory):
            with patch.object(RolloutCollector, "collect", return_value=Rollout(((),), (0.0,), 0, {})):
                with self.assertRaisesRegex(ValueError, "no actionable"):
                    run_learning(config, network, optimizer, emit=lambda event: None, checkpoint=checkpoints.append)
        self.assertEqual(checkpoints, [])


if __name__ == "__main__":
    unittest.main()
