import copy
import gzip
import io
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import torch

from src.policies.actions import imitation_loss
from src.training.imitation import (
    ImitationDataset, checked_optimizer_step, dagger_schedule, train_imitation,
)
from src.training.rollout import Rollout, RolloutCollector, sequence_chunks, unroll_sequence
from tests.test_training_rollout import (
    ConstantTeacher, ScriptedEnvironment, TinyPolicy, make_config, make_frame, make_step,
)


class ImitationDatasetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = make_config("imitation")
        self.network = TinyPolicy(self.config.model)

    def test_fifo_is_bounded_and_reuses_learner_visited_recovery_states(self):
        dataset = ImitationDataset(3)
        left = tuple(
            make_frame(
                self.network, episode_step=index, env_index=0,
                collection_index=2 * index, teacher=True,
            )
            for index in range(2)
        )
        right = tuple(
            make_frame(
                self.network, episode_step=index, env_index=1,
                collection_index=2 * index + 1, teacher=False,
            )
            for index in range(2)
        )
        dataset.extend(Rollout((left, right), (2.0, 2.0), 0, {}))
        self.assertEqual([frame.collection_index for frame in dataset.frames], [1, 2, 3])
        self.assertEqual(dataset.summary()["evicted_frames"], 1)
        self.assertEqual(dataset.summary()["recovery_frames"], 2)
        dataset.extend(Rollout((
            (make_frame(self.network, episode_step=2, collection_index=4, teacher=False),),
            (make_frame(self.network, episode_step=2, env_index=1, collection_index=5, teacher=True),),
        ), (2.0, 2.0), 1, {}))
        self.assertEqual([frame.collection_index for frame in dataset.frames], [3, 4, 5])
        self.assertEqual(dataset.total_seen, 6)
        self.assertEqual(dataset.evicted_frames, 3)
        self.assertEqual(dataset.summary()["recovery_frames"], 2)
        trajectories = dataset.trajectories()
        self.assertEqual([[frame.collection_index for frame in part] for part in trajectories], [[4], [3, 5]])

    def test_dataset_roundtrip_is_weights_only_safe_and_preserves_actual_history(self):
        config = make_config("imitation", steps=2)
        with RolloutCollector(
            config, self.network, emit=lambda event: None,
            environment_factory=lambda: ScriptedEnvironment(
                ((0,), (0,)), (1.0,), terminal_at_end=False,
            ),
            teacher_factory=ConstantTeacher,
        ) as collector:
            rollout = collector.collect(2, policy_version=0, teacher_probability=0.5)
        dataset = ImitationDataset(4)
        dataset.extend(rollout)
        payload = dataset.state_dict()

        def assert_safe(value):
            self.assertNotIsInstance(value, (np.ndarray, np.generic))
            if isinstance(value, dict):
                for child in value.values():
                    assert_safe(child)
            elif isinstance(value, list):
                for child in value:
                    assert_safe(child)
            else:
                self.assertIsInstance(value, (int, float, bool, str, torch.Tensor))

        assert_safe(payload)
        stream = io.BytesIO()
        torch.save(payload, stream)
        stream.seek(0)
        restored = ImitationDataset(4)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        self.assertEqual(dataset.summary(), restored.summary())
        for original, recovered in zip(dataset.frames, restored.frames, strict=True):
            np.testing.assert_array_equal(original.features.scalars, recovered.features.scalars)
            torch.testing.assert_close(original.initial_hidden, recovered.initial_hidden)
            torch.testing.assert_close(original.teacher_mask, recovered.teacher_mask)
            self.assertEqual(original.actions, recovered.actions)
            self.assertEqual(original.teacher_actions, recovered.teacher_actions)
            self.assertEqual(original.step, recovered.step)

    def test_invalid_counts_and_incompatible_replay_bounds_fail_explicitly(self):
        dataset = ImitationDataset(2)
        dataset.extend(Rollout(((make_frame(self.network),),), (2.0,), 0, {}))
        state = dataset.state_dict()
        state["total_seen"] = 8
        with self.assertRaisesRegex(ValueError, "counts"):
            dataset.load_state_dict(state)
        with self.assertRaisesRegex(ValueError, "max_frames"):
            ImitationDataset(3).load_state_dict(dataset.state_dict())
        with self.assertRaisesRegex(ValueError, "empty"):
            dataset.extend(Rollout(((),), (0.0,), 0, {}))

    def test_replay_lifecycle_flags_and_counters_are_not_coerced(self):
        dataset = ImitationDataset(2)
        dataset.extend(Rollout(((make_frame(self.network),),), (2.0,), 0, {}))
        original = dataset.state_dict()
        for field, value in (
            ("episode_start", "false"), ("terminated", 0), ("env_index", True),
            ("episode_step", -1), ("collection_index", 0.5), ("world_seed", 2**32),
            ("value", float("nan")), ("reward", "1.0"),
        ):
            state = copy.deepcopy(original)
            state["frames"][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                dataset.load_state_dict(state)
            self.assertEqual(len(dataset), 1)
        snapshot = dataset.json_snapshot()
        snapshot["frames"][0]["episode_start"] = "false"
        with self.assertRaisesRegex(ValueError, "episode_start"):
            dataset.load_json_snapshot(snapshot)

    def test_json_snapshot_roundtrip_rebuilds_bounded_replay_including_empty_arrays(self):
        dataset = ImitationDataset(3)
        frames = tuple(
            make_frame(
                self.network, make_step(entities=index % 2), episode_step=index,
                teacher=index % 2 == 0,
            )
            for index in range(4)
        )
        frames = (*frames[:-1], replace(frames[-1], initial_hidden=frames[-1].initial_hidden.double() + 0.1))
        dataset.extend(Rollout((frames,), (2.0,), 0, {}))
        snapshot = dataset.json_snapshot()
        encoded = json.dumps(snapshot, allow_nan=False).encode("utf-8")
        decoded = json.loads(gzip.decompress(gzip.compress(encoded)))
        self.assertEqual(len(decoded["frames"]), 3)
        self.assertEqual(decoded["evicted_frames"], 1)
        restored = ImitationDataset(3)
        restored.load_json_snapshot(decoded)
        self.assertEqual(restored.json_snapshot(), snapshot)
        self.assertEqual(restored.summary(), dataset.summary())
        self.assertEqual(restored.frames[1].features.entities.shape, (0, 10))
        self.assertEqual(restored.frames[-1].initial_hidden.dtype, torch.float64)
        torch.testing.assert_close(restored.frames[-1].initial_hidden, dataset.frames[-1].initial_hidden)
        snapshot["frames"][0]["features"]["scalars"]["data"][0][0] += 100
        self.assertNotEqual(
            snapshot["frames"][0]["features"]["scalars"]["data"][0][0],
            float(dataset.frames[0].features.scalars[0, 0]),
        )

    def test_json_snapshot_rejects_incompatible_schema_or_unknown_tensor_types(self):
        dataset = ImitationDataset(1)
        dataset.extend(Rollout(((make_frame(self.network),),), (2.0,), 0, {}))
        snapshot = dataset.json_snapshot()
        snapshot["feature_version"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            dataset.load_json_snapshot(snapshot)
        snapshot = dataset.json_snapshot()
        snapshot["frames"][0]["initial_hidden"]["dtype"] = "arbitrary_python_class"
        with self.assertRaisesRegex(ValueError, "tensor dtype"):
            dataset.load_json_snapshot(snapshot)

    def test_dagger_baseline_and_rounds_anneal_to_learner_only(self):
        config = make_config(
            "imitation", updates=6,
            imitation={"epochs": 1, "teacher_probability": 0.4, "dagger_rounds": 2, "max_frames": 4},
        )
        schedule = [dagger_schedule(config, update) for update in range(6)]
        np.testing.assert_allclose([probability for probability, _ in schedule], [0.4, 0.4, 0.2, 0.2, 0, 0])
        self.assertEqual([round_index for _, round_index in schedule], [0, 0, 1, 1, 2, 2])
        self.assertEqual(dagger_schedule(config, 0, previous_round=2), (0.0, 2))
        baseline = make_config("imitation", imitation={"teacher_probability": 0})
        self.assertEqual(dagger_schedule(baseline, 0), (0.0, 0))
        impossible = make_config("imitation", updates=2, imitation={"dagger_rounds": 2})
        with self.assertRaisesRegex(ValueError, "updates"):
            dagger_schedule(impossible, 0)


class ImitationUpdateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    @staticmethod
    def objective(network, dataset, config):
        with torch.no_grad():
            terms = [
                imitation_loss(output, frame.step, list(frame.teacher_actions))
                for chunk in sequence_chunks(
                    dataset.trajectories(), config.ppo.sequence_length, config.resources.max_tokens,
                )
                for frame, output in unroll_sequence(network, chunk)
            ]
            return float(torch.stack(terms).mean())

    def test_warm_start_changes_weights_and_decreases_supervised_objective(self):
        config = make_config(
            "imitation", updates=1, steps=3,
            imitation={"epochs": 24, "teacher_probability": 1, "max_frames": 8},
        )
        network = TinyPolicy(config.model)
        with RolloutCollector(
            config, network, emit=lambda event: None,
            environment_factory=lambda: ScriptedEnvironment(
                ((0, 1), (0, 1)), (1.0,), terminal_at_end=False,
            ),
            teacher_factory=ConstantTeacher,
        ) as collector:
            rollout = collector.collect(3, policy_version=0, teacher_probability=1)
        dataset = ImitationDataset(8)
        dataset.extend(rollout)
        before = self.objective(network, dataset, config)
        weights = network.action_bias.detach().clone()
        optimizer = torch.optim.Adam(network.parameters(), lr=0.05)
        metrics = train_imitation(network, optimizer, dataset, config)
        after = self.objective(network, dataset, config)
        self.assertLess(after, before * 0.6)
        self.assertFalse(torch.equal(weights, network.action_bias))
        self.assertEqual(metrics["optimizer_steps"], 24)
        self.assertTrue(np.isfinite(metrics["gradient_norm"]))
        self.assertEqual([frame.reward for frame in dataset.frames], [1.0, 1.0, 1.0])
        self.assertTrue(all(frame.teacher_mask.all() for frame in dataset.frames))

    def test_variable_length_microbatches_do_not_overweight_short_sequences(self):
        config = make_config(
            "imitation", steps=3, sequence_length=3,
            model={"hidden_size": 16, "entity_size": 8, "memory": "none"},
            imitation={"epochs": 1, "max_frames": 8},
        )
        network = TinyPolicy(config.model)
        clone = copy.deepcopy(network)
        frames = tuple(
            make_frame(network, make_step(ids), episode_step=index, teacher=True)
            for index, ids in enumerate(((1, 2), (2, 3), (3,)))
        )
        dataset = ImitationDataset(8)
        dataset.extend(Rollout((frames,), (2.0,), 0, {}))
        short = config.model_copy(update={"ppo": config.ppo.model_copy(update={"sequence_length": 2})})
        train_imitation(network, torch.optim.SGD(network.parameters(), lr=0.1), dataset, config)
        train_imitation(clone, torch.optim.SGD(clone.parameters(), lr=0.1), dataset, short)
        for full, split in zip(network.parameters(), clone.parameters(), strict=True):
            torch.testing.assert_close(full, split, atol=1e-7, rtol=1e-6)

    def test_nonfinite_minibatch_and_gradient_fail_without_an_optimizer_step(self):
        config = make_config("imitation")
        network = TinyPolicy(config.model)
        dataset = ImitationDataset(4)
        dataset.extend(Rollout(((make_frame(network),),), (2.0,), 0, {}))
        optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with patch("src.training.imitation.imitation_loss", return_value=torch.tensor(float("nan"))):
                with self.assertRaisesRegex(FloatingPointError, "Imitation loss"):
                    train_imitation(network, optimizer, dataset, config)
            step.assert_not_called()
            network.action_bias.grad = torch.full_like(network.action_bias, float("nan"))
            with self.assertRaises(RuntimeError):
                checked_optimizer_step(network, optimizer, 0.5)
            step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
