import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.policies.actions import imitation_components, sample_actions, VECTOR_ACTION_VERSION
from src.policies.config import ExperimentConfig, ModelConfig
from src.policies.features import encode_step
from src.policies.networks import PolicyNetwork
from src.training.artifacts import load_checkpoint, read_json, save_checkpoint
from src.training.offline_bc import migrate_angle_head
from tests.test_training_rollout import make_step


class VectorAngleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_vector_regression_has_gradient_where_bounded_angle_is_saturated(self):
        mean = torch.tensor([[0.0, 5.0, 0.0]], requires_grad=True)
        target = torch.tensor([[0.5, -0.8, 0, 0, 1, 1]], dtype=torch.float32)
        eligible = torch.tensor([True])
        old = SimpleNamespace(mean=mean, spawn_logits=torch.zeros(1))
        imitation_components(old, target, eligible)[:, 1].sum().backward()
        bounded_gradient = abs(float(mean.grad[0, 1]))
        vectors = torch.tensor([[[-1.0, 0.0], [1.0, 0.0]]], requires_grad=True)
        new = SimpleNamespace(mean=mean.detach(), angle_vectors=vectors, spawn_logits=torch.zeros(1))
        imitation_components(new, target, eligible)[:, 1].sum().backward()
        self.assertGreater(float(vectors.grad[0, 0].norm()), bounded_gradient * 1000)

    def test_migration_preserves_every_existing_parameter_and_versions_checkpoint(self):
        config = ExperimentConfig(mode="imitation", model=ModelConfig(hidden_size=16, entity_size=8))
        old = PolicyNetwork(config.model)
        new = migrate_angle_head(old, "vector_bc", 42)
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key])
        step = make_step((0,))
        out = new(encode_step(step))
        action = sample_actions(out, step, deterministic=True).actions[0]
        self.assertAlmostEqual(action.move_direction, 0)
        self.assertAlmostEqual(action.turn_angle, 0)
        with self.assertRaisesRegex(ValueError, "deterministic"):
            sample_actions(out, step)
        changed = config.model_copy(update={"model": new.config})
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "vector.pt"
            save_checkpoint(checkpoint, new, changed)
            self.assertEqual(read_json(checkpoint.with_suffix(".pt.json"))["action_version"], VECTOR_ACTION_VERSION)
            restored = load_checkpoint(checkpoint)
            self.assertEqual(restored.network.config.angle_head, "vector_bc")
        with self.assertRaisesRegex(ValueError, "PPO"):
            ExperimentConfig(mode="ppo", model=new.config)


if __name__ == "__main__":
    unittest.main()
