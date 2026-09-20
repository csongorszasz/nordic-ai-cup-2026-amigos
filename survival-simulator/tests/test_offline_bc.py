import copy
import unittest

import numpy as np
import torch

from src.policies.actions import imitation_loss
from src.policies.config import ModelConfig
from src.policies.features import encode_step
from src.policies.networks import PolicyNetwork
from src.training.offline_bc import pack_sequence, forward_sequence, sequence_loss, copying_metrics
from src.training.rollout import align_hidden
from tests.test_training_rollout import make_step, ConstantTeacher


class OfflineBCTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(53)

    def frames(self):
        previous = {}
        result = []
        for tick, ids in enumerate(((0, 1), (1, 2), (2,), (2, 3))):
            step = make_step(ids, tick=tick + 1, entities=2)
            actions = ConstantTeacher().act(step)
            actions[0].move_distance = 0.0
            result.append((step, tuple(actions), previous))
            previous = {action.agent_id: action for action in actions}
        return result

    def test_packed_sequence_matches_regular_forward_and_loss_gradients(self):
        for encoder in ("pool", "attention"):
            for memory in ("none", "gru"):
                with self.subTest(encoder=encoder, memory=memory):
                    config = ModelConfig(hidden_size=16, entity_size=8, encoder=encoder, memory=memory,
                                         public_context=True)
                    packed_model = PolicyNetwork(config)
                    reference = copy.deepcopy(packed_model)
                    frames = self.frames()
                    batch = pack_sequence(frames, config)
                    output = forward_sequence(packed_model, batch)
                    means, losses = [], []
                    hidden, previous_ids = None, ()
                    for step, actions, history in frames:
                        features = encode_step(step, history, public_context=True)
                        hidden = align_hidden(previous_ids, features.agent_ids, hidden, hidden_size=16, device="cpu")
                        current = reference(features, hidden)
                        hidden, previous_ids = current.next_hidden, features.agent_ids
                        means.append(current.mean)
                        losses.append(imitation_loss(current, step, actions))
                    torch.testing.assert_close(output.mean, torch.cat(means), atol=1e-6, rtol=1e-5)
                    left, right = sequence_loss(output, batch), torch.stack(losses).mean()
                    torch.testing.assert_close(left, right)
                    left.backward()
                    right.backward()
                    for (name, a), (_, b) in zip(packed_model.named_parameters(), reference.named_parameters()):
                        if a.grad is not None or b.grad is not None:
                            self.assertIsNotNone(a.grad, name)
                            self.assertIsNotNone(b.grad, name)
                            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=1e-4)

    def test_current_weight_recurrence_and_no_cross_frame_team_pooling(self):
        config = ModelConfig(hidden_size=16, entity_size=8)
        model = PolicyNetwork(config)
        frames = self.frames()
        first = forward_sequence(model, pack_sequence(frames, config)).mean.detach()
        altered = copy.deepcopy(frames)
        altered[-1][0].agent_status[0].energy = 1.0
        second = forward_sequence(model, pack_sequence(altered, config)).mean.detach()
        np.testing.assert_allclose(first[:5].numpy(), second[:5].numpy(), atol=1e-7)
        self.assertFalse(torch.equal(first[-2:], second[-2:]))

    def test_stop_and_empty_positive_class_metrics_are_explicit(self):
        config = ModelConfig(hidden_size=16, entity_size=8)
        model = PolicyNetwork(config)
        batch = pack_sequence(self.frames(), config)
        with torch.no_grad():
            metrics = copying_metrics(forward_sequence(model, batch), batch)
        self.assertEqual(metrics["stop_labels"], 4)
        self.assertEqual(metrics["spawn_positive_labels"], 0)
        self.assertIsNone(metrics["spawn_recall"])
        self.assertGreater(metrics["stop_distance_mean"], 0)
        self.assertGreater(metrics["stop_energy_mean_per_tick"], 0)

    def test_synthetic_fixed_data_actually_fits_with_many_optimizer_steps(self):
        config = ModelConfig(hidden_size=16, entity_size=8)
        model = PolicyNetwork(config)
        batch = pack_sequence(self.frames(), config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        first = float(sequence_loss(forward_sequence(model, batch), batch).detach())
        for _ in range(50):
            optimizer.zero_grad()
            loss = sequence_loss(forward_sequence(model, batch), batch)
            loss.backward()
            optimizer.step()
        last = float(sequence_loss(forward_sequence(model, batch), batch).detach())
        self.assertLess(last, first * 0.2)


if __name__ == "__main__":
    unittest.main()
