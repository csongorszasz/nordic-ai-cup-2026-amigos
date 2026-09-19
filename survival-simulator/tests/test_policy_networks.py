import io
import itertools
import math
import unittest
from dataclasses import replace

import numpy as np
import torch

from src.policies.config import ModelConfig
from src.policies.features import ENTITY_DIM, ENTITY_TYPES, SCALAR_DIM, FeatureBatch
from src.policies.networks import PolicyNetwork


def setUpModule():
    global _previous_threads
    _previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(_previous_threads)


def batch(*, empty_entities=False, count=4):
    rng = np.random.default_rng(513)
    rows = [] if empty_entities else [
        (owner, kind) for owner, number in ((0, 3), (2, 2))
        for kind in range(len(ENTITY_TYPES)) for _ in range(number)
    ]
    return FeatureBatch(
        tuple(10 + 7 * index for index in range(count)),
        rng.normal(size=(count, SCALAR_DIM)).astype(np.float32),
        rng.normal(size=(len(rows), ENTITY_DIM)).astype(np.float32),
        np.asarray([kind for _, kind in rows], dtype=np.int64),
        np.asarray([owner for owner, _ in rows], dtype=np.int64),
    )


def configurations():
    for encoder, memory, context, critic in itertools.product(
        ("pool", "attention"), ("none", "gru"), (False, True), ("team", "local"),
    ):
        yield ModelConfig(
            encoder=encoder, memory=memory, team_context=context, critic=critic,
            hidden_size=16, entity_size=8, entity_chunk_size=2,
        )


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.rng_context = torch.random.fork_rng(devices=[])
        self.rng_context.__enter__()
        torch.manual_seed(341)

    def tearDown(self):
        self.rng_context.__exit__(None, None, None)

    def test_all_architectures_support_finite_gradients_and_parameter_updates(self):
        features = batch()
        for config in configurations():
            with self.subTest(config=config):
                network = PolicyNetwork(config)
                optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
                before = network.mean_head.weight.detach().clone()
                output = network(features)
                self.assertEqual(output.mean.shape, (4, 3))
                self.assertEqual(output.log_std.shape, (4, 3))
                self.assertEqual(output.spawn_logits.shape, (4,))
                self.assertEqual(output.next_hidden.shape, (4, 16))
                self.assertEqual(output.value.shape, ())
                loss = (
                    (output.mean - 0.3).square().mean() + output.log_std.square().mean()
                    + output.spawn_logits.square().mean() + (output.value - 1).square()
                    + output.next_hidden.square().mean()
                )
                loss.backward()
                for name, parameter in network.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
                for encoder in network.entity_encoders:
                    self.assertGreater(encoder[0].weight.grad.abs().sum().item(), 0)
                for query in network.attention_queries:
                    self.assertGreater(query.weight.grad.abs().sum().item(), 0)
                optimizer.step()
                self.assertFalse(torch.equal(before, network.mean_head.weight))

    def test_agent_and_entity_permutations_are_equivariant(self):
        features = batch()
        agent_order = np.asarray([2, 0, 3, 1])
        inverse = np.argsort(agent_order)
        entity_order = np.random.default_rng(41).permutation(features.entities.shape[0])
        permuted = FeatureBatch(
            tuple(features.agent_ids[index] for index in agent_order),
            features.scalars[agent_order], features.entities[entity_order],
            features.entity_types[entity_order], inverse[features.entity_owners[entity_order]],
        )
        hidden = torch.linspace(-0.3, 0.5, 4 * 16).reshape(4, 16)
        for config in configurations():
            with self.subTest(config=config), torch.no_grad():
                network = PolicyNetwork(config)
                original = network(features, hidden)
                reordered = network(permuted, hidden[agent_order])
                for name in ("mean", "log_std", "spawn_logits", "next_hidden"):
                    torch.testing.assert_close(
                        getattr(reordered, name), getattr(original, name)[agent_order],
                        rtol=2e-5, atol=1e-6,
                    )
                torch.testing.assert_close(reordered.value, original.value, rtol=2e-5, atol=1e-6)

    def test_ragged_aggregates_match_unpadded_per_agent_reference(self):
        features = batch()
        for encoder in ("pool", "attention"):
            with self.subTest(encoder=encoder):
                network = PolicyNetwork(ModelConfig(
                    encoder=encoder, hidden_size=16, entity_size=8, entity_chunk_size=2,
                )).double()
                scalars, entities, types, owners, _ = network._inputs(features, None)
                scalar = network.scalar_encoder(scalars)
                pooled = network._pool_entities(scalar, entities, types, owners)
                expected = []
                for owner in range(len(features.agent_ids)):
                    row = []
                    for kind, mlp in enumerate(network.entity_encoders):
                        selected = entities[(owners == owner) & (types == kind)]
                        if selected.shape[0] == 0:
                            row.append(entities.new_zeros(17))
                            continue
                        encoded = mlp(selected)
                        if encoder == "pool":
                            average = encoded.mean(0)
                        else:
                            query = network.attention_queries[kind](scalar[owner])
                            weights = ((query * encoded).sum(1) / math.sqrt(8)).softmax(0)
                            average = (weights.unsqueeze(1) * encoded).sum(0)
                        row.append(torch.cat((
                            average, encoded.amax(0), entities.new_tensor([math.log1p(len(selected))]),
                        )))
                    expected.append(torch.cat(row))
                torch.testing.assert_close(pooled, torch.stack(expected))
                self.assertTrue(torch.equal(pooled[1], torch.zeros_like(pooled[1])))

    def test_scatter_pooling_and_attention_pass_numerical_gradient_check(self):
        for encoder in ("pool", "attention"):
            with self.subTest(encoder=encoder):
                network = PolicyNetwork(ModelConfig(
                    encoder=encoder, hidden_size=16, entity_size=8, entity_chunk_size=2,
                )).double()
                scalar = torch.randn(2, 16, dtype=torch.float64)
                entities = torch.randn(4, ENTITY_DIM, dtype=torch.float64, requires_grad=True)
                types = torch.tensor([0, 0, 1, 1])
                owners = torch.tensor([0, 0, 0, 0])
                self.assertTrue(torch.autograd.gradcheck(
                    lambda value: network._pool_entities(scalar, value, types, owners),
                    (entities,), fast_mode=True,
                ))

    def test_chunking_preserves_every_entity_and_matches_unchunked_gradients(self):
        features = batch()
        for encoder in ("pool", "attention"):
            with self.subTest(encoder=encoder):
                config = ModelConfig(encoder=encoder, hidden_size=16, entity_size=8, entity_chunk_size=2)
                network = PolicyNetwork(config).double()
                reference = PolicyNetwork(config.model_copy(update={"entity_chunk_size": 1000})).double()
                reference.load_state_dict(network.state_dict())
                chunks = []
                hooks = [mlp.register_forward_pre_hook(
                    lambda module, args: chunks.append(args[0].shape[0]),
                ) for mlp in network.entity_encoders]
                try:
                    output = network(features)
                finally:
                    for hook in hooks:
                        hook.remove()
                self.assertTrue(all(size <= 2 for size in chunks))
                self.assertEqual(sum(chunks), features.entities.shape[0])
                expected = reference(features)
                torch.testing.assert_close(output.mean, expected.mean)
                (output.mean.square().sum() + output.value.square()).backward()
                (expected.mean.square().sum() + expected.value.square()).backward()
                for (name, parameter), (_, other) in zip(network.named_parameters(), reference.named_parameters()):
                    if parameter.grad is not None:
                        torch.testing.assert_close(parameter.grad, other.grad, msg=name)

    def test_empty_entities_and_empty_population_have_finite_outputs(self):
        for config in configurations():
            with self.subTest(config=config):
                network = PolicyNetwork(config)
                populated = network(batch(empty_entities=True))
                self.assertEqual(populated.mean.shape, (4, 3))
                empty = network(batch(empty_entities=True, count=0))
                self.assertEqual(empty.mean.shape, (0, 3))
                self.assertEqual(empty.spawn_logits.shape, (0,))
                self.assertEqual(empty.next_hidden.shape, (0, 16))
                self.assertEqual(empty.value.item(), 0)
                (empty.value + empty.mean.sum()).backward()
                self.assertTrue(all(
                    torch.isfinite(parameter.grad).all().item()
                    for parameter in network.parameters() if parameter.grad is not None
                ))

    def test_explicit_hidden_state_is_used_only_by_gru(self):
        features = batch()
        for memory in ("none", "gru"):
            network = PolicyNetwork(ModelConfig(memory=memory, hidden_size=16, entity_size=8))
            first = network(features)
            second = network(features, torch.zeros(4, 16))
            changed = network(features, torch.ones(4, 16))
            torch.testing.assert_close(first.next_hidden, second.next_hidden, rtol=0, atol=0)
            torch.testing.assert_close(network(features).next_hidden, first.next_hidden, rtol=0, atol=0)
            self.assertEqual(torch.equal(first.next_hidden, changed.next_hidden), memory == "none")

    def test_critics_do_not_multiply_value_by_population(self):
        for critic in ("team", "local"):
            network = PolicyNetwork(ModelConfig(critic=critic, hidden_size=16, entity_size=8))
            single = batch(empty_entities=True, count=1)
            repeated = FeatureBatch(
                tuple(range(7)), np.repeat(single.scalars, 7, axis=0),
                single.entities, single.entity_types, single.entity_owners,
            )
            torch.testing.assert_close(network(single).value, network(repeated).value)

    def test_state_dict_round_trip_and_caller_rng_isolation(self):
        features = batch()
        for encoder, memory in itertools.product(("pool", "attention"), ("none", "gru")):
            with self.subTest(encoder=encoder, memory=memory):
                config = ModelConfig(encoder=encoder, memory=memory, hidden_size=16, entity_size=8)
                network = PolicyNetwork(config)
                buffer = io.BytesIO()
                torch.save(network.state_dict(), buffer)
                buffer.seek(0)
                global_state = torch.random.get_rng_state().clone()
                with torch.random.fork_rng(devices=[]):
                    restored = PolicyNetwork(config)
                self.assertTrue(torch.equal(global_state, torch.random.get_rng_state()))
                restored.load_state_dict(torch.load(buffer, weights_only=True))
                expected, actual = network(features), restored(features)
                for name in ("mean", "log_std", "spawn_logits", "value", "next_hidden"):
                    torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
                self.assertEqual(restored.config, config)

    def test_population_is_not_capped_and_default_hidden_size_is_honored(self):
        network = PolicyNetwork(ModelConfig(hidden_size=16, entity_size=8))
        output = network(batch(empty_entities=True, count=1025))
        self.assertEqual(output.mean.shape, (1025, 3))
        self.assertEqual(output.next_hidden.shape, (1025, 16))
        default = PolicyNetwork(ModelConfig())(batch(empty_entities=True, count=1))
        self.assertEqual(default.next_hidden.shape, (1, 128))

    def test_invalid_features_and_hidden_states_are_rejected(self):
        features = batch()
        network = PolicyNetwork(ModelConfig(hidden_size=16, entity_size=8))
        invalid = (
            replace(features, agent_ids=(0, 0, 2, 3)),
            replace(features, agent_ids=(True, 1, 2, 3)),
            replace(features, scalars=np.full_like(features.scalars, np.nan)),
            replace(features, scalars=features.scalars[:, :-1]),
            replace(features, entities=np.full_like(features.entities, np.inf)),
            replace(features, entity_types=np.full_like(features.entity_types, 5)),
            replace(features, entity_types=features.entity_types.astype(np.float32)),
            replace(features, entity_owners=np.full_like(features.entity_owners, 4)),
            replace(features, entity_owners=np.full_like(features.entity_owners, -1)),
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                network(value)
        for hidden in (torch.zeros(4, 15), torch.zeros(4, 16, 1),
                       torch.full((4, 16), float("nan")), torch.zeros(4, 16, dtype=torch.int64)):
            with self.subTest(hidden=hidden), self.assertRaises(ValueError):
                network(features, hidden)
        with torch.no_grad():
            network.mean_head.bias.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            network(features)


if __name__ == "__main__":
    unittest.main()
