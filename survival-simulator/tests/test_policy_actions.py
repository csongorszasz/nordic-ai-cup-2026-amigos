import math
import unittest
from dataclasses import replace

import torch

from src.policies.actions import ACTION_VERSION, evaluate_actions, imitation_loss, sample_actions
from src.policies.geometry import can_spawn, movement_limit
from src.policies.networks import PolicyOutput
from src.utils.DTOs import ActionRequest, ObservationResponse, StepResponse


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


def parameters(means, *, log_std=-0.4, spawn_logit=0.7, dtype=torch.float64):
    mean = torch.tensor(means, dtype=dtype).reshape(-1, 3).requires_grad_()
    return PolicyOutput(
        mean, torch.full_like(mean, log_std, requires_grad=True),
        torch.full((mean.shape[0],), spawn_logit, dtype=dtype, requires_grad=True),
        mean.sum() * 0, mean.new_zeros((mean.shape[0], 16)),
    )


def teacher(agent_id=0, *, distance=10.0, direction=0.0, turn=0.0, spawn=False):
    return ActionRequest(
        agent_id=agent_id, move_distance=distance, move_direction=direction,
        turn_angle=turn, spawn_agent=spawn,
    )


class ActionTests(unittest.TestCase):
    def test_transformed_density_and_entropy_match_analytic_reference(self):
        output = parameters([[0.1, -0.2, 0.3], [-0.2, 0.4, 0.7], [0.6, -0.4, 0.5]])
        latent = output.mean.new_tensor([[0, 0.2, -0.4], [2, -1.2, 0.7], [-4, 1.1, -2]])
        spawn = output.mean.new_tensor([1, 0, 0])
        eligible = torch.tensor([True, False, True])
        actual, entropy = evaluate_actions(output, latent, spawn, eligible)
        normal = torch.distributions.Normal(output.mean, output.log_std.exp())
        binary = torch.distributions.Bernoulli(logits=output.spawn_logits)
        distance = latent[:, 0].sigmoid()
        jacobian = (distance * (1 - distance)).log()
        jacobian = jacobian + (1 - latent[:, 1:].tanh().square()).log().sum(1)
        expected = normal.log_prob(latent).sum(1) - jacobian
        expected = expected + torch.where(eligible, binary.log_prob(spawn), 0)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            entropy, normal.entropy().sum(1) + torch.where(eligible, binary.entropy(), 0),
        )
        self.assertEqual(ACTION_VERSION, "bounded-v1")

    def test_collection_likelihood_identity_and_score_function_gradients(self):
        output = parameters([[0.2, -0.4, 0.6], [0.7, 0.4, -0.2]])
        sample = sample_actions(output, frame([agent(9), agent(4, energy=80)]),
                                generator=torch.Generator().manual_seed(123))
        log_prob, entropy = evaluate_actions(output, sample.latent, sample.spawn, sample.eligible)
        torch.testing.assert_close((log_prob - sample.log_prob).exp(), torch.ones_like(log_prob))
        torch.testing.assert_close(entropy, sample.entropy)
        self.assertFalse(sample.latent.requires_grad)
        (-sample.log_prob.mean()).backward()
        for value in (output.mean, output.log_std, output.spawn_logits):
            self.assertTrue(torch.isfinite(value.grad).all().item())
        self.assertGreater(output.mean.grad.abs().sum().item(), 0)
        self.assertEqual(output.spawn_logits.grad[1].item(), 0)

    def test_saturated_latents_keep_finite_likelihoods_and_gradients(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                output = parameters([[-100, 100, -100], [100, -100, 100]], dtype=dtype)
                sample = sample_actions(output, frame([agent(0), agent(1)]), deterministic=True)
                latent = sample.latent.clone().requires_grad_()
                log_prob, entropy = evaluate_actions(output, latent, sample.spawn, sample.eligible)
                self.assertTrue(torch.isfinite(log_prob).all().item())
                torch.testing.assert_close(log_prob, sample.log_prob)
                (-log_prob.mean() - 0.01 * entropy.mean()).backward()
                for value in (latent, output.mean, output.log_std, output.spawn_logits):
                    self.assertTrue(torch.isfinite(value.grad).all().item())
                self.assertGreater(latent.grad[:, 1:].abs().sum().item(), 0)

    def test_bounds_cover_low_energy_mutated_speeds_and_saturation(self):
        agents = [
            agent(0, energy=500), agent(1, energy=99), agent(2, speed=20, sprint_speed=5),
            agent(3, energy=1, speed=20, sprint_speed=5), agent(4, speed=0, sprint_speed=0),
            agent(5, energy=100),
        ]
        output = parameters([[100, -100, 100]] * len(agents), dtype=torch.float32)
        sample = sample_actions(output, frame(agents), deterministic=True)
        self.assertEqual([action.move_distance for action in sample.actions], [20, 10, 5, 5, 0, 20])
        for observation, action in zip(agents, sample.actions):
            self.assertGreaterEqual(action.move_distance, 0)
            self.assertLessEqual(action.move_distance, movement_limit(observation))
            self.assertEqual(action.move_direction, -math.pi)
            self.assertEqual(action.turn_angle, math.pi)
            self.assertEqual(
                action.spawn_agent, can_spawn(observation, action.move_distance, action.turn_angle),
            )

    def test_reproduction_strictly_checks_energy_after_movement_and_turn(self):
        output = parameters([[0, 0, 0], [0, 0, 0],
                             [0, 0, math.atanh(0.5)], [0, 0, math.atanh(0.5)]])
        step = frame([agent(0, energy=100.5), agent(1, energy=100.50001),
                      agent(2, energy=100.75), agent(3, energy=100.75001)])
        sample = sample_actions(output, step, deterministic=True)
        self.assertEqual(sample.eligible.tolist(), [False, True, False, True])
        self.assertEqual(sample.spawn.tolist(), [0, 1, 0, 1])
        continuous, latent_entropy = evaluate_actions(
            output, sample.latent, torch.zeros_like(sample.spawn), torch.zeros_like(sample.eligible),
        )
        torch.testing.assert_close(sample.log_prob[~sample.eligible], continuous[~sample.eligible],
                                   rtol=0, atol=0)
        torch.testing.assert_close(sample.entropy[~sample.eligible], latent_entropy[~sample.eligible],
                                   rtol=0, atol=0)
        sample.log_prob.sum().backward()
        self.assertEqual(output.spawn_logits.grad[0].item(), 0)
        self.assertEqual(output.spawn_logits.grad[2].item(), 0)

    def test_update_uses_collection_eligibility_not_new_means(self):
        output = parameters([[0, 0, 0], [0, 0, 0]])
        step = frame([agent(0, energy=100.6), agent(1, energy=100.4)])
        sample = sample_actions(output, step, deterministic=True)
        changed = parameters([[100, 0, 0], [-100, 0, 0]], spawn_logit=-3.0)
        new_sample = sample_actions(changed, step, deterministic=True)
        self.assertEqual(sample.eligible.tolist(), [True, False])
        self.assertEqual(new_sample.eligible.tolist(), [False, True])
        fixed, _ = evaluate_actions(changed, sample.latent, sample.spawn, sample.eligible)
        other = replace(changed, spawn_logits=torch.full_like(changed.spawn_logits, 3))
        alternate, _ = evaluate_actions(other, sample.latent, sample.spawn, sample.eligible)
        self.assertAlmostEqual((fixed[0] - alternate[0]).item(), -3, places=9)
        self.assertEqual(fixed[1].item(), alternate[1].item())

    def test_normalized_density_does_not_depend_on_environment_scale(self):
        output = parameters([[0.2, -0.7, 0.4]])
        first = sample_actions(output, frame([agent(energy=500, sprint_speed=10)]), deterministic=True)
        second = sample_actions(output, frame([agent(energy=500, sprint_speed=30)]), deterministic=True)
        self.assertAlmostEqual(second.actions[0].move_distance, 3 * first.actions[0].move_distance)
        torch.testing.assert_close(first.log_prob, second.log_prob, rtol=0, atol=0)

    def test_private_generators_and_deterministic_calls_preserve_global_rng(self):
        output = parameters([[0, 0, 0], [0.1, -0.2, 0.3]])
        step = frame([agent(1), agent(2)])
        first_rng, second_rng = torch.Generator().manual_seed(17), torch.Generator().manual_seed(17)
        global_state = torch.random.get_rng_state().clone()
        first = sample_actions(output, step, generator=first_rng)
        second = sample_actions(output, step, generator=second_rng)
        torch.testing.assert_close(first.latent, second.latent, rtol=0, atol=0)
        self.assertEqual(first.actions, second.actions)
        before = first_rng.get_state().clone()
        sample_actions(output, step, deterministic=True, generator=first_rng)
        self.assertTrue(torch.equal(before, first_rng.get_state()))
        self.assertTrue(torch.equal(global_state, torch.random.get_rng_state()))

    def test_empty_population_is_well_defined(self):
        output = parameters([])
        sample = sample_actions(output, frame([]), generator=torch.Generator().manual_seed(0))
        self.assertEqual(sample.actions, [])
        self.assertEqual(sample.latent.shape, (0, 3))
        self.assertEqual(sample.eligible.dtype, torch.bool)
        for value in (sample.spawn, sample.eligible, sample.log_prob, sample.entropy):
            self.assertEqual(value.shape, (0,))
        loss = imitation_loss(output, frame([]), [])
        self.assertEqual(loss.item(), 0)
        loss.backward()

    def test_invalid_parameters_latents_and_execution_masks_are_rejected(self):
        output = parameters([[0, 0, 0]])
        invalid = (
            replace(output, mean=torch.zeros(1, 2)),
            replace(output, mean=torch.full_like(output.mean, float("nan"))),
            replace(output, log_std=torch.zeros(3)),
            replace(output, log_std=torch.full_like(output.log_std, 1000)),
            replace(output, log_std=torch.full_like(output.log_std, -1000)),
            replace(output, spawn_logits=torch.full_like(output.spawn_logits, float("inf"))),
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                sample_actions(value, frame([agent()]))
        for latent, spawn, eligible in (
            (torch.zeros(1, 2), torch.zeros(1), torch.tensor([False])),
            (torch.full_like(output.mean, float("inf")), torch.zeros(1), torch.tensor([False])),
            (output.mean.detach(), torch.tensor([0.5]), torch.tensor([True])),
            (output.mean.detach(), torch.tensor([1]), torch.tensor([False])),
            (output.mean.detach(), torch.tensor([0]), torch.tensor([0])),
        ):
            with self.subTest(latent=latent, spawn=spawn, eligible=eligible), self.assertRaises(ValueError):
                evaluate_actions(output, latent, spawn, eligible)
        with self.assertRaises(ValueError):
            sample_actions(output, frame([agent(), agent(1)]))


class ImitationTests(unittest.TestCase):
    def test_angles_are_circular_across_the_branch_cut(self):
        near_pi = math.atanh(1 - 0.01 / math.pi)
        output = parameters([[0, near_pi, near_pi]], spawn_logit=-100)
        step = frame([agent()])
        near = imitation_loss(
            output, step, [teacher(direction=-math.pi + 0.01, turn=-math.pi + 0.01)],
        )
        opposite = imitation_loss(output, step, [teacher()])
        self.assertLess(near.item(), 0.001)
        self.assertGreater(opposite.item(), 3.9)
        near.backward()
        self.assertTrue(torch.isfinite(output.mean.grad).all().item())

    def test_zero_distance_ignores_travel_direction_but_not_turn(self):
        output = parameters([[0.2, 0.3, 0.4]])
        step = frame([agent(energy=80)])
        first = imitation_loss(output, step, [teacher(distance=0, direction=-2)])
        second = imitation_loss(output, step, [teacher(distance=0, direction=2)])
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        first.backward()
        self.assertEqual(output.mean.grad[0, 1].item(), 0)
        self.assertNotEqual(output.mean.grad[0, 2].item(), 0)
        self.assertEqual(output.spawn_logits.grad[0].item(), 0)

    def test_zero_movement_limit_has_no_distance_target(self):
        step = frame([agent(energy=80, speed=0, sprint_speed=0)])
        for value in (-100, 100):
            output = parameters([[value, 1, 0]])
            loss = imitation_loss(output, step, [teacher(distance=0)])
            self.assertEqual(loss.item(), 0)
            loss.backward()
            self.assertEqual(output.mean.grad[0, 0].item(), 0)

    def test_teacher_identity_and_teacher_execution_determine_targets(self):
        output = parameters([[100, 0, 0], [-100, 0, 0]])
        step = frame([agent(7, energy=100.6), agent(2, energy=100.4)])
        teachers = [teacher(7), teacher(2)]
        loss = imitation_loss(output, step, teachers)
        torch.testing.assert_close(loss, imitation_loss(output, step, list(reversed(teachers))))
        loss.backward()
        self.assertGreater(output.spawn_logits.grad[0].item(), 0)
        self.assertEqual(output.spawn_logits.grad[1].item(), 0)

    def test_teacher_actions_require_exact_valid_id_coverage(self):
        output = parameters([[0, 0, 0]])
        step = frame([agent()])
        for actions in (
            [], [teacher(1)], [teacher(), teacher()], [teacher(distance=-1)],
            [teacher(distance=21)], [teacher(direction=float("nan"))],
        ):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                imitation_loss(output, step, actions)
        with self.assertRaisesRegex(ValueError, "ineligible"):
            imitation_loss(output, frame([agent(energy=80)]), [teacher(spawn=True)])


if __name__ == "__main__":
    unittest.main()
