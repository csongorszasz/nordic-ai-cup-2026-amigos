import math
import unittest

from src.policies.config import HeuristicConfig
from src.policies.geometry import turn_cost
from src.policies.population import PopulationPolicy
from src.policies.runtime import create_policy
from src.utils.DTOs import StepResponse
from test_policy_hierarchical import agent


def frame(*agents, sim_time=0.1):
    return StepResponse(
        game_status="ok", score=sim_time, sim_time=sim_time,
        n_agents=len(agents), agent_status=list(agents),
    )


class PopulationPolicyTests(unittest.TestCase):
    def policy(self, **changes):
        return PopulationPolicy(1, HeuristicConfig(
            backend="population", min_population=1, breeding_age=45.0, **changes,
        ))

    def test_factory_selects_a_stateful_non_neural_controller(self):
        policy = create_policy(1, {"heuristic": {"backend": "population"}})
        self.assertIsInstance(policy, PopulationPolicy)
        self.assertTrue(policy.stateful)

    def test_old_last_breeder_can_replace_itself_without_a_tree(self):
        policy = self.policy()
        action = policy.act(frame(agent(0, age=70.0, energy=140.0)))[0]
        self.assertTrue(action.spawn_agent)
        self.assertEqual((action.move_distance, action.turn_angle), (0.0, 0.0))

    def test_birth_is_blocked_at_the_strict_energy_threshold(self):
        policy = self.policy(population_reserve=0.0)
        self.assertFalse(policy.act(frame(agent(0, age=70.0, energy=100.0)))[0].spawn_agent)

    def test_small_but_reproductive_energy_caps_do_not_deadlock_reserves(self):
        for count in (1, 2):
            with self.subTest(count=count):
                policy = self.policy()
                actions = policy.act(frame(*(
                    agent(index, age=70.0, energy=101.0, max_energy=101.0)
                    for index in range(count)
                )))
                self.assertEqual(sum(action.spawn_agent for action in actions), 1)

    def test_only_one_birth_is_reserved_and_spacing_survives_an_unseen_newborn(self):
        policy = self.policy()
        parents = [agent(i, age=70, energy=220) for i in range(3)]
        actions = policy.act(frame(*parents))
        self.assertEqual(sum(action.spawn_agent for action in actions), 1)
        actions = policy.act(frame(
            *(a.model_copy(update={"age": 70.1, "energy": 119.9}) for a in parents),
            sim_time=0.2,
        ))
        self.assertEqual(sum(action.spawn_agent for action in actions), 0)

    def test_spacing_does_not_block_a_last_lineage_rescue_after_newborn_loss(self):
        policy = self.policy()
        self.assertTrue(policy.act(frame(agent(0, age=70, energy=211.0)))[0].spawn_agent)
        rescue = policy.act(frame(agent(0, age=70.1, energy=110.199), sim_time=0.2))[0]
        self.assertTrue(rescue.spawn_agent)

    def test_birth_cost_and_skipped_age_do_not_falsely_detect_senescence(self):
        policy = self.policy()
        first = agent(0, age=70.0, energy=220.0)
        self.assertTrue(policy.act(frame(first))[0].spawn_agent)
        policy.act(frame(
            first.model_copy(update={"age": 70.1, "energy": 119.9}),
            agent(1, age=0.1, energy=74.9), sim_time=0.2,
        ))
        self.assertFalse(policy.lives[0].elder)
        self.assertEqual(policy.lives[0].children, {1})
        previous = policy.previous_actions[0]
        from src.policies.geometry import movement_cost

        cost = movement_cost(policy.lives[0].status, previous.move_distance) + turn_cost(previous.turn_angle)
        policy.act(frame(
            first.model_copy(update={"age": 70.1, "energy": 119.9 - cost}),
            agent(1, age=0.2, energy=74.8), sim_time=0.3,
        ))
        self.assertFalse(policy.lives[0].elder)

    def test_senescence_needs_repeated_clean_residuals(self):
        policy = self.policy()
        value = agent(0, age=70.0, energy=95.0)
        for tick in range(3):
            if tick:
                from src.policies.geometry import movement_cost

                action = policy.previous_actions[0]
                age = value.age + 0.1
                energy = value.energy - movement_cost(value, action.move_distance)
                energy -= turn_cost(action.turn_angle) + 0.1 + 0.01 * age
                value = value.model_copy(update={"age": age, "energy": energy})
            policy.act(frame(value, sim_time=(tick + 1) * 0.1))
        self.assertTrue(policy.lives[0].elder)

    def test_unaligned_components_do_not_share_resource_coordinates(self):
        policy = self.policy()
        policy.act(frame(
            agent(0, observations=[{"type": "Tree", "distance": 40.0, "angle": 0.0}]),
            agent(1),
        ))
        self.assertNotEqual(policy.lives[0].frame, policy.lives[1].frame)
        self.assertIsNotNone(policy.lives[0].home)
        self.assertIsNone(policy.lives[1].home)

    def test_root_death_retains_the_surviving_agents_frame(self):
        policy = self.policy()
        policy.act(frame(
            agent(0, observations=[{
                "type": "Agent", "distance": 30.0, "angle": 0.0,
                "rel_dir": -math.pi, "id": 1,
            }]), agent(1),
        ))
        original_frame = policy.lives[1].frame
        policy.act(frame(agent(1, age=20.1), sim_time=0.2))
        self.assertEqual(policy.lives[1].frame, original_frame)
        self.assertNotIn(0, policy.lives)
        self.assertGreater(policy.lives[1].pose.x, 20)

    def test_newborn_can_use_a_fresh_sighting_of_its_departed_parent(self):
        policy = self.policy()
        parent = agent(0, age=70, energy=140, observations=[
            {"type": "Tree", "distance": 40.0, "angle": 0.0},
        ])
        self.assertTrue(policy.act(frame(parent))[0].spawn_agent)
        original_frame = policy.lives[0].frame
        child = agent(1, age=0.1, energy=74.9, observations=[
            {"type": "Agent", "distance": 20.0, "angle": math.pi, "rel_dir": 0.0, "id": 0},
            {"type": "Tree", "distance": 20.0, "angle": 0.0},
        ])
        policy.act(frame(child, sim_time=0.2))
        self.assertEqual(policy.lives[1].frame, original_frame)
        self.assertAlmostEqual(policy.lives[1].pose.x, 20.0)
        self.assertEqual(policy.diagnostics()["confirmed_successors"], 1)

    def test_coincident_teammates_use_the_zero_vector_sensing_convention(self):
        policy = self.policy()
        first = agent(0, observations=[{
            "type": "Agent", "distance": 0.0, "angle": -0.2, "rel_dir": -0.7, "id": 1,
        }])
        policy.act(frame(first, agent(1)))
        self.assertAlmostEqual(policy.lives[1].pose.heading - policy.lives[0].pose.heading, 0.5)

    def test_new_teammate_sighting_merges_frames_and_splitting_does_not_reset_them(self):
        policy = self.policy()
        policy.act(frame(agent(0), agent(1)))
        self.assertNotEqual(policy.lives[0].frame, policy.lives[1].frame)
        policy.act(frame(agent(0, age=20.1, observations=[{
            "type": "Agent", "distance": 100.0, "angle": 0.0,
            "rel_dir": -math.pi, "id": 1,
        }]), agent(1, age=20.1), sim_time=0.2))
        self.assertEqual(policy.lives[0].frame, policy.lives[1].frame)
        merged = policy.lives[1].frame
        policy.act(frame(agent(0, age=20.2), agent(1, age=20.2), sim_time=0.3))
        self.assertEqual(policy.lives[0].frame, merged)
        self.assertEqual(policy.lives[1].frame, merged)

    def test_excessive_odometry_uncertainty_falls_back_to_a_fresh_local_frame(self):
        policy = self.policy()
        policy.act(frame(agent(0)))
        old_frame = policy.lives[0].frame
        policy.lives[0].uncertainty = 100.0
        policy.act(frame(agent(0), sim_time=0.2))
        self.assertNotEqual(old_frame, policy.lives[0].frame)
        self.assertEqual(policy.diagnostics()["localization_resets"], 1)

    def test_missing_tree_inside_hearing_is_retired_not_camped_forever(self):
        policy = self.policy()
        policy.act(frame(agent(0, observations=[{"type": "Tree", "distance": 20.0, "angle": 0.0}])))
        self.assertTrue(any(r.kind == "Tree" for r in policy.resources.values()))
        for tick in (2, 3):
            policy.act(frame(agent(0, age=20 + tick * 0.1), sim_time=tick * 0.1))
        self.assertFalse(any(r.kind == "Tree" for r in policy.resources.values()))

    def test_dangerous_spawns_are_not_treated_as_free_escapes(self):
        policy = self.policy()
        action = policy.act(frame(agent(0, age=70, energy=250, observations=[{
            "type": "Predator", "distance": 20.0, "angle": 0.0, "rel_dir": 0.0,
        }])))[0]
        self.assertFalse(action.spawn_agent)

    def test_escape_avoids_running_from_one_predator_into_another(self):
        policy = self.policy()
        action = policy.act(frame(agent(0, energy=200.0, observations=[
            {"type": "Predator", "distance": 30.0, "angle": 0.0, "rel_dir": 0.0},
            {"type": "Predator", "distance": 30.0, "angle": math.pi, "rel_dir": 0.0},
        ])))[0]
        self.assertGreater(abs(math.sin(action.move_direction)), 0.8)

    def test_colocated_predators_are_not_lost_or_order_dependent(self):
        observations = [
            {"type": "Predator", "distance": 60.0, "angle": 0.0, "rel_dir": -1.0},
            {"type": "Predator", "distance": 60.0, "angle": 0.0, "rel_dir": 1.0},
        ]
        first, second = self.policy(), self.policy()
        a = first.act(frame(agent(0, observations=observations)))
        b = second.act(frame(agent(0, observations=list(reversed(observations)))))
        self.assertEqual(a, b)
        self.assertEqual(sum(r.kind == "Predator" for r in first.resources.values()), 2)

    def test_decoy_role_is_opt_in_and_requires_a_viable_protected_successor(self):
        for enabled in (False, True):
            policy = self.policy(elder_decoys=enabled)
            parent = agent(0, age=130, energy=60, observations=[{
                "type": "Agent", "distance": 200.0, "angle": 0.0,
                "rel_dir": -math.pi, "id": 1,
            }])
            child = agent(1, age=10, energy=150)
            policy.act(frame(parent, child))
            policy.lives[0].children = {1}
            threatened = parent.model_copy(update={"age": 130.1, "observations": [*parent.observations, {
                "type": "Predator", "distance": 60.0, "angle": 0.0, "rel_dir": math.pi,
            }]})
            actions = policy.act(frame(threatened, child.model_copy(update={"age": 10.1}), sim_time=0.2))
            with self.subTest(enabled=enabled):
                self.assertEqual(policy.diagnostics()["decoys"], int(enabled))
                self.assertFalse(actions[0].spawn_agent)
                self.assertLess(math.cos(actions[0].move_direction), 0)

    def test_skipped_native_updates_do_not_refresh_or_invalidate_cached_landmarks(self):
        policy = self.policy()
        value = agent(0, observations=[{"type": "Tree", "distance": 20.0, "angle": 0.0}])
        policy.act(frame(value))
        patch = next(r for r in policy.resources.values() if r.kind == "Tree")
        seen = patch.last_seen
        for tick in (2, 3):
            action = policy.act(frame(value, sim_time=tick * 0.1))[0]
            self.assertEqual(action.move_distance, 0.0)
            self.assertEqual(patch.last_seen, seen)
        self.assertIn(patch.id, policy.resources)
        self.assertEqual(policy.diagnostics()["stale_observers"], 1)

    def test_agent_and_observation_permutations_preserve_actions(self):
        first, second = self.policy(), self.policy()
        one = agent(0, observations=[
            {"type": "Tree", "distance": 30.0, "angle": 0.0},
            {"type": "Fruit", "distance": 40.0, "angle": 0.5},
        ])
        two = agent(1, observations=[{
            "type": "Agent", "distance": 50.0, "angle": math.pi,
            "rel_dir": 0.0, "id": 0,
        }])
        for tick in range(3):
            a = first.act(frame(one, two, sim_time=(tick + 1) * 0.1))
            b = second.act(frame(
                two, one.model_copy(update={"observations": list(reversed(one.observations))}),
                sim_time=(tick + 1) * 0.1,
            ))
            self.assertEqual(a, b)

    def test_reset_clears_all_population_and_map_state(self):
        policy = self.policy()
        policy.act(frame(agent(0, age=70, energy=220)))
        policy.reset()
        self.assertEqual(policy.lives, {})
        self.assertEqual(policy.resources, {})
        self.assertEqual(policy.previous_actions, {})
        self.assertEqual(policy.diagnostics(), {})


if __name__ == "__main__":
    unittest.main()
