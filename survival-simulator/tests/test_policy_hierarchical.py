import math
import unittest

from src.policies.config import HeuristicConfig
from src.policies.hierarchical import HierarchicalPolicy, _build_contexts
from src.utils.DTOs import ObservationResponse, StepResponse


def agent(agent_id, observations=None, **changes):
    values = {
        "agent_id": agent_id, "energy": 150.0, "age": 20.0, "biome": "forest",
        "speed": 10.0, "sprint_speed": 20.0, "hearing_radius": 50.0,
        "vision_range": 200.0, "vision_angle": math.pi / 3, "max_energy": 500.0,
        "observations": observations or [],
    }
    return ObservationResponse(**(values | changes))


def frame(*agents, score=1.0):
    return StepResponse(
        game_status="ok", score=score, sim_time=1.0,
        n_agents=len(agents), agent_status=list(agents),
    )


class HierarchicalPolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = HeuristicConfig(backend="hierarchical")
        self.policy = HierarchicalPolicy(1, self.config)

    def test_guaranteed_stale_contact_fruit_is_not_chased(self):
        value = agent(
            0, energy=50.0,
            observations=[{"type": "Fruit", "distance": 9.0, "angle": 0.0}],
        )
        action = self.policy.act(frame(value))[0]
        self.assertEqual(action.move_distance, 0.0)

    def test_agents_reserve_distinct_shared_fruit_targets(self):
        first = agent(0, observations=[
            {"type": "Agent", "distance": 100.0, "angle": 0.0, "rel_dir": -math.pi, "id": 1},
            {"type": "Fruit", "distance": 30.0, "angle": 0.0},
            {"type": "Fruit", "distance": 70.0, "angle": 0.0},
        ])
        second = agent(1, observations=[
            {"type": "Agent", "distance": 100.0, "angle": math.pi, "rel_dir": 0.0, "id": 0},
            {"type": "Fruit", "distance": 70.0, "angle": math.pi},
            {"type": "Fruit", "distance": 30.0, "angle": math.pi},
        ])
        actions = {action.agent_id: action for action in self.policy.act(frame(first, second))}
        self.assertGreater(math.cos(actions[0].move_direction), 0.9)
        self.assertLess(math.cos(actions[1].move_direction), -0.9)
        self.assertGreater(actions[0].move_distance, 0.0)
        self.assertGreater(actions[1].move_distance, 0.0)

    def test_one_way_teammate_sighting_builds_one_pose_component(self):
        first = agent(0)
        second = agent(1, observations=[{
            "type": "Agent", "distance": 100.0, "angle": math.pi,
            "rel_dir": 0.0, "id": 0,
        }])
        contexts = _build_contexts(frame(first, second))
        self.assertEqual(contexts[0].component, contexts[1].component)
        self.assertAlmostEqual(contexts[1].pose.x, 100.0)
        self.assertAlmostEqual(contexts[1].pose.y, 0.0)

    def test_agent_camps_when_already_at_assigned_patch_slot(self):
        slot_angle = 0.73
        tree_angle = slot_angle + math.pi
        value = agent(0, observations=[{
            "type": "Tree", "distance": self.config.patch_radius, "angle": tree_angle,
        }])
        action = self.policy.act(frame(value))[0]
        self.assertEqual(action.move_distance, 0.0)

    def test_retreats_while_turning_to_face_predator(self):
        value = agent(0, observations=[{
            "type": "Predator", "distance": 60.0, "angle": 0.0, "rel_dir": math.pi,
        }])
        action = self.policy.act(frame(value))[0]
        self.assertGreater(action.move_distance, value.speed)
        self.assertLess(math.cos(action.move_direction), -0.5)
        self.assertGreater(math.cos(action.turn_angle), 0.5)

    def test_only_selected_high_trait_breeder_reproduces(self):
        food = [{"type": "Tree", "distance": 40.0, "angle": 0.0}]
        strong = agent(
            0, food, energy=300.0, age=70.0, speed=20.0, sprint_speed=40.0,
            hearing_radius=100.0, vision_range=400.0, max_energy=1000.0,
        )
        weak = agent(
            1, food, energy=300.0, age=70.0, speed=5.0, sprint_speed=8.0,
            hearing_radius=20.0, vision_range=80.0, max_energy=250.0,
        )
        actions = {action.agent_id: action for action in self.policy.act(frame(strong, weak))}
        self.assertTrue(actions[0].spawn_agent)
        self.assertFalse(actions[1].spawn_agent)

    def test_reset_clears_cross_episode_score_state(self):
        self.policy.act(frame(agent(0), score=10.0))
        self.assertEqual(self.policy._last_score, 10.0)
        self.policy.reset()
        self.assertIsNone(self.policy._last_score)
        self.assertEqual(self.policy.previous_actions, {})

    def test_macro_action_survival_score_is_not_food_income(self):
        hungry = agent(0, energy=300.0, age=0.1)
        first = frame(hungry, score=0.1).model_copy(update={"sim_time": 0.1})
        self.policy.act(first)
        later = frame(
            hungry.model_copy(update={"age": 0.6}), score=0.6,
        ).model_copy(update={"sim_time": 0.6})
        action = self.policy.act(later)[0]
        self.assertAlmostEqual(self.policy._food_rate, 0.0)
        self.assertFalse(action.spawn_agent)


if __name__ == "__main__":
    unittest.main()
