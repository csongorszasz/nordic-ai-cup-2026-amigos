import math
import random
import unittest
from unittest.mock import patch

from src.policies.config import HeuristicConfig
from src.policies.heuristic import HeuristicPolicy, assess_walls, candidate_walls
from src.utils.DTOs import ObservationResponse, StepResponse


class CandidateWallTests(unittest.TestCase):
    def test_broad_phase_preserves_scores_and_collisions(self):
        rng = random.Random(27)
        for _ in range(20):
            walls = tuple(
                ((rng.uniform(-200, 200), rng.uniform(-200, 200)),
                 (rng.uniform(-200, 200), rng.uniform(-200, 200)))
                for _ in range(30)
            )
            nearby = candidate_walls(walls, 20.0, 12.0)
            for index in range(16):
                endpoint = (20 * math.cos(index * math.tau / 16), 20 * math.sin(index * math.tau / 16))
                full = assess_walls(endpoint, walls, 12.0)
                filtered = assess_walls(endpoint, nearby, 12.0)
                self.assertEqual(full.blocked, filtered.blocked)
                self.assertAlmostEqual(full.score, filtered.score, places=12)

    def test_full_policy_decision_matches_unfiltered_scoring(self):
        observations = [
            {"type": "Edge", "coords": ((x, -25), (x, 25))}
            for x in (12, 30, 60, 85, -20, -55, -90)
        ] + [{"type": "Fruit", "distance": 35, "angle": math.pi / 2}]
        agent = ObservationResponse(
            agent_id=2, energy=200, age=20, biome="forest", speed=10, sprint_speed=20,
            hearing_radius=50, vision_range=200, vision_angle=1, max_energy=500,
            observations=observations,
        )
        step = StepResponse(game_status="ok", sim_time=20, score=20, n_agents=1, agent_status=[agent])
        policy = HeuristicPolicy(1, HeuristicConfig())
        filtered = policy.act(step)
        with patch("src.policies.heuristic.candidate_walls", side_effect=lambda walls, *_: walls):
            self.assertEqual(filtered, policy.act(step))


if __name__ == "__main__":
    unittest.main()
