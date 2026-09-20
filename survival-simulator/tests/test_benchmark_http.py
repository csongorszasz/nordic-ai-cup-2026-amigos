import unittest

from src.benchmarking.http import HTTPPolicy, HTTPPolicyConfig
from src.utils.DTOs import ObservationResponse, StepResponse


class Response:
    status_code = 200
    text = ""

    @staticmethod
    def json():
        return {"actions": [{
            "agent_id": 4, "move_distance": 0.0, "move_direction": 0.0,
            "turn_angle": 0.0, "spawn_agent": False,
        }]}


class Session:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return Response()


def frame():
    return StepResponse(
        game_status="ok", score=1.0, sim_time=1.0, n_agents=1,
        agent_status=[ObservationResponse(
            agent_id=4, energy=100.0, age=1.0, biome="forest", speed=10.0,
            sprint_speed=20.0, hearing_radius=50.0, vision_angle=1.0,
            vision_range=200.0, max_energy=500.0, observations=[],
        )],
    )


class HTTPPolicyTests(unittest.TestCase):
    def test_posts_step_and_validates_actions(self):
        session = Session()
        clock = iter((1.0, 1.25)).__next__
        policy = HTTPPolicy(
            HTTPPolicyConfig(url="https://example.test/predict"),
            session=session, clock=clock,
        )
        actions = policy.act(frame())
        self.assertEqual(actions[0].agent_id, 4)
        self.assertEqual(policy.calls, 1)
        self.assertEqual(policy.total_wait_seconds, 0.25)
        self.assertEqual(session.calls[0][0], "https://example.test/predict")
        self.assertEqual(session.calls[0][1]["timeout"], 10.0)
        self.assertIn('"game_status":"ok"', session.calls[0][1]["data"])

    def test_cumulative_budget_is_a_hard_failure(self):
        session = Session()
        clock = iter((0.0, 0.6, 1.0, 1.6)).__next__
        policy = HTTPPolicy(
            HTTPPolicyConfig(
                url="https://example.test/predict",
                per_request_timeout_seconds=1.0,
                cumulative_timeout_seconds=1.0,
            ),
            session=session, clock=clock,
        )
        policy.act(frame())
        with self.assertRaisesRegex(TimeoutError, "accumulated"):
            policy.act(frame())

    def test_invalid_response_shape_fails(self):
        class InvalidResponse(Response):
            @staticmethod
            def json():
                return {"actions": []}

        class InvalidSession(Session):
            def post(self, url, **kwargs):
                return InvalidResponse()

        policy = HTTPPolicy(
            HTTPPolicyConfig(url="https://example.test/predict"),
            session=InvalidSession(), clock=iter((0.0, 0.1)).__next__,
        )
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            policy.act(frame())

    def test_response_decoding_is_inside_the_budget(self):
        times = []

        class DecodingResponse(Response):
            @staticmethod
            def json():
                times.append("decode")
                return Response.json()

        class DecodingSession(Session):
            def post(self, url, **kwargs):
                times.append("post")
                return DecodingResponse()

        def clock():
            times.append("clock")
            return 0.0 if len(times) == 1 else 11.0

        policy = HTTPPolicy(HTTPPolicyConfig(url="https://example.test/predict"),
                            session=DecodingSession(), clock=clock)
        with self.assertRaises(TimeoutError):
            policy.act(frame())
        self.assertEqual(times, ["clock", "post", "decode", "clock"])


if __name__ == "__main__":
    unittest.main()
