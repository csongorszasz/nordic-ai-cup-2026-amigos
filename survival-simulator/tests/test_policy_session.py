import unittest

from src.serving.session import PolicySession, SessionConflict
from src.utils.DTOs import ActionRequest, ObservationResponse, StepResponse


def frame(time=0.1, game_status="ok", count=1):
    return StepResponse(
        game_status=game_status, sim_time=time, score=time, n_agents=count,
        agent_status=[ObservationResponse(
            agent_id=index, energy=150, age=time, biome="forest", speed=10, sprint_speed=20,
            hearing_radius=50, vision_range=200, vision_angle=1, max_energy=500, observations=[],
        ) for index in range(count)],
    )


class StatefulFixture:
    def __init__(self):
        self.calls = 0
        self.resets = 0
        self.times = []

    def reset(self):
        self.resets += 1

    def act(self, step):
        self.calls += 1
        self.times.append(step.sim_time)
        return [ActionRequest(
            agent_id=agent.agent_id, move_distance=self.calls, move_direction=0,
            turn_angle=0, spawn_agent=False,
        ) for agent in step.agent_status]


class SessionTests(unittest.TestCase):
    def test_stateful_session_can_run_without_time_order_contract(self):
        session = PolicySession(StatefulFixture, stateful=True)
        session.predict(frame())
        session.predict(frame(count=2))
        self.assertEqual(session.policy.calls, 2)

    def test_retry_is_idempotent_and_cached_actions_are_isolated(self):
        session = PolicySession(StatefulFixture, stateful=True, single_stream=True)
        first = session.predict(frame())
        first[0].move_distance = 99
        self.assertEqual(session.predict(frame())[0].move_distance, 1)
        self.assertEqual(session.policy.calls, 1)
        session.predict(frame(0.2))
        self.assertEqual(session.policy.calls, 2)

    def test_conflicting_request_does_not_mutate_recurrent_state(self):
        session = PolicySession(StatefulFixture, stateful=True, single_stream=True)
        session.predict(frame())
        with self.assertRaises(SessionConflict):
            session.predict(frame(count=2))
        self.assertEqual(session.policy.calls, 1)

    def test_bootstrap_and_new_game_without_final_request_reset(self):
        session = PolicySession(StatefulFixture, stateful=True, single_stream=True)
        bootstrap = frame(0, count=0)
        bootstrap.n_agents = 5
        self.assertEqual(session.predict(bootstrap), [])
        session.predict(frame(0.1))
        session.predict(frame(10))
        session.predict(frame(0.1))
        self.assertEqual(session.policy.resets, 2)
        self.assertEqual(session.policy.calls, 3)
        session.predict(frame(0.2, game_status="game_over"))
        self.assertEqual(session.policy.resets, 3)

    def test_late_request_is_rejected_without_resetting(self):
        session = PolicySession(StatefulFixture, stateful=True, single_stream=True)
        session.predict(frame(2))
        with self.assertRaises(SessionConflict):
            session.predict(frame(1))
        self.assertEqual(session.policy.resets, 0)

    def test_omitted_time_uses_retry_signature_and_first_tick_reset(self):
        session = PolicySession(StatefulFixture, stateful=True)
        first = StepResponse.model_validate({
            "game_status": "ok", "score": 0.1, "n_agents": 1,
            "agent_status": [frame().agent_status[0].model_copy(update={"age": 0.1})],
        })
        later = StepResponse.model_validate({
            "game_status": "ok", "score": 0.2, "n_agents": 1,
            "agent_status": [frame().agent_status[0].model_copy(update={"age": 0.2})],
        })
        self.assertNotIn("sim_time", first.model_fields_set)
        self.assertEqual(session.predict(first)[0].move_distance, 1)
        self.assertEqual(session.predict(first)[0].move_distance, 1)
        self.assertEqual(session.predict(later)[0].move_distance, 2)
        self.assertEqual(session.policy.calls, 2)
        self.assertEqual(session.policy.times, [0.1, 0.2])

        next_game = first.model_copy(update={"score": 0.1})
        next_game.model_fields_set.discard("sim_time")
        self.assertEqual(session.predict(next_game)[0].move_distance, 3)
        self.assertEqual(session.policy.resets, 1)
        self.assertEqual(session.policy.times[-1], 0.1)

    def test_stateless_requests_have_no_cross_game_order_constraint(self):
        session = PolicySession(StatefulFixture)
        session.predict(frame(20))
        session.predict(frame(0.1))
        self.assertEqual(session.policy.calls, 2)
        self.assertEqual(session.predict(frame(30, game_status="game_over")), [])


if __name__ == "__main__":
    unittest.main()
