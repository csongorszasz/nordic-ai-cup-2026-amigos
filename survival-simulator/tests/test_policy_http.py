import socket
import threading
import time
import unittest

import requests
import uvicorn

from agent_server import create_app
from src.benchmarking.config import PROJECT_ROOT, read_json
from src.policies.config import RuntimeConfig
from src.policies.runtime import create_policy
from src.utils.DTOs import ObservationResponse, StepResponse


class HTTPPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.options = RuntimeConfig.model_validate(
            read_json(PROJECT_ROOT / "configs" / "controller.json"),
        ).model_dump(mode="json")
        application = create_app(policy_factory=lambda: create_policy(1, cls.options))
        cls.socket = socket.socket()
        cls.socket.bind(("127.0.0.1", 0))
        cls.port = cls.socket.getsockname()[1]
        cls.server = uvicorn.Server(uvicorn.Config(
            application, host="127.0.0.1", port=cls.port, log_level="error",
        ))
        cls.thread = threading.Thread(
            target=cls.server.run, kwargs={"sockets": [cls.socket]}, daemon=True,
        )
        cls.thread.start()
        deadline = time.monotonic() + 15
        while not cls.server.started and cls.thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not cls.server.started:
            cls.server.should_exit = True
            cls.thread.join(5)
            cls.socket.close()
            raise RuntimeError("Local policy HTTP fixture did not start.")
        cls.url = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(10)
        cls.socket.close()
        if cls.thread.is_alive():
            raise RuntimeError("Local policy HTTP fixture did not stop.")

    def test_health_bootstrap_and_same_decisions_as_direct_policy(self):
        response = requests.get(self.url, timeout=5)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Agent endpoint running!")
        bootstrap = StepResponse(game_status="ok", sim_time=0, score=0, n_agents=5, agent_status=[])
        response = requests.post(self.url + "/predict", json=bootstrap.model_dump(), timeout=5)
        self.assertEqual(response.json(), {"actions": []})
        agent = ObservationResponse(
            agent_id=3, energy=150, age=1, biome="forest", speed=10, sprint_speed=20,
            hearing_radius=50, vision_angle=1, vision_range=200, max_energy=500,
            observations=[{"type": "Fruit", "distance": 20, "angle": 0.3}],
        )
        frame = StepResponse(game_status="ok", sim_time=1, score=1, n_agents=1, agent_status=[agent])
        expected = [a.model_dump(mode="json") for a in create_policy(1, self.options).act(frame)]
        response = requests.post(self.url + "/predict", json=frame.model_dump(), timeout=5)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"actions": expected})

    def test_official_verifier_sample_is_normalized(self):
        response = requests.post(self.url + "/predict", timeout=5, json={
            "game_status": "running",
            "score": 123.4,
            "agent_status": [{
                "agent_id": 1,
                "observations": [
                    {"type": "tree", "distance": 12.5, "angle": 1.57},
                    {"type": "predator", "distance": 30.0, "angle": -1.57, "rel_dir": -2.57},
                    {"type": "edge", "coords": [[50.0, 50.0], [100.0, 100.0]]},
                ],
                "energy": 85.0,
                "biome": "forest",
                "age": 5.2,
                "speed": 12.5,
                "sprint_speed": 13.5,
                "hearing_radius": 10.0,
                "vision_angle": 1.57,
                "vision_range": 50.0,
                "max_energy": 500.0,
            }],
        })
        self.assertEqual(response.status_code, 200, response.text)
        actions = response.json()["actions"]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["agent_id"], 1)

    def test_invalid_count_returns_explicit_client_error(self):
        response = requests.post(self.url + "/predict", timeout=5, json={
            "game_status": "ok", "sim_time": 1, "score": 1, "n_agents": 1, "agent_status": [],
        })
        self.assertEqual(response.status_code, 422)

    def test_invalid_scalar_type_returns_explicit_client_error(self):
        response = requests.post(self.url + "/predict", timeout=5, json={
            "game_status": "running", "score": "1.0",
            "agent_status": [],
        })
        self.assertEqual(response.status_code, 422)

    def test_stateful_default_accepts_consecutive_official_frames_without_sim_time(self):
        payload = {
            "game_status": "running", "score": 0.1,
            "agent_status": [{
                "agent_id": 0, "observations": [], "energy": 150.0,
                "biome": "forest", "age": 0.1, "speed": 10.0,
                "sprint_speed": 20.0, "hearing_radius": 50.0,
                "vision_angle": 1.0, "vision_range": 200.0, "max_energy": 500.0,
            }],
        }
        first = requests.post(self.url + "/predict", timeout=5, json=payload)
        payload["score"] = 0.2
        payload["agent_status"][0]["age"] = 0.2
        second = requests.post(self.url + "/predict", timeout=5, json=payload)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)


if __name__ == "__main__":
    unittest.main()
