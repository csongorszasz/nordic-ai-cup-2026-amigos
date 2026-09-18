import contextlib
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import local_playground
from src.utils.DTOs import ActionRequest
from test_policy_hierarchical import agent


def state(*agents, sim_time=0.1):
    return {
        "score": sim_time, "sim_time": sim_time, "num_agents": len(agents),
        "observations": [value.model_dump() for value in agents],
    }


class PlaygroundTests(unittest.TestCase):
    def setUp(self):
        self.sim = Mock(env_width=1600, env_height=1200)
        self.policy = Mock()
        self.output = io.StringIO()
        contexts = contextlib.ExitStack()
        self.addCleanup(contexts.close)
        for target, replacement in (
            ("SimulationCore", Mock(return_value=self.sim)),
            ("create_policy", Mock(return_value=self.policy)),
            ("read_json", Mock(return_value={"policy": "neural"})),
            ("pygame", Mock()),
        ):
            contexts.enter_context(patch.object(local_playground, target, replacement))
        contexts.enter_context(contextlib.redirect_stdout(self.output))

    def test_config_constructs_one_policy_and_passes_the_whole_team_each_tick(self):
        self.sim.step.side_effect = [
            state(agent(2), agent(8)), state(agent(2), agent(9), sim_time=0.2), state(sim_time=0.3),
        ]
        self.policy.act.side_effect = lambda step: [
            ActionRequest(
                agent_id=value.agent_id, move_distance=0.0, move_direction=0.0,
                turn_angle=0.0, spawn_agent=False,
            )
            for value in step.agent_status
        ]
        config = Path("trained-policy.json")
        local_playground.local_simulation(verbose=False, config=config, seed=42)
        local_playground.create_policy.assert_called_once_with(42, {"policy": "neural"})
        local_playground.read_json.assert_called_once_with(config)
        local_playground.SimulationCore.assert_called_once_with(seed=42)
        self.sim.step.assert_any_call([])
        self.assertEqual(self.policy.act.call_count, 2)
        self.assertEqual(
            [[value.agent_id for value in call.args[0].agent_status] for call in self.policy.act.call_args_list],
            [[2, 8], [2, 9]],
        )
        self.assertEqual([item[0] for item in self.sim.step.call_args_list[1].args[0]], [2, 8])
        local_playground.pygame.display.set_mode.assert_not_called()
        local_playground.pygame.quit.assert_called_once()

    def test_no_config_preserves_the_random_policy(self):
        self.sim.step.side_effect = [state(agent(2)), state()]
        local_playground.local_simulation(verbose=False, seed=42)
        local_playground.create_policy.assert_not_called()
        self.assertEqual(self.sim.step.call_args_list[1].args[0][0][0], 2)

    def test_step_cap_includes_the_initial_empty_action_tick(self):
        self.sim.step.return_value = state(agent(2))
        local_playground.local_simulation(verbose=False, config="policy.json", seed=42, max_steps=1)
        self.sim.step.assert_called_once_with([])
        self.policy.act.assert_not_called()

    def test_invalid_policy_actions_fail_and_close_pygame(self):
        self.sim.step.return_value = state(agent(2))
        self.policy.act.return_value = []
        with self.assertRaises(ValueError):
            local_playground.local_simulation(verbose=False, config="policy.json", seed=42)
        local_playground.pygame.quit.assert_called_once()

    def test_rendering_uses_requested_fps_and_stays_inside_display(self):
        local_playground.pygame.display.Info.return_value = SimpleNamespace(current_w=1000, current_h=800)
        local_playground.pygame.event.get.return_value = []
        self.sim.step.return_value = state(agent(2))
        local_playground.local_simulation(seed=42, fps=30, max_steps=1)
        size = local_playground.pygame.display.set_mode.call_args.args[0]
        self.assertLessEqual(size[0], 1000)
        self.assertLessEqual(size[1], 800)
        self.sim.env.draw.assert_called_once()
        local_playground.pygame.time.Clock.return_value.tick.assert_called_once_with(30)
        local_playground.pygame.display.flip.assert_called_once()

    def test_window_close_stops_without_an_extra_simulation_step(self):
        local_playground.pygame.display.Info.return_value = SimpleNamespace(current_w=1000, current_h=800)
        local_playground.pygame.event.get.return_value = [SimpleNamespace(type=local_playground.pygame.QUIT)]
        local_playground.local_simulation(seed=42)
        self.sim.step.assert_not_called()
        local_playground.pygame.quit.assert_called_once()

    def test_cli_forwards_options(self):
        with patch.object(local_playground, "local_simulation") as run:
            result = local_playground.main([
                "--config", "policy.json", "--seed", "42", "--fps", "30", "--headless", "--max-steps", "2",
            ])
        self.assertEqual(result, 0)
        run.assert_called_once_with(verbose=False, config=Path("policy.json"), seed=42, fps=30, max_steps=2)

    def test_cli_rejects_invalid_numeric_options(self):
        for arguments in (["--seed", "-1"], ["--seed", str(2**32)], ["--fps", "0"], ["--max-steps", "0"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    local_playground.main(arguments)
                self.assertEqual(raised.exception.code, 2)

    def test_missing_checkpoint_reports_failure_instead_of_random_fallback(self):
        local_playground.create_policy.side_effect = FileNotFoundError("missing checkpoint")
        with contextlib.redirect_stderr(io.StringIO()) as error:
            result = local_playground.main(["--config", "policy.json", "--seed", "42"])
        self.assertEqual(result, 1)
        self.assertIn("missing checkpoint", error.getvalue())
        local_playground.SimulationCore.assert_not_called()


if __name__ == "__main__":
    unittest.main()