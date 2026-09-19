import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.benchmarking.config import EpisodeResult, GitInfo, Timings, read_json
from src.policies.networks import PolicyNetwork
from src.training.artifacts import write_json, load_checkpoint
from src.training.evaluation import (
    EvaluationBackpressure, check_capacity, evaluate_pending, evaluation_lock,
    initialize_schedule, pending_updates, publish_snapshot, retry_failed, snapshot_directory,
)
from src.training.progress import progress_data, render_progress
from src.training.learner import run_learning
from src.training.rollout import RolloutCollector
from src.training.preflight import build_run_plan
from idun.watch_checkpoints import monitor_scheduled
from tests.test_pipeline_contract import pipeline_config
from tests.test_training_rollout import ScriptedEnvironment
from src.benchmarking.config import PROJECT_ROOT
import train


def fake_episode(case, descriptor, output, timeout):
    score = 20.0 if descriptor["policy"] == "heuristic" else 5.0
    return EpisodeResult(
        case=case, status="ok", termination="extinction", score=score + case.world_seed % 3,
        sim_time=0.2, survival_seconds=0.2, ticks=2, initial_agents=5,
        final_agents=0, peak_agents=5, mean_population=2.5,
        timings=Timings(policy_calls=1, policy_seconds=0.001, batch_mean_ms=1,
                        batch_p50_ms=1, batch_p95_ms=1, batch_max_ms=1),
    )


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.run = Path(directory.name) / "run"
        self.run.mkdir()
        self.config = pipeline_config()
        self.network = PolicyNetwork(self.config.model)
        self.guard = patch("src.core.SimulationCore.__init__", side_effect=AssertionError("No simulator"))
        self.guard.start()
        self.addCleanup(self.guard.stop)
        git = patch("src.benchmarking.artifacts._git_info",
                    return_value=GitInfo(revision=None, dirty=None, status="synthetic fixture"))
        git.start()
        self.addCleanup(git.stop)
        self.schedule = initialize_schedule(self.run, self.config, {"run_id": "fixture"})

    def publish(self, update):
        return publish_snapshot(self.run, self.network, self.config, {
            "next_update": update, "native_ticks_total": update * 11,
            "prior_native_ticks": 7, "optimizer_steps": update,
        }, self.schedule)

    def test_publication_is_immutable_slim_and_rng_neutral(self):
        rng = torch.random.get_rng_state().clone()
        first = self.publish(0)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(first, self.publish(0))
        loaded = load_checkpoint(snapshot_directory(self.run, 0) / "checkpoint.pt")
        self.assertIsNone(loaded.optimizer_state)
        self.assertEqual(loaded.rng_state, {})
        self.assertNotIn("imitation_dataset", loaded.training_state)
        self.assertEqual(pending_updates(self.run), [0])
        with torch.no_grad():
            next(self.network.parameters()).add_(1)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.publish(0)

    def test_scheduled_points_full_cases_plots_and_idempotent_restart(self):
        self.publish(0)
        self.publish(1)
        with patch("subprocess.Popen", side_effect=AssertionError("No child jobs")), \
                patch("src.training.evaluation._execute_episode", side_effect=fake_episode) as executor:
            evaluate_pending(self.run)
            self.assertEqual(executor.call_count, 9)
            evaluate_pending(self.run)
            self.assertEqual(executor.call_count, 9)
        data = progress_data(self.run)
        scores = [5.0 + seed % 3 for seed in self.schedule["suite"]["seeds"]]
        self.assertAlmostEqual(data["points"][0]["mean_score"], sum(scores) / 3)
        self.assertEqual(data["points"][0]["native_ticks"], 7)
        self.assertEqual(data["points"][1]["native_ticks"], 18)
        self.assertEqual(data["points"][2]["state"], "not_published")
        self.assertEqual(data["latest_evaluated_update"], 1)
        paths = render_progress(self.run)
        for path in paths:
            self.assertGreater(path.stat().st_size, 1000)
            self.assertTrue(path.read_bytes().startswith(b"\x89PNG"))
        self.assertIn("mean_score", (self.run / "progress" / "learning-curve.csv").read_text())

    def test_failed_or_interrupted_cases_never_produce_partial_mean(self):
        self.publish(0)
        counts = []

        def fail_candidate(case, descriptor, output, timeout):
            counts.append(case)
            if descriptor["policy"] == "neural":
                raise RuntimeError("fixture evaluation failure")
            return fake_episode(case, descriptor, output, timeout)

        evaluate_pending(self.run, executor=fail_candidate, refresh=False)
        row = read_json(snapshot_directory(self.run, 0) / "result.json")
        self.assertEqual(row["state"], "failed")
        self.assertNotIn("mean_score", row)
        before = len(counts)
        evaluate_pending(self.run, executor=fail_candidate, refresh=False)
        self.assertEqual(len(counts), before)
        with self.assertRaisesRegex(ValueError, "max_attempts"):
            retry_failed(self.run, "0", "User requests a retry")
        self.assertIsNone(progress_data(self.run)["points"][0]["mean_score"])

    def test_explicit_retry_reuses_successes_within_the_planned_cap(self):
        self.run = self.run / "retry-plan"
        self.run.mkdir()
        self.config = self.config.model_copy(update={
            "evaluation": self.config.evaluation.model_copy(update={"max_attempts": 2}),
        })
        self.schedule = initialize_schedule(self.run, self.config, {"run_id": "fixture"})
        self.publish(0)
        failed = []

        def once(case, descriptor, output, timeout):
            if descriptor["policy"] == "neural" and not failed:
                failed.append(case)
                raise RuntimeError("recoverable fixture")
            return fake_episode(case, descriptor, output, timeout)

        evaluate_pending(self.run, executor=once, refresh=False)
        retry_failed(self.run, "0", "Fixture cause fixed by user")
        evaluate_pending(self.run, executor=once, refresh=False)
        self.assertEqual(progress_data(self.run)["points"][0]["state"], "complete")
        with self.assertRaisesRegex(ValueError, "Only a failed"):
            retry_failed(self.run, "0", "Do not rerun success")

    def test_queue_limit_pauses_without_dropping_a_request(self):
        self.publish(0)
        config = self.config.model_copy(update={"evaluation": self.config.evaluation.model_copy(update={"max_pending": 1})})
        with self.assertRaises(EvaluationBackpressure):
            check_capacity(self.run, config)
        self.assertTrue((snapshot_directory(self.run, 0) / "ready.json").exists())
        self.assertEqual(pending_updates(self.run), [0])
        evaluate_pending(self.run, executor=fake_episode, refresh=False)
        check_capacity(self.run, config)

    def test_graph_rejects_partial_or_incompatible_results(self):
        self.publish(0)
        evaluate_pending(self.run, executor=fake_episode, refresh=False)
        path = snapshot_directory(self.run, 0) / "result.json"
        record = read_json(path)
        record["scores"].pop()
        write_json(path, record)
        with self.assertRaisesRegex(ValueError, "every expected"):
            progress_data(self.run)
        record["protocol_id"] = "other"
        write_json(path, record)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            progress_data(self.run)

    def test_partial_publication_is_invisible_and_conflicting_evaluator_is_locked(self):
        staging = self.run / "evaluation" / "checkpoints" / ".pending-incomplete"
        staging.mkdir(parents=True)
        write_json(staging / "ready.json", {"update": 99})
        self.assertEqual(pending_updates(self.run), [])
        with evaluation_lock(self.run / "evaluation"):
            with self.assertRaises(OSError):
                with evaluation_lock(self.run / "evaluation"):
                    self.fail("Second evaluator acquired the lock")

    def test_resume_keeps_curve_history_without_repeating_old_episodes(self):
        self.publish(0)
        self.publish(1)
        evaluate_pending(self.run, executor=fake_episode, refresh=False)
        resumed = self.run.parent / "resumed"
        resumed.mkdir()
        schedule = initialize_schedule(
            resumed, self.config, {"run_id": "resumed"}, lineage={
                "kind": "resume", "parent_run": str(self.run), "resume_update": 1,
            },
        )
        self.assertEqual(schedule["updates"], [2])
        data = progress_data(resumed)
        self.assertEqual([point["update"] for point in data["points"]], [0, 1, 2])
        self.assertEqual(data["latest_evaluated_update"], 1)
        self.assertIsNotNone(data["teacher_mean"])

    def test_watcher_drains_only_authorized_requests_with_mock_scheduler(self):
        self.publish(0)
        write_json(self.run / "manifest.json", {"status": "complete"})
        with patch("idun.watch_checkpoints.training_active", return_value=False) as scheduler, \
                patch("src.training.evaluation._execute_episode", side_effect=fake_episode):
            result = monitor_scheduled(self.run, 123, poll_seconds=0)
        scheduler.assert_called_once_with(123)
        self.assertEqual(result["state"], "incomplete")  # Future updates weren't fabricated.

    def test_both_learning_modes_publish_due_weights_using_synthetic_rollouts(self):
        for mode in ("ppo", "imitation"):
            with self.subTest(mode=mode):
                config = pipeline_config(mode)
                destination = self.run / mode
                destination.mkdir()
                schedule = initialize_schedule(destination, config, {"run_id": mode})
                network = PolicyNetwork(config.model)
                optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
                before = next(network.parameters()).detach().clone()
                checkpoints = []

                def collector(config, network, *, emit, state):
                    return RolloutCollector(
                        config, network, emit=emit, state=state,
                        environment_factory=lambda: ScriptedEnvironment(
                            ((0, 1), (1, 2), (2, 3)), (0.1, 0.2), terminal_at_end=False,
                        ),
                    )

                publish_snapshot(destination, network, config, {"next_update": 0}, schedule)
                with patch("src.training.learner.RolloutCollector", side_effect=collector):
                    result = run_learning(
                        config, network, optimizer, emit=lambda _: None, checkpoint=checkpoints.append,
                        update_callback=lambda state: publish_snapshot(destination, network, config, state, schedule),
                    )
                self.assertEqual(result["next_update"], 2)
                self.assertEqual(pending_updates(destination), [0, 1, 2])
                self.assertEqual(result["native_ticks_total"], 5)
                self.assertEqual(len(checkpoints), 2)
                # This fixture has no entities, so check actor parameters rather than unused entity MLPs.
                self.assertTrue(any(parameter.grad is not None for parameter in network.parameters()))
                evaluate_pending(destination, executor=fake_episode, refresh=False)
                self.assertEqual(progress_data(destination)["latest_evaluated_update"], 2)

    def test_cli_run_pause_evaluate_resume_uses_only_synthetic_environment(self):
        config = pipeline_config("imitation")
        config = config.model_copy(update={
            "evaluation": config.evaluation.model_copy(update={"max_pending": 2}),
        })
        evidence = read_json(PROJECT_ROOT / "configs" / "evidence-diagnostic.json")
        plan_file = self.run.parent / "run-plan.json"
        write_json(plan_file, build_run_plan(config, evidence))
        first = self.run.parent / "first"

        def collector(settings, network, *, emit, state):
            return RolloutCollector(
                settings, network, emit=emit, state=state,
                environment_factory=lambda: ScriptedEnvironment(
                    ((0, 1), (1, 2), (2, 3)), (0.1, 0.2), terminal_at_end=False,
                ),
            )

        with patch("src.training.learner.RolloutCollector", side_effect=collector):
            self.assertEqual(train.execute(config, first, run_plan=plan_file), 3)
        self.assertEqual(read_json(first / "manifest.json")["status"], "paused")
        self.assertEqual(load_checkpoint(first / "checkpoint.pt").training_state["next_update"], 1)
        evaluate_pending(first, executor=fake_episode)
        resume_file = self.run.parent / "resume-plan.json"
        write_json(resume_file, build_run_plan(config, evidence, resume=first / "checkpoint.pt"))
        resumed = self.run.parent / "second"
        with patch("src.training.learner.RolloutCollector", side_effect=collector):
            self.assertEqual(train.execute(config, resumed, first / "checkpoint.pt", run_plan=resume_file), 0)
        evaluate_pending(resumed, executor=fake_episode)
        data = progress_data(resumed)
        self.assertEqual(data["latest_evaluated_update"], 2)
        self.assertEqual([point["update"] for point in data["points"]], [0, 1, 2])
        self.assertEqual([point["native_ticks"] for point in data["points"]], [0, 3, 6])

    def test_episode_timeout_terminates_only_the_owned_child(self):
        from src.training.evaluation import _execute_episode
        from src.benchmarking.config import make_cases, Suite
        import subprocess
        from unittest.mock import Mock

        process = Mock()
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 1), 0]
        with patch("src.training.evaluation.subprocess.Popen", return_value=process):
            with self.assertRaises(subprocess.TimeoutExpired):
                _execute_episode(make_cases(Suite(name="fixture", seeds=[1]))[0],
                                 {}, self.run / "timed-out.json", 1)
        process.terminate.assert_called_once()
        process.kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
