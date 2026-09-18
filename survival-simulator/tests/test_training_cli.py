import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

import train
from src.benchmarking.config import EpisodeCase, EpisodeResult
from src.policies.config import ExperimentConfig, ResourceConfig, SearchConfig
from src.training.artifacts import write_json


class TrainingCLITests(unittest.TestCase):
    def test_candidate_world_batches_preserve_identities_and_incremental_events(self):
        class ImmediatePool:
            def __init__(self, **kwargs):
                self.workers = kwargs["max_workers"]

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def submit(self, function, argument):
                future = Future()
                future.set_result(function(argument))
                return future

        def episode(arguments):
            seed, options = arguments
            return EpisodeResult(
                case=EpisodeCase(case_id=f"world-{seed}-repeat-0", world_seed=seed,
                                 policy_seed=seed, repeat_index=0),
                status="ok", termination="extinction", score=seed + options["danger_weight"],
                sim_time=1.0, survival_seconds=1.0, ticks=10, initial_agents=5,
                final_agents=0, peak_agents=5, mean_population=3.0,
            )

        with tempfile.TemporaryDirectory() as directory:
            events = []
            run = SimpleNamespace(path=Path(directory), emit=events.append)
            config = ExperimentConfig(
                mode="search", policy="heuristic", search=SearchConfig(candidates=3, worlds=2),
                resources=ResourceConfig(workers=4),
            )
            with patch("train.training_seeds", return_value=[91, 97]), \
                    patch("train._search_episode", side_effect=episode), \
                    patch("train.ProcessPoolExecutor", ImmediatePool), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = train.run_search(config, run)
            completed = [event for event in events if event["event"] == "search_episode"]
            summaries = [
                event for event in events if event["event"] == "search_candidate_summary"
            ]
            self.assertEqual(len(completed), 6)
            self.assertEqual(len(summaries), 3)
            self.assertEqual({event["candidate"] for event in completed}, {0, 1, 2})
            for candidate in range(3):
                self.assertEqual(
                    {event["result"]["case"]["world_seed"] for event in completed
                     if event["candidate"] == candidate}, {91, 97},
                )
            self.assertEqual(result["trials"], 3)
            self.assertIn("best_training_lower_tail_score", result)
            self.assertTrue((run.path / "policy.json").is_file())

    def test_config_override_and_profile_orchestration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "profile.json"
            write_json(config, {"mode": "profile", "policy": "heuristic"})
            with patch("train.run_profile", return_value={"workers": 2}) as profile:
                with contextlib.redirect_stdout(io.StringIO()):
                    code = train.main(["--config", str(config), "--output", str(root / "run"),
                                       "--set", "resources.workers=2"])
            self.assertEqual(code, 0)
            self.assertEqual(profile.call_args.args[0].resources.workers, 2)
            self.assertTrue((root / "run" / "summary.json").is_file())

    def test_invalid_override_does_not_create_a_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "profile.json"
            write_json(config, {"mode": "profile", "policy": "heuristic"})
            with contextlib.redirect_stderr(io.StringIO()):
                code = train.main(["--config", str(config), "--output", str(root / "run"),
                                   "--set", "model.unsupported=true"])
            self.assertEqual(code, 1)
            self.assertFalse((root / "run").exists())

    def test_failure_is_recorded_and_propagated_not_a_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            with patch("train.run_profile", side_effect=RuntimeError("worker failed")):
                with self.assertRaisesRegex(RuntimeError, "worker failed"):
                    train.execute(ExperimentConfig(mode="profile", policy="heuristic"), root)
            self.assertIn('"status": "failed"', (root / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("worker failed", (root / "summary.json").read_text(encoding="utf-8"))

    def test_resume_rejected_for_search(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "--resume"):
                train.execute(ExperimentConfig(mode="search", policy="heuristic"),
                              root / "run", root / "checkpoint.pt")
            self.assertFalse((root / "run").exists())


if __name__ == "__main__":
    unittest.main()
