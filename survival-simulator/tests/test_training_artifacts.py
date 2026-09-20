import random
import copy
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch

import torch

from idun.watch_checkpoints import monitor, snapshot_latest
from src.benchmarking.config import read_json
from src.policies.config import ExperimentConfig, ModelConfig
from src.policies.networks import PolicyNetwork
from src.training.artifacts import (
    TrainingRun, load_checkpoint, restore_rng, save_checkpoint, verify_resume_provenance, write_json,
)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig(model=ModelConfig(hidden_size=32, entity_size=8))
        self.network = PolicyNetwork(self.config.model)

    def test_weights_optimizer_configuration_and_rng_round_trip(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            digest = save_checkpoint(path, self.network, self.config, optimizer, {"next_update": 3})
            expected_random = random.random()
            expected_tensor = torch.rand(3)
            state_before_load = torch.random.get_rng_state().clone()
            loaded = load_checkpoint(path, expected_model=self.config.model, expected_sha256=digest)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), state_before_load))
            for key, value in self.network.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.network.state_dict()[key]))
            self.assertEqual(loaded.training_state["next_update"], 3)
            self.assertEqual(loaded.config, self.config)
            self.assertEqual(loaded.optimizer_state["param_groups"][0]["lr"], 0.001)
            restore_rng(loaded.rng_state)
            self.assertEqual(random.random(), expected_random)
            self.assertTrue(torch.equal(torch.rand(3), expected_tensor))

    def test_tampered_weights_and_wrong_architecture_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, self.network, self.config)
            with self.assertRaisesRegex(ValueError, "architecture"):
                load_checkpoint(path, expected_model=ModelConfig(hidden_size=64))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_checkpoint(path, expected_sha256="0" * 64)
            with path.open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_checkpoint(path)

    def test_nonfinite_weights_are_rejected(self):
        with torch.no_grad():
            next(self.network.parameters()).fill_(float("nan"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, self.network, self.config)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_checkpoint(path)


class CheckpointSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.run_path = self.root / "training"
        self.run_path.mkdir()
        self.output = self.root / "evaluation"
        self.config = ExperimentConfig(model=ModelConfig(hidden_size=32, entity_size=8))
        self.network = PolicyNetwork(self.config.model)

    def publish(self, update):
        checkpoint = self.run_path / "checkpoint.pt"
        digest = save_checkpoint(checkpoint, self.network, self.config, training_state={"next_update": update})
        write_json(self.run_path / "policy.json", {
            "policy": "neural", "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest, "device": "cuda",
        })
        return digest

    def test_snapshot_stays_fixed_and_uses_cpu_when_training_advances(self):
        digest = self.publish(3)
        snapshot = snapshot_latest(self.run_path, self.output)
        self.assertEqual(snapshot["update"], 3)
        frozen = Path(snapshot["path"])
        descriptor = read_json(frozen / "policy.json")
        self.assertEqual(descriptor["device"], "cpu")
        self.assertEqual(Path(descriptor["checkpoint"]), frozen / "checkpoint.pt")
        self.assertEqual(snapshot_latest(self.run_path, self.output), snapshot)
        self.assertIsNone(snapshot_latest(self.run_path, self.output, digest))
        self.publish(4)
        loaded = load_checkpoint(frozen / "checkpoint.pt", expected_sha256=digest)
        self.assertEqual(loaded.training_state["next_update"], 3)
        self.assertEqual(snapshot_latest(self.run_path, self.output, digest)["update"], 4)

    def test_incomplete_or_mixed_generations_are_retried_without_publication(self):
        self.assertIsNone(snapshot_latest(self.run_path, self.output))
        self.publish(1)
        old_descriptor = read_json(self.run_path / "policy.json")
        self.publish(2)
        new_descriptor = read_json(self.run_path / "policy.json")
        write_json(self.run_path / "policy.json", old_descriptor)
        self.assertIsNone(snapshot_latest(self.run_path, self.output))
        self.assertEqual(list((self.output / "snapshots").iterdir()), [])
        write_json(self.run_path / "policy.json", new_descriptor)
        self.assertEqual(snapshot_latest(self.run_path, self.output)["update"], 2)

    def test_capture_continues_while_evaluation_is_busy_and_drains_on_exit(self):
        self.publish(1)
        evaluating = Event()
        release = Event()
        checks = []

        def job_active(job_id):
            checks.append(job_id)
            if len(checks) == 2:
                self.assertTrue(evaluating.wait(5))
                self.publish(2)
            if len(checks) == 3:
                release.set()
                return False
            return True

        def evaluate(snapshot, output, reference):
            if snapshot["update"] == 1:
                evaluating.set()
                self.assertTrue(release.wait(5))
            return {"mean_score": snapshot["update"] * 10}

        with patch("idun.watch_checkpoints.training_active", side_effect=job_active), \
                patch("idun.watch_checkpoints.evaluate_snapshot", side_effect=evaluate):
            status = monitor(self.run_path, self.output, self.root / "reference", 123,
                             poll_seconds=0, scheduler_seconds=0)
        self.assertEqual(status["state"], "complete")
        self.assertEqual(status["missing_updates"], [])
        self.assertEqual(set(status["updates"]), {"1", "2"})
        self.assertTrue(all(record["state"] == "complete" for record in status["updates"].values()))

    def test_failed_evaluation_is_recorded_and_not_resubmitted_on_restart(self):
        self.publish(1)
        with patch("idun.watch_checkpoints.training_active", return_value=False), \
                patch("idun.watch_checkpoints.evaluate_snapshot", side_effect=RuntimeError("benchmark failed")) as evaluate:
            status = monitor(self.run_path, self.output, self.root / "reference", 123, poll_seconds=0)
            self.assertEqual(status["state"], "finished_with_errors")
            self.assertEqual(status["updates"]["1"]["state"], "failed")
            monitor(self.run_path, self.output, self.root / "reference", 123, poll_seconds=0)
            self.assertEqual(evaluate.call_count, 1)


class TrainingRunTests(unittest.TestCase):
    def test_resume_rejects_changed_world_stream_and_runtime(self):
        files = {name: "a" * 64 for name in (
            "src\\training\\seeds.py", "benchmarks\\suites\\quick.json",
            "benchmarks\\suites\\standard.json", "benchmarks\\suites\\holdout.json",
        )}
        manifest = {
            "provenance": {"engine": {"sha256": "engine"}, "runtime": {"python": "3.12"},
                           "artifacts": {"files": files}},
            "training_dependencies": {"torch": "2.8.0"},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            with self.assertRaisesRegex(ValueError, "manifest"):
                verify_resume_provenance(checkpoint, manifest)
            write_json(root / "manifest.json", manifest)
            verify_resume_provenance(checkpoint, manifest)
            for field in ("suite", "engine", "runtime", "torch"):
                changed = copy.deepcopy(manifest)
                if field == "suite":
                    changed["provenance"]["artifacts"]["files"]["benchmarks\\suites\\holdout.json"] = "b" * 64
                elif field == "torch":
                    changed["training_dependencies"]["torch"] = "different"
                else:
                    changed["provenance"][field] = {"changed": True}
                with self.subTest(field=field), self.assertRaises(ValueError):
                    verify_resume_provenance(checkpoint, changed)

    def test_records_are_incremental_and_outputs_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            run = TrainingRun(path, ExperimentConfig())
            run.emit({"event": "update", "score": -1.0})
            with self.assertRaises(ValueError):
                run.emit({"event": "bad", "value": float("nan")})
            run.finish("complete", {"updates": 1})
            self.assertIn('"score":-1.0', (path / "events.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(run.manifest["status"], "complete")
            with self.assertRaises(ValueError):
                run.finish("complete", {})
            with self.assertRaises(FileExistsError):
                TrainingRun(path, ExperimentConfig())


if __name__ == "__main__":
    unittest.main()
