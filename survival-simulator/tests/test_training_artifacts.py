import random
import copy
import tempfile
import unittest
from pathlib import Path

import torch

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
