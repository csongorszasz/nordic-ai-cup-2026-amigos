import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.benchmarking.config import GitInfo, read_json
from src.policies.networks import PolicyNetwork
from src.training.artifacts import file_hash, load_checkpoint, save_checkpoint, write_json
from src.training.bc_corpus_fit import main as fit
from src.training.dagger_collection import aggregate_corpora, collect_recovery_episode, prepare_anchor
from src.training.demonstrations import read_episode
from src.training.teacher import resolve_teacher
from tests.test_bc_corpus_fit import synthetic_corpus
from tests.test_pipeline_contract import pipeline_config
from tests.test_training_rollout import ConstantTeacher, ScriptedEnvironment


class CorrectedDAggerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = pipeline_config("imitation")
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.checkpoint = self.root / "student.pt"
        network = PolicyNetwork(self.config.model)
        with torch.no_grad():
            network.mean_head.weight.zero_()
            network.mean_head.bias.zero_()
            network.spawn_head.bias.fill_(-20)
        save_checkpoint(self.checkpoint, network, self.config)
        guard = patch("src.core.SimulationCore.__init__", side_effect=AssertionError("No real simulation"))
        guard.start()
        self.addCleanup(guard.stop)
        git = patch("src.benchmarking.artifacts._git_info",
                    return_value=GitInfo(revision=None, dirty=None, status="synthetic"))
        git.start()
        self.addCleanup(git.stop)

    def test_deterministic_student_whole_team_mixing_and_actual_history(self):
        env = ScriptedEnvironment(((0, 1), (1, 2), (2, 3), ()), (0.1, 0.2, 0.3))
        class Mixture:
            def __init__(self, *_):
                self.values = iter((0.1, 0.9, 0.1))
            def random(self):
                return next(self.values)
        with patch("src.training.dagger_collection.build_policy", side_effect=ConstantTeacher), \
                patch("src.training.dagger_collection.random.Random", Mixture), \
                patch("torch.randn", side_effect=AssertionError("No stochastic student noise")), \
                patch("torch.rand", side_effect=AssertionError("No stochastic student birth draws")):
            result = collect_recovery_episode(
                (91, str(self.root / "recovery"), str(self.checkpoint), 0.5, 5),
                environment_factory=lambda **_: env,
            )
        self.assertEqual(result["coverage"]["teacher_team_ticks"], 2)
        self.assertEqual(result["coverage"]["student_team_ticks"], 1)
        self.assertTrue(all(action.move_distance == 4 for action in env.executed[0]))
        self.assertTrue(all(action.move_distance == 10 for action in env.executed[1]))
        frames = read_episode(self.root / "recovery", result)
        self.assertTrue(all(action.move_distance == 4 for action in frames[2][1]))  # Expert labels.
        self.assertTrue(all(action.move_distance == 10 for action in frames[2][2].values()))  # Executed history.

    def test_truncated_recovery_is_never_published_as_complete(self):
        env = ScriptedEnvironment(terminal_at_end=False)
        with patch("src.training.dagger_collection.build_policy", side_effect=ConstantTeacher):
            with self.assertRaisesRegex(RuntimeError, "truncated"):
                collect_recovery_episode(
                    (91, str(self.root / "short"), str(self.checkpoint), 0.5, 1),
                    environment_factory=lambda **_: env,
                )
        self.assertFalse((self.root / "short" / "train" / "world-91").exists())

    def make_recovery(self, expert):
        destination = self.root / "recovery"
        env = ScriptedEnvironment(((0, 1), (1, 2), ()), (0.1, 0.2))
        with patch("src.training.dagger_collection.build_policy", side_effect=ConstantTeacher):
            result = collect_recovery_episode(
                (123987, str(destination), str(self.checkpoint), 0.5, 5),
                environment_factory=lambda **_: env,
            )
        write_json(destination / "manifest.json", {
            "status": "complete", "teacher": resolve_teacher(self.config).provenance,
            "expert_manifest_sha256": file_hash(expert / "manifest.json"),
            "train_seeds": [123987], "validation_seeds": [], "episodes": [result],
            "simulation": read_json(expert / "manifest.json")["simulation"],
            "native_ticks": result["native_ticks"],
        })
        return destination

    def test_aggregation_retains_every_expert_and_validation_shard(self):
        expert = self.root / "expert"
        expert.mkdir()
        synthetic_corpus(expert, self.config)
        manifest = read_json(expert / "manifest.json")
        manifest["simulation"] = {}
        for row in manifest["episodes"]:
            row["coverage"] = manifest["coverage"][row["split"]]
        write_json(expert / "manifest.json", manifest)
        recovery = self.make_recovery(expert)
        combined = aggregate_corpora(expert, recovery, self.root / "combined")
        original = {(row["seed"], row["split"], row["sha256"]) for row in manifest["episodes"]}
        retained = {(row["seed"], row["split"], row["sha256"]) for row in combined["episodes"]
                    if row["origin"] == "expert"}
        self.assertEqual(original, retained)
        self.assertEqual(combined["validation_seeds"], manifest["validation_seeds"])
        self.assertGreaterEqual(combined["expert_frame_fraction"], 0.5)

    def test_peer_anchor_and_matched_data_forks_share_initial_weights_and_lr(self):
        expert = self.root / "expert"
        expert.mkdir()
        synthetic_corpus(expert, self.config)
        manifest = read_json(expert / "manifest.json")
        manifest["simulation"] = {}
        for row in manifest["episodes"]:
            row["coverage"] = manifest["coverage"][row["split"]]
        write_json(expert / "manifest.json", manifest)
        common = ["--sequence-length", "2", "--burn-in", "1", "--batch-sequences", "1",
                  "--validate-every", "1", "--device", "cpu"]
        parent = self.root / "parent"
        self.assertEqual(fit(["--corpus", str(expert), "--checkpoint", str(self.checkpoint),
                              "--output", str(parent), "--steps", "1", *common]), 0)
        anchor = self.root / "anchor"
        prepare_anchor(parent / "checkpoint.pt", expert, anchor, peer_context=True)
        anchored = load_checkpoint(anchor / "checkpoint.pt")
        self.assertTrue(anchored.config.model.peer_context)
        self.assertEqual(anchored.training_state["optimizer_steps"], 1)
        recovery = self.make_recovery(expert)
        combined = self.root / "combined"
        aggregate_corpora(expert, recovery, combined)
        starts = []
        for name, data in (("control", expert), ("dagger", combined)):
            output = self.root / name
            self.assertEqual(fit(["--corpus", str(data), "--data-fork", str(anchor / "checkpoint.pt"),
                                  "--output", str(output), "--steps", "2", *common]), 0)
            ready = read_json(output / "evaluation" / "checkpoints" / "update-00000001" / "ready.json")
            starts.append(ready["model_sha256"])
            checkpoint = load_checkpoint(output / "checkpoint.pt")
            self.assertTrue(checkpoint.training_state["data_fork"]["constant_learning_rate"])
            self.assertEqual(checkpoint.optimizer_state["param_groups"][0]["lr"],
                             anchored.optimizer_state["param_groups"][0]["lr"])
            self.assertEqual(checkpoint.training_state["optimizer_steps"], 2)
        self.assertEqual(starts[0], starts[1])


if __name__ == "__main__":
    unittest.main()
