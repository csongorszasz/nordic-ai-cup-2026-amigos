import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.policies.actions import imitation_loss
from src.policies.networks import PolicyNetwork
from src.training.artifacts import file_hash, save_checkpoint, write_json
from src.training.bc_corpus_fit import EpisodeStore, WindowSampler, evaluate_split, main
from src.training.evaluation import evaluate_pending
from src.training.offline_bc import cache_frames, copying_metrics, forward_sequence, pack_sequence
from src.training.teacher import resolve_teacher
from src.benchmarking.config import GitInfo, read_json
from tests.test_pipeline_contract import pipeline_config
from tests.test_pipeline_evaluation import fake_episode
from tests.test_training_rollout import make_step, ConstantTeacher


def synthetic_corpus(root, config):
    rows = []
    for split, seed in (("train", 91), ("validation", 97)):
        directory = root / split / f"world-{seed}"
        directory.mkdir(parents=True)
        with gzip.open(directory / "frames.jsonl.gz", "wt") as stream:
            for tick in range(4):
                step = make_step((0, 1), tick=tick + 1, entities=1)
                actions = ConstantTeacher().act(step)
                actions[0].move_distance = 0
                actions[1].spawn_agent = tick == 2
                stream.write(json.dumps({"step": step.model_dump(mode="json"),
                                         "actions": [action.model_dump() for action in actions]}) + "\n")
        rows.append({"seed": seed, "split": split, "frames": 4, "native_ticks": 5,
                     "path": directory.relative_to(root).as_posix(),
                     "sha256": file_hash(directory / "frames.jsonl.gz")})
    write_json(root / "manifest.json", {
        "status": "complete", "dataset_id": "synthetic", "teacher": resolve_teacher(config).provenance,
        "train_seeds": [91], "validation_seeds": [97], "episodes": rows, "native_ticks": 10,
        "coverage": {split: {key: 1 for key in ("idle_labels", "spawn_labels", "predator_frames",
                                                "descendant_frames", "late_frames")}
                     for split in ("train", "validation")},
    })


class CorpusFitTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.guard = patch("src.core.SimulationCore.__init__", side_effect=AssertionError("No real simulator"))
        self.guard.start()
        self.addCleanup(self.guard.stop)
        git = patch("src.benchmarking.artifacts._git_info",
                    return_value=GitInfo(revision=None, dirty=None, status="synthetic"))
        git.start()
        self.addCleanup(git.stop)

    def test_sampler_resume_retains_order_counts_and_epoch(self):
        metadata = [{"split": "train", "frames": 7}, {"split": "validation", "frames": 5},
                    {"split": "train", "frames": 4}]
        first = WindowSampler(metadata, 2, 41)
        first.take(3)
        second = WindowSampler(metadata, 2, 41, first.state_dict())
        for _ in range(5):
            self.assertEqual(first.take(2), second.take(2))
        self.assertEqual(first.exposures, second.exposures)
        self.assertEqual(first.epochs, second.epochs)

    def test_cached_features_and_chunked_validation_match_full_current_history(self):
        base = pipeline_config("imitation")
        config = base.model_copy(update={"model": base.model.model_copy(update={"angle_head": "vector_bc"})})
        network = PolicyNetwork(config.model)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_corpus(root, config)
            store = EpisodeStore(root, config.model)
            frames = store.episode(1)
            with torch.no_grad():
                batch = pack_sequence(frames, config.model)
                expected = copying_metrics(forward_sequence(network, batch), batch)
                actual = evaluate_split(network, store, "validation", "cpu", chunk_frames=2)
            for key in ("loss", "distance_mae", "turn_mae_radians", "stop_distance_mean", "spawn_recall"):
                self.assertAlmostEqual(expected[key], actual[key], places=5)

    def test_corpus_fit_checkpoints_and_resumes_with_frozen_sampler(self):
        config = pipeline_config("imitation")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            corpus.mkdir()
            synthetic_corpus(corpus, config)
            checkpoint = root / "initial.pt"
            save_checkpoint(checkpoint, PolicyNetwork(config.model), config,
                            training_state={"native_ticks_total": 0, "optimizer_steps": 0})
            common = ["--corpus", str(corpus), "--checkpoint", str(checkpoint),
                      "--sequence-length", "2", "--burn-in", "1", "--batch-sequences", "1",
                      "--validate-every", "1", "--device", "cpu"]
            first, second = root / "first", root / "second"
            self.assertEqual(main([*common, "--steps", "2", "--output", str(first)]), 0)
            self.assertEqual(read_json(first / "summary.json")["optimizer_steps"], 2)
            evaluate_pending(first, executor=fake_episode, refresh=False)
            self.assertEqual(main([*common, "--steps", "3", "--resume", str(first / "checkpoint.pt"),
                                   "--output", str(second)]), 0)
            result = read_json(second / "summary.json")
            self.assertEqual(result["optimizer_steps"], 3)
            self.assertGreater(result["data_exposures"], read_json(first / "summary.json")["data_exposures"])
            self.assertFalse(result["teacher_level_qualified"])


if __name__ == "__main__":
    unittest.main()
