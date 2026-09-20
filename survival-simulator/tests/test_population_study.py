import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from population_experiments import Study, seeds
from src.benchmarking.config import read_json
from src.policies.config import ExperimentConfig


class PopulationStudyTests(unittest.TestCase):
    def test_seed_pools_are_reproducible_unique_and_disjoint(self):
        first = seeds(1, 100, {1, 17, 42})
        second = seeds(2, 100, set(first) | {1, 17, 42})
        self.assertEqual(first, seeds(1, 100, {1, 17, 42}))
        self.assertEqual(len(set(first)), 100)
        self.assertFalse(set(first) & set(second))
        self.assertFalse({1, 17, 42} & set(first))

    def test_study_preserves_source_snapshot_and_reports_stage_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            study = Study(Path(temporary) / "study", workers=1)
            self.assertTrue((study.path / "source.zip").is_file())
            study.command("ok", ["-c", "print('fixture')"])
            self.assertEqual(read_json(study.path / "study.json")["stages"][-1]["status"], "complete")
            with self.assertRaisesRegex(RuntimeError, "failed with exit 7"):
                study.command("bad", ["-c", "raise SystemExit(7)"])
            self.assertEqual(read_json(study.path / "study.json")["stages"][-1]["status"], "failed")
            with patch("population_experiments.hashlib.sha256") as digest:
                digest.return_value.hexdigest.return_value = "changed"
                with self.assertRaisesRegex(RuntimeError, "sources changed"):
                    study.verify_sources()

    def test_survival_search_rejects_gpu_or_macro_actions(self):
        for resources in ({"device": "cuda"}, {"action_repeat": 2}):
            with self.subTest(resources=resources), self.assertRaises(ValueError):
                ExperimentConfig.model_validate({
                    "mode": "search", "policy": "heuristic",
                    "search": {"objective": "survival"}, "resources": resources,
                })


if __name__ == "__main__":
    unittest.main()
