import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.training.demonstrations import collect_corpus, read_episode
from tests.test_pipeline_contract import pipeline_config
from tests.test_training_rollout import ScriptedEnvironment, ConstantTeacher


class DemonstrationTests(unittest.TestCase):
    def test_complete_immutable_episode_shards_and_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "corpus"
            with patch("src.training.demonstrations.EnvironmentAdapter", side_effect=lambda **_: ScriptedEnvironment()), \
                    patch("src.training.demonstrations.build_policy", side_effect=ConstantTeacher):
                manifest = collect_corpus(pipeline_config("imitation"), root,
                                          train_worlds=2, validation_worlds=1, workers=1)
                again = collect_corpus(pipeline_config("imitation"), root,
                                       train_worlds=2, validation_worlds=1, workers=1)
            self.assertEqual(manifest, again)
            self.assertFalse(set(manifest["train_seeds"]) & set(manifest["validation_seeds"]))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(manifest["episodes"]), 3)
            frames = read_episode(root, manifest["episodes"][0])
            self.assertEqual(len(frames), 3)
            self.assertEqual(frames[0][2], {})
            self.assertEqual(set(frames[1][2]), {0, 1})
            shard = root / manifest["episodes"][0]["path"] / "frames.jsonl.gz"
            with shard.open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_episode(root, manifest["episodes"][0])


if __name__ == "__main__":
    unittest.main()
