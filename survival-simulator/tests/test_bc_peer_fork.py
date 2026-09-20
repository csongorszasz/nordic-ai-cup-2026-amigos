import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from src.benchmarking.config import GitInfo, content_hash, read_json
from src.policies.features import PEER_FEATURE_VERSION, PUBLIC_FEATURE_VERSION, feature_version
from src.policies.networks import PolicyNetwork
from src.training.artifacts import TrainingRun, file_hash, load_checkpoint, save_checkpoint, write_json
from src.training.bc_corpus_fit import (
    EpisodeStore, WindowSampler, _DEFINITION_FIELDS, _verify_continuation, main,
)
from src.training.evaluation import initialize_schedule, load_schedule, publish_snapshot, snapshot_directory
from src.training.offline_bc import forward_sequence, pack_sequence
from src.training.preflight import source_fingerprint
from tests.test_bc_corpus_fit import synthetic_corpus
from tests.test_peer_context import peer_step
from tests.test_pipeline_contract import pipeline_config


class CorpusPeerForkTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.enterContext(torch.random.fork_rng(devices=[]))
        torch.manual_seed(671)
        self.enterContext(patch("src.core.SimulationCore.__init__", side_effect=AssertionError("No real simulator")))
        self.enterContext(patch("src.benchmarking.artifacts._git_info",
                                return_value=GitInfo(revision=None, dirty=None, status="synthetic")))
        self.enterContext(patch("subprocess.Popen", side_effect=AssertionError("No jobs or external processes")))
        self.enterContext(patch("src.training.bc_corpus_fit.render_curve"))
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right, strict=True):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def make_parent(self, *, legacy=False, first=12, peer=False):
        config = pipeline_config("imitation", updates=first)
        config = config.model_copy(update={
            "model": config.model.model_copy(update={"public_context": True, "peer_context": peer,
                                                    "angle_head": "vector_bc"}),
            "evaluation": config.evaluation.model_copy(update={"every_updates": 2}),
        })
        corpus = self.root / "corpus"
        corpus.mkdir()
        with patch("tests.test_bc_corpus_fit.make_step",
                   side_effect=lambda *_, **kw: peer_step().model_copy(update={"sim_time": kw["tick"]})):
            synthetic_corpus(corpus, config)
        store = EpisodeStore(corpus, config.model)
        source = source_fingerprint()
        if legacy:
            files = {"src/policies/features.py": "a" * 64}
            source = {"files": files, "sha256": content_hash(files)}
        definition = {
            "dataset_id": store.manifest["dataset_id"], "dataset_manifest_sha256": store.manifest_sha256,
            "model": config.model.model_dump(mode="json"), "sequence_length": 2, "burn_in": 1,
            "batch_sequences": 1, "validate_every": 2, "initial_lr": config.optimizer.learning_rate,
            "loss": "unchanged-bc-v1", "seed": config.seed,
        }
        if legacy:
            definition["model"].pop("peer_context")
        definition_id = content_hash(definition)
        run = TrainingRun(self.root / "parent", config)
        artifact = {**definition, "definition_id": definition_id, "source": source}
        if not legacy:
            artifact.update(feature_version=feature_version(True, peer), start_optimizer_step=0, fork=None)
        write_json(run.path / "offline-definition.json", artifact)
        sampler = WindowSampler(store.metadata, 2, config.seed)
        for _ in range(first):
            sampler.take(1)
        network = PolicyNetwork(config.model)
        optimizer = torch.optim.Adam(network.parameters(), lr=config.optimizer.learning_rate)
        # Synthetic nonzero moments expose accidental optimizer resets and incomplete migration.
        for parameter in network.parameters():
            optimizer.state[parameter] = {
                "step": torch.tensor(float(first)), "exp_avg": torch.full_like(parameter, 0.0125),
                "exp_avg_sq": torch.full_like(parameter, 0.02),
            }
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=4, threshold=0.001, min_lr=1e-6,
        )
        scheduler.step(1.0)
        for _ in range(5):
            scheduler.step(2.0)
        state = {
            "next_update": first, "optimizer_steps": first, "native_ticks_total": 5,
            "prior_native_ticks": 7, "offline_definition_id": definition_id,
            "sampler": sampler.state_dict(), "scheduler": scheduler.state_dict(),
            "scheduler_epoch": sampler.epochs + (sampler.cursor == len(sampler.order)),
            "lr_reductions": 1, "best_validation_loss": 1e-9,
        }
        if not legacy:
            state.update(offline_definition_sha256=file_hash(run.path / "offline-definition.json"),
                         offline_source_sha256=source["sha256"])
        checkpoint = run.path / "checkpoint.pt"
        save_checkpoint(checkpoint, network, config, optimizer, state)
        if legacy:
            payload = torch.load(checkpoint, weights_only=True)
            payload["config"]["model"].pop("peer_context")
            torch.save(payload, checkpoint)
            sidecar = read_json(checkpoint.with_suffix(".pt.json"))
            sidecar["model"].pop("peer_context")
            sidecar["sha256"] = file_hash(checkpoint)
            write_json(checkpoint.with_suffix(".pt.json"), sidecar)
        with patch("src.training.evaluation.source_fingerprint", return_value=source):
            schedule = initialize_schedule(run.path, config, run.manifest)
            publish_snapshot(run.path, network, config, state, schedule)
        write_json(run.path / "evaluation" / "history.json", [{"update": 1, "mean_score": 9999, "state": "complete"}])
        (run.path / "evaluation" / "teacher").mkdir()
        write_json(run.path / "evaluation" / "teacher" / "result.json", {"state": "complete", "scores": [9999]})
        (run.path / "copying-metrics.jsonl").write_text(
            json.dumps({"split": "validation", "optimizer_steps": 1, "loss": 9999}) + "\n", encoding="utf-8",
        )
        return corpus, checkpoint

    def arguments(self, corpus, checkpoint, output, *, fork=True, steps=13):
        return [
            "--corpus", str(corpus), "--fork" if fork else "--resume", str(checkpoint),
            "--output", str(output), "--steps", str(steps), "--sequence-length", "2",
            "--burn-in", "1", "--batch-sequences", "1", "--validate-every", "2", "--device", "cpu",
        ]

    def run_fit(self, arguments):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(arguments), 0)

    def assert_legacy_ab_fork(self, first):
        corpus, checkpoint = self.make_parent(legacy=True, first=first)
        parent = load_checkpoint(checkpoint)
        parent_hash = file_hash(checkpoint)
        old_schedule = load_schedule(checkpoint.parent)
        old_definition = read_json(checkpoint.parent / "offline-definition.json")
        forks = []
        for peer in (False, True):
            output = self.root / ("peer" if peer else "control")
            arguments = self.arguments(corpus, checkpoint, output, steps=first + 1)
            # Stop at the fork boundary: forward-only synthetic validation, no optimizer update.
            arguments += ["--max-seconds", "0.000000001"]
            if peer:
                arguments.append("--peer-context")
            self.run_fit(arguments)
            loaded = load_checkpoint(output / "checkpoint.pt")
            forks.append(loaded)
            self.assertEqual(loaded.training_state["optimizer_steps"], first)
            self.assertEqual(loaded.training_state["prior_native_ticks"], 7)
            self.assertEqual(loaded.training_state["native_ticks_total"], 5)
            for key in ("sampler", "scheduler", "scheduler_epoch", "lr_reductions"):
                self.assert_tree_equal(parent.training_state[key], loaded.training_state[key])
            self.assert_tree_equal(parent.rng_state, loaded.rng_state)
            self.assertEqual(loaded.config.teacher, parent.config.teacher)
            for field in ("optimizer", "ppo", "imitation", "heuristic", "resources"):
                self.assertEqual(getattr(loaded.config, field), getattr(parent.config, field))
            for name, value in loaded.network.state_dict().items():
                expected = parent.network.state_dict()[name].clone()
                if peer and name == "entity_encoders.3.0.weight":
                    expected[:, 8:10] = 0
                self.assertTrue(torch.equal(value, expected), name)
            expected_optimizer = copy.deepcopy(parent.optimizer_state)
            if peer:
                index = [name for name, _ in parent.network.named_parameters()].index("entity_encoders.3.0.weight")
                parameter_id = expected_optimizer["param_groups"][0]["params"][index]
                for moment in ("exp_avg", "exp_avg_sq"):
                    expected_optimizer["state"][parameter_id][moment][:, 8:10] = 0
            self.assert_tree_equal(expected_optimizer, loaded.optimizer_state)
            definition = read_json(output / "offline-definition.json")
            lineage = definition["fork"]
            self.assertEqual(lineage, loaded.training_state["fork"])
            self.assertEqual(lineage["parent_checkpoint_sha256"], parent_hash)
            self.assertEqual(lineage["parent_definition_id"], old_definition["definition_id"])
            self.assertEqual(lineage["parent_definition_sha256"], file_hash(checkpoint.parent / "offline-definition.json"))
            self.assertEqual(lineage["parent_source_sha256"], old_definition["source"]["sha256"])
            self.assertEqual(lineage["source_sha256"], definition["source"]["sha256"])
            self.assertEqual(lineage["dataset_manifest_sha256"], file_hash(corpus / "manifest.json"))
            self.assertEqual(lineage["start_optimizer_step"], first)
            self.assertEqual(lineage["allowed_model_input_change"],
                             {"peer_context": {"from": False, "to": True}} if peer else {})
            self.assertFalse(lineage["exact_resume"])
            schedule = load_schedule(output)
            self.assertEqual(schedule["updates"], [first, first + 1])
            self.assertEqual(schedule["lineage"], lineage)
            self.assertEqual(schedule["source"], definition["source"])
            self.assertNotEqual(schedule["protocol_id"], old_schedule["protocol_id"])
            self.assertFalse((output / "evaluation" / "history.json").exists())
            self.assertFalse((output / "evaluation" / "historical-teacher.json").exists())
            records = [json.loads(row) for row in (output / "copying-metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["optimizer_steps"] for row in records], [first])
            self.assertEqual(loaded.training_state["best_validation_loss"], records[0]["loss"])
            self.assertGreater(records[0]["loss"], parent.training_state["best_validation_loss"])
            self.assertTrue((output / "best.pt").is_file())
            initial = load_checkpoint(snapshot_directory(output, first) / "checkpoint.pt")
            self.assertEqual(initial.config.model.peer_context, peer)
            self.assertEqual(read_json((output / "checkpoint.pt").with_suffix(".pt.json"))["feature_version"],
                             PEER_FEATURE_VERSION if peer else PUBLIC_FEATURE_VERSION)
        with torch.no_grad():
            frames = EpisodeStore(corpus, parent.config.model).episode(0)
            expected = forward_sequence(parent.network, pack_sequence(frames, parent.config.model))
            for loaded in forks:
                frames = EpisodeStore(corpus, loaded.config.model).episode(0)
                actual = forward_sequence(loaded.network, pack_sequence(frames, loaded.config.model))
                for field in ("mean", "log_std", "spawn_logits", "value", "next_hidden", "angle_vectors"):
                    torch.testing.assert_close(getattr(actual, field), getattr(expected, field), atol=1e-7, rtol=1e-6)
        self.assertEqual(file_hash(checkpoint), parent_hash)

    def test_legacy_ab_fork_uses_8750_step_source(self):
        self.assert_legacy_ab_fork(8750)

    def test_legacy_ab_fork_uses_10000_step_source(self):
        self.assert_legacy_ab_fork(10000)

    def test_legacy_ab_fork_starts_between_validation_intervals(self):
        self.assert_legacy_ab_fork(10001)

    def test_control_fork_synthetic_updates_match_exact_continuation(self):
        first = 10000
        corpus, checkpoint = self.make_parent(first=first)
        write_json(snapshot_directory(checkpoint.parent, first) / "result.json",
                   {"update": first, "state": "complete", "mean_score": 1})
        resumed, forked = self.root / "resumed", self.root / "forked"
        self.run_fit(self.arguments(corpus, checkpoint, resumed, fork=False, steps=first + 3))
        self.run_fit(self.arguments(corpus, checkpoint, forked, steps=first + 3))
        left, right = load_checkpoint(resumed / "checkpoint.pt"), load_checkpoint(forked / "checkpoint.pt")
        self.assertEqual(left.training_state["optimizer_steps"], first + 3)
        self.assert_tree_equal(left.network.state_dict(), right.network.state_dict())
        self.assert_tree_equal(left.optimizer_state, right.optimizer_state)
        self.assert_tree_equal(left.rng_state, right.rng_state)
        for key in ("sampler", "scheduler", "scheduler_epoch", "lr_reductions", "prior_native_ticks", "native_ticks_total"):
            self.assert_tree_equal(left.training_state[key], right.training_state[key])
        expected = load_checkpoint(checkpoint).training_state["sampler"]["exposures"] + 6
        self.assertEqual(left.training_state["sampler"]["exposures"], expected)
        self.assertTrue((resumed / "evaluation" / "history.json").exists())
        self.assertFalse((forked / "evaluation" / "history.json").exists())
        self.assertEqual(load_schedule(forked)["updates"], [first, first + 2, first + 3])

    def test_fork_requires_an_absolute_target_beyond_the_source_counter(self):
        first = 10000
        corpus, checkpoint = self.make_parent(first=first)
        for peer in (False, True):
            for target in (first - 1, first):
                output = self.root / f"rejected-{peer}-{target}"
                arguments = self.arguments(corpus, checkpoint, output, steps=target)
                if peer:
                    arguments.append("--peer-context")
                with self.subTest(peer=peer, target=target), self.assertRaisesRegex(ValueError, "target optimizer-step"):
                    main(arguments)
                self.assertFalse(output.exists())

    def test_peer_fork_can_resume_strictly_without_repeating_migration(self):
        corpus, checkpoint = self.make_parent()
        first, second = self.root / "peer", self.root / "continued"
        self.run_fit([*self.arguments(corpus, checkpoint, first), "--peer-context", "--max-seconds", "0.000000001"])
        write_json(snapshot_directory(first, 12) / "result.json", {"update": 12, "state": "complete", "mean_score": 1})
        self.run_fit([*self.arguments(corpus, first / "checkpoint.pt", second, fork=False),
                      "--peer-context", "--max-seconds", "0.000000001"])
        before, after = load_checkpoint(first / "checkpoint.pt"), load_checkpoint(second / "checkpoint.pt")
        self.assert_tree_equal(before.network.state_dict(), after.network.state_dict())
        self.assert_tree_equal(before.optimizer_state, after.optimizer_state)
        self.assertEqual(before.training_state["fork"], after.training_state["fork"])
        self.assertEqual(load_schedule(second)["updates"], [13])

    def test_fork_rejects_every_other_model_training_and_data_change(self):
        _, checkpoint = self.make_parent()
        loaded = load_checkpoint(checkpoint)
        previous = read_json(checkpoint.parent / "offline-definition.json")
        base = {key: previous[key] for key in _DEFINITION_FIELDS}
        source = source_fingerprint()
        changes = [
            ("sequence_length", 3), ("burn_in", 0), ("batch_sequences", 2), ("validate_every", 1),
            ("initial_lr", 0.01), ("loss", "different"), ("seed", base["seed"] + 1),
            ("dataset_id", "different"), ("dataset_manifest_sha256", "0" * 64),
        ]
        for field, value in changes:
            changed = copy.deepcopy(base)
            changed[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Fork cannot change"):
                _verify_continuation(checkpoint, loaded, changed, source, fork=True)
        for field, value in (("hidden_size", 32), ("angle_head", "bounded"), ("memory", "none"),
                             ("public_context", False), ("encoder", "attention"), ("log_std_min", -6.0)):
            changed = copy.deepcopy(base)
            changed["model"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Fork cannot change"):
                _verify_continuation(checkpoint, loaded, changed, source, fork=True)
        permitted = copy.deepcopy(base)
        permitted["model"]["peer_context"] = True
        self.assertEqual(_verify_continuation(checkpoint, loaded, permitted, source, fork=True), previous)

    def test_exact_resume_rejects_peer_migration_changed_sources_and_definition_tampering(self):
        corpus, checkpoint = self.make_parent()
        output = self.root / "rejected"
        with self.assertRaisesRegex(ValueError, "use --fork"):
            main([*self.arguments(corpus, checkpoint, output, fork=False), "--peer-context"])
        self.assertFalse(output.exists())
        files = {"src/policies/features.py": "b" * 64}
        with patch("src.training.bc_corpus_fit.source_fingerprint",
                   return_value={"files": files, "sha256": content_hash(files)}):
            with self.assertRaisesRegex(ValueError, "Resume requires unchanged"):
                main(self.arguments(corpus, checkpoint, output, fork=False))
        self.assertFalse(output.exists())
        definition_path = checkpoint.parent / "offline-definition.json"
        definition = read_json(definition_path)
        definition["source"]["files"]["new.py"] = "a" * 64
        definition["source"]["sha256"] = content_hash(definition["source"]["files"])
        write_json(definition_path, definition)
        with self.assertRaisesRegex(ValueError, "checksum"):
            main(self.arguments(corpus, checkpoint, output))
        self.assertFalse(output.exists())

    def test_missing_continuation_state_and_invalid_sampler_cannot_silently_restart(self):
        _, checkpoint = self.make_parent()
        loaded = load_checkpoint(checkpoint)
        previous = read_json(checkpoint.parent / "offline-definition.json")
        definition = {key: previous[key] for key in _DEFINITION_FIELDS}
        variants = [replace(loaded, optimizer_state=None), replace(loaded, rng_state={})]
        for key in ("sampler", "scheduler", "scheduler_epoch", "lr_reductions", "optimizer_steps"):
            state = copy.deepcopy(loaded.training_state)
            state.pop(key)
            variants.append(replace(loaded, training_state=state))
        for changed in variants:
            with self.assertRaisesRegex(ValueError, "Continuation requires"):
                _verify_continuation(checkpoint, changed, definition, source_fingerprint(), fork=True)
        metadata = [{"split": "train", "frames": 4}]
        sampler = WindowSampler(metadata, 2, 3)
        sampler.take(1)
        for key, value in (("cursor", -1), ("cursor", 3), ("epochs", True), ("exposures", 0), ("order", [0, 0])):
            state = sampler.state_dict()
            state[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                WindowSampler(metadata, 2, 3, state)

    def test_cli_fork_and_resume_are_mutually_exclusive_and_source_is_required(self):
        common = ["--corpus", str(self.root), "--output", str(self.root / "out")]
        with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit) as error:
            main([*common, "--fork", "parent.pt", "--resume", "other.pt"])
        self.assertEqual(error.exception.code, 2)
        with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit) as error:
            main(common)
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
