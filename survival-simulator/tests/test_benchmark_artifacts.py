import csv
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.benchmarking.artifacts import (
    RunWriter, _runtime_info, _timing_context, build_manifest, load_run, summarize_run,
)
from src.benchmarking.config import (
    BASELINE_NAME, EpisodeResult, FailureInfo, PolicySpec, Suite, canonical_json, make_cases,
)
from tests.benchmark_fixtures import episode, manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.storage = tempfile.TemporaryDirectory(prefix=".benchmark-artifacts-", dir=PROJECT_ROOT)
        self.addCleanup(self.storage.cleanup)
        self.root = Path(self.storage.name)
        self.manifest = manifest()

    def writer(self, name="run", record=None):
        return RunWriter(self.root / name, record or self.manifest)

    def complete(self, name="run"):
        writer = self.writer(name)
        for index, case in enumerate(writer.manifest.cases):
            writer.append(episode(case, index - 1.0))
        writer.finalize("complete")
        return writer

    def change_manifest(self, output, update):
        path = output / "manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        update(data)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_empty_running_run_has_explicit_unavailable_measurements(self):
        writer = self.writer()
        loaded = load_run(writer.output)
        summary = summarize_run(loaded.manifest, loaded.episodes)
        self.assertEqual(loaded.manifest.status, "running")
        self.assertEqual(summary["score"]["count"], 0)
        self.assertIsNone(summary["score"]["mean"])
        self.assertIsNone(summary["completion"]["fraction"])
        self.assertIsNone(summary["policy_latency"]["pooled_batch_mean_ms"])
        self.assertEqual(summary["runtime"]["episode_seconds"]["count"], 0)
        self.assertEqual(summary["missing_cases"], 3)
        self.assertFalse(summary["rankable"])
        self.assertEqual(summary["aggregate_scope"], "diagnostic_subset")

    def test_append_is_durable_and_finalization_exports_complete_measurements(self):
        writer = self.writer()
        with patch("src.benchmarking.artifacts.os.fsync", wraps=os.fsync) as fsync:
            writer.append(episode(writer.manifest.cases[0], -1))
        self.assertEqual(fsync.call_count, 1)
        self.assertEqual(len(load_run(writer.output).episodes), 1)
        for index, case in enumerate(writer.manifest.cases[1:]):
            writer.append(episode(case, index))
        summary = writer.finalize("complete")
        loaded = load_run(writer.output)
        self.assertEqual(loaded.manifest.status, "complete")
        self.assertTrue(summary["is_complete"])
        self.assertTrue(summary["rankable"])
        self.assertEqual(summary["score"]["mean"], 0)
        self.assertEqual(summary["score"]["sample_std"], 1)
        self.assertEqual(summary["policy_latency"]["total_policy_calls"], 3)
        self.assertEqual(summary["policy_latency"]["pooled_batch_mean_ms"], 2)
        self.assertEqual(summary["policy_latency"]["batch_p95_ms"]["count"], 3)
        self.assertFalse(summary["policy_latency"]["pooled_percentiles_available"])
        self.assertEqual(
            json.loads((writer.output / "summary.json").read_text(encoding="utf-8")), summary,
        )
        with (writer.output / "episodes.csv").open(encoding="utf-8", newline="") as stream:
            exported = list(csv.DictReader(stream))
        self.assertEqual(len(exported), 3)
        self.assertEqual(exported[0]["case_id"], writer.manifest.cases[0].case_id)
        self.assertEqual(exported[0]["batch_mean_ms"], "2.0")
        self.assertIn("Engine SHA-256", (writer.output / "report.md").read_text(encoding="utf-8"))
        self.assertFalse(list(writer.output.glob("*.pending")))

    def test_partial_run_cannot_be_finalized_complete(self):
        writer = self.writer()
        writer.append(episode(writer.manifest.cases[0], 7))
        with self.assertRaisesRegex(ValueError, "every expected case"):
            writer.finalize("complete")
        loaded = load_run(writer.output)
        self.assertEqual(loaded.manifest.status, "running")
        self.assertEqual(loaded.episodes[0].score, 7)
        self.assertFalse((writer.output / "summary.json").exists())
        summary = writer.finalize("interrupted")
        self.assertEqual(summary["recorded_cases"], 1)
        self.assertFalse(summary["rankable"])
        self.assertIn("not a ranking", (writer.output / "report.md").read_text(encoding="utf-8"))

    def test_failed_subset_does_not_impute_zero_score(self):
        writer = self.writer()
        writer.append(episode(writer.manifest.cases[0], -7))
        failure = FailureInfo(stage="policy_decision", error_type="RuntimeError", message="broken")
        writer.append(EpisodeResult(
            case=writer.manifest.cases[1], status="failed", termination="error", failure=failure,
        ))
        summary = writer.finalize("failed")
        self.assertEqual(summary["score"]["mean"], -7)
        self.assertEqual(summary["score"]["count"], 1)
        self.assertIsNone(summary["score"]["sample_std"])
        self.assertEqual(summary["failed_cases"], 1)
        self.assertEqual(summary["missing_cases"], 1)
        self.assertEqual(summary["runtime"]["episode_seconds"]["count"], 2)
        self.assertFalse(summary["rankable"])
        self.assertEqual(load_run(writer.output).manifest.failure, failure)
        with (writer.output / "episodes.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[1]["score"], "")
        self.assertEqual(rows[1]["failure_message"], "broken")

    def test_weighted_batch_mean_uses_real_calls_without_inventing_pooled_percentiles(self):
        writer = self.writer()
        writer.append(episode(self.manifest.cases[0], ticks=2, batch_ms=2))
        writer.append(episode(self.manifest.cases[1], ticks=4, batch_ms=10))
        writer.append(episode(self.manifest.cases[2], ticks=1))
        summary = writer.finalize("complete")
        latency = summary["policy_latency"]
        self.assertEqual(latency["total_policy_calls"], 4)
        self.assertEqual(latency["pooled_batch_mean_ms"], 8)
        self.assertEqual(latency["batch_mean_ms"]["mean"], 6)
        self.assertEqual(latency["batch_p95_ms"]["count"], 2)
        self.assertNotIn("pooled_batch_p95_ms", latency)
        self.assertEqual(summary["runtime"]["episode_seconds"]["count"], 3)

    def test_explicit_between_episode_failure_and_empty_interrupt_are_saved(self):
        writer = self.writer()
        writer.append(episode(writer.manifest.cases[0]))
        failure = FailureInfo(stage="run", error_type="RuntimeError", message="outside episode")
        writer.finalize("failed", failure)
        self.assertEqual(load_run(writer.output).manifest.failure, failure)
        empty = self.writer("empty")
        empty.finalize("interrupted")
        self.assertEqual(load_run(empty.output).manifest.status, "interrupted")

    def test_truncated_and_early_extinction_capped_runs_remain_diagnostics(self):
        record = manifest(max_steps=2)
        writer = self.writer(record=record)
        writer.append(episode(record.cases[0], 4, termination="step_limit"))
        summary = writer.finalize("truncated")
        self.assertEqual(summary["truncated_cases"], 1)
        self.assertEqual(summary["score"]["mean"], 4)
        self.assertFalse(summary["rankable"])
        self.assertEqual(load_run(writer.output).manifest.status, "truncated")
        extinct = self.writer("early", record)
        for case in record.cases:
            extinct.append(episode(case))
        summary = extinct.finalize("complete")
        self.assertTrue(summary["is_complete"])
        self.assertFalse(summary["rankable"])

    def test_writer_rejects_duplicate_unknown_and_unconfigured_truncation(self):
        writer = self.writer()
        row = episode(writer.manifest.cases[0])
        writer.append(row)
        other = make_cases(Suite(name="other", seeds=[99]))[0]
        for invalid in (row, episode(other), episode(writer.manifest.cases[1], termination="step_limit")):
            with self.subTest(case=invalid.case.case_id), self.assertRaises(ValueError):
                writer.append(invalid)
        self.assertEqual(len(load_run(writer.output).episodes), 1)

    def test_directories_and_finalized_runs_are_never_reused(self):
        writer = self.complete()
        with self.assertRaises(FileExistsError):
            self.writer()
        empty = self.root / "existing-empty"
        empty.mkdir()
        with self.assertRaises(FileExistsError):
            RunWriter(empty, self.manifest)
        with self.assertRaises(ValueError):
            writer.append(episode(self.manifest.cases[0]))
        with self.assertRaises(ValueError):
            writer.finalize("complete")
        with self.assertRaises(ValueError):
            self.writer("running-status").finalize("running")

    def test_io_failure_propagates_without_marking_run_complete(self):
        writer = self.writer()
        for case in writer.manifest.cases:
            writer.append(episode(case))
        with patch("src.benchmarking.artifacts.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                writer.finalize("complete")
        loaded = load_run(writer.output)
        self.assertEqual(loaded.manifest.status, "running")
        self.assertEqual(len(loaded.episodes), 3)
        self.assertFalse(list(writer.output.glob("*.pending")))

    def test_finalization_checks_persisted_rows_not_only_memory(self):
        writer = self.writer()
        for case in writer.manifest.cases:
            writer.append(episode(case))
        path = writer.output / "episodes.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text("".join(lines[:-1]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Run files changed"):
            writer.finalize("complete")
        self.assertEqual(load_run(writer.output).manifest.status, "running")

    def test_corrupt_duplicate_missing_and_unknown_results_fail_loading(self):
        for mode in ("duplicate", "missing", "unknown"):
            with self.subTest(mode=mode):
                writer = self.complete(mode)
                path = writer.output / "episodes.jsonl"
                lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
                if mode == "duplicate":
                    lines.append(lines[0])
                elif mode == "missing":
                    lines.pop()
                else:
                    other = make_cases(Suite(name="other", seeds=[99]))[0]
                    lines[0] = canonical_json(episode(other).model_dump(mode="json")) + "\n"
                path.write_text("".join(lines), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_run(writer.output)

    def test_truncated_jsonl_is_never_silently_discarded(self):
        writer = self.writer()
        writer.append(episode(self.manifest.cases[0]))
        path = writer.output / "episodes.jsonl"
        line = path.read_text(encoding="utf-8")
        for suffix in ('{"case":', line.rstrip("\n")):
            with self.subTest(suffix=suffix[:20]):
                path.write_text(line + suffix, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Truncated episodes.jsonl line 2"):
                    load_run(writer.output)
        path.write_text(line + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid episodes.jsonl line 2"):
            load_run(writer.output)

    def test_missing_jsonl_and_nonfinite_scores_are_corruption(self):
        writer = self.complete()
        path = writer.output / "episodes.jsonl"
        path.write_text(path.read_text(encoding="utf-8").replace('"score":-1.0', '"score":NaN'), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_run(writer.output)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            load_run(writer.output)

    def test_schema_manifest_case_hash_and_status_corruption_are_rejected(self):
        changes = (
            lambda data: data.update(schema_version=999),
            lambda data: data.update(schema_version=True),
            lambda data: data.pop("schema_version"),
            lambda data: data["simulation"].pop("dt"),
            lambda data: data["policy"].pop("config"),
            lambda data: data["cases"].append(data["cases"][0]),
            lambda data: data.update(suite_sha256="0" * 64),
            lambda data: data.update(status="failed"),
            lambda data: data.update(failure={"stage": "x", "error_type": "x", "message": "x"}),
        )
        for index, update in enumerate(changes):
            with self.subTest(index=index):
                writer = self.complete(str(index))
                self.change_manifest(writer.output, update)
                with self.assertRaises(ValueError):
                    load_run(writer.output)

    def test_missing_stored_timing_fields_do_not_become_zero_measurements(self):
        for field in ("timings", "episode_seconds"):
            with self.subTest(field=field):
                writer = self.complete(field)
                path = writer.output / "episodes.jsonl"
                lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
                data = json.loads(lines[0])
                if field == "timings":
                    data.pop(field)
                else:
                    data["timings"].pop(field)
                lines[0] = canonical_json(data) + "\n"
                path.write_text("".join(lines), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "missing stored fields"):
                    load_run(writer.output)

    def test_completed_manifest_with_failed_row_is_rejected(self):
        writer = self.complete()
        failed = EpisodeResult(
            case=self.manifest.cases[0], status="failed", termination="error",
            failure=FailureInfo(stage="policy", error_type="Error", message="failed"),
        )
        path = writer.output / "episodes.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        lines[0] = canonical_json(failed.model_dump(mode="json")) + "\n"
        path.write_text("".join(lines), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_run(writer.output)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.storage = tempfile.TemporaryDirectory(prefix=".benchmark-provenance-", dir=PROJECT_ROOT)
        self.addCleanup(self.storage.cleanup)
        self.root = Path(self.storage.name)
        for relative in (
            ("src", "core.py"), ("src", "utils", "simulation.py"), ("src", "utils", "DTOs.py"),
            ("src", "utils", "sensing.py"), ("src", "elements", "agent.py"),
            ("src", "benchmarking", "config.py"), ("src", "benchmarking", "policies.py"),
            ("src", "benchmarking", "runner.py"), ("src", "benchmarking", "artifacts.py"),
            ("src", "benchmarking", "comparison.py"), ("src", "benchmarking", "plots.py"),
            ("src", "benchmarking", "telemetry.py"), ("src", "benchmarking", "progress.py"),
            ("src", "benchmarking", "survival.py"),
            ("src", "benchmarking", "jobs.py"),
            ("src", "utils", "controllers", "dummy_agent_policy.py"), ("benchmark.py",),
        ):
            self.source(*relative)
        sample = manifest().provenance
        for name, value in (
            ("_runtime_info", sample.runtime), ("_timing_context", sample.timing_context),
            ("_git_info", sample.git),
        ):
            patcher = patch(f"src.benchmarking.artifacts.{name}", return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.policy = PolicySpec(reference=BASELINE_NAME, label="random")
        self.suite = Suite(name="fixture", seeds=[1, 2])

    def source(self, *parts, text="# fixture source\n"):
        path = self.root.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def build(self, **kwargs):
        return build_manifest(self.root, self.suite, kwargs.pop("policy", self.policy), **kwargs)

    def test_hashes_are_separated_and_track_dirty_content_not_just_git_sha(self):
        first = self.build()
        groups = (
            ("engine", ("src", "core.py")),
            ("runner", ("src", "benchmarking", "config.py")),
            ("reporting", ("src", "benchmarking", "plots.py")),
            ("policy", ("src", "utils", "controllers", "dummy_agent_policy.py")),
        )
        for field, parts in groups:
            with self.subTest(field=field):
                before = self.build()
                self.source(*parts, text="# changed while Git SHA stays unchanged\n")
                after = self.build()
                self.assertNotEqual(getattr(before.provenance, field), getattr(after.provenance, field))
                for unchanged, _ in groups:
                    if unchanged != field:
                        self.assertEqual(getattr(before.provenance, unchanged), getattr(after.provenance, unchanged))
                self.assertEqual(before.provenance.git, after.provenance.git)
        self.assertNotEqual(first.run_id, self.build().run_id)

    def test_baseline_provenance_excludes_virtualenv_results_and_unrelated_sources(self):
        before = self.build()
        self.source(".venv", "unrelated.py", text="secret = 'not read'\n")
        self.source("benchmark-results", "saved.py")
        self.source("unrelated.py")
        after = self.build()
        self.assertEqual(before.provenance, after.provenance)

    def test_plugin_source_sibling_imported_helpers_and_untracked_files_are_hashed(self):
        source = self.source("plugins", "policy.py", text="from shared.helper import value\n")
        helper = self.source("plugins", "helper.py", text="value = 1\n")
        imported = self.source("shared", "helper.py", text="value = 2\n")
        initializer = self.source("shared", "__init__.py", text="raise RuntimeError('never execute provenance')\n")
        self.source("plugins", ".venv", "not_policy.py")
        policy = PolicySpec(reference="plugins.policy:create", label="candidate")
        before = self.build(policy=policy, policy_sources=[source])
        files = before.provenance.policy.files
        self.assertIn(str(source.relative_to(self.root)), files)
        self.assertIn(str(helper.relative_to(self.root)), files)
        self.assertIn(str(imported.relative_to(self.root)), files)
        self.assertIn(str(initializer.relative_to(self.root)), files)
        self.assertFalse(any(".venv" in name for name in files))
        helper.write_text("value = 3\n", encoding="utf-8")
        after = self.build(policy=policy, policy_sources=[source])
        self.assertNotEqual(before.provenance.policy, after.provenance.policy)
        self.assertEqual(before.provenance.engine, after.provenance.engine)
        self.assertEqual(before.provenance.runner, after.provenance.runner)

    def test_package_relative_and_dynamic_sibling_helpers_are_captured(self):
        self.source("plugins", "__init__.py")
        source = self.source("plugins", "policy.py", text="from .helpers import value\n")
        self.source("plugins", "helpers.py")
        dynamic = self.source("plugins", "dynamic", "loaded_by_name.py")
        self.source("plugins", "results", "ignore.py")
        policy = PolicySpec(reference="plugins.policy:create", label="candidate")
        record = self.build(policy=policy, policy_sources=[source])
        self.assertIn(str(dynamic.relative_to(self.root)), record.provenance.policy.files)
        self.assertFalse(any("results" in name for name in record.provenance.policy.files))

    def test_config_and_explicit_model_artifact_hashes_are_recorded(self):
        weights = self.root / "weights.bin"
        weights.write_bytes(b"\x00\x01weights")
        before = self.build(extra_artifacts=[weights])
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        self.assertEqual(before.provenance.artifacts.files["weights.bin"], digest)
        weights.write_bytes(b"\x00\x02weights")
        after = self.build(extra_artifacts=[weights])
        self.assertNotEqual(before.provenance.artifacts, after.provenance.artifacts)
        options = self.policy.model_copy(update={"config": {"threshold": 3}})
        configured = self.build(policy=options)
        self.assertNotEqual(before.provenance.policy.sha256, configured.provenance.policy.sha256)
        self.assertEqual(configured.policy.config, {"threshold": 3})
        renamed = self.build(policy=self.policy.model_copy(update={"label": "different display label"}))
        self.assertEqual(before.provenance.policy.sha256, renamed.provenance.policy.sha256)

    def test_fixed_policy_seed_is_recorded_with_every_case(self):
        record = self.build(repeats=2, fixed_policy_seed=7)
        self.assertEqual(record.fixed_policy_seed, 7)
        self.assertEqual({case.policy_seed for case in record.cases}, {7})
        self.assertEqual({case.policy_seed_mode for case in record.cases}, {"fixed"})

    def test_v1_artifacts_are_migrated_explicitly(self):
        writer = RunWriter(self.root / "v1-run", self.build())
        for case in writer.manifest.cases:
            writer.append(episode(case))
        writer.finalize("complete")
        manifest_path = writer.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(schema_version=1, protocol_version="local-benchmark-v1")
        manifest.pop("fixed_policy_seed")
        for case in manifest["cases"]:
            case.pop("policy_seed_mode")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        path = writer.output / "episodes.jsonl"
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            row["case"].pop("policy_seed_mode")
            rows.append(json.dumps(row, separators=(",", ":")))
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        loaded = load_run(writer.output)
        self.assertEqual(loaded.manifest.schema_version, 1)
        self.assertEqual(loaded.manifest.protocol_version, "local-benchmark-v1")
        self.assertIsNone(loaded.manifest.fixed_policy_seed)
        self.assertEqual(
            {case.policy_seed_mode for case in loaded.manifest.cases}, {"derived"},
        )

    def test_missing_explicit_artifacts_and_sources_fail(self):
        missing = self.root / "missing.bin"
        with self.assertRaises(FileNotFoundError):
            self.build(extra_artifacts=[missing])
        with self.assertRaises(FileNotFoundError):
            self.build(policy_sources=[missing])
        with self.assertRaises(ValueError):
            self.build(extra_artifacts=[self.root])
        with self.assertRaises(ValueError):
            self.build(policy=PolicySpec(reference="plugin:create", label="unlocated"))


class RuntimeProvenanceTests(unittest.TestCase):
    def test_installed_packages_native_libraries_and_timer_are_recorded(self):
        with patch.dict(os.environ, {"PYGAME_HIDE_SUPPORT_PROMPT": "1"}):
            runtime = _runtime_info()
        self.assertTrue(runtime.python_version)
        self.assertTrue(runtime.os)
        self.assertIn("numpy", runtime.dependencies)
        self.assertNotEqual(runtime.dependencies["numpy"], "not-installed")
        for name in ("GEOS", "SDL", "numpy.blas", "scipy.lapack"):
            self.assertTrue(runtime.native_libraries[name])
        timing = _timing_context()
        self.assertTrue(timing["timer"]["monotonic"])
        self.assertIn("hostname", timing)
        self.assertIn("cpu", timing)
        self.assertIn("OMP_NUM_THREADS", timing["thread_environment"])


if __name__ == "__main__":
    unittest.main()
