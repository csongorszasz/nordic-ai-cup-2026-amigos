import csv
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src.benchmarking.artifacts import RunWriter, SavedRun, load_run
from src.benchmarking.comparison import compare_runs, write_comparison_report
from src.benchmarking.config import FailureInfo, PolicySpec, SimulationSettings, Suite, make_cases
from tests.benchmark_fixtures import episode, fingerprint, manifest, saved


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ComparisonStatisticsTests(unittest.TestCase):
    def test_hand_computed_absolute_scores_deltas_and_measurements(self):
        reference = saved([-4, 0, 8], label="reference-policy")
        candidate = saved([-2, 0, 5], label="candidate-policy")
        report = compare_runs(reference, [candidate])
        pair = report["comparisons"][0]
        self.assertAlmostEqual(pair["mean_delta"], -1 / 3)
        self.assertEqual((pair["wins"], pair["ties"], pair["losses"]), (1, 1, 1))
        self.assertEqual([row["delta"] for row in pair["case_deltas"]], [2, 0, -3])
        summary = report["runs"][1]["summary"]
        self.assertEqual(summary["score"]["mean"], 1)
        self.assertEqual(summary["score"]["median"], 0)
        self.assertAlmostEqual(summary["score"]["sample_std"], math.sqrt(13))
        self.assertEqual((summary["score"]["min"], summary["score"]["max"]), (-2, 5))
        self.assertEqual(summary["score"]["count"], 3)
        self.assertAlmostEqual(summary["survival_seconds"]["mean"], 0.2)
        self.assertEqual(summary["completion"]["fraction"], 0)
        self.assertEqual(summary["population"]["final"]["mean"], 0)
        self.assertEqual(summary["runtime"]["episode_seconds"]["count"], 3)
        self.assertEqual(report["leaderboard"][0]["key"], "reference")
        self.assertEqual(report["bootstrap"]["n_resamples"], 10_000)
        self.assertEqual(report["bootstrap"]["method"], "percentile")
        self.assertEqual(report["bootstrap"]["sampling_unit"], "world_seed")
        self.assertLessEqual(pair["ci95"]["low"], 0)
        self.assertGreaterEqual(pair["ci95"]["high"], 0)
        self.assertTrue(any("includes zero" in warning for warning in pair["warnings"]))
        json.dumps(report, allow_nan=False)

    def test_rows_are_joined_by_case_identity_not_file_order(self):
        reference = saved([10, -1, 4], label="reference")
        candidate = saved([13, 1, 3], label="candidate")
        original = compare_runs(reference, [candidate])
        permuted_manifest = candidate.manifest.model_copy(update={"cases": list(reversed(candidate.manifest.cases))})
        candidate = replace(candidate, manifest=permuted_manifest, episodes=tuple(reversed(candidate.episodes)))
        reference = replace(reference, episodes=(reference.episodes[1], reference.episodes[2], reference.episodes[0]))
        permuted = compare_runs(reference, [candidate])
        self.assertEqual(original["comparisons"], permuted["comparisons"])
        self.assertEqual(original["runs"][1]["scores"], permuted["runs"][1]["scores"])
        self.assertEqual(original["runs"][1]["summary"], permuted["runs"][1]["summary"])

    def test_bootstrap_and_pair_statistics_do_not_depend_on_candidate_order(self):
        reference = saved([0, 1, 2, 3])
        left = saved([1, 5, -2, 9], label="left")
        right = saved([-3, 7, 0, 4], label="right")
        first = compare_runs(reference, [left, right])
        second = compare_runs(reference, [right, left])
        for label in ("left", "right"):
            a = next(pair for pair in first["comparisons"] if pair["label"] == label)
            b = next(pair for pair in second["comparisons"] if pair["label"] == label)
            for field in (
                "mean_delta", "ci95", "wins", "ties", "losses", "case_deltas", "seed_deltas",
                "case_delta_summary", "seed_delta_summary",
            ):
                self.assertEqual(a[field], b[field], field)

    def test_repeats_are_clustered_before_world_seed_bootstrap(self):
        reference = saved([0, 0, 0, 0], repeats=2)
        candidate = saved([-100, 102, 2, 4], repeats=2, label="candidate")
        report = compare_runs(reference, [candidate])
        pair = report["comparisons"][0]
        self.assertEqual(pair["world_count"], 2)
        self.assertEqual(pair["case_count"], 4)
        self.assertEqual([row["mean_delta"] for row in pair["seed_deltas"]], [1, 3])
        self.assertEqual([row["repeats"] for row in pair["seed_deltas"]], [2, 2])
        self.assertEqual(pair["mean_delta"], 2)
        self.assertEqual(pair["ci95"], {"low": 1.0, "high": 3.0, "status": "estimated"})
        collapsed = compare_runs(saved([0, 0]), [saved([1, 3])])["comparisons"][0]
        self.assertEqual(pair["ci95"], collapsed["ci95"])
        self.assertEqual((pair["wins"], pair["ties"], pair["losses"]), (3, 0, 1))
        self.assertEqual(pair["win_tie_loss_unit"], "paired_case")

    def test_single_world_never_has_an_interval_even_with_repeats_or_self_comparison(self):
        for repeats in (1, 3):
            with self.subTest(repeats=repeats):
                reference = saved([0] * repeats, repeats=repeats)
                for candidate in (reference, saved(list(range(repeats)), repeats=repeats)):
                    pair = compare_runs(reference, [candidate])["comparisons"][0]
                    self.assertIsNone(pair["ci95"]["low"])
                    self.assertIsNone(pair["ci95"]["high"])
                    self.assertEqual(pair["ci95"]["status"], "unavailable_single_world")
                    self.assertTrue(any("One independent world" in warning for warning in pair["warnings"]))

    def test_constant_and_self_deltas_are_point_intervals_with_zero_or_negative_means(self):
        reference = saved([-5, -1, 0], label="duplicate")
        candidate = saved([-3, 1, 2], label="duplicate")
        with patch("src.benchmarking.comparison.bootstrap", side_effect=AssertionError("not needed")):
            report = compare_runs(reference, [candidate, reference, reference])
        self.assertEqual([run["key"] for run in report["runs"]], [
            "reference", "candidate-1", "candidate-2", "candidate-3",
        ])
        self.assertEqual(len({run["run_id"] for run in report["runs"]}), 1)
        self.assertEqual(report["comparisons"][0]["ci95"], {"low": 2.0, "high": 2.0, "status": "point"})
        for pair in report["comparisons"][1:]:
            self.assertEqual(pair["mean_delta"], 0)
            self.assertEqual(pair["ci95"], {"low": 0.0, "high": 0.0, "status": "point"})
            self.assertEqual((pair["wins"], pair["ties"], pair["losses"]), (0, 3, 0))
        self.assertEqual(report["leaderboard"][0]["summary"]["score"]["mean"], 0)
        self.assertNotIn("percent_change", report["comparisons"][0])

    def test_any_policy_can_be_the_reference_and_ties_use_absolute_fixed_tolerance(self):
        reference = saved([0, 0, 0, 0], label="future-policy")
        candidate = saved([-1e-9, 1e-9, -2e-9, 2e-9], label="another-policy")
        report = compare_runs(reference, [candidate])
        self.assertEqual(report["reference_label"], "future-policy")
        pair = report["comparisons"][0]
        self.assertEqual((pair["wins"], pair["ties"], pair["losses"]), (1, 2, 1))
        self.assertEqual(report["tie_tolerance"], 1e-9)
        reverse = compare_runs(candidate, [reference])
        self.assertEqual(reverse["reference_label"], "another-policy")
        self.assertEqual(reverse["comparisons"][0]["mean_delta"], -pair["mean_delta"])

    def test_quick_suite_is_exploratory_and_candidate_is_required(self):
        run = saved([1, 2, 3], suite_name="quick")
        self.assertTrue(any("exploratory" in warning for warning in compare_runs(run, [run])["warnings"]))
        with self.assertRaises(ValueError):
            compare_runs(run, [])


class ComparisonValidationTests(unittest.TestCase):
    def setUp(self):
        self.reference = saved([1, 2, 3], label="reference")
        self.candidate = saved([2, 3, 4], label="candidate")

    def changed_provenance(self, **changes):
        provenance = self.candidate.manifest.provenance.model_copy(update=changes)
        record = self.candidate.manifest.model_copy(update={"provenance": provenance})
        return replace(self.candidate, manifest=record)

    def test_policy_config_git_reporting_and_model_changes_are_expected(self):
        candidate = self.changed_provenance(
            policy=fingerprint("edited-untracked-policy"), artifacts=fingerprint("different-model"),
            reporting=fingerprint("new-reporting"),
            git=self.candidate.manifest.provenance.git.model_copy(update={
                "revision": "a-different-revision", "dirty": True, "status": "?? untracked-policy.py",
            }),
        )
        candidate = replace(candidate, manifest=candidate.manifest.model_copy(update={
            "policy": PolicySpec(reference="new.plugin:create", label="changed", config={"weight": 3}),
        }))
        report = compare_runs(self.reference, [candidate])
        self.assertEqual(report["comparisons"][0]["mean_delta"], 1)
        self.assertEqual(report["runs"][1]["manifest"], candidate.manifest.model_dump(mode="json"))

    def test_engine_and_runner_fingerprint_mismatches_reject_comparison(self):
        for field in ("engine", "runner"):
            with self.subTest(field=field):
                candidate = self.changed_provenance(**{field: fingerprint("changed")})
                with self.assertRaisesRegex(ValueError, f"{field} fingerprint"):
                    compare_runs(self.reference, [candidate])

    def test_relevant_runtime_mismatches_reject_but_timing_context_only_warns(self):
        runtime = self.candidate.manifest.provenance.runtime
        changes = (
            {"python_version": "3.13.0"}, {"implementation": "OtherPython"}, {"os": "Linux"},
            {"os_release": "different"}, {"architecture": "arm64"},
            {"dependencies": {**runtime.dependencies, "numpy": "different"}},
            {"native_libraries": {"GEOS": "different"}},
        )
        for change in changes:
            with self.subTest(change=change):
                candidate = self.changed_provenance(runtime=runtime.model_copy(update=change))
                with self.assertRaisesRegex(ValueError, "runtime Python/OS"):
                    compare_runs(self.reference, [candidate])
        candidate = self.changed_provenance(timing_context={"hostname": "other-host", "threads": 8})
        report = compare_runs(self.reference, [candidate])
        self.assertFalse(report["timing_compatible"])
        self.assertFalse(report["comparisons"][0]["timing_compatible"])
        self.assertTrue(any("not a controlled speed" in warning for warning in report["warnings"]))
        self.assertEqual(report["comparisons"][0]["mean_delta"], 1)
        self.assertTrue(compare_runs(self.reference, [self.candidate])["timing_compatible"])

    def test_suite_hash_case_set_repeats_simulation_and_protocol_must_match(self):
        candidates = (
            saved([1, 2, 3], seeds=[7, 8, 9]),
            saved([1, 2, 3], suite_name="different"),
            saved([1, 2, 3, 4, 5, 6], repeats=2),
            saved([1, 2, 3], settings=SimulationSettings(time_limit=4000)),
            replace(self.candidate, manifest=self.candidate.manifest.model_copy(update={"suite_sha256": "0" * 64})),
            replace(self.candidate, manifest=self.candidate.manifest.model_copy(update={"protocol_version": "new"})),
            replace(self.candidate, manifest=self.candidate.manifest.model_copy(update={"seed_version": "new"})),
            replace(self.candidate, manifest=self.candidate.manifest.model_copy(update={"execution": "parallel"})),
        )
        for index, candidate in enumerate(candidates):
            with self.subTest(index=index), self.assertRaises(ValueError):
                compare_runs(self.reference, [candidate])

    def test_incomplete_failure_interrupt_running_and_diagnostic_runs_cannot_rank(self):
        failure = FailureInfo(stage="run", error_type="Error", message="failed")
        for status in ("running", "failed", "interrupted", "truncated"):
            record = self.candidate.manifest.model_copy(update={
                "status": status, "failure": failure if status == "failed" else None,
            })
            with self.subTest(status=status), self.assertRaises(ValueError):
                compare_runs(self.reference, [replace(self.candidate, manifest=record)])
        capped = replace(
            self.candidate, manifest=self.candidate.manifest.model_copy(update={"max_steps": 100}),
        )
        with self.assertRaisesRegex(ValueError, "diagnostic max_steps"):
            compare_runs(capped, [capped])
        cases = self.candidate.manifest.cases
        truncated = replace(capped, manifest=capped.manifest.model_copy(update={"status": "truncated"}), episodes=(
            episode(cases[0], termination="step_limit"), *self.candidate.episodes[1:],
        ))
        with self.assertRaises(ValueError):
            compare_runs(self.reference, [truncated])

    def test_missing_duplicate_unknown_and_malformed_cases_are_rejected_before_pairing(self):
        other = make_cases(Suite(name="other", seeds=[99]))[0]
        candidates = (
            replace(self.candidate, episodes=self.candidate.episodes[:-1]),
            replace(self.candidate, episodes=(*self.candidate.episodes, self.candidate.episodes[0])),
            replace(self.candidate, episodes=(episode(other), *self.candidate.episodes[1:])),
            replace(self.candidate, manifest=self.candidate.manifest.model_copy(update={
                "cases": [*self.candidate.manifest.cases, self.candidate.manifest.cases[0]],
            })),
        )
        for index, candidate in enumerate(candidates):
            with self.subTest(index=index), self.assertRaises(ValueError):
                compare_runs(self.reference, [candidate])

    def test_measurement_time_ticks_horizon_and_population_are_validated(self):
        original = self.candidate.episodes[0]
        for changes in (
            {"sim_time": 0.4}, {"ticks": 3}, {"survival_seconds": 2},
            {"initial_agents": 99}, {"peak_agents": 1}, {"mean_population": 6},
        ):
            with self.subTest(changes=changes):
                invalid = original.model_copy(update=changes)
                candidate = replace(self.candidate, episodes=(invalid, *self.candidate.episodes[1:]))
                with self.assertRaises(ValueError):
                    compare_runs(self.reference, [candidate])

    def test_latency_requires_real_calls_and_consistent_episode_statistics(self):
        original = self.candidate.episodes[0]
        for changes in (
            {"policy_calls": 0}, {"batch_mean_ms": None}, {"batch_p95_ms": 1},
            {"batch_mean_ms": 3}, {"policy_seconds": 0.1},
        ):
            with self.subTest(changes=changes):
                invalid = original.model_copy(update={"timings": original.timings.model_copy(update=changes)})
                candidate = replace(self.candidate, episodes=(invalid, *self.candidate.episodes[1:]))
                with self.assertRaises(ValueError):
                    compare_runs(self.reference, [candidate])
        rows = tuple(episode(case, ticks=1) for case in self.candidate.manifest.cases)
        candidate = replace(self.candidate, episodes=rows)
        report = compare_runs(self.reference, [candidate])
        self.assertEqual(report["runs"][1]["batch_mean_ms"], [None, None, None])
        self.assertEqual(report["runs"][1]["summary"]["policy_latency"]["batch_mean_ms"]["count"], 0)

    def test_strict_float_horizon_first_terminal_tick_and_extinction_precedence(self):
        settings = SimulationSettings(dt=0.1, time_limit=0.3)
        record = manifest(seeds=(1, 2), simulation=settings, status="complete")
        rows = tuple(episode(case, ticks=3, settings=settings, termination="time_limit") for case in record.cases)
        run = SavedRun(Path("float-boundary"), record, rows)
        report = compare_runs(run, [run])
        self.assertGreater(rows[0].sim_time, 0.3)
        self.assertEqual(rows[0].ticks, 3)
        self.assertEqual(report["runs"][0]["summary"]["completion"]["fraction"], 1)
        extinction = tuple(episode(case, ticks=3, settings=settings) for case in record.cases)
        extinct = replace(run, episodes=extinction)
        self.assertEqual(compare_runs(run, [extinct])["runs"][1]["summary"]["completion"]["fraction"], 0)
        late = tuple(episode(case, ticks=4, settings=settings, termination="time_limit") for case in record.cases)
        with self.assertRaisesRegex(ValueError, "first terminal tick"):
            compare_runs(run, [replace(run, episodes=late)])
        exact_settings = SimulationSettings(dt=1, time_limit=2)
        exact_record = record.model_copy(update={"simulation": exact_settings})
        exact_rows = tuple(
            episode(case, ticks=2, settings=exact_settings, termination="time_limit") for case in record.cases
        )
        with self.assertRaisesRegex(ValueError, "requires sim_time >"):
            compare_runs(replace(run, manifest=exact_record, episodes=exact_rows), [run])

    def test_default_horizon_accepts_actual_accumulated_float_time_not_fixed_tick_count(self):
        settings = SimulationSettings()
        ticks = 0
        elapsed = 0.0
        while elapsed <= settings.time_limit:
            elapsed += settings.dt
            ticks += 1
        record = manifest(seeds=(1, 2), status="complete")
        rows = tuple(
            episode(case, ticks=ticks, settings=settings, termination="time_limit")
            for case in record.cases
        )
        run = SavedRun(Path("native-horizon"), record, rows)
        report = compare_runs(run, [run])
        self.assertEqual(report["runs"][0]["summary"]["survival_seconds"]["mean"], 3000)
        self.assertEqual(report["comparisons"][0]["mean_delta"], 0)


class ComparisonArtifactTests(unittest.TestCase):
    def test_report_is_written_from_saved_inputs_with_provenance_and_plot_data_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix=".benchmark-comparison-", dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            record = manifest(seeds=(1, 2))
            writer = RunWriter(root / "saved", record)
            for index, case in enumerate(record.cases):
                writer.append(episode(case, index + 1, batch_ms=index + 2))
            writer.finalize("complete")
            run = load_run(writer.output)
            report = compare_runs(run, [run, run])
            output = root / "report"
            write_comparison_report(output, report)
            stored = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(stored, report)
            self.assertEqual(stored["runs"][0]["manifest"], run.manifest.model_dump(mode="json"))
            self.assertEqual(stored["runs"][0]["scores"], [1, 2])
            self.assertEqual(stored["runs"][0]["batch_mean_ms"], [2, 3])
            self.assertEqual(
                stored["runs"][0]["case_ids"], [case.case_id for case in run.manifest.cases],
            )
            with (output / "paired_cases.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["candidate_key"] for row in rows}, {"candidate-1", "candidate-2"})
            self.assertTrue(all(float(row["delta"]) == 0 for row in rows))
            markdown = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("Descriptive leaderboard", markdown)
            self.assertIn("fixed PCG64 seed", markdown)
            before = (output / "summary.json").read_bytes()
            with self.assertRaises(FileExistsError):
                write_comparison_report(output, report)
            self.assertEqual((output / "summary.json").read_bytes(), before)
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaises(FileExistsError):
                write_comparison_report(empty, report)


if __name__ == "__main__":
    unittest.main()
