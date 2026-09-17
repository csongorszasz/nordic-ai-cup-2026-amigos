import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import benchmark
from src.benchmarking.config import (
    EpisodeResult, FailureInfo, SimulationSettings, load_suite, make_cases,
)
from src.benchmarking.runner import EpisodeExecutionError, EpisodeInterrupted


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.cases = make_cases(load_suite("quick"))
        self.manifest = SimpleNamespace(
            cases=self.cases, simulation=SimulationSettings(), max_steps=None,
        )

    def result(self, case, truncated=False):
        return EpisodeResult(
            case=case, status="truncated" if truncated else "ok",
            termination="step_limit" if truncated else "extinction",
            score=0.1, sim_time=0.1, survival_seconds=0.1, ticks=1,
            initial_agents=5, final_agents=5 if truncated else 0, peak_agents=5,
            mean_population=5.0 if truncated else 0.0,
        )

    def execute_run(self, results, *extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.object(benchmark, "build_manifest", return_value=self.manifest),
            patch.object(benchmark, "RunWriter") as writer_class,
            patch.object(benchmark, "run_episode", side_effect=results) as runner,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            writer = writer_class.return_value
            writer.results = []
            writer.append.side_effect = writer.results.append
            writer.finalize.return_value = {"score": {"mean": 0.1, "median": 0.1}}
            exit_code = benchmark.main(["run", "--output", "unused-fixture-directory", *extra])
        return exit_code, writer, runner, stdout.getvalue(), stderr.getvalue()

    def test_complete_run_persists_each_episode_then_finalizes(self):
        results = [self.result(case) for case in self.cases]
        exit_code, writer, runner, stdout, stderr = self.execute_run(results)
        self.assertEqual(exit_code, 0)
        self.assertEqual(writer.results, results)
        self.assertEqual(runner.call_count, 3)
        writer.finalize.assert_called_once_with("complete")
        self.assertIn("3/3", stdout)
        self.assertEqual(stderr, "")

    def test_failure_is_saved_and_stops_subsequent_cases(self):
        failure = EpisodeResult(
            case=self.cases[1], status="failed", termination="error",
            failure=FailureInfo(stage="policy_decision", error_type="RuntimeError", message="bad policy"),
        )
        exit_code, writer, runner, _, stderr = self.execute_run([
            self.result(self.cases[0]), EpisodeExecutionError(failure),
        ])
        self.assertEqual(exit_code, 1)
        self.assertEqual(writer.results[-1], failure)
        self.assertEqual(runner.call_count, 2)
        writer.finalize.assert_called_once_with("failed", failure.failure)
        self.assertIn("bad policy", stderr)

    def test_interrupt_preserves_completed_and_partial_episodes(self):
        interrupted = EpisodeResult(
            case=self.cases[1], status="interrupted", termination="interrupted",
            failure=FailureInfo(
                stage="simulation_step", error_type="KeyboardInterrupt", message="Interrupted by user.",
            ),
        )
        exit_code, writer, _, _, _ = self.execute_run([
            self.result(self.cases[0]), EpisodeInterrupted(interrupted),
        ])
        self.assertEqual(exit_code, 130)
        self.assertEqual(len(writer.results), 2)
        writer.finalize.assert_called_once_with("interrupted", interrupted.failure)

    def test_between_episode_interrupt_preserves_prior_results(self):
        exit_code, writer, _, _, _ = self.execute_run([
            self.result(self.cases[0]), KeyboardInterrupt(),
        ])
        self.assertEqual(exit_code, 130)
        self.assertEqual(len(writer.results), 1)
        self.assertEqual(writer.finalize.call_args.args[0], "interrupted")

    def test_diagnostic_run_is_labeled_not_rankable(self):
        self.manifest.max_steps = 1
        exit_code, writer, _, stdout, _ = self.execute_run(
            [self.result(case, truncated=True) for case in self.cases], "--max-steps", "1",
        )
        self.assertEqual(exit_code, 0)
        writer.finalize.assert_called_once_with("truncated")
        self.assertIn("not eligible for ranking", stdout)

    def test_invalid_numbers_are_argparse_errors(self):
        for option, value in (("--repeats", "0"), ("--max-steps", "-1"), ("--repeats", "1.5")):
            with self.subTest(option=option, value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    benchmark.main(["run", "--output", "unused", option, value])
                self.assertEqual(caught.exception.code, 2)

    def test_invalid_config_cannot_create_an_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.json"
            config.write_text("[]", encoding="utf-8")
            with (
                patch.object(benchmark, "RunWriter") as writer,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                code = benchmark.main(["run", "--output", "unused", "--config", str(config)])
            self.assertEqual(code, 1)
            writer.assert_not_called()

    def test_comparison_without_plots_does_not_import_matplotlib(self):
        report = {
            "comparisons": [{"key": "candidate-1", "mean_delta": 1.0}],
            "leaderboard": [{
                "key": "candidate-1", "label": "candidate",
                "summary": {
                    "score": {"mean": 2.0}, "survival_seconds": {"mean": 3.0},
                    "completion": {"completed": 0, "count": 3},
                },
            }],
            "warnings": ["Quick-suite results are exploratory."],
        }
        stdout = io.StringIO()
        with (
            patch.object(benchmark, "load_run") as load,
            patch.object(benchmark, "compare_runs", return_value=report),
            patch.object(benchmark, "write_comparison_report") as write,
            patch.dict("sys.modules", {"matplotlib": None}),
            contextlib.redirect_stdout(stdout),
        ):
            code = benchmark.main([
                "compare", "--reference", "r", "--candidate", "c1",
                "--candidate", "c2", "--output", "out", "--no-plots",
            ])
        self.assertEqual(code, 0)
        self.assertEqual(load.call_count, 3)
        write.assert_called_once_with(Path("out"), report)
        self.assertIn("candidate (candidate-1) | 2.000000", stdout.getvalue())
        self.assertIn("1.000000", stdout.getvalue())
        self.assertIn("Quick-suite", stdout.getvalue())

    def test_inspect_summarizes_saved_records_without_running_policies(self):
        stdout = io.StringIO()
        saved = SimpleNamespace(manifest=object(), episodes=[])
        with (
            patch.object(benchmark, "load_run", return_value=saved),
            patch.object(benchmark, "summarize_run", return_value={"status": "incomplete"}),
            patch.object(benchmark, "run_episode") as runner,
            contextlib.redirect_stdout(stdout),
        ):
            code = benchmark.main(["inspect", "--run", "partial-run"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue()), {"status": "incomplete"})
        runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
