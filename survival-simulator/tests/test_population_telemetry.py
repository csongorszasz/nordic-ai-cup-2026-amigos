import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

from benchmark_fixtures import manifest
from src.benchmarking.artifacts import RunWriter, _atomic_text
from src.benchmarking.config import SimulationSettings, Suite, make_cases, read_json
from src.benchmarking.progress import last_complete_sample, progress_data, read_complete_lines, render_progress
from src.benchmarking.runner import EpisodeExecutionError, run_episode
from src.benchmarking.telemetry import EpisodeTelemetry
from src.utils.DTOs import ActionRequest
from test_policy_semantics import agent, core, environment


class OneBirth:
    def __init__(self):
        self.calls = 0

    def act(self, step):
        self.calls += 1
        return [
            ActionRequest(
                agent_id=a.agent_id, move_distance=0.0, move_direction=0.0,
                turn_angle=0.0, spawn_agent=self.calls == 1 and a.agent_id == 0,
            )
            for a in step.agent_status
        ]


class PopulationTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.storage = tempfile.TemporaryDirectory()
        self.addCleanup(self.storage.cleanup)
        self.root = Path(self.storage.name)
        self.case = make_cases(Suite(name="fixture", seeds=[1]))[0]
        self.settings = SimulationSettings(starting_agents=1, time_limit=0.3)

    def run_fixture(self, telemetry=None, policy=None):
        env = environment(agent(energy=250.0, max_age=60.0))
        env.agents[0].age = 60.0
        env.spawn_fruit(x=100.0, y=100.0)
        result = run_episode(
            self.case, lambda **kwargs: policy or OneBirth(), {}, self.settings,
            simulation_factory=lambda **kwargs: core(env), telemetry=telemetry,
        )
        return result, env

    def test_energy_accounting_and_instrumentation_do_not_change_the_fixture(self):
        ordinary, first = self.run_fixture()
        telemetry = EpisodeTelemetry(self.root, self.case, time_limit=0.3, sample_every=1)
        recorded, second = self.run_fixture(telemetry)
        self.assertEqual(ordinary.score, recorded.score)
        self.assertEqual(first.rng.getstate(), second.rng.getstate())
        self.assertEqual(
            [(a.agent_id, a.x, a.y, a.age, a.energy) for a in first.agents],
            [(a.agent_id, a.x, a.y, a.age, a.energy) for a in second.agents],
        )
        summary = read_json(telemetry.path / "summary.json")
        self.assertEqual(summary["counts"]["birth"], 1)
        self.assertEqual(summary["totals"]["cost_birth"], 100.0)
        self.assertEqual(summary["totals"]["newborn_energy"], 75.0)
        self.assertAlmostEqual(summary["last_sample"]["energy_balance_residual"], 0.0)
        births = [r for r in read_complete_lines(telemetry.path / "events.jsonl") if r["kind"] == "birth"]
        self.assertEqual(births[0]["parent_id"], 0)
        self.assertIsNone(second.event_sink)
        self.assertTrue(telemetry.samples.closed)
        decisions = read_complete_lines(telemetry.path / "decisions.jsonl")
        self.assertEqual(decisions[0]["tick"], 1)
        self.assertTrue(decisions[0]["actions"][0]["spawn_agent"])

    def test_failure_is_explicit_and_closes_the_trace(self):
        class Failing:
            def act(self, step):
                raise ValueError("fixture failure")

        telemetry = EpisodeTelemetry(self.root, self.case)
        with self.assertRaises(EpisodeExecutionError):
            self.run_fixture(telemetry, Failing())
        summary = read_json(telemetry.path / "summary.json")
        self.assertEqual(summary["state"], "failed")
        self.assertIsNone(summary["result"]["score"])
        self.assertTrue(telemetry.events.closed)

    def test_policy_mutation_cannot_corrupt_the_recorded_public_request(self):
        class Mutating(OneBirth):
            def act(self, step):
                actions = super().act(step)
                step.agent_status[0].energy = 0.0
                return actions

        telemetry = EpisodeTelemetry(self.root, self.case)
        self.run_fixture(telemetry, Mutating())
        recorded = read_complete_lines(telemetry.path / "decisions.jsonl")
        self.assertGreater(recorded[0]["step"]["agent_status"][0]["energy"], 200.0)

    def test_storage_exhaustion_fails_instead_of_dropping_events(self):
        telemetry = EpisodeTelemetry(self.root, self.case)
        telemetry.max_bytes = 1
        with self.assertRaises(EpisodeExecutionError) as caught:
            self.run_fixture(telemetry)
        self.assertIn("storage budget", caught.exception.result.failure.message)
        self.assertEqual(read_json(telemetry.path / "summary.json")["state"], "failed")
        self.assertTrue(telemetry.samples.closed)

    def test_plot_reader_only_tolerates_an_unfinished_trailing_record(self):
        path = self.root / "partial.jsonl"
        path.write_text('{"ok":1}\n{"unfinished":', encoding="utf-8")
        self.assertEqual(read_complete_lines(path), [{"ok": 1}])
        path.write_text('{"ok":1}\n{invalid}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            read_complete_lines(path)

    def test_live_reader_tails_append_only_samples_across_large_records(self):
        path = self.root / "samples.jsonl"
        path.write_text('{"tick":1}\n{"tick":2,"large":"' + "x" * 20000 + '"}\n{"tick":', encoding="utf-8")
        self.assertEqual(last_complete_sample(path)["tick"], 2)
        path.write_text('{"tick":', encoding="utf-8")
        self.assertIsNone(last_complete_sample(path))
        path.write_text('{"tick":1}\n{invalid}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            last_complete_sample(path)

    def test_live_progress_never_replaces_a_file_that_readers_may_lock(self):
        telemetry = EpisodeTelemetry(self.root, self.case, sample_every=1)
        old_latest = telemetry.path / "latest.json"
        old_latest.write_text('{"legacy":true}\n', encoding="utf-8")
        original = telemetry._json
        names = []

        def track(name, value):
            names.append(name)
            original(name, value)

        with (
            old_latest.open("rb"),
            (telemetry.path / "samples.jsonl").open("rb"),
            patch.object(telemetry, "_json", side_effect=track),
        ):
            self.run_fixture(telemetry)
        self.assertNotIn("latest.json", names)
        self.assertEqual(names.count("last-request.json"), 1)
        self.assertEqual(read_json(old_latest), {"legacy": True})
        self.assertIsNotNone(last_complete_sample(telemetry.path / "samples.jsonl"))

    def test_progress_is_durable_and_full_case_only(self):
        run = self.root / "run"
        writer = RunWriter(run, manifest(seeds=(1,), simulation=self.settings))
        telemetry = EpisodeTelemetry(run, self.case, time_limit=0.3, sample_every=1)
        self.assertEqual(progress_data(run)["points"][0]["state"], "partial")
        result, _ = self.run_fixture(telemetry)
        writer.append(result)
        writer.finalize("complete")
        data = progress_data(run)
        self.assertEqual(data["points"][0]["state"], "complete")
        self.assertEqual(data["points"][0]["metrics"]["completed_worlds"], 1)
        paths = render_progress(run)
        self.assertEqual(len(paths), 5)
        self.assertTrue(all(path.stat().st_size > 5000 for path in paths))
        page = (run / "progress" / "index.html").read_text(encoding="utf-8")
        self.assertIn('http-equiv="refresh"', page)
        self.assertEqual(read_json(run / "progress" / "status.json")["run_state"], "complete")

    def test_windows_sharing_retry_is_bounded_and_preserves_atomic_output(self):
        path = self.root / "status.json"
        _atomic_text(path, "old")
        original = os.replace
        error = PermissionError("fixture file lock")
        error.winerror = 32
        attempts = []

        def locked_once(source, destination):
            attempts.append(source)
            if len(attempts) == 1:
                raise error
            original(source, destination)

        with (
            patch("src.benchmarking.artifacts.os.replace", side_effect=locked_once),
            patch("src.benchmarking.artifacts.time.sleep"),
            self.assertLogs("src.benchmarking.artifacts", level="WARNING"),
        ):
            _atomic_text(path, "new")
        self.assertEqual(path.read_text(encoding="utf-8"), "new")
        self.assertEqual(len(attempts), 2)
        self.assertLess(len(attempts[0].name), 40)
        with (
            patch("src.benchmarking.artifacts.os.replace", side_effect=error) as replace,
            patch("src.benchmarking.artifacts.time.sleep"),
            self.assertLogs("src.benchmarking.artifacts", level="WARNING"),
            self.assertRaises(PermissionError),
        ):
            _atomic_text(path, "never")
        self.assertEqual(replace.call_count, 6)
        self.assertEqual(path.read_text(encoding="utf-8"), "new")


if __name__ == "__main__":
    unittest.main()
