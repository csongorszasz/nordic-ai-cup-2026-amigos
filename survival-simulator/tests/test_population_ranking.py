import unittest

from benchmark_fixtures import episode, manifest
from src.benchmarking.config import SimulationSettings
from src.benchmarking.survival import extinction_upper_bound, survival_curve, survival_metrics


class SurvivalRankingTests(unittest.TestCase):
    def setUp(self):
        self.cases = manifest(seeds=(1, 2)).cases
        self.settings = SimulationSettings(time_limit=0.2)
        self.expected = [case.case_id for case in self.cases]

    def metrics(self, rows):
        return survival_metrics(rows, expected_case_ids=self.expected, time_limit=0.2)

    def test_no_score_bonus_can_compensate_for_an_extra_extinction(self):
        safe = [episode(c, -1e9, settings=self.settings, ticks=3, termination="time_limit") for c in self.cases]
        unsafe = [episode(c, 1e9, settings=self.settings) for c in self.cases]
        self.assertGreater(self.metrics(safe)["rank"], self.metrics(unsafe)["rank"])

    def test_missing_duplicate_and_truncated_cases_cannot_rank(self):
        rows = [episode(c, settings=self.settings) for c in self.cases]
        for bad in (rows[:1], [rows[0], rows[0]], [
            rows[0], episode(self.cases[1], settings=self.settings, termination="step_limit"),
        ]):
            with self.subTest(rows=bad), self.assertRaises(ValueError):
                self.metrics(bad)

    def test_repeats_do_not_multiply_independent_worlds(self):
        cases = manifest(seeds=(1,), repeats=3).cases
        rows = [
            episode(c, settings=self.settings, ticks=3, termination="time_limit")
            for c in cases
        ]
        rows[1] = episode(cases[1], settings=self.settings)
        result = survival_metrics(
            rows, expected_case_ids=[c.case_id for c in cases], time_limit=0.2,
        )
        self.assertEqual(result["worlds"], 1)
        self.assertEqual(result["extinct_worlds"], 1)

    def test_zero_observed_failures_is_not_a_zero_risk_bound(self):
        self.assertAlmostEqual(extinction_upper_bound(0, 100), 0.0295130496)
        self.assertEqual(extinction_upper_bound(10, 10), 1.0)
        self.assertIsNone(extinction_upper_bound(0, 0))
        self.assertGreater(extinction_upper_bound(1, 100), extinction_upper_bound(0, 100))

    def test_horizon_completion_is_censored_in_survival_curve(self):
        rows = [
            episode(self.cases[0], settings=self.settings, ticks=1),
            episode(self.cases[1], settings=self.settings, ticks=3, termination="time_limit"),
        ]
        _, ys = survival_curve(rows)
        self.assertEqual(ys, [1.0, 0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
