import time
import unittest

from src.benchmarking.jobs import JobExecutionError, run_jobs
from src.policies.config import HeuristicConfig, SearchConfig
from src.training.search import optimize_controller


def square(value):
    return value * value


def fail(value):
    raise ValueError(f"fixture job {value}")


def block(value):
    time.sleep(60)
    return value


class PopulationJobsTests(unittest.TestCase):
    def test_spawn_jobs_preserve_identity_without_requiring_completion_order(self):
        results = list(run_jobs([3, 1, 2], square, workers=2, timeout_seconds=30))
        self.assertEqual(dict(results), {3: 9, 1: 1, 2: 4})

    def test_worker_failure_is_explicit(self):
        with self.assertRaises(JobExecutionError) as caught:
            list(run_jobs([7], fail, workers=1, timeout_seconds=30))
        self.assertEqual(caught.exception.job, 7)
        self.assertIn("fixture job 7", str(caught.exception))

    def test_owned_process_watchdog_stops_a_blocked_job(self):
        with self.assertRaisesRegex(JobExecutionError, "watchdog"):
            list(run_jobs([1], block, workers=1, timeout_seconds=0.5))

    def test_raw_survival_order_selects_the_winner_across_cma_generations(self):
        base = HeuristicConfig()
        settings = SearchConfig(method="cma", objective="survival", candidates=18, parameters={
            "breeding_age": (30.0, 60.0), "population_reserve": (5.0, 30.0),
        })

        def score(config):
            return (float(config.breeding_age < 50), -abs(config.population_reserve - 15), config.breeding_age)

        result = optimize_controller(base, settings, 42, score)
        self.assertEqual(result.best_score, max(trial.score for trial in result.trials))
        self.assertEqual(result.best_score, score(result.best_config))
        self.assertGreater(result.best_score[0], 0)

    def test_search_rejects_categorical_and_boolean_bounds(self):
        for name in ("elder_decoys", "backend", "population_scan_ticks"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                SearchConfig(parameters={name: (0, 1)})


if __name__ == "__main__":
    unittest.main()
