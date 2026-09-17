import builtins
import contextlib
import importlib.util
import io
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.policies.config import HeuristicConfig, SearchConfig
from src.training.search import SearchResult, optimize_controller


def objective(config):
    return -((config.food_weight - 1.3) ** 2 + (config.danger_weight - 3.7) ** 2)


def settings(method="random", candidates=13, **changes):
    values = {
        "method": method, "candidates": candidates,
        "parameters": {"food_weight": (0.0, 4.0), "danger_weight": (0.0, 8.0)},
    }
    return SearchConfig(**(values | changes))


class ControllerSearchTests(unittest.TestCase):
    def setUp(self):
        self.base = HeuristicConfig(food_weight=3.7, danger_weight=7.5, wall_weight=3.25)

    def assert_bounds(self, result, search):
        for trial in result.trials[1:]:
            for name, (lower, upper) in search.parameters.items():
                self.assertGreaterEqual(getattr(trial.config, name), lower)
                self.assertLessEqual(getattr(trial.config, name), upper)

    def test_random_search_is_baseline_first_and_maximizes(self):
        calls = []

        def evaluate(config):
            calls.append(config)
            return objective(config)

        search = settings(candidates=17)
        result = optimize_controller(self.base, search, 23, evaluate)
        self.assertIsInstance(result, SearchResult)
        self.assertEqual(result.method, "random")
        self.assertEqual(len(calls), search.candidates)
        self.assertEqual([trial.index for trial in result.trials], list(range(search.candidates)))
        self.assertIs(calls[0], self.base)
        self.assertEqual(result.trials[0].score, objective(self.base))
        self.assertEqual(result.best_score, max(trial.score for trial in result.trials))
        self.assertEqual(result.best_score, objective(result.best_config))
        self.assertGreater(result.best_score, objective(self.base))
        self.assert_bounds(result, search)

    def test_random_reproducibility_and_parameter_order_independence(self):
        search = settings()
        permuted = settings(parameters=dict(reversed(list(search.parameters.items()))))
        first = optimize_controller(self.base, search, 0, objective)
        second = optimize_controller(self.base, permuted, 0, objective)
        self.assertEqual(first, second)
        self.assertNotEqual(first.trials, optimize_controller(self.base, search, 1, objective).trials)

    def test_trial_parameters_are_full_and_do_not_allow_mutating_configs(self):
        result = optimize_controller(self.base, settings(candidates=2), 1, objective)
        for trial in result.trials:
            self.assertEqual(trial.parameters, trial.config.model_dump(mode="json"))
            self.assertEqual(trial.config.wall_weight, 3.25)
            changed = trial.parameters
            changed["wall_weight"] = 100
            self.assertEqual(trial.config.wall_weight, 3.25)
        self.assertEqual(self.base.wall_weight, 3.25)
        self.assertIsInstance(result.trials, tuple)

    def test_baseline_outside_search_bounds_is_not_clipped(self):
        base = HeuristicConfig(food_weight=10.0)
        search = settings(parameters={"food_weight": (0.0, 1.0)}, candidates=5)
        result = optimize_controller(base, search, 1, objective)
        self.assertEqual(result.trials[0].config.food_weight, 10.0)
        self.assert_bounds(result, search)

    def test_ties_preserve_the_baseline(self):
        result = optimize_controller(self.base, settings(), 99, lambda _: 4.0)
        self.assertIs(result.best_config, self.base)
        self.assertEqual(result.best_score, 4.0)

    def test_single_candidate_only_evaluates_baseline_without_importing_cma(self):
        for method in ("random", "cma"):
            calls = []
            with self.subTest(method=method), patch(
                "src.training.search._cma_trials", side_effect=AssertionError("unneeded CMA import"),
            ):
                result = optimize_controller(
                    self.base, settings(method, candidates=1), 0,
                    lambda value: calls.append(value) or 1.0,
                )
                self.assertEqual(calls, [self.base])
                self.assertEqual(len(result.trials), 1)
                self.assertIs(result.best_config, self.base)
                self.assertEqual(result.method, method)

    def test_random_search_preserves_global_rngs(self):
        python_state, numpy_state = random.getstate(), np.random.get_state()
        optimize_controller(self.base, settings(), 91, objective)
        self.assertEqual(random.getstate(), python_state)
        after = np.random.get_state()
        self.assertEqual(after[0], numpy_state[0])
        np.testing.assert_array_equal(after[1], numpy_state[1])
        self.assertEqual(after[2:], numpy_state[2:])

    def test_nonfinite_and_non_numeric_scores_fail_with_candidate_context(self):
        for invalid in (float("nan"), float("inf"), -float("inf"), None, "1", True, 1j):
            for index in (0, 1):
                with self.subTest(invalid=invalid, index=index):
                    values = iter([1.0] * index + [invalid])
                    with self.assertRaisesRegex(ValueError, f"candidate {index}.*finite real score"):
                        optimize_controller(self.base, settings(), 1, lambda _: next(values))

    def test_numeric_numpy_scores_are_accepted(self):
        result = optimize_controller(self.base, settings(candidates=1), 1, lambda _: np.float32(1.5))
        self.assertEqual(result.best_score, 1.5)
        self.assertIsInstance(result.best_score, float)

    def test_evaluator_error_propagates_unchanged_with_full_candidate_context(self):
        failure = RuntimeError("training world failed")
        calls = []

        def evaluate(config):
            calls.append(config)
            if len(calls) == 2:
                raise failure
            return 1.0

        with self.assertRaises(RuntimeError) as caught:
            optimize_controller(self.base, settings(), 3, evaluate)
        self.assertIs(caught.exception, failure)
        self.assertEqual(len(calls), 2)
        self.assertIn("candidate 1", failure.__notes__[0])
        self.assertIn("breeding_reserve", failure.__notes__[0])

    def test_invalid_seed_and_unvalidated_configuration_are_rejected(self):
        for invalid in (True, 1.5, "1", None):
            with self.subTest(seed=invalid), self.assertRaises(ValueError):
                optimize_controller(self.base, settings(), invalid, objective)
        with self.assertRaises(TypeError):
            optimize_controller({}, settings(), 1, objective)

    def test_missing_cma_has_an_install_hint_and_preserves_cause(self):
        original_import = builtins.__import__
        failure = ModuleNotFoundError("no cma", name="cma")
        calls = []

        def missing(name, *args, **kwargs):
            if name == "cma":
                raise failure
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=missing):
            with self.assertRaisesRegex(ModuleNotFoundError, "pip install cma==4.4.4") as caught:
                optimize_controller(self.base, settings("cma"), 1, lambda value: calls.append(value) or 1.0)
        self.assertIs(caught.exception.__cause__, failure)
        self.assertEqual(calls, [self.base])

    def test_missing_transitive_dependency_is_not_misreported_as_missing_cma(self):
        original_import = builtins.__import__
        failure = ModuleNotFoundError("broken dependency", name="some_dependency")

        def missing(name, *args, **kwargs):
            if name == "cma":
                raise failure
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=missing):
            with self.assertRaises(ModuleNotFoundError) as caught:
                optimize_controller(self.base, settings("cma"), 1, objective)
        self.assertIs(caught.exception, failure)


@unittest.skipUnless(importlib.util.find_spec("cma") is not None, "optional cma dependency is not installed")
class CMASearchTests(unittest.TestCase):
    def setUp(self):
        self.base = HeuristicConfig(food_weight=3.7, danger_weight=7.5)

    def test_cma_improves_a_cheap_quadratic_and_reproduces_with_seed_zero(self):
        search = settings("cma", candidates=20)
        first = optimize_controller(self.base, search, 0, objective)
        second = optimize_controller(self.base, search, 0, objective)
        self.assertEqual(first, second)
        self.assertEqual(first.method, "cma")
        self.assertEqual(len(first.trials), search.candidates)
        self.assertIs(first.trials[0].config, self.base)
        self.assertGreater(first.best_score, objective(self.base))
        for trial in first.trials[1:]:
            self.assertTrue(0 <= trial.config.food_weight <= 4)
            self.assertTrue(0 <= trial.config.danger_weight <= 8)

    def test_small_and_partial_budgets_are_exact(self):
        for budget in (2, 3, 4, 5, 8, 18):
            with self.subTest(budget=budget):
                result = optimize_controller(self.base, settings("cma", budget), 4, objective)
                self.assertEqual(len(result.trials), budget)
                self.assertEqual([trial.index for trial in result.trials], list(range(budget)))
                self.assertEqual(result.best_score, max(trial.score for trial in result.trials))

    def test_only_full_generations_are_told_and_population_fits_budget(self):
        import cma

        factory = cma.CMAEvolutionStrategy
        constructed = []

        def record(*args, **kwargs):
            strategy = factory(*args, **kwargs)
            constructed.append(strategy)
            return strategy

        for budget in (2, 3, 4, 18):
            with self.subTest(budget=budget), patch("cma.CMAEvolutionStrategy", side_effect=record):
                optimize_controller(self.base, settings("cma", budget), 19, objective)
                strategy = constructed[-1]
                self.assertLessEqual(strategy.popsize, budget)
                full_generations = (budget - 1) // strategy.popsize
                self.assertEqual(strategy.countiter, full_generations)
                self.assertEqual(strategy.countevals, full_generations * strategy.popsize)

    def test_best_partial_generation_candidate_is_not_lost(self):
        calls = []

        def increasing(config):
            calls.append(config)
            return float(len(calls))

        result = optimize_controller(self.base, settings("cma", 8), 3, increasing)
        self.assertEqual(result.best_score, 8.0)
        self.assertIs(result.best_config, result.trials[-1].config)

    def test_one_parameter_normalization_and_out_of_bounds_baseline(self):
        search = settings("cma", 10, parameters={"food_weight": (100.0, 200.0)})
        result = optimize_controller(
            self.base, search, 5, lambda value: -(value.food_weight - 140.0) ** 2,
        )
        self.assertIs(result.trials[0].config, self.base)
        for trial in result.trials[1:]:
            self.assertTrue(100.0 <= trial.config.food_weight <= 200.0)
            self.assertEqual(trial.config.danger_weight, self.base.danger_weight)

    def test_cma_parameter_order_does_not_change_sampling(self):
        search = settings("cma")
        permuted = settings("cma", parameters=dict(reversed(list(search.parameters.items()))))
        self.assertEqual(
            optimize_controller(self.base, search, 11, objective),
            optimize_controller(self.base, permuted, 11, objective),
        )

    def test_cma_is_quiet_has_no_files_and_preserves_global_rngs(self):
        import cma

        self.assertTrue(hasattr(cma, "CMAEvolutionStrategy"))
        python_state, numpy_state = random.getstate(), np.random.get_state()
        stdout, stderr = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.chdir(directory), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                optimize_controller(self.base, settings("cma", 18), 17, objective)
                self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(random.getstate(), python_state)
        after = np.random.get_state()
        self.assertEqual(after[0], numpy_state[0])
        np.testing.assert_array_equal(after[1], numpy_state[1])
        self.assertEqual(after[2:], numpy_state[2:])

    def test_cma_nonfinite_candidate_score_is_not_replaced(self):
        values = iter([1.0, float("nan")])
        with self.assertRaisesRegex(ValueError, "candidate 1.*finite real score"):
            optimize_controller(self.base, settings("cma"), 1, lambda _: next(values))


if __name__ == "__main__":
    unittest.main()
