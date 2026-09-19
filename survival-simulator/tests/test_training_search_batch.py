import unittest

from src.policies.config import HeuristicConfig, SearchConfig
from src.training.search import optimize_controller


class BatchedSearchTests(unittest.TestCase):
    @staticmethod
    def objective(config):
        return -((config.breeding_age - 52) ** 2 + (config.danger_weight - 7) ** 2)

    def test_batching_preserves_proposals_scores_order_and_winner(self):
        for method in ("random", "cma"):
            for count in (1, 2, 4, 10):
                settings = SearchConfig(method=method, candidates=count)
                base = HeuristicConfig()
                reference = optimize_controller(base, settings, 42, self.objective)
                batches = []

                def evaluate_many(configs):
                    batches.append(tuple(configs))
                    return [self.objective(config) for config in configs]

                batched = optimize_controller(
                    base, settings, 42, self.objective, evaluate_many=evaluate_many,
                )
                with self.subTest(method=method, count=count):
                    self.assertEqual(reference, batched)
                    self.assertEqual(batched.trials[0].config, base)
                    self.assertEqual([trial.index for trial in batched.trials], list(range(count)))
                    if count > 1:
                        self.assertEqual(batches[0][0], base)

    def test_batch_result_shape_and_values_are_validated(self):
        settings = SearchConfig(candidates=3)
        for scores in ([1], [1, 2, float("nan")], [True, 2, 3]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                optimize_controller(
                    HeuristicConfig(), settings, 1, self.objective,
                    evaluate_many=lambda _: scores,
                )

    def test_batch_failure_is_propagated_with_candidate_context(self):
        error = RuntimeError("worker failed")

        def fail(_):
            raise error

        with self.assertRaises(RuntimeError) as caught:
            optimize_controller(
                HeuristicConfig(), SearchConfig(candidates=3), 1, self.objective,
                evaluate_many=fail,
            )
        self.assertIs(caught.exception, error)
        self.assertTrue(any("candidate batch 0-2" in note for note in error.__notes__))


if __name__ == "__main__":
    unittest.main()
