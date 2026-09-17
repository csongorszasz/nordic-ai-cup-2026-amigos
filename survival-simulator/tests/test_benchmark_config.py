import unittest

from pydantic import ValidationError

from src.benchmarking.config import (
    EpisodeCase, PolicySpec, Suite, load_suite, make_cases, policy_seed,
)


class SuiteTests(unittest.TestCase):
    def test_committed_suites(self):
        quick = load_suite("quick")
        standard = load_suite("standard")
        holdout = load_suite("holdout")
        self.assertEqual(quick.seeds, standard.seeds[:3])
        self.assertEqual(len(standard.seeds), 20)
        self.assertEqual(len(holdout.seeds), 20)
        self.assertFalse(set(standard.seeds) & set(holdout.seeds))

    def test_seed_validation(self):
        for seeds in ([1, 1], [-1], [2**32], [True], ["1"], []):
            with self.subTest(seeds=seeds), self.assertRaises(ValidationError):
                Suite(name="invalid", seeds=seeds)
        with self.assertRaises(ValueError):
            load_suite("missing")

    def test_repeat_cases_are_stable_and_unique(self):
        cases = make_cases(load_suite("quick"), 3)
        self.assertEqual(cases, make_cases(load_suite("quick"), 3))
        self.assertEqual(len({case.case_id for case in cases}), 9)
        self.assertEqual(cases[0].policy_seed, cases[0].world_seed)
        self.assertNotEqual(cases[1].policy_seed, cases[0].policy_seed)
        self.assertEqual(policy_seed(42, 0), 42)
        for repeats in (0, -1, True, 1.5):
            with self.subTest(repeats=repeats), self.assertRaises(ValueError):
                make_cases(load_suite("quick"), repeats)

    def test_mismatched_case_identity_is_rejected(self):
        with self.assertRaises(ValidationError):
            EpisodeCase(case_id="world-1-repeat-0", world_seed=1, repeat_index=0, policy_seed=2)
        with self.assertRaises(ValidationError):
            EpisodeCase(case_id="wrong", world_seed=1, repeat_index=0, policy_seed=1)

    def test_policy_config_is_finite_json(self):
        with self.assertRaises(ValidationError):
            PolicySpec(reference="random", label="bad", config={"threshold": float("nan")})
        with self.assertRaises(ValidationError):
            PolicySpec(reference="random", label="bad", config={"value": object()})


if __name__ == "__main__":
    unittest.main()
