import unittest

from pydantic import ValidationError

from src.policies.config import ExperimentConfig, SearchConfig, apply_overrides


class PolicyConfigTests(unittest.TestCase):
    def test_nested_architecture_and_parameter_overrides(self):
        original = ExperimentConfig()
        changed = apply_overrides(original, [
            "model.memory=none", "model.encoder=attention", "model.team_context=false",
            "optimizer.learning_rate=0.001", "resources.workers=2",
        ])
        self.assertEqual(changed.model.memory, "none")
        self.assertEqual(changed.model.encoder, "attention")
        self.assertFalse(changed.model.team_context)
        self.assertEqual(changed.optimizer.learning_rate, 0.001)
        self.assertEqual(changed.resources.workers, 2)
        self.assertEqual(original.model.memory, "gru")

    def test_unknown_nonfinite_and_incompatible_values_fail(self):
        for changes in (
            ["model.memmory=gru"], ["unknown=1"], ["seed.child=3"], ["model.memory"],
            ["resources.workers=0"], ["resources.workers=true"],
            ["optimizer.learning_rate=NaN"], ["ppo.gamma=0"], ["mode=search"],
            ["rollout_steps=2"], ["model.encoder=unimplemented"],
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                apply_overrides(ExperimentConfig(), changes)
        with self.assertRaises(ValidationError):
            ExperimentConfig.model_validate({"extra": 1})

    def test_search_bounds_must_match_a_valid_continuous_parameter(self):
        for parameters in (
            {}, {"missing": (0, 1)}, {"directions": (8, 10)}, {"breeding_age": (10, 5)},
            {"breeding_energy": (0, 100)}, {"danger_weight": (0, float("inf"))},
        ):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                SearchConfig(parameters=parameters)

    def test_round_trip_preserves_resolved_config(self):
        config = apply_overrides(ExperimentConfig(), ["model.memory=none"])
        self.assertEqual(ExperimentConfig.model_validate_json(config.model_dump_json()), config)


if __name__ == "__main__":
    unittest.main()
