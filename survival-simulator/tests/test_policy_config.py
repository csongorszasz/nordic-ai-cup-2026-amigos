import unittest

from pydantic import ValidationError

from src.benchmarking.config import PROJECT_ROOT, read_json
from src.policies.config import ExperimentConfig, RuntimeConfig, SearchConfig, apply_overrides


class PolicyConfigTests(unittest.TestCase):
    def test_default_serving_config_selects_hierarchical_controller_without_checkpoint(self):
        options = RuntimeConfig.model_validate(read_json(PROJECT_ROOT / "configs" / "controller.json"))
        self.assertEqual(options.policy, "heuristic")
        self.assertEqual(options.heuristic.backend, "hierarchical")
        self.assertIsNone(options.checkpoint)

    def test_escape_strategy_is_strict_and_predictive_by_default(self):
        self.assertEqual(ExperimentConfig().heuristic.escape_strategy, "predictive_wall_aware")
        for strategy in ("direct", "direct_wall_aware", "predictive_wall_aware"):
            config = ExperimentConfig.model_validate({"heuristic": {"escape_strategy": strategy}})
            self.assertEqual(config.heuristic.escape_strategy, strategy)
        with self.assertRaises(ValidationError):
            ExperimentConfig.model_validate({"heuristic": {"escape_strategy": "unknown"}})

    def test_turnaway_runtime_configs_only_change_escape_strategy(self):
        configs = {}
        for name in ("direct", "wall-aware", "predictive"):
            path = PROJECT_ROOT / "configs" / f"controller-turnaway-{name}.json"
            configs[name] = RuntimeConfig.model_validate(read_json(path))
            self.assertEqual(configs[name].heuristic.backend, "hierarchical")
        self.assertEqual(configs["direct"].heuristic.escape_strategy, "direct")
        self.assertEqual(configs["wall-aware"].heuristic.escape_strategy, "direct_wall_aware")
        self.assertEqual(configs["predictive"].heuristic.escape_strategy, "predictive_wall_aware")
        base = configs["predictive"].model_dump(mode="json")
        for name, config in configs.items():
            candidate = config.model_dump(mode="json")
            candidate["heuristic"]["escape_strategy"] = "predictive_wall_aware"
            self.assertEqual(candidate, base, name)

    def test_nested_architecture_and_parameter_overrides(self):
        original = ExperimentConfig()
        changed = apply_overrides(original, [
            "model.memory=none", "model.encoder=attention", "model.team_context=false",
            "optimizer.learning_rate=0.001", "resources.workers=2", "resources.action_repeat=5",
        ])
        self.assertEqual(changed.model.memory, "none")
        self.assertEqual(changed.model.encoder, "attention")
        self.assertFalse(changed.model.team_context)
        self.assertEqual(changed.optimizer.learning_rate, 0.001)
        self.assertEqual(changed.resources.workers, 2)
        self.assertEqual(changed.resources.action_repeat, 5)
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
        with self.assertRaises(ValidationError):
            RuntimeConfig.model_validate({
                "policy": "heuristic",
                "heuristic": {
                    "backend": "hierarchical", "min_population": 20,
                    "target_population": 10, "max_population": 15,
                },
            })

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
