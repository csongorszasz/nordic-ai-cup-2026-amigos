import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.benchmarking.config import PROJECT_ROOT, read_json
from src.policies.config import ExperimentConfig, ModelConfig, ResourceConfig, load_experiment
from src.policies.features import encode_step, PUBLIC_FEATURE_VERSION, PUBLIC_SCALAR_DIM
from src.policies.heuristic import build_policy
from src.policies.networks import PolicyNetwork
from src.policies.neural import NeuralPolicy
from src.policies.runtime import create_policy
from src.training.artifacts import save_checkpoint, load_checkpoint, write_json
from src.training.preflight import build_run_plan, verify_run_plan
from src.training.teacher import resolve_teacher
from tests.test_policy_features import agent, frame
from tests.test_training_rollout import ScriptedEnvironment, make_config
from src.training.rollout import RolloutCollector
import train


def pipeline_config(mode="ppo", updates=2, **changes):
    values = load_experiment(PROJECT_ROOT / "configs" / ("imitation-gru.json" if mode == "imitation" else "ppo-gru.json")).model_dump(mode="json")
    values.update(mode=mode, updates=updates, rollout_steps=2)
    values["model"].update(hidden_size=16, entity_size=8)
    values["resources"].update(device="cpu", workers=1, max_tokens=256)
    values["ppo"].update(sequence_length=2, epochs=1, target_kl=1.0)
    values["imitation"].update(epochs=1, max_frames=8)
    values["evaluation"].update(every_updates=1, max_pending=4, max_snapshot_mb=16)
    values["budget"].update(max_updates=10, max_native_ticks=100, max_evaluation_episodes=36)
    values.update(changes)
    return ExperimentConfig.model_validate(values)


class PipelineContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.guard = patch("src.core.SimulationCore.__init__", side_effect=AssertionError("Real simulation forbidden"))
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def test_teacher_is_exact_descriptor_and_keeps_independent_public_history(self):
        config = pipeline_config("imitation")
        teacher = resolve_teacher(config)
        self.assertEqual(teacher.heuristic.backend, "hierarchical")
        self.assertEqual(teacher.heuristic.escape_strategy, "direct_wall_aware")
        selected = create_policy(7, teacher.provenance["descriptor"])
        from_collection = build_policy(7, teacher.heuristic)
        for tick in (1, 2, 3):
            step = frame([agent(0), agent(1)])
            step.sim_time = tick
            step.score = tick + 0.5
            self.assertEqual(from_collection.act(step), selected.act(step))
        from_collection.reset()
        self.assertEqual(from_collection.act(frame([agent(0)])), build_policy(7, teacher.heuristic).act(frame([agent(0)])))
        altered = config.model_dump(mode="json")
        altered["teacher"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checksum"):
            resolve_teacher(ExperimentConfig.model_validate(altered))
        altered = config.model_dump(mode="json")
        altered["heuristic"]["backend"] = "turnaway"
        with self.assertRaisesRegex(ValueError, "conflicting"):
            ExperimentConfig.model_validate(altered)

    def test_public_features_cover_score_and_symmetry_but_preserve_permutation(self):
        step = frame([agent(0), agent(1)])
        old = encode_step(step)
        np.testing.assert_array_equal(old.scalars[0], old.scalars[1])
        teacher = build_policy(1, resolve_teacher(pipeline_config()).heuristic)
        decisions = teacher.act(step)
        self.assertNotEqual(decisions[0].model_dump(exclude={"agent_id"}),
                            decisions[1].model_dump(exclude={"agent_id"}))
        public = encode_step(step, public_context=True)
        self.assertEqual(public.scalars.shape[1], PUBLIC_SCALAR_DIM)
        self.assertFalse(np.array_equal(public.scalars[0], public.scalars[1]))
        permuted = step.model_copy(update={"agent_status": list(reversed(step.agent_status))})
        np.testing.assert_array_equal(public.scalars, encode_step(permuted, public_context=True).scalars[::-1])
        changed = step.model_copy(update={"score": 10.0})
        self.assertFalse(np.array_equal(public.scalars, encode_step(changed, public_context=True).scalars))

    def test_new_and_legacy_feature_checkpoint_roundtrip(self):
        for public in (False, True):
            config = pipeline_config().model_copy(update={"model": ModelConfig(hidden_size=16, entity_size=8, public_context=public)})
            network = PolicyNetwork(config.model)
            step = frame([agent(3)])
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "checkpoint.pt"
                save_checkpoint(checkpoint, network, config)
                loaded = load_checkpoint(checkpoint)
                self.assertEqual(NeuralPolicy(network, 1).act(step), NeuralPolicy(loaded.network, 1).act(step))
                if public:
                    self.assertEqual(read_json(checkpoint.with_suffix(".pt.json"))["feature_version"], PUBLIC_FEATURE_VERSION)

    def test_no_teacher_queries_in_scratch_rollout(self):
        config = make_config()
        from tests.test_training_rollout import TinyPolicy, ConstantTeacher

        class ForbiddenTeacher(ConstantTeacher):
            def act(self, step):
                raise AssertionError("Teacher label queried in scratch PPO")

        with RolloutCollector(config, TinyPolicy(config.model), emit=lambda _: None,
                              environment_factory=ScriptedEnvironment, teacher_factory=ForbiddenTeacher) as collector:
            result = collector.collect(2, policy_version=0, label_teacher=False)
            self.assertTrue(all(not value.teacher_actions for value in result.trajectories[0]))
            self.assertEqual(collector.native_tick_count, 3)

    def test_large_gpu_budgets_are_not_rejected_by_laptop_limit(self):
        self.assertEqual(ResourceConfig(max_vram_mb=60000).max_vram_mb, 60000)

    def test_preflight_is_pure_and_exact_effective_config_is_required(self):
        config = pipeline_config()
        evidence = read_json(PROJECT_ROOT / "configs" / "evidence-diagnostic.json")
        with patch("subprocess.Popen", side_effect=AssertionError("No jobs")), \
                patch("torch.optim.Adam", side_effect=AssertionError("No optimizer")):
            plan = build_run_plan(config, evidence)
        self.assertEqual(plan["accounting"]["decision_ticks"], 4)
        self.assertEqual(plan["accounting"]["scheduled_updates"], [0, 1, 2])
        self.assertEqual(plan["accounting"]["evaluation_episodes_including_teacher"], 12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            write_json(path, plan)
            self.assertEqual(verify_run_plan(path, config), plan)
            changed = config.model_copy(update={"seed": config.seed + 1})
            with self.assertRaises(ValueError):
                verify_run_plan(path, changed)
            plan["accounting"]["decision_ticks"] = 100
            write_json(path, plan)
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_run_plan(path, config)

    def test_extended_budgets_require_specific_measured_evidence(self):
        config = pipeline_config()
        evidence = read_json(PROJECT_ROOT / "configs" / "evidence-diagnostic.json")
        values = config.model_dump(mode="json")
        values["budget"]["kind"] = "extended"
        with self.assertRaisesRegex(ValueError, "measured"):
            build_run_plan(ExperimentConfig.model_validate(values), evidence)
        values["budget"]["kind"] = "diagnostic"
        values["budget"]["max_evaluation_episodes"] = 1
        with self.assertRaisesRegex(ValueError, "episode budget"):
            build_run_plan(ExperimentConfig.model_validate(values), evidence)
        values["resources"]["action_repeat"] = 5
        with self.assertRaisesRegex(ValueError, "action_repeat"):
            ExperimentConfig.model_validate(values)

    def test_extended_plan_checks_completed_evaluated_pilot_not_just_a_file(self):
        import copy
        config = pipeline_config()
        entries = read_json(PROJECT_ROOT / "configs" / "evidence-diagnostic.json")
        provisional = build_run_plan(config, entries)
        values = config.model_dump(mode="json")
        values["budget"]["kind"] = "extended"
        extended = ExperimentConfig.model_validate(values)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pilot = root / "manifest.json"
            write_json(pilot, {"status": "running"})
            evidence = copy.deepcopy(entries)
            for group, entry in evidence.items():
                entry.update(basis="measured", values=provisional["evidence"][group]["values"],
                             artifacts=[str(pilot)])
            with self.assertRaisesRegex(ValueError, "completed, evaluated"):
                build_run_plan(extended, evidence)
            write_json(pilot, {
                "status": "complete", "config": config.model_dump(mode="json"),
                "teacher": resolve_teacher(config).provenance,
                "result": {"native_ticks_total": 5, "optimizer_steps_this_run": 2,
                           "next_update": 2, "device": "cpu"},
            })
            (root / "progress").mkdir()
            write_json(root / "progress" / "status.json", {
                "latest_evaluated_update": 2, "teacher_state": "complete", "teacher_mean": 20,
            })
            plan = build_run_plan(extended, evidence)
            self.assertEqual(plan["config"]["budget"]["kind"], "extended")

    def test_learning_without_run_plan_cannot_start_a_simulator(self):
        with tempfile.TemporaryDirectory() as directory, patch("train.TrainingRun") as run:
            with self.assertRaisesRegex(ValueError, "run-plan"):
                train.execute(pipeline_config(), Path(directory) / "run")
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
