"""Pure run planning: no simulator, optimizer, device initialization, or job launch."""

from pathlib import Path

from pydantic import Field

from src.benchmarking.config import PROJECT_ROOT, content_hash, load_suite, read_json
from src.policies.config import ExperimentConfig, ModelConfig, Settings
from src.training.teacher import resolve_teacher, sha256


REQUIRED_TEACHER = "configs/controller-turnaway-wall-aware.json"
GROUPS = ("schedule", "model", "optimizer", "ppo", "imitation", "resources", "evaluation")


def _qualified_pilot(path: Path, config: ExperimentConfig) -> bool:
    if path.name != "manifest.json":
        return False
    data = read_json(path)
    if not isinstance(data, dict):
        return False
    pilot = data.get("config", {})
    result = data.get("result", {})
    if not isinstance(pilot, dict) or not isinstance(result, dict):
        return False
    progress = path.parent / "progress" / "status.json"
    if (
        data.get("status") != "complete" or pilot.get("mode") != config.mode
        or pilot.get("model") != config.model.model_dump(mode="json")
        or data.get("teacher", {}).get("sha256") != config.teacher.sha256
        or type(result.get("native_ticks_total")) is not int or result["native_ticks_total"] <= 0
        or type(result.get("optimizer_steps_this_run")) is not int or result["optimizer_steps_this_run"] <= 0
        or (config.resources.device != "auto" and result.get("device") != config.resources.device)
        or not progress.is_file()
    ):
        return False
    observed = read_json(progress)
    return (
        observed.get("latest_evaluated_update") == result.get("next_update")
        and observed.get("teacher_state") == "complete"
        and observed.get("teacher_mean") is not None
    )


class Evidence(Settings):
    basis: str = Field(pattern=r"^(invariant|reference|provisional|measured)$")
    rationale: str = Field(min_length=12)
    units: str = Field(min_length=1)
    alternatives: str = Field(min_length=1)
    uncertainty: str = Field(min_length=1)
    values: dict | None = None
    artifacts: list[str] = Field(default_factory=list)


def source_fingerprint(root: Path = PROJECT_ROOT) -> dict:
    files = set()
    for directory in ("src", "idun"):
        for path in (root / directory).rglob("*"):
            if path.is_file() and path.suffix in (".py", ".sh", ".slurm") and "__pycache__" not in path.parts:
                files.add(path)
    files.update((root / "benchmarks" / "suites").glob("*.json"))
    files.update(root.glob("requirements*.txt"))
    files.update(path for path in (root / "train.py", root / "benchmark.py") if path.is_file())
    hashes = {path.relative_to(root).as_posix(): sha256(path) for path in sorted(files)}
    return {"sha256": content_hash(hashes), "files": hashes}


def scheduled_updates(config: ExperimentConfig) -> list[int]:
    if not config.evaluation.enabled:
        return []
    return sorted({0, config.updates, *range(config.evaluation.every_updates, config.updates + 1,
                                           config.evaluation.every_updates)})


def build_run_plan(
    config: ExperimentConfig, evidence_data: dict, *, root: Path = PROJECT_ROOT,
    resume: Path | None = None,
) -> dict:
    if config.mode not in ("imitation", "ppo"):
        raise ValueError("Learning preflight requires imitation or PPO mode.")
    if resume is not None and config.checkpoint is not None:
        raise ValueError("Choose resume or a warm-start checkpoint, not both.")
    if config.teacher is None:
        raise ValueError("Pin the required teacher descriptor before planning learning.")
    if config.teacher.path.replace("\\", "/") != REQUIRED_TEACHER:
        raise ValueError(f"This pipeline requires teacher.path={REQUIRED_TEACHER}.")
    teacher = resolve_teacher(config, root)
    if config.resources.action_repeat != 1:
        raise ValueError("Use action_repeat=1 until a matching deployment cadence is implemented.")
    if config.budget is None or not config.evaluation.enabled:
        raise ValueError("Learning requires explicit run budgets and periodic evaluation.")
    if set(evidence_data) != set(GROUPS):
        raise ValueError(f"Evidence must cover exactly these groups: {', '.join(GROUPS)}.")
    resolved = config.model_dump(mode="json")
    resolved["schedule"] = {"updates": config.updates, "rollout_steps": config.rollout_steps,
                            "seed": config.seed, "mode": config.mode, "checkpoint": config.checkpoint}
    evidence = {}
    for group in GROUPS:
        entry = Evidence.model_validate(evidence_data[group])
        if entry.values is not None and entry.values != resolved[group]:
            raise ValueError(f"Evidence values do not match the effective {group} settings.")
        if config.budget.kind == "extended":
            if entry.values is None or entry.basis != "measured" or not entry.artifacts:
                raise ValueError(f"Extended runs need exact values and measured pilot artifacts for {group}.")
        artifacts = {}
        qualified = False
        for name in entry.artifacts:
            path = Path(name.replace("\\", "/"))
            path = path if path.is_absolute() else root / path
            artifacts[name] = sha256(path.resolve(strict=True))
            if config.budget.kind == "extended":
                is_pilot = _qualified_pilot(path, config)
                qualified |= is_pilot
                if is_pilot:
                    artifacts[name + "#progress"] = sha256(path.parent / "progress" / "status.json")
        if config.budget.kind == "extended" and not qualified:
            raise ValueError(
                f"{group} needs a completed, evaluated pilot manifest with matching mode/model/teacher/device."
            )
        evidence[group] = {**entry.model_dump(mode="json"), "values": resolved[group],
                           "artifact_hashes": artifacts}
    checkpoints = scheduled_updates(config)
    suite = load_suite(config.evaluation.suite)
    cases = len(suite.seeds) * config.evaluation.repeats
    # Each collector tick can terminate/reset each worker; reserve every possible bootstrap.
    decisions = config.updates * config.resources.workers * config.rollout_steps
    native_upper = decisions * (config.resources.action_repeat + 1) + config.resources.workers
    episodes = (len(checkpoints) + 1) * cases * config.evaluation.max_attempts
    limits = config.budget
    if config.updates > limits.max_updates or native_upper > limits.max_native_ticks:
        raise ValueError("Planned updates/native ticks exceed the declared run budget.")
    if episodes > limits.max_evaluation_episodes:
        raise ValueError("Scheduled evaluations plus teacher baseline exceed the episode budget.")
    if len(checkpoints) * config.evaluation.max_snapshot_mb > config.evaluation.max_storage_mb:
        raise ValueError("Snapshot storage reservation exceeds evaluation.max_storage_mb.")
    discount_trace = config.ppo.gamma * config.ppo.gae_lambda
    lineage = None
    source = resume or (Path(config.checkpoint.replace("\\", "/")) if config.checkpoint else None)
    if source is not None:
        source = source if source.is_absolute() else root / source
        header = read_json(source.with_suffix(source.suffix + ".json"))
        if (header.get("model") is None or ModelConfig.model_validate(header["model"]) != config.model
                or header.get("action_repeat") != config.resources.action_repeat):
            raise ValueError("Checkpoint architecture/cadence is incompatible or lacks review metadata.")
        if (header.get("teacher_sha256") != config.teacher.sha256
                or type(header.get("native_ticks_total")) is not int):
            raise ValueError("Checkpoint needs matching teacher and actual native-tick provenance.")
        if sha256(source) != header.get("sha256"):
            raise ValueError("Source checkpoint checksum mismatch.")
        lineage = {"checkpoint_sha256": sha256(source),
                   "sidecar_sha256": sha256(source.with_suffix(source.suffix + ".json")),
                   "kind": "resume" if resume else "warm_start"}
    plan = {
        "version": 1, "launch_policy": "manual_only", "config": config.model_dump(mode="json"),
        "teacher": teacher.provenance, "source": source_fingerprint(root),
        "evidence": evidence, "lineage": lineage,
        "evaluation_worlds": suite.model_dump(mode="json"),
        "accounting": {
            "decision_ticks": decisions, "native_ticks_upper_including_bootstraps": native_upper,
            "scheduled_updates": checkpoints, "evaluation_episodes_including_teacher": episodes,
            "gae_trace_sim_seconds": 0.1 * config.resources.action_repeat / (1 - discount_trace)
            if discount_trace < 1 else None,
            "truncated_bptt_sim_seconds": 0.1 * config.resources.action_repeat * config.ppo.sequence_length,
            "action_interval_sim_seconds": 0.1 * config.resources.action_repeat,
        },
        "qualification": "diagnostic only; not a claim of optimal parameters" if limits.kind == "diagnostic"
        else "evidence linked; the user must review measurement relevance before launching",
    }
    return {**plan, "plan_id": content_hash(plan)}


def verify_run_plan(path: Path, config: ExperimentConfig, *, resume: Path | None = None) -> dict:
    plan = read_json(path)
    if not isinstance(plan, dict):
        raise ValueError("Run plan must be a JSON object.")
    payload = {key: value for key, value in plan.items() if key != "plan_id"}
    if plan.get("plan_id") != content_hash(payload):
        raise ValueError("Run-plan checksum mismatch.")
    evidence = {
        group: {key: value for key, value in item.items() if key != "artifact_hashes"}
        for group, item in plan["evidence"].items()
    }
    current = build_run_plan(config, evidence, resume=resume)
    if current != plan:
        raise ValueError("Effective config, sources, teacher, evidence, or lineage changed; regenerate and review preflight.")
    return plan
