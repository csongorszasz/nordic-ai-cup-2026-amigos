"""Durable training records and version-checked, weights-only checkpoints."""

import hashlib
import gzip
import importlib.metadata
import json
import os
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from src.benchmarking.artifacts import _atomic_text, build_manifest
from src.benchmarking.config import (
    PROJECT_ROOT, PolicySpec, Suite, canonical_json, read_json,
)
from src.policies.config import ExperimentConfig, ModelConfig
from src.policies.features import FEATURE_VERSIONS, feature_version
from src.training.teacher import resolve_teacher

if TYPE_CHECKING:
    import torch
    from src.policies.networks import PolicyNetwork


CHECKPOINT_VERSION = 1


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_text(path, payload)


def write_json_gzip(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with gzip.open(temporary, "xt", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


class TrainingRun:
    def __init__(self, path: Path, config: ExperimentConfig):
        self.path = path.resolve()
        self.config = config
        sources = list((PROJECT_ROOT / "src" / "training").glob("*.py"))
        sources.extend((PROJECT_ROOT / "benchmarks" / "suites").glob("*.json"))
        sources.extend([PROJECT_ROOT / "requirements-training.txt"])
        if (PROJECT_ROOT / "train.py").is_file():
            sources.append(PROJECT_ROOT / "train.py")
        if config.checkpoint:
            sources.append(Path(config.checkpoint).resolve(strict=True))
        teacher = resolve_teacher(config)
        if config.teacher:
            sources.append(PROJECT_ROOT / config.teacher.path.replace("\\", "/"))
        provenance = build_manifest(
            PROJECT_ROOT, Suite(name="training-provenance", seeds=[config.seed]),
            PolicySpec(reference="src.policies.runtime:create_policy", label=f"{config.mode}-training",
                       config=config.model_dump(mode="json")),
            policy_sources=[PROJECT_ROOT / "src" / "policies" / "runtime.py"],
            extra_artifacts=sources,
        ).provenance
        dependencies = {}
        for name in ("torch", "cma"):
            try:
                dependencies[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                dependencies[name] = "not-installed"
        self.manifest = {
            "schema_version": 1, "run_id": uuid.uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(), "status": "running",
            "mode": config.mode, "config": config.model_dump(mode="json"),
            "provenance": provenance.model_dump(mode="json"), "training_dependencies": dependencies,
            "evaluation_claim": "training/diagnostic only; use full benchmark comparisons for ranking",
            "resume_semantics": "model/optimizer/RNG resume; reference environments restart",
            "teacher": teacher.provenance,
        }
        self.path.mkdir(parents=True, exist_ok=False)
        write_json(self.path / "manifest.json", self.manifest)
        write_json(self.path / "config.json", config.model_dump(mode="json"))

    def emit(self, event: dict) -> None:
        payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(payload) + "\n")
            stream.flush()

    def finish(self, status: str, result: dict) -> None:
        if status not in ("complete", "failed", "interrupted", "paused"):
            raise ValueError(f"Invalid training run status: {status}")
        if self.manifest["status"] != "running":
            raise ValueError("A finalized training run cannot be overwritten.")
        manifest = {**self.manifest, "status": status, "result": result}
        write_json(self.path / "summary.json", result)
        write_json(self.path / "manifest.json", manifest)
        self.manifest = manifest


@dataclass(frozen=True)
class LoadedCheckpoint:
    network: "PolicyNetwork"
    config: ExperimentConfig
    optimizer_state: dict | None
    training_state: dict
    rng_state: dict
    sha256: str


def save_checkpoint(
    path: Path, network: "PolicyNetwork", config: ExperimentConfig,
    optimizer: "torch.optim.Optimizer | None" = None, training_state: dict | None = None,
    *, inference_only: bool = False,
    rng_state: dict | None = None,
) -> str:
    import torch
    from src.policies.actions import model_action_version

    if network.config != config.model:
        raise ValueError("Cannot save weights with a different model configuration.")
    payload = {
        "version": CHECKPOINT_VERSION,
        "feature_version": feature_version(config.model.public_context, config.model.peer_context),
        "action_version": model_action_version(config.model), "config": config.model_dump(mode="json"),
        "model": {key: tensor.detach().cpu() for key, tensor in network.state_dict().items()},
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "training_state": training_state or {},
        "teacher": resolve_teacher(config).provenance,
        "rng": {} if inference_only else rng_state if rng_state is not None else {
            "python": random.getstate(), "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        },
    }
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = file_hash(path)
    write_json(path.with_suffix(path.suffix + ".json"), {
        "sha256": digest, "version": CHECKPOINT_VERSION,
        "feature_version": payload["feature_version"], "action_version": payload["action_version"],
        "model": config.model.model_dump(mode="json"),
        "teacher_sha256": config.teacher.sha256 if config.teacher else None,
        "action_repeat": config.resources.action_repeat,
        "native_ticks_total": payload["training_state"].get("native_ticks_total"),
        "prior_native_ticks": payload["training_state"].get("prior_native_ticks", 0),
    })
    return digest


def load_checkpoint(
    path: Path, *, expected_model: ModelConfig | None = None, device: str = "cpu",
    expected_sha256: str | None = None,
) -> LoadedCheckpoint:
    import torch
    from src.policies.actions import ACTION_VERSION, VECTOR_ACTION_VERSION, model_action_version
    from src.policies.networks import PolicyNetwork

    digest = file_hash(path)
    sidecar = read_json(path.with_suffix(path.suffix + ".json"))
    if digest != sidecar.get("sha256") or (expected_sha256 is not None and digest != expected_sha256):
        raise ValueError("Checkpoint checksum mismatch.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or (
        payload.get("version") != CHECKPOINT_VERSION
        or payload.get("feature_version") not in FEATURE_VERSIONS
        or payload.get("action_version") not in (ACTION_VERSION, VECTOR_ACTION_VERSION)
    ):
        raise ValueError("Incompatible checkpoint format, feature schema, or action codec.")
    config = ExperimentConfig.model_validate(payload["config"])
    if payload["action_version"] != model_action_version(config.model):
        raise ValueError("Checkpoint action codec does not match its model.")
    expected_features = feature_version(config.model.public_context, config.model.peer_context)
    if payload["feature_version"] != expected_features:
        raise ValueError("Checkpoint feature schema does not match its model.")
    if expected_model is not None and config.model != expected_model:
        raise ValueError("Checkpoint architecture does not match the requested model configuration.")
    with torch.random.fork_rng(devices=[]):
        network = PolicyNetwork(config.model)
    network.load_state_dict(payload["model"], strict=True)
    if not all(torch.isfinite(parameter).all().item() for parameter in network.parameters()):
        raise ValueError("Checkpoint contains non-finite model parameters.")
    network.to(device).eval()
    return LoadedCheckpoint(
        network, config, payload["optimizer"], payload["training_state"], payload["rng"], digest,
    )


def restore_rng(state: dict) -> None:
    import torch

    random.setstate(state["python"])
    torch.random.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("Cannot restore a CUDA training RNG on an unavailable CUDA device.")
        torch.cuda.set_rng_state_all(state["cuda"])


def verify_resume_provenance(checkpoint: Path, current_manifest: dict) -> None:
    manifest_path = checkpoint.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Resume requires the original manifest.json beside the checkpoint; use a warm start otherwise.")
    previous = read_json(manifest_path)
    old = previous.get("provenance", {})
    current = current_manifest["provenance"]
    if previous.get("teacher") != current_manifest.get("teacher"):
        raise ValueError("Resume teacher provenance changed; use an explicit new experiment.")
    old_config = previous.get("config", {})
    new_config = current_manifest.get("config", {})
    for name in ("evaluation", "mode", "ppo", "imitation", "rollout_steps"):
        if old_config.get(name) != new_config.get(name):
            raise ValueError(f"Resume cannot change the {name} protocol.")
    if (old_config.get("model") != new_config.get("model")
            or old_config.get("seed") != new_config.get("seed")):
        raise ValueError("Resume cannot change the model or seed.")
    if old_config.get("resources", {}).get("action_repeat", 1) != new_config.get("resources", {}).get("action_repeat", 1):
        raise ValueError("Resume cannot change action_repeat.")
    if old.get("engine") != current["engine"] or old.get("runtime") != current["runtime"]:
        raise ValueError("Resume requires compatible simulator/runtime provenance; use a warm start for a changed environment.")
    if previous.get("training_dependencies", {}).get("torch") != current_manifest["training_dependencies"]["torch"]:
        raise ValueError("Resume requires the recorded PyTorch version.")
    old_files = {name.replace("\\", "/"): value for name, value in old.get("artifacts", {}).get("files", {}).items()}
    new_files = {name.replace("\\", "/"): value for name, value in current["artifacts"]["files"].items()}
    for name in (
        "src/training/seeds.py", "benchmarks/suites/quick.json",
        "benchmarks/suites/standard.json", "benchmarks/suites/holdout.json",
    ):
        if name not in old_files or name not in new_files or old_files[name] != new_files[name]:
            raise ValueError(f"Resume seed-stream provenance changed or is missing: {name}. Use a warm start.")
