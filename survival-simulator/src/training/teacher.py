"""Resolve the named observation-only teacher once, never from a worker's CWD."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from src.benchmarking.config import PROJECT_ROOT, content_hash, read_json
from src.policies.config import ExperimentConfig, HeuristicConfig, RuntimeConfig


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class ResolvedTeacher:
    heuristic: HeuristicConfig
    provenance: dict


def resolve_teacher(config: ExperimentConfig, root: Path = PROJECT_ROOT) -> ResolvedTeacher:
    if config.teacher is None:
        return ResolvedTeacher(config.heuristic, {
            "kind": "legacy_inline", "descriptor": config.heuristic.model_dump(mode="json"),
        })
    path = Path(config.teacher.path.replace("\\", "/"))
    path = path if path.is_absolute() else root / path
    path = path.resolve(strict=True)
    digest = sha256(path)
    if digest != config.teacher.sha256:
        raise ValueError("Teacher descriptor checksum mismatch; select a new experiment explicitly.")
    descriptor = RuntimeConfig.model_validate(read_json(path))
    if descriptor.policy != "heuristic":
        raise ValueError("This pipeline requires an observation-only heuristic teacher.")
    # Hash shared policy helpers as well: changing geometry can change teacher labels.
    files = {
        source.relative_to(root).as_posix(): sha256(source)
        for source in sorted((root / "src" / "policies").glob("*.py"))
    }
    return ResolvedTeacher(descriptor.heuristic, {
        "kind": "descriptor", "path": config.teacher.path, "sha256": digest,
        "descriptor": descriptor.model_dump(mode="json"),
        "sources": files, "sources_sha256": content_hash(files),
    })
