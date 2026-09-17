"""Local source provenance and durable, non-resumable benchmark artifacts."""

import ast
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
import platform
import statistics
import struct
import subprocess
import sys
import time
import tokenize
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.benchmarking.config import (
    BASELINE_NAME, SCHEMA_VERSION, EpisodeCase, EpisodeResult, FailureInfo, Fingerprint, GitInfo, PolicySpec,
    Provenance, RunManifest, RunStatus, RuntimeInfo, SimulationSettings, Suite,
    Timings, canonical_json, content_hash, make_cases, read_json,
)


_ENGINE_FILES = (
    Path("src") / "core.py",
    Path("src") / "utils" / "simulation.py",
    Path("src") / "utils" / "DTOs.py",
    Path("src") / "utils" / "sensing.py",
)
_RUNNER_FILES = tuple(
    Path("src") / "benchmarking" / name for name in ("config.py", "policies.py", "runner.py")
)
_REPORTING_FILES = (
    *(Path("src") / "benchmarking" / name for name in ("artifacts.py", "comparison.py", "plots.py")),
    Path("benchmark.py"),
)
_BASELINE_SOURCE = Path("src") / "utils" / "controllers" / "dummy_agent_policy.py"
_EXCLUDED_DIRECTORIES = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", "site-packages",
    "benchmark-results", "results", "build", "dist", ".tox", ".mypy_cache", ".pytest_cache",
}
_DEPENDENCIES = ("numpy", "scipy", "pygame", "Shapely", "pydantic", "pydantic_core")
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OMP_DYNAMIC", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "MKL_DYNAMIC", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)


@dataclass(frozen=True)
class SavedRun:
    path: Path
    manifest: RunManifest
    episodes: Sequence[EpisodeResult]


def _file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _file_hashes(root: Path, paths: Sequence[Path]) -> dict[str, str]:
    files = {}
    for path in sorted({path.resolve(strict=True) for path in paths}, key=str):
        if not path.is_file():
            raise ValueError(f"Provenance artifact is not a file: {path}")
        name = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        files[name] = _file_hash(path)
    return files


def _fingerprint(files: dict[str, str]) -> Fingerprint:
    return Fingerprint(sha256=content_hash(files), files=files)


def _walk_error(error: OSError) -> None:
    raise error


def _python_tree(directory: Path) -> set[Path]:
    sources = set()
    for current, directories, filenames in os.walk(
        directory, followlinks=False, onerror=_walk_error,
    ):
        directories[:] = sorted(
            name for name in directories
            if name.lower() not in _EXCLUDED_DIRECTORIES and not (Path(current) / name).is_symlink()
        )
        for name in filenames:
            path = Path(current) / name
            if path.suffix.lower() == ".py" and not path.is_symlink():
                sources.add(path.resolve())
    return sources


def _policy_sources(root: Path, policy: PolicySpec, declared: Sequence[Path]) -> list[Path]:
    sources = {Path(path).resolve(strict=True) for path in declared}
    baseline = policy.reference in ("random", BASELINE_NAME)
    if baseline:
        sources.add((root / _BASELINE_SOURCE).resolve(strict=True))
    if not sources:
        raise ValueError("Declare the local policy factory source with policy_sources.")
    if any(not path.is_file() for path in sources):
        raise ValueError("Every declared policy source must be a file.")
    if baseline:
        return sorted(sources, key=str)

    # Capture sibling helpers and the containing package, including untracked files.
    # Static imports extend this closure without importing or executing policy code.
    search_roots = {root}
    for source in tuple(sources):
        package = source.parent
        while package != root and (package.parent / "__init__.py").is_file():
            package = package.parent
        search_roots.update((source.parent, package.parent))
        if source.parent == root or (
            not source.parent.is_relative_to(root) and not (source.parent / "__init__.py").is_file()
        ):
            neighbors = {
                path.resolve() for path in source.parent.glob("*.py") if not path.is_symlink()
            }
        else:
            neighbors = _python_tree(source.parent)
        sources.update(neighbors)
        for parent in (source.parent, *source.parent.parents):
            initializer = parent / "__init__.py"
            if not initializer.is_file():
                break
            sources.add(initializer.resolve())
    pending = list(sources)
    scanned = set()
    while pending:
        source = pending.pop()
        if source in scanned or source.suffix.lower() != ".py":
            continue
        scanned.add(source)
        with tokenize.open(source) as stream:
            tree = ast.parse(stream.read(), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
                roots = search_roots
            elif isinstance(node, ast.ImportFrom):
                prefix = node.module or ""
                names = [prefix, *(f"{prefix}.{alias.name}".strip(".") for alias in node.names)]
                if node.level:
                    base = source.parent
                    for _ in range(node.level - 1):
                        base = base.parent
                    roots = {base}
                else:
                    roots = search_roots
            else:
                continue
            for base in roots:
                for name in names:
                    if not name or "*" in name:
                        continue
                    module = base.joinpath(*name.split("."))
                    matches = (
                        {module.with_suffix(".py").resolve()} if module.with_suffix(".py").is_file()
                        else _python_tree(module) if module.is_dir() else set()
                    )
                    matches = {
                        path for path in matches
                        if not any(part.lower() in _EXCLUDED_DIRECTORIES for part in path.parts)
                    }
                    for depth in range(1, len(name.split("."))):
                        initializer = base.joinpath(*name.split(".")[:depth]) / "__init__.py"
                        if initializer.is_file():
                            matches.add(initializer.resolve())
                    sources.update(matches)
                    pending.extend(matches - scanned)
    return sorted(sources, key=str)


def _git_info(root: Path) -> GitInfo:
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
        )

    try:
        revision = git("rev-parse", "HEAD")
    except FileNotFoundError:
        return GitInfo(revision=None, dirty=None, status="Git executable unavailable.")
    if revision.returncode:
        return GitInfo(revision=None, dirty=None, status=revision.stderr.strip())
    status = git("status", "--porcelain=v1", "--untracked-files=normal", "--", ".")
    return GitInfo(
        revision=revision.stdout.strip(),
        dirty=bool(status.stdout.strip()) if status.returncode == 0 else None,
        status=status.stdout.rstrip() if status.returncode == 0 else status.stderr.strip(),
    )


def _runtime_info() -> RuntimeInfo:
    versions = {}
    for name in _DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    native = {}
    for name in ("numpy", "scipy"):
        module = importlib.import_module(name)
        config = getattr(module.__config__, "CONFIG", {}).get("Build Dependencies", {})
        for library in ("blas", "lapack"):
            details = config.get(library, {})
            native[f"{name}.{library}"] = canonical_json({
                key: details[key]
                for key in ("name", "version", "openblas configuration")
                if key in details
            })
    pygame = importlib.import_module("pygame")
    native["SDL"] = ".".join(map(str, pygame.get_sdl_version(linked=True)))
    native["SDL_image"] = ".".join(map(str, pygame.image.get_sdl_image_version()))
    native["SDL_ttf"] = ".".join(map(str, pygame.font.get_sdl_ttf_version()))
    shapely = importlib.import_module("shapely")
    native["GEOS"] = shapely.geos_version_string
    native["GEOS_CAPI"] = shapely.geos_capi_version_string
    native["libc"] = " ".join(part for part in platform.libc_ver() if part) or "unavailable"
    return RuntimeInfo(
        python_version=platform.python_version(), implementation=platform.python_implementation(),
        os=platform.system(), os_release=f"{platform.release()} ({platform.version()})",
        architecture=f"{platform.machine()}-{8 * struct.calcsize('P')}bit",
        dependencies=versions, native_libraries=native,
    )


def _timing_context() -> dict:
    clock = time.get_clock_info("perf_counter")
    return {
        "hostname": platform.node(),
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "thread_environment": {name: os.environ.get(name) for name in _THREAD_VARIABLES},
        "timer": {
            "name": "perf_counter_ns", "implementation": clock.implementation,
            "resolution_seconds": clock.resolution, "monotonic": clock.monotonic,
            "adjustable": clock.adjustable,
        },
        "python_compiler": platform.python_compiler(),
        "python_executable": sys.executable,
    }


def build_manifest(
    project_root: Path,
    suite: Suite,
    policy: PolicySpec,
    *,
    repeats: int = 1,
    simulation: SimulationSettings | None = None,
    max_steps: int | None = None,
    policy_sources: Sequence[Path] = (),
    extra_artifacts: Sequence[Path] = (),
) -> RunManifest:
    """Hash local factory/package helpers and static local imports without executing them.

    Declare external dynamic helpers, configuration files and model weights explicitly
    in ``extra_artifacts``; every explicitly declared file must exist.
    """
    root = Path(project_root).resolve(strict=True)
    cases = make_cases(suite, repeats)
    engine_paths = [root / path for path in _ENGINE_FILES]
    elements = root / "src" / "elements"
    if not elements.is_dir():
        raise FileNotFoundError(f"Engine source directory is missing: {elements}")
    engine_paths.extend(elements.glob("*.py"))
    policy_files = _file_hashes(root, _policy_sources(root, policy, policy_sources))
    artifact_files = _file_hashes(root, [Path(path) for path in extra_artifacts])
    return RunManifest(
        run_id=uuid.uuid4().hex, created_at=datetime.now(timezone.utc).isoformat(),
        suite=suite, suite_sha256=content_hash(suite.model_dump()), repeats=repeats,
        simulation=simulation or SimulationSettings(), max_steps=max_steps, cases=cases,
        policy=policy,
        provenance=Provenance(
            git=_git_info(root),
            engine=_fingerprint(_file_hashes(root, engine_paths)),
            runner=_fingerprint(_file_hashes(root, [root / path for path in _RUNNER_FILES])),
            policy=Fingerprint(
                files=policy_files,
                sha256=content_hash({
                    "files": policy_files, "reference": policy.reference, "config": policy.config,
                }),
            ),
            reporting=_fingerprint(_file_hashes(root, [root / path for path in _REPORTING_FILES])),
            artifacts=_fingerprint(artifact_files),
            runtime=_runtime_info(), timing_context=_timing_context(),
        ),
    )


def _validate_measurements(manifest: RunManifest, episodes: Sequence[EpisodeResult]) -> None:
    settings = manifest.simulation
    measured = [row for row in episodes if row.status in ("ok", "truncated")]
    terminal_times = {}
    elapsed = 0.0
    ticks = 0
    # Replay only the scalar clock. Repeated float addition determines the
    # first strict-horizon tick, including the simulator's 0.1-second drift.
    for target in sorted({row.ticks for row in measured}):
        while ticks < target:
            if elapsed > settings.time_limit:
                raise ValueError("An episode continued beyond the first terminal tick.")
            elapsed += settings.dt
            ticks += 1
        terminal_times[target] = elapsed
    for row in measured:
        expected_time = terminal_times[row.ticks]
        time_tolerance = max(1e-12, 2 * math.ulp(expected_time))
        if not math.isclose(row.sim_time, expected_time, rel_tol=0, abs_tol=time_tolerance):
            raise ValueError(f"{row.case.case_id}: simulated time disagrees with ticks and dt.")
        if not math.isclose(
            row.survival_seconds, min(row.sim_time, settings.time_limit),
            rel_tol=0, abs_tol=time_tolerance,
        ):
            raise ValueError(f"{row.case.case_id}: survival time disagrees with the configured horizon.")
        if row.termination == "time_limit" and not (
            row.sim_time > settings.time_limit and expected_time > settings.time_limit
        ):
            raise ValueError(f"{row.case.case_id}: time-limit termination requires sim_time > time_limit.")
        if row.initial_agents != settings.starting_agents:
            raise ValueError(f"{row.case.case_id}: initial population disagrees with simulation settings.")
        if row.peak_agents < max(row.initial_agents, row.final_agents):
            raise ValueError(f"{row.case.case_id}: peak population is inconsistent.")
        if row.mean_population > row.peak_agents:
            raise ValueError(f"{row.case.case_id}: mean population exceeds the peak.")
        timing = row.timings
        if timing.policy_calls != row.ticks - 1:
            raise ValueError(f"{row.case.case_id}: batch-call count disagrees with the initial empty tick.")
        latencies = (timing.batch_mean_ms, timing.batch_p50_ms, timing.batch_p95_ms, timing.batch_max_ms)
        if timing.policy_calls == 0:
            if any(value is not None for value in latencies) or timing.policy_seconds != 0:
                raise ValueError(f"{row.case.case_id}: latency was recorded without a policy call.")
        elif any(value is None for value in latencies):
            raise ValueError(f"{row.case.case_id}: measured policy calls require batch-latency statistics.")
        elif (
            not timing.batch_p50_ms <= timing.batch_p95_ms <= timing.batch_max_ms
            or (
                timing.batch_mean_ms > timing.batch_max_ms
                and not math.isclose(timing.batch_mean_ms, timing.batch_max_ms, rel_tol=1e-9, abs_tol=1e-12)
            )
            or not math.isclose(
                timing.batch_mean_ms, 1000 * timing.policy_seconds / timing.policy_calls,
                rel_tol=1e-9, abs_tol=1e-9,
            )
        ):
            raise ValueError(f"{row.case.case_id}: batch-latency statistics disagree with recorded calls.")


def validate_run(manifest: RunManifest, episodes: Sequence[EpisodeResult]) -> None:
    """Validate saved contracts and progress; running and legitimate partial runs are allowed."""
    RunManifest.model_validate(manifest.model_dump())
    expected = {case.case_id: case for case in manifest.cases}
    seen = set()
    for result in episodes:
        EpisodeResult.model_validate(result.model_dump())
        case_id = result.case.case_id
        if case_id in seen:
            raise ValueError(f"Duplicate episode case: {case_id}")
        if expected.get(case_id) != result.case:
            raise ValueError(f"Unknown or mismatched episode case: {case_id}")
        if manifest.max_steps is not None and result.ticks > manifest.max_steps:
            raise ValueError(f"{case_id}: episode ticks exceed the recorded diagnostic step cap.")
        if result.status == "truncated" and (
            result.ticks != manifest.max_steps or result.final_agents == 0
            or result.sim_time > manifest.simulation.time_limit
        ):
            raise ValueError(f"{case_id}: truncation disagrees with the step cap or terminal state.")
        seen.add(case_id)
    statuses = {result.status for result in episodes}
    if "truncated" in statuses and manifest.max_steps is None:
        raise ValueError("Truncated results require a recorded diagnostic step cap.")
    if manifest.status == "complete":
        if seen != set(expected) or statuses != {"ok"} or manifest.failure is not None:
            raise ValueError("Complete runs require every expected case to succeed without failure.")
    elif manifest.status == "running":
        if manifest.failure is not None:
            raise ValueError("A running manifest cannot contain a final run failure.")
    elif manifest.status == "failed":
        if manifest.failure is None and "failed" not in statuses:
            raise ValueError("Failed runs require explicit failure details.")
        if "interrupted" in statuses:
            raise ValueError("Failed manifest disagrees with interrupted episode status.")
    elif manifest.status == "interrupted":
        if "failed" in statuses:
            raise ValueError("Interrupted manifest disagrees with failed episode status.")
    elif manifest.status == "truncated":
        if (
            manifest.max_steps is None or "truncated" not in statuses
            or statuses - {"ok", "truncated"} or manifest.failure is not None
        ):
            raise ValueError("Truncated manifests require diagnostic results without execution failures.")
    _validate_measurements(manifest, episodes)


def describe(values: Sequence[float | int | None]) -> dict:
    samples = [value for value in values if value is not None]
    return {
        "count": len(samples),
        "mean": statistics.fmean(samples) if samples else None,
        "median": statistics.median(samples) if samples else None,
        "sample_std": statistics.stdev(samples) if len(samples) > 1 else None,
        "min": min(samples) if samples else None,
        "max": max(samples) if samples else None,
    }


def summarize_run(manifest: RunManifest, episodes: Sequence[EpisodeResult]) -> dict:
    validate_run(manifest, episodes)
    counts = Counter(result.status for result in episodes)
    missing = sorted({case.case_id for case in manifest.cases} - {row.case.case_id for row in episodes})
    measured = [row for row in episodes if row.status in ("ok", "truncated")]
    complete = manifest.status == "complete" and not missing and counts["ok"] == len(manifest.cases)
    rankable = complete and manifest.max_steps is None
    warnings = []
    if not rankable:
        warnings.append("Diagnostic results only; observed subsets are not a complete policy ranking.")
    if manifest.max_steps is not None:
        warnings.append("A diagnostic step cap was configured, even if an episode ended before the cap.")
    if not measured:
        warnings.append("No scored episodes are available; missing and failed scores are not zero.")
    warnings.append(
        "Latency summaries describe per-episode batch-call statistics, not pooled latency percentiles."
    )
    timing_names = (
        "initialization_seconds", "policy_construction_seconds", "simulation_seconds",
        "policy_seconds", "episode_seconds",
    )
    runtime = {
        name: {
            **describe([getattr(row.timings, name) for row in episodes]),
            "total": math.fsum(getattr(row.timings, name) for row in episodes),
        }
        for name in timing_names
    }
    calls = sum(row.timings.policy_calls for row in episodes)
    return {
        "schema_version": manifest.schema_version, "run_id": manifest.run_id,
        "label": manifest.policy.label, "suite": manifest.suite.name,
        "status": manifest.status, "is_complete": complete, "rankable": rankable,
        "aggregate_scope": "full_run" if rankable else "diagnostic_subset",
        "measurement_scope": "ok_and_truncated_episodes", "timing_scope": "all_recorded_attempts",
        "expected_cases": len(manifest.cases), "recorded_cases": len(episodes),
        "successful_cases": counts["ok"], "failed_cases": counts["failed"],
        "interrupted_cases": counts["interrupted"], "truncated_cases": counts["truncated"],
        "missing_cases": len(missing), "missing_case_ids": missing,
        "world_count": len(manifest.suite.seeds), "repeats": manifest.repeats,
        "score": describe([row.score for row in measured]),
        "survival_seconds": describe([row.survival_seconds for row in measured]),
        "sim_time": describe([row.sim_time for row in measured]),
        "ticks": describe([row.ticks for row in measured]),
        "completion": {
            "count": len(measured), "completed": sum(row.completed for row in measured),
            "fraction": sum(row.completed for row in measured) / len(measured) if measured else None,
        },
        "population": {
            name: describe([getattr(row, field) for row in measured])
            for name, field in (
                ("initial", "initial_agents"), ("final", "final_agents"),
                ("peak", "peak_agents"), ("mean", "mean_population"),
            )
        },
        "runtime": runtime,
        "policy_latency": {
            "summary_unit": "episode", "batch_call_unit": "all_current_agents",
            **{
                name: describe([getattr(row.timings, name) for row in episodes])
                for name in ("batch_mean_ms", "batch_p50_ms", "batch_p95_ms", "batch_max_ms")
            },
            "policy_calls": describe([row.timings.policy_calls for row in episodes]),
            "total_policy_calls": calls,
            "pooled_batch_mean_ms": 1000 * runtime["policy_seconds"]["total"] / calls if calls else None,
            "pooled_percentiles_available": False,
        },
        "failure": manifest.failure.model_dump(mode="json") if manifest.failure else None,
        "warnings": warnings,
    }


def _atomic_text(path: Path, text: str) -> None:
    pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        with pending.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        if pending.exists():
            pending.unlink()


def _json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _csv_text(episodes: Sequence[EpisodeResult]) -> str:
    episode_fields = [
        name for name in EpisodeResult.model_fields if name not in ("case", "timings", "failure")
    ]
    fields = [
        *EpisodeCase.model_fields, *episode_fields, *Timings.model_fields,
        *(f"failure_{name}" for name in FailureInfo.model_fields),
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for episode in episodes:
        row = episode.model_dump(mode="json")
        case = row.pop("case")
        timings = row.pop("timings")
        failure = row.pop("failure")
        writer.writerow({
            **case, **row, **timings,
            **{f"failure_{key}": value for key, value in (failure or {}).items()},
        })
    return stream.getvalue()


def _markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def measurement_rows(summary: dict) -> list[tuple[str, float | int | None, int]]:
    rows = [
        ("Survival (simulated seconds)", summary["survival_seconds"]["mean"], summary["survival_seconds"]["count"]),
        ("Time-limit completion fraction", summary["completion"]["fraction"], summary["completion"]["count"]),
    ]
    rows.extend(
        (f"Population: {name}", values["mean"], values["count"])
        for name, values in summary["population"].items()
    )
    rows.extend(
        (f"Runtime: {name}", values["mean"], values["count"])
        for name, values in summary["runtime"].items()
    )
    rows.extend(
        (f"Per-episode latency: {name}", summary["policy_latency"][name]["mean"],
         summary["policy_latency"][name]["count"])
        for name in ("batch_mean_ms", "batch_p50_ms", "batch_p95_ms", "batch_max_ms")
    )
    return rows


def _run_markdown(manifest: RunManifest, summary: dict) -> str:
    title = "Benchmark run" if summary["rankable"] else "Benchmark diagnostics - not a ranking"
    lines = [
        f"# {title}", "", f"Policy: {_markdown(manifest.policy.label)}",
        f"Run ID: `{manifest.run_id}`",
        f"Suite: {_markdown(manifest.suite.name)}; status: {manifest.status}",
        f"Cases: {summary['recorded_cases']}/{summary['expected_cases']} recorded; "
        f"{summary['successful_cases']} successful; {summary['missing_cases']} missing.",
        "", "| Native score statistic | Value |", "| --- | --- |",
        *(f"| {name} | {value if value is not None else 'unavailable'} |"
          for name, value in summary["score"].items()),
        "", "## Survival, population and runtime",
        "", "| Measurement | Mean across recorded episodes | n |", "| --- | --- | --- |",
        *(f"| {name} | {value if value is not None else 'unavailable'} | {count} |"
          for name, value, count in measurement_rows(summary)),
        "", "## Limitations", *(f"- {warning}" for warning in summary["warnings"]),
        "", "## Provenance",
        f"- Engine SHA-256: `{manifest.provenance.engine.sha256}`",
        f"- Runner SHA-256: `{manifest.provenance.runner.sha256}`",
        f"- Policy SHA-256: `{manifest.provenance.policy.sha256}`",
        f"- Git revision: `{manifest.provenance.git.revision}`; dirty: {manifest.provenance.git.dirty}",
        "- Full source hashes, policy configuration, runtime and timing context are in `manifest.json`.",
        "- All measurements and sample counts are in `summary.json` and `episodes.csv`.", "",
    ]
    if manifest.failure:
        lines.extend([
            "## Failure", f"{_markdown(manifest.failure.stage)}: "
            f"{_markdown(manifest.failure.error_type)}: {_markdown(manifest.failure.message)}", "",
        ])
    return "\n".join(lines)


class RunWriter:
    def __init__(self, output: Path, manifest: RunManifest):
        validate_run(manifest, ())
        if manifest.status != "running":
            raise ValueError("New run writers require a running manifest.")
        self.output = Path(output)
        self.manifest = RunManifest.model_validate(manifest.model_dump())
        self._results: list[EpisodeResult] = []
        self._finalized = False
        self.output.mkdir(parents=True, exist_ok=False)
        with (self.output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        _atomic_text(self.output / "manifest.json", _json_text(self.manifest.model_dump(mode="json")))

    @property
    def results(self) -> tuple[EpisodeResult, ...]:
        return tuple(self._results)

    def append(self, result: EpisodeResult) -> None:
        if self._finalized:
            raise ValueError("Cannot append to a finalized run.")
        result = EpisodeResult.model_validate(result.model_dump())
        validate_run(self.manifest, [*self.results, result])
        line = canonical_json(result.model_dump(mode="json")) + "\n"
        with (self.output / "episodes.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        self._results.append(result)

    def finalize(self, status: RunStatus, failure: FailureInfo | None = None) -> dict:
        if self._finalized:
            raise ValueError("Run has already been finalized.")
        if status == "running":
            raise ValueError("Finalization requires a terminal run status.")
        persisted = load_run(self.output)
        if persisted.manifest != self.manifest or tuple(persisted.episodes) != self.results:
            raise ValueError("Run files changed after writing; refusing to finalize inconsistent artifacts.")
        if failure is None and status in ("failed", "interrupted"):
            failure = next((row.failure for row in reversed(self.results) if row.failure), None)
        manifest = RunManifest.model_validate({
            **self.manifest.model_dump(), "status": status,
            "failure": failure.model_dump() if failure else None,
        })
        summary = summarize_run(manifest, self.results)
        _atomic_text(self.output / "episodes.csv", _csv_text(self.results))
        _atomic_text(self.output / "summary.json", _json_text(summary))
        _atomic_text(self.output / "report.md", _run_markdown(manifest, summary))
        _atomic_text(self.output / "manifest.json", _json_text(manifest.model_dump(mode="json")))
        self.manifest = manifest
        self._finalized = True
        return summary


def _require_stored_fields(value: object, model, name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    missing = set(model.model_fields) - value.keys()
    if missing:
        raise ValueError(f"{name} is missing stored fields: {', '.join(sorted(missing))}.")


def load_run(path: Path) -> SavedRun:
    path = Path(path)
    data = read_json(path / "manifest.json")
    _require_stored_fields(data, RunManifest, "Manifest")
    if type(data["schema_version"]) is not int or data["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema version: {data['schema_version']!r}")
    for field, model in (("suite", Suite), ("simulation", SimulationSettings), ("policy", PolicySpec)):
        _require_stored_fields(data[field], model, f"Manifest {field}")
    manifest = RunManifest.model_validate(data)
    episodes = []
    with (path / "episodes.jsonl").open(encoding="utf-8-sig", newline="") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"Truncated episodes.jsonl line {number} in {path}.")
            try:
                data = json.loads(line)
                _require_stored_fields(data, EpisodeResult, "Episode")
                _require_stored_fields(data["timings"], Timings, "Episode timings")
                episodes.append(EpisodeResult.model_validate(data))
            except ValueError as exc:
                raise ValueError(f"Invalid episodes.jsonl line {number} in {path}: {exc}") from exc
    validate_run(manifest, episodes)
    return SavedRun(path=path.resolve(), manifest=manifest, episodes=tuple(episodes))
