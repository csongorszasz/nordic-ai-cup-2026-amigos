"""Strict saved-run comparisons, paired by case and bootstrapped by world seed.

``compare_runs`` returns a JSON-compatible dictionary. ``runs`` contains role
keys, original manifests, summaries, and aligned ``case_ids``, ``scores`` and
``batch_mean_ms`` arrays. ``comparisons`` contains candidate-minus-reference
``case_deltas`` and repeat-averaged ``seed_deltas``, wins/ties/losses, and
``ci95`` (low/high/status). ``leaderboard`` is descriptive, ordered by mean
score. Role keys (reference/candidate-1/...) never identify the underlying run.
Top-level ``suite`` is the complete Suite dictionary; ``reference_label``,
``case_count``, ``world_count``, ``repeats``, ``warnings`` and
``timing_compatible`` supply plot captions. Each ``seed_deltas`` entry contains
``world_seed``, ``reference_mean``, ``candidate_mean``, ``mean_delta`` and
``repeats``. Use the role key as well as the label to distinguish legend entries.
"""

import csv
import io
import statistics
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import bootstrap

from src.benchmarking.artifacts import (
    SavedRun, _atomic_text, _json_text, _markdown, describe, measurement_rows,
    summarize_run, validate_run,
)
from src.benchmarking.config import RunManifest


TIE_TOLERANCE = 1e-9
BOOTSTRAP_SEED = 20260917
BOOTSTRAP_RESAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95


def _validate_terminal_results(run: SavedRun) -> None:
    manifest = run.manifest
    validate_run(manifest, run.episodes)
    if manifest.status != "complete":
        raise ValueError(f"Run {manifest.run_id} is {manifest.status}, not complete.")
    if manifest.max_steps is not None:
        raise ValueError(f"Run {manifest.run_id} used a diagnostic max_steps cap and cannot be ranked.")


def _compatibility(reference: RunManifest, candidate: RunManifest) -> None:
    checks = {
        "suite/version/hash": (
            (reference.suite.model_dump(), reference.suite_sha256),
            (candidate.suite.model_dump(), candidate.suite_sha256),
        ),
        "case definitions": (
            {row.case_id: row for row in reference.cases},
            {row.case_id: row for row in candidate.cases},
        ),
        "repeats": (reference.repeats, candidate.repeats),
        "simulation settings": (reference.simulation, candidate.simulation),
        "engine fingerprint": (reference.provenance.engine, candidate.provenance.engine),
        "runner fingerprint": (reference.provenance.runner, candidate.provenance.runner),
        "runner protocol": (reference.protocol_version, candidate.protocol_version),
        "seed derivation version": (reference.seed_version, candidate.seed_version),
        "execution mode": (reference.execution, candidate.execution),
        "runtime Python/OS/dependencies/native libraries": (
            reference.provenance.runtime, candidate.provenance.runtime,
        ),
    }
    differences = [name for name, (left, right) in checks.items() if left != right]
    if differences:
        raise ValueError(
            f"Incompatible run {candidate.run_id}: {', '.join(differences)}."
        )


def _interval(deltas: Sequence[float]) -> dict:
    if len(deltas) < 2:
        return {"low": None, "high": None, "status": "unavailable_single_world"}
    if all(delta == deltas[0] for delta in deltas):
        return {"low": deltas[0], "high": deltas[0], "status": "point"}
    result = bootstrap(
        (np.asarray(deltas, dtype=float),), np.mean, vectorized=True,
        n_resamples=BOOTSTRAP_RESAMPLES, batch=1000, confidence_level=CONFIDENCE_LEVEL,
        method="percentile", rng=np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)),
    )
    return {
        "low": float(result.confidence_interval.low),
        "high": float(result.confidence_interval.high), "status": "estimated",
    }


def compare_runs(reference: SavedRun, candidates: Sequence[SavedRun]) -> dict:
    if not candidates:
        raise ValueError("Provide at least one candidate; a reference may also be its own candidate.")
    saved_runs = [reference, *candidates]
    for run in saved_runs:
        _validate_terminal_results(run)
    for candidate in candidates:
        _compatibility(reference.manifest, candidate.manifest)
    cases = sorted(
        reference.manifest.cases, key=lambda case: (case.world_seed, case.repeat_index, case.case_id)
    )
    reference_rows = {row.case.case_id: row for row in reference.episodes}
    runs = []
    comparisons = []
    warnings = ["Rankings are descriptive; confidence intervals are not automatic significance claims."]
    if reference.manifest.suite.name == "quick":
        warnings.append("Quick-suite results are exploratory: only a small set of world seeds is sampled.")
    for index, run in enumerate(saved_runs):
        key = "reference" if index == 0 else f"candidate-{index}"
        rows = {row.case.case_id: row for row in run.episodes}
        ordered = [rows[case.case_id] for case in cases]
        runs.append({
            "key": key, "run_id": run.manifest.run_id, "label": run.manifest.policy.label,
            "path": str(run.path), "manifest": run.manifest.model_dump(mode="json"),
            "summary": summarize_run(run.manifest, ordered),
            "case_ids": [case.case_id for case in cases],
            "scores": [row.score for row in ordered],
            "batch_mean_ms": [row.timings.batch_mean_ms for row in ordered],
        })
        if index == 0:
            continue
        pairs = [
            {
                **case.model_dump(), "reference_score": reference_rows[case.case_id].score,
                "candidate_score": rows[case.case_id].score,
                "delta": rows[case.case_id].score - reference_rows[case.case_id].score,
            }
            for case in cases
        ]
        seeds = []
        for seed in sorted(reference.manifest.suite.seeds):
            seed_pairs = [pair for pair in pairs if pair["world_seed"] == seed]
            seeds.append({
                "world_seed": seed, "repeats": len(seed_pairs),
                "reference_mean": statistics.fmean(pair["reference_score"] for pair in seed_pairs),
                "candidate_mean": statistics.fmean(pair["candidate_score"] for pair in seed_pairs),
                "mean_delta": statistics.fmean(pair["delta"] for pair in seed_pairs),
            })
        deltas = [pair["delta"] for pair in pairs]
        seed_deltas = [seed["mean_delta"] for seed in seeds]
        ci = _interval(seed_deltas)
        timing = run.manifest.provenance.timing_context == reference.manifest.provenance.timing_context
        pair_warnings = []
        if not timing:
            message = (
                f"{key} ({run.manifest.policy.label}) has a different machine/timing context; "
                "scores are comparable, but runtime is not a controlled speed comparison."
            )
            pair_warnings.append(message)
            warnings.append(message)
        if ci["status"] == "unavailable_single_world":
            pair_warnings.append("One independent world seed: a 95% interval is unavailable, even with repeats.")
        elif ci["low"] <= 0 <= ci["high"]:
            pair_warnings.append("The 95% interval includes zero; it does not establish a clear improvement.")
        comparisons.append({
            "key": key, "reference_key": "reference", "label": run.manifest.policy.label,
            "case_count": len(pairs), "world_count": len(seeds),
            "mean_delta": statistics.fmean(seed_deltas),
            "case_delta_summary": describe(deltas), "seed_delta_summary": describe(seed_deltas),
            "wins": sum(delta > TIE_TOLERANCE for delta in deltas),
            "ties": sum(abs(delta) <= TIE_TOLERANCE for delta in deltas),
            "losses": sum(delta < -TIE_TOLERANCE for delta in deltas),
            "win_tie_loss_unit": "paired_case",
            "ci95": ci, "case_deltas": pairs, "seed_deltas": seeds,
            "timing_compatible": timing, "warnings": pair_warnings,
        })
    ordered_runs = sorted(
        runs, key=lambda run: (-run["summary"]["score"]["mean"], run["label"], run["run_id"], run["key"])
    )
    leaderboard = []
    rank = 0
    previous_score = None
    for position, run in enumerate(ordered_runs, start=1):
        score = run["summary"]["score"]["mean"]
        if score != previous_score:
            rank = position
        previous_score = score
        leaderboard.append({
            "key": run["key"], "label": run["label"], "run_id": run["run_id"],
            "rank": rank, "summary": run["summary"],
        })
    report = {
        "schema_version": 1, "kind": "paired_comparison",
        "suite": reference.manifest.suite.model_dump(mode="json"),
        "suite_sha256": reference.manifest.suite_sha256,
        "reference_key": "reference", "reference_label": reference.manifest.policy.label,
        "case_count": len(cases), "world_count": len(reference.manifest.suite.seeds),
        "repeats": reference.manifest.repeats, "tie_tolerance": TIE_TOLERANCE,
        "bootstrap": {
            "confidence_level": CONFIDENCE_LEVEL, "n_resamples": BOOTSTRAP_RESAMPLES,
            "method": "percentile", "rng": "PCG64", "seed": BOOTSTRAP_SEED,
            "sampling_unit": "world_seed", "repeat_aggregation": "mean_paired_delta_within_world",
            "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        },
        "timing_compatible": all(pair["timing_compatible"] for pair in comparisons),
        "warnings": warnings, "runs": runs, "comparisons": comparisons, "leaderboard": leaderboard,
    }
    return report


def _paired_csv(report: dict) -> str:
    stream = io.StringIO(newline="")
    fields = [
        "candidate_key", "candidate_label", "reference_key", "reference_label",
        "case_id", "world_seed", "repeat_index", "policy_seed", "policy_seed_mode",
        "reference_score", "candidate_score", "delta",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for pair in report["comparisons"]:
        for row in pair["case_deltas"]:
            writer.writerow({
                "candidate_key": pair["key"], "candidate_label": pair["label"],
                "reference_key": report["reference_key"], "reference_label": report["reference_label"],
                **row,
            })
    return stream.getvalue()


def _report_markdown(report: dict) -> str:
    lines = [
        "# Paired benchmark comparison", "",
        f"Suite: {_markdown(report['suite']['name'])}; {report['world_count']} world seeds, "
        f"{report['case_count']} cases per run; repeats: {report['repeats']}.",
        f"Reference: {_markdown(report['reference_label'])} (`reference`).", "",
        "## Descriptive leaderboard - native final score", "",
        "| Rank | Role | Policy | Mean | Median | Sample std | Min | Max | n |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in report["leaderboard"]:
        score = run["summary"]["score"]
        values = [
            run["rank"], run["key"], _markdown(run["label"]),
            *(score[name] for name in ("mean", "median", "sample_std", "min", "max", "count")),
        ]
        lines.append("| " + " | ".join("unavailable" if value is None else str(value) for value in values) + " |")
    lines.extend([
        "", "## Absolute paired deltas - candidate minus reference", "",
        "| Candidate | Mean delta | 95% interval | Wins | Ties | Losses | Independent worlds |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for pair in report["comparisons"]:
        ci = pair["ci95"]
        interval = "unavailable (one world)" if ci["low"] is None else f"[{ci['low']}, {ci['high']}] ({ci['status']})"
        lines.append(
            f"| {_markdown(pair['label'])} ({pair['key']}) | {pair['mean_delta']} | {interval} | "
            f"{pair['wins']} | {pair['ties']} | {pair['losses']} | {pair['world_count']} |"
        )
    bootstrap_info = report["bootstrap"]
    lines.extend([
        "", "Repeat-level paired differences are averaged within each world seed before bootstrapping.",
        f"Intervals: {bootstrap_info['method']} method, {bootstrap_info['n_resamples']} resamples, "
        f"fixed {bootstrap_info['rng']} seed {bootstrap_info['seed']}.",
        f"Wins/ties/losses count paired cases with absolute tie tolerance {report['tie_tolerance']}.",
        "", "## Survival, population and runtime", "",
        "| Role | Policy | Measurement | Mean across recorded episodes | n |",
        "| --- | --- | --- | --- | --- |",
    ])
    for run in report["runs"]:
        for name, value, count in measurement_rows(run["summary"]):
            rendered = "unavailable" if value is None else str(value)
            lines.append(f"| {run['key']} | {_markdown(run['label'])} | {name} | {rendered} | {count} |")
    lines.extend(["", "## Limitations", *(f"- {_markdown(warning)}" for warning in report["warnings"])])
    for pair in report["comparisons"]:
        lines.extend(f"- {pair['key']}: {_markdown(warning)}" for warning in pair["warnings"])
    lines.extend([
        "", "## Provenance and measurements",
        "Original run manifests, source fingerprints, policy configurations, runtime/timing context, "
        "per-run metric summaries and aligned plotting data are embedded in `summary.json`.",
        "The paired case export is `paired_cases.csv`. Failed or truncated runs are not admitted.",
        "Latency summaries are per-episode batch measurements, not pooled latency percentiles.", "",
    ])
    return "\n".join(lines)


def write_comparison_report(output: Path, report: dict) -> None:
    """Persist the comparison payload without rerunning policies or reading source files."""
    json_text = _json_text(report)
    csv_text = _paired_csv(report)
    markdown = _report_markdown(report)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _atomic_text(output / "summary.json", json_text)
    _atomic_text(output / "paired_cases.csv", csv_text)
    _atomic_text(output / "report.md", markdown)
