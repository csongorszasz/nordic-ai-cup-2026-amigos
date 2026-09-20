"""Complete-case survival ordering, independent of native score magnitudes."""

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, median

from src.benchmarking.config import EpisodeResult, read_json


OBJECTIVE_VERSION = "survival-v1"


def extinction_upper_bound(failures: int, worlds: int, confidence: float = 0.95) -> float | None:
    if type(failures) is not int or type(worlds) is not int or not 0 <= failures <= worlds:
        raise ValueError("Extinction counts must satisfy 0 <= failures <= worlds.")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("Confidence must be strictly between zero and one.")
    if not worlds:
        return None
    if failures == worlds:
        return 1.0
    if not failures:
        return -math.expm1(math.log1p(-confidence) / worlds)
    from scipy.special import betaincinv

    return float(betaincinv(failures + 1, worlds - failures, confidence))


def survival_metrics(
    results: Sequence[EpisodeResult], *, expected_case_ids: Sequence[str],
    time_limit: float = 3000.0, tail_fraction: float = 0.05,
    terminal_health: Mapping[str, tuple[float, float]] | None = None,
) -> dict:
    expected = set(expected_case_ids)
    actual = [row.case.case_id for row in results]
    if (
        not expected or len(expected) != len(expected_case_ids)
        or len(actual) != len(set(actual)) or set(actual) != expected
    ):
        raise ValueError("Survival ranking requires every expected case exactly once.")
    if not math.isfinite(time_limit) or time_limit <= 0 or not 0 < tail_fraction <= 1:
        raise ValueError("Survival ranking requires a valid horizon and tail fraction.")
    groups: dict[int, list[EpisodeResult]] = defaultdict(list)
    for row in results:
        if row.status != "ok" or row.survival_seconds is None or row.score is None:
            raise ValueError("Failed, interrupted, and truncated episodes cannot enter a survival ranking.")
        if row.survival_seconds > time_limit or (
            row.completed and not math.isclose(row.survival_seconds, time_limit)
        ):
            raise ValueError("Episode horizon disagrees with survival ranking settings.")
        groups[row.case.world_seed].append(row)
    if terminal_health is not None:
        if set(terminal_health) != expected:
            raise ValueError("Terminal health must cover every expected case.")
        if any(
            len(values) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            for values in terminal_health.values()
        ):
            raise ValueError("Terminal health must contain finite renewal/reserve pairs.")
    lifetimes = sorted(min(row.survival_seconds for row in rows) for rows in groups.values())
    completed = sum(all(row.completed for row in rows) for rows in groups.values())
    tail_count = max(1, math.ceil(len(lifetimes) * tail_fraction))
    renewal = min(values[0] for values in terminal_health.values()) if terminal_health else 0.0
    reserve = min(values[1] for values in terminal_health.values()) if terminal_health else 0.0
    tail = fmean(lifetimes[:tail_count])
    score = fmean(fmean(row.score for row in rows) for rows in groups.values())
    return {
        "objective_version": OBJECTIVE_VERSION,
        "worlds": len(groups), "cases": len(results),
        "completed_worlds": completed, "extinct_worlds": len(groups) - completed,
        "completion_fraction": completed / len(groups),
        "lower_tail_seconds": tail, "worst_seconds": lifetimes[0],
        "lower_tail_fraction": tail_fraction,
        "median_seconds": float(median(lifetimes)),
        "renewal_min": renewal if terminal_health is not None else None,
        "reserve_min": reserve if terminal_health is not None else None,
        "mean_native_score": score,
        "rank": (completed, tail, lifetimes[0], renewal, reserve, score),
        "iid_extinction_upper95": extinction_upper_bound(len(groups) - completed, len(groups)),
        "uncertainty_assumption": (
            "The probability bound requires independently sampled audit worlds; "
            "it is not an assurance for tuned, fixed-development, or adversarial suites. "
            "Repeated cases are collapsed to the worst outcome per world."
        ),
    }


def terminal_health(path: Path, results: Sequence[EpisodeResult], *, required: bool) -> dict | None:
    files = {
        row.case.case_id: path / "telemetry" / row.case.case_id / "summary.json" for row in results
    }
    if not required and not any(file.exists() for file in files.values()):
        return None
    health = {}
    for row in results:
        summary = read_json(files[row.case.case_id])
        if summary["result"] != row.model_dump(mode="json") or summary["last_sample"] is None:
            raise ValueError("Terminal telemetry must agree with its complete episode result.")
        sample = summary["last_sample"]
        health[row.case.case_id] = (sample["young_reproductive"], sample["energy_p10"] or 0.0)
    return health


def survival_curve(results: Sequence[EpisodeResult]) -> tuple[list[float], list[float]]:
    """Kaplan-Meier curve; completed horizons are right-censored, not extinctions."""
    groups: dict[float, list[bool]] = defaultdict(list)
    for row in results:
        if row.status == "ok" and row.survival_seconds is not None:
            groups[row.survival_seconds].append(row.termination == "extinction")
    remaining = sum(len(values) for values in groups.values())
    xs, ys = [0.0], [1.0]
    for elapsed, events in sorted(groups.items()):
        probability = ys[-1] * (1.0 - sum(events) / remaining)
        xs.append(elapsed)
        ys.append(probability)
        remaining -= len(events)
    return xs, ys
