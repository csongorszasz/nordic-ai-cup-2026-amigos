from pathlib import Path

from src.benchmarking.artifacts import SavedRun
from src.benchmarking.config import (
    EpisodeResult, Fingerprint, GitInfo, PolicySpec, Provenance, RunManifest, RuntimeInfo,
    SimulationSettings, Suite, Timings, content_hash, make_cases,
)


def fingerprint(name="fixture"):
    files = {f"{name}.py": content_hash(name)}
    return Fingerprint(sha256=content_hash(files), files=files)


def manifest(*, seeds=(1, 2, 3), repeats=1, label="policy", suite_name="fixture", **changes):
    suite = Suite(name=suite_name, seeds=list(seeds))
    values = dict(
        run_id=label, created_at="2026-09-17T12:00:00+00:00",
        suite=suite, suite_sha256=content_hash(suite.model_dump()),
        repeats=repeats, cases=make_cases(suite, repeats),
        policy=PolicySpec(reference="fixture:factory", label=label),
        provenance=Provenance(
            git=GitInfo(revision="fixture", dirty=False, status=""),
            engine=fingerprint("engine"), runner=fingerprint("runner"),
            policy=fingerprint("policy"), reporting=fingerprint("reporting"),
            artifacts=fingerprint("artifacts"),
            runtime=RuntimeInfo(
                python_version="3.12.0", implementation="CPython",
                os="Windows", os_release="11", architecture="AMD64-64bit",
                dependencies={"numpy": "fixture", "scipy": "fixture"},
                native_libraries={"GEOS": "fixture"},
            ),
            timing_context={"hostname": "fixture-host", "cpu": "fixture-cpu", "threads": 1},
        ),
    )
    values.update(changes)
    return RunManifest(**values)


def episode(case, score=1.0, *, settings=None, ticks=2, termination="extinction", batch_ms=2.0):
    settings = settings or SimulationSettings()
    sim_time = 0.0
    for _ in range(ticks):
        sim_time += settings.dt
    calls = ticks - 1
    policy_seconds = calls * batch_ms / 1000 if batch_ms is not None else 0.0
    population = settings.starting_agents if termination != "extinction" else 0
    return EpisodeResult(
        case=case, status="truncated" if termination == "step_limit" else "ok",
        termination=termination, score=score, sim_time=sim_time,
        survival_seconds=min(sim_time, settings.time_limit), ticks=ticks,
        completed=termination == "time_limit", initial_agents=settings.starting_agents,
        final_agents=population, peak_agents=settings.starting_agents,
        mean_population=float(population),
        timings=Timings(
            initialization_seconds=0.003, policy_construction_seconds=0.001,
            simulation_seconds=0.004, policy_seconds=policy_seconds,
            episode_seconds=0.02 + policy_seconds, policy_calls=calls,
            batch_mean_ms=batch_ms if calls else None,
            batch_p50_ms=batch_ms if calls else None,
            batch_p95_ms=batch_ms if calls else None,
            batch_max_ms=batch_ms if calls else None,
        ),
    )


def saved(scores, *, seeds=None, repeats=1, label="policy", settings=None, **changes):
    seeds = tuple(range(1, len(scores) // repeats + 1)) if seeds is None else seeds
    record = manifest(
        seeds=seeds, repeats=repeats, label=label, simulation=settings or SimulationSettings(),
        status="complete", **changes,
    )
    if len(scores) != len(record.cases):
        raise ValueError("Fixture scores must cover all cases.")
    rows = tuple(
        episode(case, score, settings=record.simulation) for case, score in zip(record.cases, scores)
    )
    return SavedRun(path=Path("fixture-runs") / label, manifest=record, episodes=rows)
