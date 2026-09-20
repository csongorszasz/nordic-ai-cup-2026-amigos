"""Run local policies and compare saved benchmark results."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from src.benchmarking.artifacts import RunWriter, build_manifest, load_run, summarize_run
from src.benchmarking.comparison import compare_runs, write_comparison_report
from src.benchmarking.config import (
    PROJECT_ROOT, SUITE_NAMES, EpisodeResult, FailureInfo, SimulationSettings, Suite, load_suite, read_json,
)
from src.benchmarking.jobs import JobExecutionError, run_jobs
from src.benchmarking.policies import load_policy
from src.benchmarking.runner import EpisodeExecutionError, EpisodeInterrupted, run_episode


def positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Must be a positive integer.") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return number


def uint32(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Must be an integer between 0 and 2^32 - 1.") from exc
    if not 0 <= number <= 2**32 - 1:
        raise argparse.ArgumentTypeError("Must be an integer between 0 and 2^32 - 1.")
    return number


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Must be a positive finite number.") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive finite number.")
    return number


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        description="Compare local policies without changing the competition simulator.",
    )
    commands = cli.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Evaluate one policy on a fixed suite.")
    run.add_argument("--policy", default="random", help="'random' or an importable module:factory.")
    run.add_argument("--label", help="Display name (recorded separately from source identity).")
    run.add_argument("--config", type=Path, help="JSON object passed to the policy factory.")
    suites = run.add_mutually_exclusive_group()
    suites.add_argument("--suite", choices=SUITE_NAMES, default="quick")
    suites.add_argument("--suite-file", type=Path, help="Versioned custom seed manifest.")
    run.add_argument("--time-limit", type=positive_float, default=3000.0,
                     help="Stopping horizon only; non-native horizons remain separately identified.")
    run.add_argument("--telemetry", action="store_true", help="Record evaluator-only population diagnostics.")
    run.add_argument("--progress", action="store_true", help="Enable telemetry and refresh local progress plots.")
    run.add_argument("--sample-every", type=positive_integer, default=10)
    run.add_argument("--trace-max-mb", type=positive_integer, default=64)
    run.add_argument("--workers", type=positive_integer,
                     help="Use isolated episode processes (1-32); omitted preserves the in-process runner.")
    run.add_argument("--episode-timeout", type=positive_float, default=1800.0,
                     help="Per-process episode watchdog, including startup; requires --workers.")
    run.add_argument("--repeats", type=positive_integer, default=1, help="Policy trials per world seed.")
    run.add_argument(
        "--fixed-policy-seed", type=uint32,
        help="Use one deployment-equivalent policy seed for every world and repeat.",
    )
    run.add_argument(
        "--max-steps", type=positive_integer,
        help="Diagnostic cap, including the initial empty-action tick. Not rankable.",
    )
    run.add_argument(
        "--artifact", action="append", type=Path, default=[],
        help="Additional source/config/model file to fingerprint; repeat for multiple files.",
    )
    run.add_argument("--output", type=Path, required=True, help="New result directory; never overwritten.")
    run.set_defaults(handler=run_command)

    compare = commands.add_parser("compare", help="Compare complete, compatible saved runs.")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, action="append", required=True)
    compare.add_argument("--output", type=Path, required=True, help="New comparison directory.")
    compare.add_argument("--no-plots", action="store_true", help="Generate numerical reports without Matplotlib.")
    compare.add_argument("--survival-first", action="store_true", help="Order by extinction avoidance and tail survival.")
    compare.set_defaults(handler=compare_command)

    inspect = commands.add_parser("inspect", help="Summarize saved results, including incomplete runs.")
    inspect.add_argument("--run", type=Path, required=True)
    inspect.set_defaults(handler=inspect_command)
    return cli


def _episode_job(job):
    case, reference, config, settings, max_steps, trace, sample_every, max_trace_mb = job
    loaded = load_policy(reference, config=config)
    telemetry = None
    if trace is not None:
        from src.benchmarking.telemetry import EpisodeTelemetry

        telemetry = EpisodeTelemetry(
            trace, case, time_limit=settings.time_limit, max_steps=max_steps,
            sample_every=sample_every, max_trace_mb=max_trace_mb,
        )
    return run_episode(case, loaded.factory, config, settings, max_steps, telemetry=telemetry)


def run_command(args: argparse.Namespace) -> int:
    config = read_json(args.config) if args.config is not None else {}
    if not isinstance(config, dict):
        raise ValueError("Policy configuration must be a JSON object.")
    if args.workers is not None and args.workers > 32:
        raise ValueError("--workers cannot exceed 32.")
    refresh = None
    if args.progress:
        from src.benchmarking.plots import require_plotting
        from src.benchmarking.progress import render_progress

        require_plotting()
        refresh = lambda: render_progress(args.output)
    suite = Suite.model_validate(read_json(args.suite_file)) if args.suite_file else load_suite(args.suite)
    loaded = load_policy(args.policy, label=args.label, config=config)
    if args.workers is not None and args.workers > 1 and loaded.spec.reference == "src.benchmarking.http:create_policy":
        raise ValueError("One stateful HTTP endpoint cannot serve parallel worlds; use --workers 1.")
    extra_artifacts = args.artifact + ([args.suite_file] if args.suite_file else [])
    if args.workers is not None:
        extra_artifacts += [PROJECT_ROOT / "src" / "training" / "workers.py"]
    manifest = build_manifest(
        PROJECT_ROOT, suite, loaded.spec,
        repeats=args.repeats, fixed_policy_seed=args.fixed_policy_seed,
        simulation=SimulationSettings(time_limit=args.time_limit), max_steps=args.max_steps,
        policy_sources=loaded.source_files,
        extra_artifacts=extra_artifacts,
    )
    if args.telemetry or args.progress:
        provenance = manifest.provenance.model_copy(update={"timing_context": {
            **manifest.provenance.timing_context,
            "population_telemetry": True, "progress_plots": args.progress,
            "telemetry_sample_every": args.sample_every,
        }})
        manifest = manifest.model_copy(update={"provenance": provenance})
    if args.workers is not None:
        provenance = manifest.provenance.model_copy(update={"timing_context": {
            **manifest.provenance.timing_context,
            "episode_workers": args.workers, "episode_watchdog_seconds": args.episode_timeout,
        }})
        manifest = manifest.model_copy(update={"execution": "processes", "provenance": provenance})
    writer = RunWriter(args.output, manifest)
    last_refresh = 0.0

    def refresh_if_due():
        nonlocal last_refresh
        if refresh is not None and time.monotonic() - last_refresh >= 10:
            refresh()
            last_refresh = time.monotonic()

    def sequential_results():
        for case in manifest.cases:
            telemetry = None
            if args.telemetry or args.progress:
                from src.benchmarking.telemetry import EpisodeTelemetry

                telemetry = EpisodeTelemetry(
                    args.output, case, time_limit=args.time_limit, max_steps=args.max_steps,
                    sample_every=args.sample_every, max_trace_mb=args.trace_max_mb, refresh=refresh,
                )
            yield run_episode(
                case, loaded.factory, loaded.spec.config, manifest.simulation, manifest.max_steps,
                **({"telemetry": telemetry} if telemetry is not None else {}),
            )

    try:
        if args.workers is None:
            results = sequential_results()
        else:
            jobs = [
                (case, loaded.spec.reference, loaded.spec.config, manifest.simulation, manifest.max_steps,
                 args.output.resolve() if args.telemetry or args.progress else None,
                 args.sample_every, args.trace_max_mb)
                for case in manifest.cases
            ]
            results = (result for _, result in run_jobs(
                jobs, _episode_job, workers=args.workers, timeout_seconds=args.episode_timeout,
                progress=refresh_if_due,
            ))
        for index, result in enumerate(results, start=1):
            writer.append(result)
            print(
                f"{index}/{len(manifest.cases)} {result.case.case_id}: "
                f"score={result.score:.6f}, survival={result.survival_seconds:.1f}s, "
                f"{result.termination}",
                flush=True,
            )
    except JobExecutionError as exc:
        failure = FailureInfo(stage="episode_worker", error_type=type(exc).__name__, message=str(exc))
        result = exc.result or EpisodeResult(
            case=exc.job[0], status="failed", termination="error", failure=failure,
        )
        writer.append(result)
        writer.finalize("failed", result.failure)
        if refresh is not None:
            refresh()
        print(f"Benchmark worker failed: {exc}\nDetails preserved in {args.output}.", file=sys.stderr)
        return 1
    except EpisodeExecutionError as exc:
        writer.append(exc.result)
        writer.finalize("failed", exc.result.failure)
        print(f"Benchmark failed: {exc}\nDetails preserved in {args.output}.", file=sys.stderr)
        return 1
    except EpisodeInterrupted as exc:
        writer.append(exc.result)
        writer.finalize("interrupted", exc.result.failure)
        print(f"Benchmark interrupted; partial results preserved in {args.output}.", file=sys.stderr)
        return 130
    except KeyboardInterrupt:
        writer.finalize("interrupted", FailureInfo(
            stage="between_episodes", error_type="KeyboardInterrupt", message="Interrupted by user.",
        ))
        print(f"Benchmark interrupted; completed results preserved in {args.output}.", file=sys.stderr)
        return 130
    status = "truncated" if any(result.status == "truncated" for result in writer.results) else "complete"
    summary = writer.finalize(status)
    if args.telemetry or args.progress:
        from src.benchmarking.artifacts import _atomic_text
        from src.benchmarking.config import canonical_json
        from src.benchmarking.progress import progress_data

        points = progress_data(args.output)["points"]
        _atomic_text(args.output / "survival-summary.json", canonical_json({
            "schema_version": 1, "scope": "complete_cases_only", "points": points,
        }) + "\n")
    if refresh is not None:
        refresh()
    print(
        f"Mean native score: {summary['score']['mean']:.6f}; "
        f"median: {summary['score']['median']:.6f}."
    )
    suffix = " (diagnostic; not eligible for ranking)" if manifest.max_steps is not None else ""
    print(f"Saved {status} run to {args.output}{suffix}.")
    return 0


def compare_command(args: argparse.Namespace) -> int:
    reference = load_run(args.reference)
    candidates = [load_run(path) for path in args.candidate]
    report = compare_runs(reference, candidates, **({"survival_first": True} if args.survival_first else {}))
    if not args.no_plots:
        from src.benchmarking.plots import require_plotting

        require_plotting()
    write_comparison_report(args.output, report)
    if not args.no_plots:
        from src.benchmarking.plots import comparison_plot_data, render_plots

        render_plots(comparison_plot_data(report), args.output)
    pairs = {pair["key"]: pair for pair in report["comparisons"]}
    print("Policy (role) | Mean score | Survival s | Time-limit completions | Mean delta")
    for entry in report["leaderboard"]:
        summary = entry["summary"]
        pair = pairs.get(entry["key"])
        delta = f"{pair['mean_delta']:.6f}" if pair is not None else "--"
        label = entry["label"].replace("\r", " ").replace("\n", " ")
        print(
            f"{label} ({entry['key']}) | {summary['score']['mean']:.6f} | "
            f"{summary['survival_seconds']['mean']:.2f} | "
            f"{summary['completion']['completed']}/{summary['completion']['count']} | {delta}"
        )
    for warning in report["warnings"]:
        print(f"Note: {warning}")
    print(f"Saved comparison to {args.output}.")
    return 0


def inspect_command(args: argparse.Namespace) -> int:
    saved = load_run(args.run)
    print(json.dumps(summarize_run(saved.manifest, saved.episodes), indent=2, allow_nan=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, ValueError, ImportError) as exc:
        print(f"Benchmark error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Benchmark interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
