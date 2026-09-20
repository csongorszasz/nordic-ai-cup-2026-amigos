"""Run local policies and compare saved benchmark results."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from src.benchmarking.artifacts import RunWriter, build_manifest, load_run, summarize_run
from src.benchmarking.comparison import compare_runs, write_comparison_report
from src.benchmarking.config import (
    PROJECT_ROOT, SUITE_NAMES, FailureInfo, SimulationSettings, load_suite, read_json,
)
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


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        description="Compare local policies without changing the competition simulator.",
    )
    commands = cli.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Evaluate one policy on a fixed suite.")
    run.add_argument("--policy", default="random", help="'random' or an importable module:factory.")
    run.add_argument("--label", help="Display name (recorded separately from source identity).")
    run.add_argument("--config", type=Path, help="JSON object passed to the policy factory.")
    run.add_argument("--suite", choices=SUITE_NAMES, default="quick")
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
    compare.set_defaults(handler=compare_command)

    inspect = commands.add_parser("inspect", help="Summarize saved results, including incomplete runs.")
    inspect.add_argument("--run", type=Path, required=True)
    inspect.set_defaults(handler=inspect_command)
    return cli


def run_command(args: argparse.Namespace) -> int:
    config = read_json(args.config) if args.config is not None else {}
    if not isinstance(config, dict):
        raise ValueError("Policy configuration must be a JSON object.")
    loaded = load_policy(args.policy, label=args.label, config=config)
    manifest = build_manifest(
        PROJECT_ROOT, load_suite(args.suite), loaded.spec,
        repeats=args.repeats, fixed_policy_seed=args.fixed_policy_seed,
        simulation=SimulationSettings(), max_steps=args.max_steps,
        policy_sources=loaded.source_files, extra_artifacts=args.artifact,
    )
    writer = RunWriter(args.output, manifest)
    try:
        for index, case in enumerate(manifest.cases, start=1):
            result = run_episode(
                case, loaded.factory, loaded.spec.config, manifest.simulation, manifest.max_steps,
            )
            writer.append(result)
            print(
                f"{index}/{len(manifest.cases)} {case.case_id}: "
                f"score={result.score:.6f}, survival={result.survival_seconds:.1f}s, "
                f"{result.termination}",
                flush=True,
            )
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
    report = compare_runs(reference, candidates)
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
