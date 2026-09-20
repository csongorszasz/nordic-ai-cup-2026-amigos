"""Run the bounded, local-only population study using the existing benchmark and search CLIs."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from src.benchmarking.artifacts import _atomic_text, load_run
from src.benchmarking.config import PROJECT_ROOT, Suite, canonical_json, load_suite, read_json
from src.benchmarking.survival import survival_metrics, terminal_health
from src.policies.config import RuntimeConfig, load_experiment
from src.training.seeds import training_seeds


def seeds(seed: int, count: int, excluded: set[int]) -> list[int]:
    rng = random.Random(seed)
    used = set(excluded)
    result = []
    while len(result) < count:
        value = rng.getrandbits(32)
        if value not in used:
            used.add(value)
            result.append(value)
    return result


def metrics(path: Path) -> dict:
    run = load_run(path)
    if run.manifest.status != "complete" or run.manifest.max_steps is not None:
        raise ValueError(f"Only complete uncapped results may inform selection: {path}")
    return survival_metrics(
        run.episodes, expected_case_ids=[case.case_id for case in run.manifest.cases],
        time_limit=run.manifest.simulation.time_limit,
        terminal_health=terminal_health(path, run.episodes, required=True),
    )


class Study:
    def __init__(self, output: Path, workers: int):
        self.path = output.resolve()
        if os.name == "nt" and len(str(self.path)) > 164:
            raise ValueError("Use a short study output path to leave room for Windows per-case artifacts.")
        self.path.mkdir(parents=True, exist_ok=False)
        self.workers = workers
        self.inputs: dict[Path, str] = {}
        paths = sorted([
            *PROJECT_ROOT.joinpath("src").rglob("*.py"),
            *PROJECT_ROOT.joinpath("configs").glob("*.json"),
            *PROJECT_ROOT.glob("requirements*.txt"),
            PROJECT_ROOT / "benchmark.py", PROJECT_ROOT / "train.py",
            PROJECT_ROOT / "agent_server.py",
            PROJECT_ROOT / "population_experiments.py",
            *PROJECT_ROOT.joinpath("benchmarks", "suites").glob("*.json"),
        ])
        self.sources = {
            str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        }
        with zipfile.ZipFile(self.path / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, arcname=str(path.relative_to(PROJECT_ROOT)))
        self.state = {
            "version": 1, "kind": "local_population_study", "status": "running",
            "workers": workers, "sources": self.sources, "stages": [],
            "promotion": "Not permitted until frozen survival and separate HTTP gates pass.",
        }
        self.publish()

    def write(self, name: str, value: object) -> Path:
        path = self.path / name
        _atomic_text(path, canonical_json(value) + "\n")
        return path

    def publish(self) -> None:
        self.write("study.json", self.state)
        rows = "\n".join(
            f'<tr><td>{html.escape(stage["name"])}</td><td>{stage["status"]}</td>'
            f'<td><a href="{stage["plots"]}">plots</a></td>'
            f'<td><a href="{stage["name"]}.log">log</a></td></tr>'
            for stage in self.state["stages"]
        )
        current = next((stage for stage in reversed(self.state["stages"])
                        if stage["status"] == "running"), None)
        frame = (
            f'<iframe src="{current["plots"]}" style="width:100%;height:1000px"></iframe>'
            if current else ""
        )
        _atomic_text(self.path / "index.html", (
            '<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="15">'
            '<title>Local population study</title><style>body{font:16px sans-serif;margin:2em}'
            'td,th{padding:.4em;border:1px solid #ddd}table{border-collapse:collapse}</style></head><body>'
            f'<h1>Local population study: {self.state["status"]}</h1>'
            '<p>Original simulator; no RL or cloud experiments. Finite measurements are not an infinity guarantee.</p>'
            '<table><tr><th>Stage</th><th>Status</th><th>Progress</th><th>Output</th></tr>'
            f'{rows}</table>{frame}</body></html>\n'
        ))

    def verify_sources(self) -> None:
        for relative, expected in self.sources.items():
            if hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Study sources changed after freezing: {relative}. Start a new study.")
        for path, expected in self.inputs.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Study input changed after freezing: {path}. Start a new study.")

    def pin(self, path: Path) -> None:
        self.inputs[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.state["pinned_inputs"] = {str(path.relative_to(self.path)): value for path, value in self.inputs.items()}

    def command(self, name: str, arguments: list[str]) -> float:
        self.verify_sources()
        stage = {
            "name": name, "status": "running", "arguments": arguments,
            "plots": f"{name}/scores.png" if arguments[:2] == ["benchmark.py", "compare"]
            else f"{name}/progress/index.html",
        }
        self.state["stages"].append(stage)
        self.publish()
        print(f"Starting {name}; plots: {self.path / name / 'progress' / 'index.html'}", flush=True)
        started = time.monotonic()
        environment = {
            **os.environ, "PYGAME_HIDE_SUPPORT_PROMPT": "1", "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        }
        with (self.path / f"{name}.log").open("x", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, *arguments], cwd=PROJECT_ROOT, env=environment,
                stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        stage.update(
            status="complete" if result.returncode == 0 else "failed",
            returncode=result.returncode, wall_seconds=time.monotonic() - started,
        )
        self.publish()
        if result.returncode:
            raise RuntimeError(f"Stage {name} failed with exit {result.returncode}; see {self.path / (name + '.log')}.")
        self.verify_sources()
        return stage["wall_seconds"]

    def benchmark(
        self, name: str, config: Path, *, suite: str = "quick", suite_file: Path | None = None,
        max_steps: int | None = None, horizon: float = 3000.0, workers: int | None = None,
    ) -> dict | None:
        arguments = [
            "benchmark.py", "run", "--policy", "src.policies.runtime:create_policy",
            "--config", str(config), "--label", name, "--fixed-policy-seed", "1",
            "--progress", "--workers", str(workers or self.workers),
            "--episode-timeout", "1800", "--time-limit", str(horizon),
            "--output", str(self.path / name),
        ]
        arguments += ["--suite-file", str(suite_file)] if suite_file else ["--suite", suite]
        if max_steps is not None:
            arguments += ["--max-steps", str(max_steps)]
        self.command(name, arguments)
        if max_steps is None:
            result = metrics(self.path / name)
            print(
                f"{name}: {result['completed_worlds']}/{result['worlds']} complete; "
                f"worst={result['worst_seconds']:.1f}; tail={result['lower_tail_seconds']:.1f}",
                flush=True,
            )
            return result
        return None

    def run(self, development_worlds: int, audit_worlds: int) -> None:
        standard = load_suite("standard")
        holdout = load_suite("holdout")
        search_config = load_experiment(PROJECT_ROOT / "configs" / "search-population.json")
        search_worlds = training_seeds(search_config.seed, search_config.search.worlds)
        excluded = set(standard.seeds) | set(holdout.seeds) | set(search_worlds)
        development = seeds(20260920, development_worlds, excluded)
        fresh_audit = seeds(20260921, audit_worlds, excluded | set(development))
        dev_file = self.write("dev.json", Suite(name="population-development-v1", seeds=development).model_dump())
        audit_file = self.write("audit.json", Suite(
            name="population-frozen-audit-v1", seeds=[*holdout.seeds, *fresh_audit],
        ).model_dump())
        pilot_file = self.write("pilot.json", Suite(name="population-pilot-v1", seeds=standard.seeds[:4]).model_dump())
        for path in (dev_file, audit_file, pilot_file):
            self.pin(path)
        self.state["sampling"] = {
            "development": development, "fresh_audit": fresh_audit,
            "fixed_holdout": holdout.seeds, "search": search_worlds,
            "method": "Uniform uint32 draws without replacement, excluding every other pool.",
        }
        self.state["maximum_full_episodes"] = (
            12 + search_config.search.candidates * search_config.search.worlds + 15
            + 40 + development_worlds + audit_worlds + len(holdout.seeds) + 6
        )
        if shutil.disk_usage(self.path).free < self.state["maximum_full_episodes"] * 64 * 1024**2:
            raise OSError("Insufficient free disk for the declared worst-case trace budget.")
        self.publish()
        base = PROJECT_ROOT / "configs" / "controller-population.json"
        self.benchmark("pilot1", base, suite_file=pilot_file, max_steps=60, workers=1)
        self.benchmark("pilotn", base, suite_file=pilot_file, max_steps=60)
        first, parallel = self.state["stages"][-2:]
        if parallel["wall_seconds"] >= first["wall_seconds"]:
            self.workers = 1
            self.state["workers"] = 1
            self.state["worker_decision"] = "Parallel pilot did not improve batch throughput; retaining one worker."
        else:
            self.state["worker_decision"] = "Parallel pilot improved batch throughput within the requested worker cap."
        references = {
            "bh": PROJECT_ROOT / "configs" / "controller.json",
            "bw": PROJECT_ROOT / "configs" / "controller-turnaway-wall-aware.json",
            "br": PROJECT_ROOT / "configs" / "controller-turnaway-rules.json",
        }
        results = {name: self.benchmark(name, config) for name, config in references.items()}
        self.benchmark("p0", base)
        best_reference = max(results, key=lambda name: (tuple(results[name]["rank"]), name))
        self.command("s", [
            "train.py", "--config", str(PROJECT_ROOT / "configs" / "search-population.json"),
            "--set", f"resources.workers={self.workers}", "--output", str(self.path / "s"),
        ])
        selected = RuntimeConfig.model_validate(read_json(self.path / "s" / "policy.json"))
        variants = {"ab0": None, "abc": "capacity_feedback", "abm": "patch_memory",
                    "abd": "dispersion", "abe": "elder_decoys"}
        variant_paths, variant_results = {}, {}
        for name, setting in variants.items():
            value = selected.model_dump(mode="json")
            if setting:
                value["heuristic"][setting] = setting == "elder_decoys"
            descriptor = RuntimeConfig.model_validate(value).model_dump(mode="json")
            variant_paths[name] = self.write(f"{name}.json", descriptor)
            self.pin(variant_paths[name])
            variant_results[name] = self.benchmark(name, variant_paths[name])
        baseline_rows = {row.case.world_seed: row.survival_seconds for row in load_run(self.path / "ab0").episodes}
        eligible = ["ab0"]
        for name in variants:
            if name != "ab0" and all(
                row.survival_seconds >= baseline_rows[row.case.world_seed]
                for row in load_run(self.path / name).episodes
            ):
                eligible.append(name)
        winner = max(eligible, key=lambda name: (tuple(variant_results[name]["rank"]), name == "ab0"))
        candidate = self.write("candidate.json", read_json(variant_paths[winner]))
        self.pin(candidate)
        reference_metrics = self.benchmark("ref", references[best_reference], suite="standard")
        candidate_metrics = self.benchmark("standard", candidate, suite="standard")
        self.command("comparison", [
            "benchmark.py", "compare", "--reference", str(self.path / "ref"),
            "--candidate", str(self.path / "standard"), "--survival-first",
            "--output", str(self.path / "comparison"),
        ])
        self.benchmark("development", candidate, suite_file=dev_file)
        self.write("freeze.json", {
            "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "sources": self.sources, "selection": winner,
            "reference": best_reference, "standard_candidate": candidate_metrics,
            "standard_reference": reference_metrics,
            "audit_suite_sha256": hashlib.sha256(audit_file.read_bytes()).hexdigest(),
        })
        audit_metrics = self.benchmark("audit", candidate, suite_file=audit_file)
        audit_run = load_run(self.path / "audit")
        fresh_rows = [row for row in audit_run.episodes if row.case.world_seed in set(fresh_audit)]
        self.write("audit-summary.json", {
            "all_audit_worlds": audit_metrics,
            "fresh_uniform_worlds_only": survival_metrics(
                fresh_rows, expected_case_ids=[row.case.case_id for row in fresh_rows],
            ),
            "infinite_survival_claim": False,
        })
        rows = sorted(load_run(self.path / "development").episodes,
                      key=lambda row: (-row.survival_seconds, row.case.world_seed))
        survivors = [row.case.world_seed for row in rows if row.completed][:3]
        extended = survivors or [rows[0].case.world_seed]
        extended_file = self.write("extended.json", Suite(name="extended-native-v1", seeds=extended).model_dump())
        self.pin(extended_file)
        extended_metrics = self.benchmark("long6", candidate, suite_file=extended_file, horizon=6000.0)
        if extended_metrics["completed_worlds"]:
            self.benchmark("long12", candidate, suite_file=extended_file, horizon=12000.0)
        failures = []
        for name in ("standard", "development"):
            for path in sorted((self.path / name / "telemetry").rglob("summary.json")):
                result = read_json(path)
                if result["result"]["termination"] == "extinction":
                    failures.append({
                        "run": name, "case": result["case"],
                        "survival_seconds": result["result"]["survival_seconds"],
                        "diagnosis": result["diagnosis"], "evidence": str(path.relative_to(self.path)),
                    })
        self.write("failure-bank.json", {"version": 1, "development_only": True, "failures": failures})
        self.state.update(
            status="complete", candidate=str(candidate),
            native_survival_gate_passed=audit_metrics["completed_worlds"] == audit_metrics["worlds"],
            http_gate="pending_separate_loopback_run",
            promotion="Existing controller remains unchanged; full survival and HTTP gates are required.",
        )
        self.publish()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--development-worlds", type=int, default=100)
    parser.add_argument("--audit-worlds", type=int, default=100)
    parser.add_argument("--previous-study", type=Path, help="Record a prior attempt; never reuse or overwrite its cases.")
    parser.add_argument("--retry-reason", help="Explicit reason for starting another study attempt.")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or min(args.development_worlds, args.audit_worlds) < 1:
        parser.error("Use 1-4 workers and positive development/audit world counts.")
    if bool(args.previous_study) != bool(args.retry_reason):
        parser.error("--previous-study and --retry-reason must be supplied together.")
    previous = read_json(args.previous_study / "study.json") if args.previous_study else None
    if previous is not None and previous.get("status") == "running":
        parser.error("The previous study must have stopped before retrying.")
    study = Study(args.output, args.workers)
    if previous is not None:
        study.state["previous_attempt"] = {
            "path": str(args.previous_study.resolve()), "status": previous["status"],
            "reason": args.retry_reason, "cases_reused": False,
        }
        study.publish()
    try:
        study.run(args.development_worlds, args.audit_worlds)
    except BaseException as error:
        study.state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                           error_type=type(error).__name__, error=str(error))
        for stage in study.state["stages"]:
            if stage["status"] == "running":
                stage["status"] = study.state["status"]
        study.publish()
        raise
    print(f"Local study complete; separate loopback HTTP acceptance is still required: {study.path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
