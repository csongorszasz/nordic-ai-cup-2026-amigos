"""Read-only population experiment plots; this module never starts a simulation."""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import math
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from src.benchmarking.artifacts import _atomic_replace, _atomic_text
from src.benchmarking.config import EpisodeResult, _invalid_constant, canonical_json, read_json
from src.benchmarking.plots import require_plotting
from src.benchmarking.survival import survival_curve, survival_metrics
from src.benchmarking.telemetry import TELEMETRY_VERSION


def read_complete_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            value = json.loads(line, parse_constant=_invalid_constant)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object in {path}.")
            records.append(value)
    return records


def last_complete_sample(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("rb") as stream:
        position = stream.seek(0, 2)
        suffix = b""
        while position:
            length = min(position, 8192)
            position -= length
            stream.seek(position)
            suffix = stream.read(length) + suffix
            end = suffix.rfind(b"\n")
            if end < 0:
                continue
            start = suffix.rfind(b"\n", 0, end)
            if start >= 0 or position == 0:
                value = json.loads(suffix[start + 1:end], parse_constant=_invalid_constant)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected a complete sample object in {path}.")
                return value
    return None


def progress_data(run: Path) -> dict:
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    search = manifest.get("mode") == "search"
    if search:
        world_file = run / "search-worlds.json"
        worlds = read_json(world_file)["seeds"] if world_file.exists() else []
        expected = [f"world-{seed}-repeat-0" for seed in worlds]
        expected_total = manifest["config"]["search"]["candidates"] * len(expected)
        label = "Population parameter search"
        tail_fraction = manifest["config"]["search"]["lower_tail_fraction"]
    elif "cases" in manifest:
        expected = [case["case_id"] for case in manifest["cases"]]
        expected_total = len(expected)
        label = manifest["policy"]["label"]
        tail_fraction = 0.05
    else:
        raise ValueError("Population monitoring requires a benchmark or controller-search manifest.")
    cases = []
    seen = set()
    grouped: dict[int, list[dict]] = defaultdict(list)
    for path in sorted((run / "telemetry").rglob("case.json")):
        meta = read_json(path)
        if meta["schema_version"] not in (1, TELEMETRY_VERSION):
            raise ValueError("Unsupported population telemetry version.")
        key = (meta["candidate"], meta["case"]["case_id"])
        if key in seen:
            raise ValueError(f"Duplicate telemetry case: {key}.")
        seen.add(key)
        latest_path, summary_path = path.parent / "latest.json", path.parent / "summary.json"
        samples_path = path.parent / "samples.jsonl"
        latest = read_json(latest_path) if meta["schema_version"] == 1 and latest_path.exists() else None
        summary = read_json(summary_path) if summary_path.exists() else None
        if summary is not None and summary["case"] != meta["case"]:
            raise ValueError("Telemetry result/case identities disagree.")
        result = EpisodeResult.model_validate(summary["result"]) if summary is not None else None
        sample = (
            summary["last_sample"] if summary is not None else latest["sample"]
            if latest is not None else last_complete_sample(samples_path)
        )
        item = {
            "case_id": meta["case"]["case_id"], "world_seed": meta["case"]["world_seed"],
            "candidate": meta["candidate"], "state": result.status if result else (
                "running" if manifest["status"] == "running" else "aborted"
            ),
            "scope": meta["scope"], "time_limit": meta["time_limit"],
            "sample": sample, "result": result, "path": path.parent,
            "diagnosis": summary.get("diagnosis") if summary else None,
            "modified": max(
                (p.stat().st_mtime_ns for p in (path, latest_path, samples_path, summary_path) if p.exists()),
            ),
        }
        cases.append(item)
        grouped[meta["candidate"] if meta["candidate"] is not None else 0].append(item)
    points = []
    for candidate, items in sorted(grouped.items()):
        rows = [item["result"] for item in items if item["result"] is not None]
        point = {"candidate": candidate, "state": "partial", "native_ticks": sum(
            item["sample"]["tick"] for item in items if item["sample"] is not None
        ), "metrics": None}
        if any(row.status in ("failed", "interrupted") for row in rows):
            point["state"] = "failed"
        elif any(item["scope"] == "diagnostic" for item in items):
            point["state"] = "diagnostic"
        elif expected and {row.case.case_id for row in rows} == set(expected):
            horizons = {item["time_limit"] for item in items}
            if len(horizons) != 1:
                raise ValueError("Cannot rank mixed stopping horizons.")
            health = {
                item["case_id"]: (
                    item["sample"]["young_reproductive"], item["sample"]["energy_p10"] or 0.0,
                )
                for item in items if item["sample"] is not None
            }
            point["metrics"] = survival_metrics(
                rows, expected_case_ids=expected, time_limit=next(iter(horizons)),
                terminal_health=health, tail_fraction=tail_fraction,
            )
            point["state"] = "complete"
        points.append(point)
    current = max(cases, key=lambda item: (item["state"] == "running", item["modified"])) if cases else None
    samples = read_complete_lines(current["path"] / "samples.jsonl") if current else []
    curve_rows = [
        item["result"] for item in cases
        if item["result"] is not None and (current is None or item["candidate"] == current["candidate"])
    ]
    states = Counter(item["state"] for item in cases)
    return {
        "label": label, "run_state": manifest["status"], "search": search,
        "expected_cases": expected_total, "recorded_cases": len(cases), "states": dict(states),
        "points": points, "cases": cases, "current": current, "samples": samples,
        "curve_rows": curve_rows,
    }


def render_progress(run: Path) -> list[Path]:
    require_plotting()
    import matplotlib
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    run = Path(run)
    data = progress_data(run)
    output = run / "progress"
    output.mkdir(parents=True, exist_ok=True)
    current, samples = data["current"], data["samples"]
    xs = [sample["sim_time"] for sample in samples]
    current_label = (
        f"candidate={current['candidate']} | {current['case_id']} | {current['state']}"
        if current else "Waiting for the first episode"
    )
    status = {
        key: data[key] for key in (
            "label", "run_state", "expected_cases", "recorded_cases", "states", "points",
        )
    }
    status["current"] = current_label
    status["warning"] = (
        "Partial/failed/capped cases are not complete rankings. Engine truth is diagnostic only. "
        "Probability bounds apply only to independently sampled audit worlds, not development suites."
    )
    _atomic_text(output / "status.json", canonical_json(status) + "\n")
    columns = ("candidate", "world_seed", "case_id", "state", "scope", "survival_seconds", "completed")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for item in data["cases"]:
        row = {key: item[key] for key in columns[:5]}
        result = item["result"]
        row.update(survival_seconds=result.survival_seconds if result else None,
                   completed=result.completed if result else None)
        writer.writerow(row)
    _atomic_text(output / "cases.csv", buffer.getvalue())
    paths = []

    def line(ax, key, label, *, parent=None, **kwargs):
        values = [
            sample.get(parent, {}).get(key) if parent else sample.get(key)
            for sample in samples
        ]
        if not any(value is not None for value in values):
            return
        ax.plot(xs, [math.nan if value is None else value for value in values], label=label, **kwargs)

    def save(name, title, draw):
        fig = Figure(figsize=(12, 8), dpi=110)
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, 2)
        fig.suptitle(f"{data['label']} | {title}\n{current_label}", fontsize=11)
        draw(axes)
        for ax in axes.flat:
            ax.grid(alpha=0.2)
            if not ax.has_data() and not ax.texts:
                ax.text(0.5, 0.5, "No measurements available", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7)
        fig.text(0.02, 0.01, status["warning"], fontsize=7, wrap=True)
        fig.tight_layout(rect=(0, 0.05, 1, 0.92))
        path = output / f"{name}.png"
        temporary = output / f".{uuid.uuid4().hex}.png"
        try:
            fig.savefig(temporary, format="png")
            _atomic_replace(temporary, path)
        finally:
            fig.clear()
            if temporary.exists():
                temporary.unlink()
        paths.append(path)

    def population(axes):
        a, b, c, d = axes.flat
        for key, label in (("population", "Living"), ("young", "Age < 60"),
                           ("reproductive", "Energy-eligible"), ("elders", "Senescent (truth)")):
            line(a, key, label)
        a.set(xlabel="Simulated seconds", ylabel="Agents")
        for key, label in (("birth", "Confirmed births"), ("death", "Deaths"),
                           ("spawn_requests", "Requested births")):
            line(b, key, label, parent="counts")
        b.set(xlabel="Simulated seconds", ylabel="Cumulative events")
        for key, label in (("target_population", "Capacity target"),
                           ("replacement_demand", "Replacement demand")):
            line(c, key, label, parent="policy")
        c.set(xlabel="Simulated seconds", ylabel="Public-policy estimate (previous decision)")
        ages = samples[-1]["ages"] if samples else []
        d.hist(ages, bins=np.arange(0, max(150, max(ages, default=0) + 15), 15))
        d.axvline(60, color="red", linestyle="--", label="Earliest senescence")
        d.set(xlabel="Current agent age", ylabel="Agents")

    def energy(axes):
        a, b, c, d = axes.flat
        line(a, "energy", "Total energy")
        line(a, "energy_p10", "Individual p10")
        a.set(xlabel="Simulated seconds", ylabel="Energy")
        for key in ("food_retained", "cost_living", "cost_move", "cost_turn", "cost_senescence"):
            line(b, key, key.replace("_", " "), parent="totals")
        b.plot(xs, [s["totals"]["cost_birth"] - s["totals"]["newborn_energy"] for s in samples],
               label="net birth tax")
        b.set(xlabel="Simulated seconds", ylabel="Cumulative energy (engine truth)")
        for key in ("trees", "fruit", "predators"):
            line(c, key, key)
        c.set(xlabel="Simulated seconds", ylabel="World entities (engine truth)")
        line(d, "energy_balance_residual", "Accounting residual")
        d.set(xlabel="Simulated seconds", ylabel="Measured minus accounted energy")

    def robustness(axes):
        a, b, c, d = axes.flat
        cx, cy = survival_curve(data["curve_rows"])
        if data["curve_rows"]:
            a.step(cx, cy, where="post", label="Completed cases; horizons censored")
        a.set(xlabel="Simulated seconds", ylabel="Estimated survival", ylim=(0, 1.05))
        complete = [point for point in data["points"] if point["metrics"] is not None]
        if complete:
            bx = [point["candidate"] for point in complete]
            b.plot(bx, [point["metrics"]["completion_fraction"] for point in complete], "o-", label="Complete worlds")
            c.plot(bx, [point["metrics"]["lower_tail_seconds"] for point in complete], "o-", label="Lower-tail survival")
            c.plot(bx, [point["metrics"]["worst_seconds"] for point in complete], "o-", label="Worst survival")
        b.set(xlabel="Candidate (complete matched cases only)", ylabel="Completion fraction", ylim=(0, 1.05))
        c.set(xlabel="Candidate", ylabel="Survival seconds")
        states = data["states"]
        d.bar(list(states), list(states.values()))
        d.set(title=f"{data['recorded_cases']}/{data['expected_cases']} cases started", ylabel="Cases")

    def failures(axes):
        a, b, c, d = axes.flat
        causes = Counter()
        hypotheses = Counter()
        for item in data["cases"]:
            if item["sample"]:
                causes.update(item["sample"]["death_causes"])
            if item["diagnosis"]:
                hypotheses.update(item["diagnosis"]["hypotheses"])
        a.bar(list(causes), list(causes.values()))
        a.set(title="Confirmed proximate death causes", ylabel="Deaths")
        positions = samples[-1]["positions"] if samples else []
        if positions:
            b.scatter([p[1] for p in positions], [p[2] for p in positions], s=20)
        elif samples:
            b.text(0.5, 0.5, "No living agents", transform=b.transAxes, ha="center", va="center")
        b.set(title="Current agent positions (engine truth)", xlabel="x", ylabel="y")
        for key in ("colonies", "known_patches", "migrations", "blocked_births_count"):
            line(c, key, key.replace("_", " "), parent="policy")
        c.set(xlabel="Simulated seconds", ylabel="Public-policy diagnostics")
        if hypotheses:
            d.barh(list(hypotheses), list(hypotheses.values()))
            d.set(title="Inferred collapse mechanisms (not proven)", xlabel="Episodes")
        elif samples:
            counts = samples[-1]["counts"]
            names = ["fruit_spawn", "fruit_eaten", "tree_spawn", "tree_removed", "deaths_while_senescent"]
            d.barh(names, [counts.get(name, 0) for name in names])
            d.set(title="Lifecycle evidence; no collapse diagnosis yet", xlabel="Events")

    def performance(axes):
        a, b, c, d = axes.flat
        line(a, "batch_p95_ms", "Recent batch p95")
        a.axhline(20, color="red", linestyle="--", label="Conservative average budget reference")
        a.set(xlabel="Simulated seconds", ylabel="Batch latency ms")
        line(b, "http_wait_seconds", "Accumulated HTTP wait")
        b.axhline(600, color="red", linestyle="--", label="Local native-episode gate")
        b.set(xlabel="Simulated seconds", ylabel="Seconds")
        if samples:
            c.plot(xs, [s["tick"] / max(s["wall_seconds"], 1e-9) for s in samples], label="Recorded episode throughput")
            d.plot([s["wall_seconds"] for s in samples], [s["tick"] for s in samples], label="Native ticks")
        c.set(xlabel="Simulated seconds", ylabel="Native ticks / wall second")
        d.set(xlabel="Episode wall seconds", ylabel="Native ticks")

    with matplotlib.rc_context({"text.usetex": False, "font.family": "DejaVu Sans"}):
        for name, title, draw in (
            ("population-renewal", "Population and renewal", population),
            ("energy-resources", "Energy and resources", energy),
            ("survival-progress", "Robustness progress", robustness),
            ("failure-spatial", "Failure and spatial diagnostics", failures),
            ("performance", "Performance", performance),
        ):
            save(name, title, draw)
    revision = time.time_ns()
    images = "\n".join(
        f'<img src="{path.name}?v={revision}" alt="{html.escape(path.stem)}">'
        for path in paths
    )
    _atomic_text(output / "index.html", (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="10"><title>Survival progress</title>'
        '<style>body{font:16px sans-serif;margin:2em;background:#f5f5f5}'
        'img{max-width:100%;background:white;margin:1em 0}pre{white-space:pre-wrap}</style>'
        f'</head><body><h1>{html.escape(data["label"])}</h1>'
        f'<p>{html.escape(current_label)}; run: {html.escape(data["run_state"])}</p>'
        f'<p>{html.escape(status["warning"])}</p>'
        '<p><a href="status.json">Status</a> | <a href="cases.csv">Cases CSV</a></p>'
        f'{images}</body></html>\n'
    ))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 1:
        parser.error("--interval must be finite and at least one second.")
    while True:
        render_progress(args.run)
        if not args.watch or read_json(args.run / "manifest.json")["status"] != "running":
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
