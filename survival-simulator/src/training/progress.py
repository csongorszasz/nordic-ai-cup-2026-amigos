"""Learning curves from immutable evaluation records; never runs a policy."""

import argparse
import csv
import io
import math
from pathlib import Path
from statistics import mean

from src.benchmarking.artifacts import _atomic_text
from src.benchmarking.config import read_json
from src.benchmarking.plots import require_plotting
from src.training.artifacts import write_json


def progress_data(run: Path) -> dict:
    from src.training.evaluation import load_schedule
    schedule = load_schedule(run)
    expected_seeds = [
        seed for seed in schedule["suite"]["seeds"]
        for _ in range(schedule["evaluation"]["repeats"])
    ]
    points = []
    history_path = run / "evaluation" / "history.json"
    history = read_json(history_path) if history_path.exists() else []
    by_update = {point["update"]: point for point in history}
    if len(by_update) != len(history):
        raise ValueError("Resume history contains duplicate updates.")
    for update in sorted(set(schedule["updates"]) | set(by_update)):
        directory = run / "evaluation" / "checkpoints" / f"update-{update:08d}"
        ready = directory / "ready.json"
        result = directory / "result.json"
        record = read_json(result) if result.exists() else read_json(ready) if ready.exists() else by_update.get(update)
        if record is None:
            points.append({"update": update, "state": "not_published", "mean_score": None,
                           "native_ticks": None})
            continue
        if record.get("protocol_id") != schedule["protocol_id"] or record["update"] != update:
            raise ValueError("Cannot mix incompatible evaluation protocols or update identities.")
        point = {**record, "state": record.get("state", "queued"), "mean_score": None}
        if point["state"] == "complete":
            scores = record.get("scores", [])
            if record.get("world_seeds") != expected_seeds or len(scores) != len(expected_seeds):
                raise ValueError("A curve point requires every expected episode in seed/repeat order.")
            if not all(type(value) in (int, float) and math.isfinite(value) for value in scores):
                raise ValueError("Evaluation scores must be finite.")
            averages = [
                mean(scores[index:index + schedule["evaluation"]["repeats"]])
                for index in range(0, len(scores), schedule["evaluation"]["repeats"])
            ]
            point.update(mean_score=mean(scores), seed_min=min(averages), seed_max=max(averages))
            if not math.isclose(point["mean_score"], record["mean_score"], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("Stored mean does not match the complete episode records.")
        points.append(point)
    teacher = None
    teacher_state = "not_evaluated"
    teacher_error = None
    baseline = run / "evaluation" / "teacher" / "result.json"
    if not baseline.exists():
        baseline = run / "evaluation" / "historical-teacher.json"
    if baseline.exists():
        record = read_json(baseline)
        if record["protocol_id"] != schedule["protocol_id"]:
            raise ValueError("Teacher baseline uses another protocol.")
        teacher_state = record["state"]
        teacher_error = record.get("error")
        if record["state"] == "complete":
            if len(record["scores"]) != len(expected_seeds) or not all(
                type(score) in (int, float) and math.isfinite(score) for score in record["scores"]
            ):
                raise ValueError("Teacher baseline is incomplete or non-finite.")
            teacher = mean(record["scores"])
    completed = [point["update"] for point in points if point["state"] == "complete"]
    published = [point["update"] for point in points if point["state"] != "not_published"]
    return {
        "run_id": schedule["run_id"], "stage": schedule["stage"],
        "protocol_id": schedule["protocol_id"], "suite": schedule["suite"],
        "teacher_mean": teacher, "points": points, "lineage": schedule.get("lineage"),
        "teacher_state": teacher_state, "teacher_error": teacher_error,
        "latest_evaluated_update": max(completed) if completed else None,
        "latest_published_update": max(published) if published else None,
        "warning": "Development seeds are reused; seed range is not a confidence interval or optimizer-seed variance.",
    }


def render_progress(run: Path) -> list[Path]:
    require_plotting()
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    run = Path(run)
    data = progress_data(run)
    output = run / "progress"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "status.json", data)
    from src.benchmarking.config import canonical_json
    _atomic_text(output / "evaluations.jsonl", "".join(
        canonical_json(point) + "\n" for point in data["points"] if point["state"] != "not_published"
    ))
    columns = ("update", "state", "mean_score", "seed_min", "seed_max", "native_ticks", "sha256")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(data["points"])
    _atomic_text(output / "learning-curve.csv", buffer.getvalue())
    paths = []
    for xkey, name, xlabel in (
        ("update", "learning-curve.png", "Completed learner updates"),
        ("native_ticks", "learning-curve-ticks.png", "Cumulative native training ticks (including recorded warm start)"),
    ):
        with matplotlib.rc_context({"text.usetex": False, "font.family": "DejaVu Sans"}):
            figure = Figure(figsize=(10, 5), dpi=120)
            FigureCanvasAgg(figure)
            axes = figure.subplots()
            points = [point for point in data["points"] if point.get(xkey) is not None]
            xs = [point[xkey] for point in points]
            ys = [point["mean_score"] if point["state"] == "complete" else math.nan for point in points]
            axes.plot(xs, ys, marker="o", label=data["stage"])
            low = [point["seed_min"] if point["state"] == "complete" else math.nan for point in points]
            high = [point["seed_max"] if point["state"] == "complete" else math.nan for point in points]
            axes.fill_between(xs, low, high, alpha=0.15, label="Observed world-seed range")
            if data["teacher_mean"] is not None:
                axes.axhline(data["teacher_mean"], linestyle="--", color="black", label="Pinned teacher")
            for point in points:
                if point["state"] not in ("complete", "not_published"):
                    axes.axvline(point[xkey], color="red" if point["state"] == "failed" else "gray",
                                 linestyle=":", alpha=0.4)
                    axes.text(point[xkey], 0.02, point["state"], rotation=90, fontsize=7,
                              transform=axes.get_xaxis_transform())
            axes.set(xlabel=xlabel, ylabel="Mean native final episode score",
                     title=f"{data['stage']} | {data['suite']['name']} | latest evaluated: {data['latest_evaluated_update']}")
            axes.grid(alpha=0.2)
            axes.legend(fontsize=8)
            figure.text(0.02, 0.02, data["warning"], fontsize=7)
            figure.tight_layout(rect=(0, 0.06, 1, 1))
            path = output / name
            temporary = path.with_suffix(".pending.png")
            try:
                figure.savefig(temporary, format="png")
                temporary.replace(path)
            finally:
                figure.clear()
                if temporary.exists():
                    temporary.unlink()
            paths.append(path)
    _atomic_text(output / "report.md", (
        f"# {data['stage']} progress\n\n"
        f"Last evaluated update: {data['latest_evaluated_update']}; "
        f"last published update: {data['latest_published_update']}.\n\n"
        f"Teacher baseline: {data['teacher_state']}.\n\n"
        "![Native score](learning-curve.png)\n\n"
        f"{data['warning']}\n\n"
        "Per-checkpoint episode records: `../evaluation/checkpoints/`.\n"
    ))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    render_progress(parser.parse_args().run)


if __name__ == "__main__":
    main()
