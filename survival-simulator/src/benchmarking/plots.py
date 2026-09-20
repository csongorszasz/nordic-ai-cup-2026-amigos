import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunSeries:
    key: str
    label: str
    scores: Sequence[float]
    batch_mean_ms: Sequence[float | None]


@dataclass(frozen=True)
class DeltaSeries:
    label: str
    world_seeds: Sequence[int]
    mean_deltas: Sequence[float]


@dataclass(frozen=True)
class PlotData:
    suite: str
    reference_label: str
    world_count: int
    case_count: int
    runs: Sequence[RunSeries]
    deltas: Sequence[DeltaSeries]
    warnings: Sequence[str]


def comparison_plot_data(report: dict) -> PlotData:
    return PlotData(
        suite=report["suite"]["name"],
        reference_label=report["reference_label"],
        world_count=report["world_count"],
        case_count=report["case_count"],
        runs=[
            RunSeries(run["key"], run["label"], run["scores"], run["batch_mean_ms"])
            for run in report["runs"]
        ],
        deltas=[
            DeltaSeries(
                label=f"{comparison['label']} ({comparison['key']})",
                world_seeds=[row["world_seed"] for row in comparison["seed_deltas"]],
                mean_deltas=[row["mean_delta"] for row in comparison["seed_deltas"]],
            )
            for comparison in report["comparisons"]
        ],
        warnings=report["warnings"],
    )


def require_plotting() -> None:
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        raise ImportError(
            "Static plots require Matplotlib. Run "
            "'python -m pip install -r requirements-benchmark.txt', "
            "or compare with --no-plots."
        ) from exc


def render_plots(data: PlotData, output: Path) -> list[Path]:
    require_plotting()
    import matplotlib
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output = Path(output)
    if not output.is_dir():
        raise ValueError("Write the numerical comparison report before rendering plots.")
    filenames = ("scores.png", "paired-deltas.png", "score-vs-latency.png")
    paths = [output / name for name in filenames]
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite plot {path}.")
    if not data.runs or any(
        not run.scores or len(run.scores) != len(run.batch_mean_ms) for run in data.runs
    ):
        raise ValueError("Plot data must contain aligned score and batch-latency observations.")
    if any(len(delta.world_seeds) != len(delta.mean_deltas) for delta in data.deltas):
        raise ValueError("Plot data must align world seeds and paired deltas.")

    def figure(title):
        fig = Figure(figsize=(11, 6), dpi=120)
        FigureCanvasAgg(fig)
        subtitle = (
            f"{data.suite} | {data.world_count} world seeds, {data.case_count} cases per run"
            f" | reference: {data.reference_label}"
        )
        fig.suptitle(f"{title}\n{subtitle}", fontsize=11)
        warning = " ".join(data.warnings)
        if warning:
            fig.text(0.02, 0.02, textwrap.fill(warning, width=150), fontsize=8, va="bottom")
        return fig, fig.subplots()

    def save(fig, path):
        try:
            fig.tight_layout(rect=(0, 0.16, 1, 0.88))
            fig.savefig(path, format="png")
        finally:
            fig.clear()

    with matplotlib.rc_context({"text.usetex": False, "font.family": "DejaVu Sans"}):
        fig, ax = figure("Native final score distributions")
        labels = [f"{run.label}\n({run.key})" for run in data.runs]
        ax.boxplot([list(run.scores) for run in data.runs], tick_labels=labels, showmeans=True)
        for position, run in enumerate(data.runs, start=1):
            jitter = np.linspace(-0.12, 0.12, len(run.scores))
            ax.scatter(position + jitter, run.scores, s=14, alpha=0.6)
        ax.set_ylabel("Native final score (higher is better)")
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", alpha=0.2)
        save(fig, paths[0])

        fig, ax = figure("Paired score differences by world seed")
        ax.axhline(0, color="black", linewidth=1)
        for index, delta in enumerate(data.deltas):
            offset = (index - (len(data.deltas) - 1) / 2) * 0.12
            positions = np.arange(len(delta.world_seeds))
            ax.scatter(positions + offset, delta.mean_deltas, label=delta.label, s=24, alpha=0.8)
        if data.deltas:
            ax.set_xticks(range(len(data.deltas[0].world_seeds)), data.deltas[0].world_seeds, rotation=90)
            ax.legend(fontsize=8)
        ax.set_ylabel("Candidate minus reference score (repeat mean per seed)")
        ax.set_xlabel("World seed")
        ax.grid(axis="y", alpha=0.2)
        save(fig, paths[1])

        fig, ax = figure("Quality versus measured batch-policy latency")
        has_latency = False
        for run in data.runs:
            points = [
                (latency, score)
                for score, latency in zip(run.scores, run.batch_mean_ms)
                if latency is not None
            ]
            if points:
                has_latency = True
                ax.scatter(
                    [point[0] for point in points], [point[1] for point in points],
                    label=f"{run.label} ({run.key})", alpha=0.65, s=24,
                )
        if has_latency:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No policy calls recorded; latency unavailable.", ha="center", transform=ax.transAxes)
        ax.set_xlabel("Mean batch-call latency per episode (ms, includes cold calls)")
        ax.set_ylabel("Native final score")
        ax.grid(alpha=0.2)
        save(fig, paths[2])
    return paths
