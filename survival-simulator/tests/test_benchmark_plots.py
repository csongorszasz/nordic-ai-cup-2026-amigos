import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.benchmarking.plots import (
    DeltaSeries, PlotData, RunSeries, comparison_plot_data, render_plots, require_plotting,
)


class PlotTests(unittest.TestCase):
    def data(self, latencies=(0.1, 0.2)):
        return PlotData(
            suite="fixture", reference_label="random-local-v1", world_count=2, case_count=2,
            runs=[
                RunSeries("reference", "random-local-v1", [1.0, 2.0], latencies),
                RunSeries("candidate-1", "candidate", [2.0, 4.0], latencies),
            ],
            deltas=[DeltaSeries("candidate", [1, 17], [1.0, 2.0])],
            warnings=["Timing is diagnostic; this is not an HTTP evaluation."],
        )

    def test_noninteractive_figures_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            paths = render_plots(self.data(), output)
            self.assertEqual(len(paths), 3)
            for path in paths:
                self.assertGreater(path.stat().st_size, 1000)
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            with self.assertRaises(FileExistsError):
                render_plots(self.data(), output)

    def test_missing_latencies_are_not_fabricated_as_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = render_plots(self.data(latencies=(None, None)), Path(temporary))
            self.assertTrue(all(path.is_file() for path in paths))

    def test_optional_dependency_has_actionable_error(self):
        with patch.dict("sys.modules", {"matplotlib": None}):
            with self.assertRaisesRegex(ImportError, "requirements-benchmark.txt"):
                require_plotting()

    def test_report_adapter_preserves_paired_data_and_warnings(self):
        report = {
            "suite": {"name": "quick"}, "reference_label": "same-label",
            "world_count": 2, "case_count": 4, "warnings": ["Timing contexts differ."],
            "runs": [{
                "key": "reference", "label": "same-label", "scores": [1.0, 2.0],
                "batch_mean_ms": [0.1, None],
            }],
            "comparisons": [{
                "key": "candidate-1", "label": "same-label",
                "seed_deltas": [
                    {"world_seed": 1, "mean_delta": 0.5},
                    {"world_seed": 17, "mean_delta": -0.1},
                ],
            }],
        }
        data = comparison_plot_data(report)
        self.assertEqual(data.runs[0].scores, report["runs"][0]["scores"])
        self.assertEqual(data.runs[0].batch_mean_ms, [0.1, None])
        self.assertEqual(data.deltas[0].world_seeds, [1, 17])
        self.assertEqual(data.deltas[0].mean_deltas, [0.5, -0.1])
        self.assertEqual(data.deltas[0].label, "same-label (candidate-1)")
        self.assertEqual(data.warnings, report["warnings"])


if __name__ == "__main__":
    unittest.main()
