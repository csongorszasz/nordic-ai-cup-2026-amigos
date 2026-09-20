"""Export a trained Ultralytics detector to TensorRT when the dependency stack is available."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO  # type: ignore

    _ULTRALYTICS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    YOLO = None
    _ULTRALYTICS_AVAILABLE = False


def export(weights: Path, imgsz: int, half: bool, simplify: bool, workspace: int) -> Path:
    if not _ULTRALYTICS_AVAILABLE:
        raise RuntimeError("ultralytics is not installed; TensorRT export is unavailable")

    yolo_cls = YOLO
    assert yolo_cls is not None
    model = yolo_cls(str(weights))
    exported = model.export(
        format="engine",
        imgsz=imgsz,
        half=half,
        simplify=simplify,
        workspace=workspace,
    )
    return Path(exported)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a YOLO detector to TensorRT.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--workspace", type=int, default=4)
    arguments = parser.parse_args()

    exported = export(
        arguments.weights,
        arguments.imgsz,
        arguments.half,
        not arguments.no_simplify,
        arguments.workspace,
    )
    print(f"Exported TensorRT engine to {exported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
