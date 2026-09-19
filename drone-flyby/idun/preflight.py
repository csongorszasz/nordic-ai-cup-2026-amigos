"""Validate the allocated GPU and persist the experiment environment."""

import hashlib
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform

import torch


def configure_tracking() -> None:
    config_dir = Path(os.environ["YOLO_CONFIG_DIR"]).resolve()
    if not config_dir.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("YOLO_CONFIG_DIR must belong to this experiment")
    from ultralytics import settings

    settings.update({
        name: False for name in ("sync", "wandb", "mlflow", "comet", "clearml", "dvc", "neptune")
        if name in settings
    })


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the allocated GPU job")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Experiments require exactly one visible GPU")
    device = torch.cuda.get_device_properties(0)
    if not any(model in device.name.upper() for model in ("A100", "H100")):
        raise RuntimeError(f"Expected A100/H100, received {device.name}")
    if device.total_memory < 75 * 1024**3:
        raise RuntimeError(f"Expected an 80 GB device, received {device.total_memory} bytes")
    metadata = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": platform.node(),
        "python": platform.python_version(),
        "gpu": device.name,
        "vram_bytes": device.total_memory,
        "cuda": torch.version.cuda,
        "command": os.environ.get("RUN_CMD"),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "ultralytics", "numpy",
                         "opencv-python", "faster-coco-eval", "pydantic")
        },
    }
    lock = Path(os.environ["DRONE_ENV_PATH"]) / "requirements.lock"
    metadata["package_lock"] = lock.read_text(encoding="utf-8")
    metadata["package_lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
    snapshot = Path("snapshot.json")
    if snapshot.exists():
        manifest = json.loads(snapshot.read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            path = Path(entry["path"])
            if not path.resolve().is_relative_to(Path.cwd().resolve()):
                raise ValueError(f"Snapshot path escapes the experiment: {path}")
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != entry["sha256"]:
                raise RuntimeError(f"Snapshot content changed: {path}")
        metadata["snapshot_sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    output = Path("runs") / f"environment_{metadata['job_id']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configure-only", action="store_true")
    arguments = parser.parse_args()
    configure_tracking()
    if not arguments.configure_only:
        main()
