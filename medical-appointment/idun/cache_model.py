"""Development-only caching of a pinned public model; never used by /predict."""

import argparse
import fnmatch
import json
import os
import shutil
from pathlib import Path


PATTERNS = ("*.json", "*.jinja", "*.safetensors", "merges.txt", "spiece.model")


def planned_files(siblings, maximum_bytes):
    files = []
    for file in siblings:
        if not any(fnmatch.fnmatch(file.rfilename, pattern) for pattern in PATTERNS):
            continue
        if file.size is None:
            raise ValueError(f"Unknown download size for {file.rfilename}.")
        files.append({"path": file.rfilename, "bytes": file.size})
    if not files or not any(file["path"].endswith(".safetensors") for file in files):
        raise ValueError("Model weights are absent from the manifest.")
    if sum(file["bytes"] for file in files) > maximum_bytes:
        raise ValueError("Pinned model exceeds the explicit download-size ceiling.")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/model_cache.json"))
    args = parser.parse_args()
    if not args.revision or len(args.revision) != 40 or args.max_bytes <= 0:
        parser.error("Use an exact commit revision and a positive size ceiling.")
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = "2"
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    info = HfApi().model_info(args.model, revision=args.revision, files_metadata=True)
    if info.sha != args.revision:
        raise ValueError("Model revision changed unexpectedly.")
    files = planned_files(info.siblings, args.max_bytes)
    total = sum(file["bytes"] for file in files)
    if shutil.disk_usage(Path.home()).free < total + 20 * 1024**3:
        raise RuntimeError("Insufficient filesystem headroom for the bounded model cache.")
    print(f"Pinned {info.id}@{info.sha}: {total} bytes; user quota must be checked separately.", flush=True)
    try:
        existing = Path(snapshot_download(args.model, revision=args.revision, local_files_only=True))
    except LocalEntryNotFoundError:
        existing = None
    cached = existing is not None and all(
        (existing / file["path"]).is_file()
        and (existing / file["path"]).stat().st_size == file["bytes"]
        for file in files
    )
    if cached:
        path = existing
    else:
        print("Required pinned files are missing; caching during development.", flush=True)
        path = Path(snapshot_download(
            args.model, revision=args.revision, allow_patterns=list(PATTERNS), max_workers=2,
        ))
    if not all(
        (path / file["path"]).is_file() and (path / file["path"]).stat().st_size == file["bytes"]
        for file in files
    ):
        raise RuntimeError("Model cache does not match the pinned size manifest.")
    result = {
        "model": info.id, "revision": info.sha, "path": str(path),
        "total_bytes": total, "files": files, "already_cached": cached,
        "complete": True, "inference_network_access": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "files"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
