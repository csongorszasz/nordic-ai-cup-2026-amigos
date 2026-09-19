"""Supervise one session-scoped IDUN API and its temporary public tunnel."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener
import uuid


ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)
QUICK_URL = re.compile(r"https://[a-z0-9]+(?:-[a-z0-9]+)*\.trycloudflare\.com(?![a-z0-9.-])")
EXPECTED_STATS = {
    "device": "cuda:0",
    "detector_type": "yolo_standard",
    "tracker_type": "world_map",
    "policy_type": "hold",
    "inference_image_size": 3200,
    "inference_rect": True,
    "inference_half": True,
}


class RequestedShutdown(Exception):
    pass


def request_shutdown(signum, _frame):
    raise RequestedShutdown(f"Received signal {signum}")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_environment(weights: Path, nonce: str) -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith("DRONE_FLYBY_")
    }
    environment.update({
        "DRONE_FLYBY_DEVICE": "cuda:0",
        "DRONE_FLYBY_DETECTOR_TYPE": "yolo_standard",
        "DRONE_FLYBY_YOLO_WEIGHTS_PATH": str(weights),
        "DRONE_FLYBY_TRACKER_TYPE": "world_map",
        "DRONE_FLYBY_POLICY_TYPE": "hold",
        "DRONE_FLYBY_INFERENCE_IMAGE_SIZE": "3200",
        "DRONE_FLYBY_INFERENCE_RECT": "1",
        "DRONE_FLYBY_INFERENCE_HALF": "1",
        "DRONE_FLYBY_CONF_L0": "0.001",
        "DRONE_FLYBY_CONF_L1": "0.15",
        "DRONE_FLYBY_CONF_L2": "0.20",
        "DRONE_FLYBY_DETECTOR_MAX_DET": "300",
        "DRONE_FLYBY_DETECTOR_NMS_IOU": "0.7",
        "DRONE_FLYBY_RECORD_VALIDATION_DATA": "0",
        "DRONE_FLYBY_RUN_NONCE": nonce,
    })
    return environment


def verify_checkpoint(weights: Path, expected: str) -> str:
    with weights.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected.lower():
        raise ValueError(f"Checkpoint hash mismatch for {weights}: {actual}")
    return actual


def check_children(children: dict[str, subprocess.Popen]) -> None:
    for name, child in children.items():
        code = child.poll()
        if code is not None:
            raise RuntimeError(f"{name} exited with code {code}; inspect its serving log")


def wait_for_origin(children: dict[str, subprocess.Popen], port: int, nonce: str,
                    timeout: float = 180.0) -> dict:
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    last_error = "No readiness response"
    while time.monotonic() < deadline:
        check_children(children)
        try:
            with opener.open(f"http://127.0.0.1:{port}/stats", timeout=2) as response:
                stats = json.load(response)
        except (URLError, TimeoutError, ConnectionError) as error:
            last_error = str(error)
            time.sleep(0.25)
            continue
        expected = {**EXPECTED_STATS, "run_nonce": nonce}
        if not isinstance(stats, dict) or any(stats.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Port is serving an unexpected process or model configuration")
        return stats
    raise TimeoutError(f"API did not finish startup: {last_error}")


def wait_for_tunnel(children: dict[str, subprocess.Popen], log: Path,
                    timeout: float = 90.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        check_children(children)
        match = QUICK_URL.search(log.read_text(encoding="utf-8"))
        if match is not None:
            return match.group()
        time.sleep(0.25)
    raise TimeoutError(f"No Quick Tunnel URL was assigned; inspect {log}")


def stop_children(children: dict[str, subprocess.Popen], grace: float = 10.0) -> None:
    for child in children.values():
        if child.poll() is None:
            child.terminate()
    deadline = time.monotonic() + grace
    for child in children.values():
        try:
            child.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--port", type=int, default=9052)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    if re.fullmatch(r"[a-fA-F0-9]{64}", args.sha256) is None:
        parser.error("--sha256 must be a complete checkpoint SHA-256")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Serving requires an owned Slurm allocation")

    run = ROOT / "runs" / "serving"
    run.mkdir(exist_ok=False)
    manifest_path = run / "deployment.json"
    manifest = {
        "status": "starting", "started_at": now(),
        "job_id": os.environ["SLURM_JOB_ID"], "node": socket.gethostname(),
        "bind": f"0.0.0.0:{args.port}", "origin": f"http://127.0.0.1:{args.port}",
        "model_quality": "provisional-development-model-not-competition-validated",
    }
    children: dict[str, subprocess.Popen] = {}
    signals = [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]
    for signum in signals:
        signal.signal(signum, request_shutdown)
    exit_code = 1
    with ExitStack() as stack:
        try:
            weights = args.weights.resolve(strict=True)
            manifest["weights"] = str(weights)
            manifest["checkpoint_sha256"] = verify_checkpoint(weights, args.sha256)
            snapshot = ROOT / "snapshot.json"
            manifest["snapshot_sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            manifest["commit"] = json.loads(snapshot.read_text())["commit"]
            cloudflared = shutil.which("cloudflared")
            if cloudflared is None:
                raise FileNotFoundError("cloudflared is not available on the allocated node")
            manifest["cloudflared_version"] = subprocess.check_output(
                [cloudflared, "--version"], text=True, timeout=10,
            ).strip()
            job = subprocess.check_output(
                ["scontrol", "show", "job", manifest["job_id"], "-o"], text=True, timeout=10,
            )
            end_time = re.search(r"\bEndTime=(\S+)", job)
            if end_time is None:
                raise RuntimeError("Could not determine the serving allocation expiry")
            manifest["slurm_end_time"] = end_time.group(1)
            manifest["node_timezone"] = time.strftime("%z")
            with socket.socket() as probe:
                probe.bind(("0.0.0.0", args.port))

            nonce = uuid.uuid4().hex
            environment = model_environment(weights, nonce)
            manifest["run_nonce"] = nonce
            manifest["overrides"] = {
                key: value for key, value in environment.items() if key.startswith("DRONE_FLYBY_")
            }
            print(f"SERVING_JOB_ID={manifest['job_id']} NODE={manifest['node']}", flush=True)
            server_log = stack.enter_context((run / "server.log").open("w", encoding="utf-8"))
            children["api"] = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "api:app", "--app-dir", str(ROOT / "src"),
                 "--host", "0.0.0.0", "--port", str(args.port), "--workers", "1",
                 "--lifespan", "on", "--no-access-log"],
                cwd=ROOT, env=environment, stdout=server_log, stderr=subprocess.STDOUT,
            )
            manifest["api_pid"] = children["api"].pid
            write_manifest(manifest_path, manifest)
            manifest["startup_stats"] = wait_for_origin(children, args.port, nonce)
            print(f"ORIGIN_READY={manifest['origin']}", flush=True)

            tunnel_home = run / "cloudflared-home"
            tunnel_home.mkdir()
            tunnel_environment = {
                key: value for key, value in os.environ.items()
                if not key.startswith(("TUNNEL_", "CLOUDFLARED_"))
            }
            tunnel_environment["HOME"] = str(tunnel_home)
            tunnel_log_path = run / "cloudflared.log"
            tunnel_log = stack.enter_context(tunnel_log_path.open("w", encoding="utf-8"))
            children["cloudflared"] = subprocess.Popen(
                [cloudflared, "tunnel", "--no-autoupdate", "--protocol", "http2",
                 "--metrics", "127.0.0.1:0", "--url", manifest["origin"]],
                cwd=run, env=tunnel_environment, stdout=tunnel_log, stderr=subprocess.STDOUT,
            )
            manifest["cloudflared_pid"] = children["cloudflared"].pid
            manifest["predict_url"] = wait_for_tunnel(children, tunnel_log_path) + "/predict"
            manifest["status"] = "tunnel-assigned-awaiting-external-verification"
            write_manifest(manifest_path, manifest)
            print(f"TUNNEL_URL={manifest['predict_url']}", flush=True)
            while True:
                check_children(children)
                time.sleep(1)
        except RequestedShutdown as error:
            manifest.update(status="stopped", reason=str(error))
            LOGGER.info("%s; stopping owned serving processes", error)
            exit_code = 0
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            manifest.update(status="failed", error=str(error))
            LOGGER.exception("Serving failed; inspect %s", run)
        finally:
            for signum in signals:
                signal.signal(signum, signal.SIG_IGN)
            stop_children(children)
            manifest["stopped_at"] = now()
            write_manifest(manifest_path, manifest)
    return exit_code


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
