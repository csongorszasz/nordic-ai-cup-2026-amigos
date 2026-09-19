import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("idun_serve", ROOT / "idun" / "serve.py")
serve = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(serve)


def test_serving_clears_inherited_model_recording_and_calibration_settings(monkeypatch):
    monkeypatch.setenv("DRONE_FLYBY_DETECTOR_TYPE", "dummy")
    monkeypatch.setenv("DRONE_FLYBY_INFERENCE_IMAGE_SIZES", "960,960,960")
    monkeypatch.setenv("DRONE_FLYBY_CALIBRATION_PATH", "unrelated.json")
    monkeypatch.setenv("DRONE_FLYBY_RECORD_VALIDATION_DATA", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    environment = serve.model_environment(Path("alpha.pt"), "owned-nonce")
    assert environment["DRONE_FLYBY_DETECTOR_TYPE"] == "yolo_standard"
    assert environment["DRONE_FLYBY_POLICY_TYPE"] == "hold"
    assert environment["DRONE_FLYBY_INFERENCE_IMAGE_SIZE"] == "3200"
    assert environment["DRONE_FLYBY_INFERENCE_HALF"] == "1"
    assert environment["DRONE_FLYBY_RECORD_VALIDATION_DATA"] == "0"
    assert environment["DRONE_FLYBY_RUN_NONCE"] == "owned-nonce"
    assert "DRONE_FLYBY_CALIBRATION_PATH" not in environment
    assert "DRONE_FLYBY_INFERENCE_IMAGE_SIZES" not in environment
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"


def test_checkpoint_hash_is_required_and_verified(tmp_path):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"owned checkpoint")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert serve.verify_checkpoint(weights, digest.upper()) == digest
    with pytest.raises(ValueError, match="hash mismatch"):
        serve.verify_checkpoint(weights, "0" * 64)
    with pytest.raises(FileNotFoundError):
        serve.verify_checkpoint(tmp_path / "absent.pt", digest)


@pytest.mark.parametrize("overrides,valid", [
    ({"run_nonce": "owned"}, True),
    ({"run_nonce": "unrelated"}, False),
    ({"run_nonce": "owned", "policy_type": "belief_voi"}, False),
])
def test_readiness_requires_correct_process_identity_and_configuration(overrides, valid):
    stats = {**serve.EXPECTED_STATS, **overrides}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/stats"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(stats).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        children = {"api": Mock(poll=Mock(return_value=None))}
        if valid:
            assert serve.wait_for_origin(children, server.server_port, "owned", timeout=2) == stats
        else:
            with pytest.raises(RuntimeError, match="unexpected process"):
                serve.wait_for_origin(children, server.server_port, "owned", timeout=2)
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def test_readiness_rejects_dead_child_and_reports_timeout():
    with pytest.raises(RuntimeError, match="api exited with code 1"):
        serve.wait_for_origin({"api": Mock(poll=Mock(return_value=1))}, 9052, "n")
    with pytest.raises(TimeoutError, match="did not finish startup"):
        serve.wait_for_origin({}, 9052, "n", timeout=0)


def test_tunnel_url_must_be_an_assigned_quick_tunnel_hostname(tmp_path):
    log = tmp_path / "cloudflared.log"
    log.write_text(
        "Docs: https://trycloudflare.com\n"
        "Unrelated: https://fake.trycloudflare.com.evil.invalid\n"
        "INF | https://owned-test-endpoint.trycloudflare.com |\n"
    )
    children = {"api": Mock(poll=Mock(return_value=None)), "cloudflared": Mock(poll=Mock(return_value=None))}
    assert serve.wait_for_tunnel(children, log) == "https://owned-test-endpoint.trycloudflare.com"
    children["api"].poll.return_value = 2
    with pytest.raises(RuntimeError, match="api exited"):
        serve.wait_for_tunnel(children, log)


def test_cleanup_terminates_and_reaps_all_owned_processes():
    children = {
        name: subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        for name in ("api", "cloudflared")
    }
    try:
        serve.stop_children(children, grace=2)
        assert all(child.poll() is not None for child in children.values())
    finally:
        serve.stop_children(children, grace=0)


def test_deployment_manifest_is_persisted_without_partial_json(tmp_path):
    path = tmp_path / "deployment.json"
    serve.write_manifest(path, {"status": "starting"})
    serve.write_manifest(path, {"status": "stopped", "job_id": "123"})
    assert json.loads(path.read_text()) == {"status": "stopped", "job_id": "123"}
    assert not path.with_suffix(".tmp").exists()


def test_main_rejects_login_node_execution_before_creating_artifacts(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(serve, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["serve.py", "--weights", "model.pt", "--sha256", "a" * 64])
    with pytest.raises(RuntimeError, match="owned Slurm allocation"):
        serve.main()
    assert not (tmp_path / "runs").exists()


def test_launcher_uses_attached_srun_and_preserves_api_default():
    launcher = (ROOT / "idun" / "serve.ps1").read_text()
    assert "exec srun --unbuffered" in launcher
    assert "'-tt'" in launcher
    assert "--constraint='gpu80g&(a100|h100)'" in launcher
    assert "--time=12:00:00" in launcher
    assert "--ntasks=1" in launcher
    assert "sbatch" not in launcher
    assert "PORT = 9053" in (ROOT / "src" / "api.py").read_text()
