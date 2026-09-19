"""CPU-only transport diagnosis; no model, inference endpoint, or competition calls."""

import http.server
import json
import re
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path

import requests

from benchmark import write_json
from http_benchmark import stop_owned_process


def main():
    output = Path("results/tunnel_probe")
    output.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    incoming = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            incoming.append({"time": time.monotonic(), "path": self.path})
            body = json.dumps({"diagnostic_only": True, "nonce": nonce}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    local = f"http://127.0.0.1:{server.server_port}"
    process = None
    observations = []
    try:
        with (output / "tunnel.log").open("w") as log:
            process = subprocess.Popen(
                [str(Path.home() / ".local/bin/cloudflared"), "tunnel", "--url", local, "--no-autoupdate"],
                stdout=log, stderr=subprocess.STDOUT,
            )
            started = time.monotonic()
            public = None
            success = False
            while time.monotonic() - started < 360:
                if process.poll() is not None:
                    raise RuntimeError("Diagnostic tunnel exited.")
                matches = re.findall(
                    r"https://[a-z0-9-]+\.trycloudflare\.com",
                    (output / "tunnel.log").read_text(errors="replace"),
                )
                if not matches:
                    time.sleep(2)
                    continue
                public = matches[-1]
                observation = {"elapsed_s": round(time.monotonic() - started, 2)}
                try:
                    response = requests.get(public + "/", timeout=5)
                    observation["status"] = response.status_code
                    observation["body"] = response.text[:150]
                    success = response.status_code == 200 and response.json().get("nonce") == nonce
                except requests.RequestException as exc:
                    observation["error_type"] = type(exc).__name__
                    observation["error"] = str(exc)[:700]
                try:
                    observation["dns"] = sorted({
                        item[4][0] for item in socket.getaddrinfo(
                            public.removeprefix("https://"), 443, type=socket.SOCK_STREAM
                        )
                    })
                except socket.gaierror as exc:
                    observation["dns_error"] = str(exc)
                observations.append(observation)
                write_json(output / "observations.json", observations)
                write_json(output / "candidate.json", {"url": public, "nonce": nonce})
                print(json.dumps(observation), flush=True)
                if success:
                    break
                time.sleep(5)
            write_json(output / "summary.json", {
                "success": success, "url": public, "incoming_count": len(incoming),
                "elapsed_s": time.monotonic() - started, "observations": observations,
                "diagnostic_only": True,
            })
            return int(not success)
    finally:
        if process is not None:
            stop_owned_process(process)
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
