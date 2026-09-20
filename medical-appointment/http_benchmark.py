"""Uncached, local HTTP acceptance gate for an isolated IDUN candidate.

This launches only a loopback endpoint on an owned socket. It does not create
a tunnel, change the live service, or contact the competition.
"""

import argparse
import json
import math
import os
import random
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

from answerers.llm_prompt import VARIANTS
from benchmark import write_json
from local_evaluator import replay
from utils import gold_evidence, group_questions_by_conversation

ROOT = Path(__file__).resolve().parent


def wait_ready(process, url, timeout=960.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Candidate API exited during warm-up ({process.returncode}).")
        try:
            response = requests.get(url + "/", timeout=2)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError("Candidate API did not become ready.")


def stop_owned_process(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def release_qualified(summary, minimum_score):
    return (
        summary["complete"] and summary["full_corpus"]
        and summary["failed_conversations"] == 0
        and summary["timeouts"] == 0 and not summary["aborted"]
        and summary["latency_gate"] and summary["score"] >= minimum_score
    )


def publish_and_hold(process, url, output, summary):
    """Expose only an already-qualified local process; never register an attempt."""
    binary = Path.home() / ".local" / "bin" / "cloudflared"
    if not binary.is_file():
        raise FileNotFoundError(f"Cloudflared is not installed at {binary}.")
    tunnel_log = output / "tunnel.log"
    tunnel = None
    try:
        with tunnel_log.open("w", encoding="utf-8") as log:
            tunnel = subprocess.Popen(
                [str(binary), "tunnel", "--url", url, "--no-autoupdate"],
                stdout=log, stderr=subprocess.STDOUT,
            )
            public = None
            verify_after = 0.0
            deadline = time.monotonic() + 600
            last_failure = None
            while time.monotonic() < deadline:
                if process.poll() is not None or tunnel.poll() is not None:
                    raise RuntimeError("An owned serving process exited before publication.")
                matches = re.findall(
                    r"https://[a-z0-9-]+\.trycloudflare\.com",
                    tunnel_log.read_text(encoding="utf-8", errors="replace"),
                )
                if matches:
                    if public != matches[-1]:
                        public = matches[-1]
                        verify_after = time.time() + 180
                        write_json(output / "publication_pending.json", {
                            "url": public, "job_id": os.environ.get("SLURM_JOB_ID"),
                            "local_score": summary["score"], "verified": False,
                            "verify_after": verify_after,
                        })
                    external_path = output / "external_verification.json"
                    if external_path.exists():
                        external = json.loads(external_path.read_text())
                        if (
                            external.get("url") == public
                            and external.get("root_status") == 200
                            and external.get("predict_status") == 200
                            and external.get("predictions_match") is True
                        ):
                            break
                    if time.time() < verify_after:
                        time.sleep(2)
                        continue
                    try:
                        response = requests.get(public + "/", timeout=5, allow_redirects=False)
                        failure = (
                            "Compute-node root is reachable; awaiting external prediction verification."
                            if response.status_code == 200 else f"HTTP {response.status_code}"
                        )
                    except requests.RequestException as exc:
                        failure = f"{type(exc).__name__}: {exc}"
                    if failure != last_failure:
                        print("PUBLIC_HEALTH_PENDING " + failure, flush=True)
                        last_failure = failure
                time.sleep(2)
            else:
                raise TimeoutError(
                    f"The qualified candidate's tunnel did not become ready: {last_failure}"
                )
            if public is None:
                raise RuntimeError("Cloudflared supplied no public hostname.")
            endpoint = {
                "url": public + "/predict", "local_url": url + "/predict",
                "api_pid": process.pid, "tunnel_pid": tunnel.pid,
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "local_score": summary["score"],
                "model": summary["arguments"]["model"],
                "revision": summary["arguments"]["revision"],
                "span_calibration": os.environ.get("MEDAPP_SPAN_CALIBRATION"),
                "official_attempt_queued": False,
            }
            write_json(output / "endpoint.json", endpoint)
            print("QUALIFIED_ENDPOINT " + json.dumps(endpoint, sort_keys=True), flush=True)
            while process.poll() is None and tunnel.poll() is None:
                time.sleep(5)
            raise RuntimeError("An owned serving process exited; stopping the release.")
    finally:
        if tunnel is not None:
            stop_owned_process(tunnel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--variant", default="base", choices=VARIANTS)
    parser.add_argument("--tokenization", choices=("template", "legacy"), default="legacy")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--output", default="results/http_benchmark")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--min-score", type=float)
    args = parser.parse_args()
    if os.name != "posix" or not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run this GPU acceptance gate in an isolated IDUN allocation.")
    if not (ROOT / "run_request.json").exists():
        parser.error("An isolated source snapshot is required.")
    if args.serve and (args.min_score is None or args.limit is not None):
        parser.error("--serve requires an explicit --min-score and the full corpus.")
    if args.min_score is not None and (
        not math.isfinite(args.min_score) or not 0.0 <= args.min_score <= 1.0
    ):
        parser.error("--min-score must be a finite value in [0, 1].")
    if args.serve:
        def terminate(signum, frame):
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, terminate)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.pop("MEDAPP_SKIP_WARMUP", None)
    environment.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "MEDAPP_CAPTURE": "0", "MEDAPP_ANSWERER": "llm",
        "MEDAPP_INFERENCE_WORKER": "1", "MEDAPP_REQUIRE_WARMUP": "1",
        "MEDAPP_LLM_MODEL": args.model, "MEDAPP_LLM_REVISION": args.revision,
        "MEDAPP_LLM_PROMPT": args.variant,
        "MEDAPP_LLM_LEGACY_SPECIAL_TOKENS": str(int(args.tokenization == "legacy")),
        "MEDAPP_LLM_MAX_NEW_TOKENS": str(args.max_new_tokens),
        "MEDAPP_ASR_CACHE": "0",
        "WHISPER_MODEL": "large-v3-turbo", "WHISPER_COMPUTE_TYPE": "int8",
    })
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    write_json(output / "summary.json", {"complete": False, "arguments": vars(args)})
    process = None
    try:
        with (output / "api.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    sys.executable, "-m", "uvicorn", "api:app",
                    "--fd", str(listener.fileno()), "--log-level", "info",
                ],
                cwd=ROOT, env=environment, pass_fds=(listener.fileno(),),
                stdout=log, stderr=subprocess.STDOUT,
            )
            listener.close()
            print(f"Candidate API pid={process.pid}, loopback port={port}; warming.", flush=True)
            wait_ready(process, url)
            groups = group_questions_by_conversation()
            if args.limit:
                groups = groups[:args.limit]
            if args.shuffle_seed is not None:
                rng = random.Random(args.shuffle_seed)
                reordered = []
                for filename, rows in groups:
                    shuffled = list(rows)
                    rng.shuffle(shuffled)
                    reordered.append((filename, shuffled))
                groups = reordered
            print(f"Replaying {len(groups)} conversations with ASR cache disabled.", flush=True)
            records = []

            def record_response(filename, rows, answers, spans, latency_ms, error):
                for row, prediction, span in zip(rows, answers, spans):
                    gold = gold_evidence(row)
                    records.append({
                        "question_id": row["question_id"],
                        "transcript_id": row["transcript_id"],
                        "question_type": row["question_type"], "label": int(row["label"]),
                        "prediction": prediction, "answer": prediction == 1,
                        "span": list(span) if span is not None else None,
                        "gold": list(gold) if gold is not None else None,
                        "audio_filename": filename, "request_latency_ms": latency_ms,
                        "request_error": error,
                    })
                write_json(output / "questions.json", records)

            stats = replay(
                url + "/predict", verbose=False, conversations=groups,
                on_response=record_response,
            )
            latency = sorted(value / 1000 for value in stats.latencies_ms)
            p95 = latency[min(len(latency) - 1, int(0.95 * len(latency)))] if latency else None
            maximum = max(latency) if latency else None
            summary = {
                "complete": stats.total == sum(len(rows) for _, rows in groups),
                "full_corpus": len(groups) == len(group_questions_by_conversation()),
                "arguments": vars(args), "score": stats.final_score,
                "accuracy": stats.accuracy, "mean_tiou": stats.mean_tiou,
                "questions": stats.total, "conversations": stats.conversations,
                "failed_conversations": stats.failed_conversations,
                "unsent_conversations": stats.unsent_conversations,
                "timeouts": stats.timeouts, "aborted": stats.aborted,
                "latency_p95_s": p95, "latency_max_s": maximum,
                "latencies_s": latency,
                "latency_gate": p95 is not None and p95 < 50 and maximum < 58,
                "by_type": dict(stats.by_type),
                "model_path": "owned-worker", "asr_cache_enabled": False,
            }
            write_json(output / "summary.json", summary)
            print(stats.report(), flush=True)
            print(json.dumps(summary, sort_keys=True), flush=True)
            failed = (
                stats.failed_conversations > 0 or stats.aborted or not summary["latency_gate"]
            )
            if args.serve:
                if not release_qualified(summary, args.min_score):
                    raise RuntimeError("Candidate did not pass release score/latency/contract gates.")
                log.flush()
                errors = re.findall(
                    r"ERROR|Traceback|Hard inference deadline|returning guesses|guessing no|"
                    r"worker is (?:busy|unavailable)|Cannot ground|Unusable LLM answer",
                    (output / "api.log").read_text(encoding="utf-8", errors="replace"),
                )
                if errors:
                    raise RuntimeError("Candidate logged runtime fallbacks; refusing publication.")
                publish_and_hold(process, url, output, summary)
            return int(failed)
    finally:
        listener.close()
        if process is not None:
            stop_owned_process(process)


if __name__ == "__main__":
    raise SystemExit(main())
