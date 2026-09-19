"""HTTP gate supervision never mistakes a crashed candidate for readiness."""

import subprocess

import pytest

from http_benchmark import release_qualified, stop_owned_process, wait_ready


def test_http_gate_rejects_startup_failure():
    class FailedProcess:
        returncode = 1

        def poll(self):
            return self.returncode

    with pytest.raises(RuntimeError, match="exited during warm-up"):
        wait_ready(FailedProcess(), "http://unused")


def test_http_gate_kills_only_its_process_when_graceful_shutdown_stalls():
    class Process:
        def __init__(self):
            self.terminated = self.killed = False
            self.waits = 0

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("owned-api", timeout)
            return 0

    process = Process()
    stop_owned_process(process)
    assert process.terminated and process.killed and process.waits == 2


def test_publication_requires_every_release_gate():
    summary = {
        "complete": True, "full_corpus": True, "failed_conversations": 0,
        "timeouts": 0, "aborted": False, "latency_gate": True, "score": 0.8007,
    }
    assert release_qualified(summary, 0.799)
    for field, value in (
        ("complete", False), ("full_corpus", False), ("failed_conversations", 1),
        ("timeouts", 1), ("aborted", True), ("latency_gate", False), ("score", 0.7886),
    ):
        assert not release_qualified({**summary, field: value}, 0.799)
