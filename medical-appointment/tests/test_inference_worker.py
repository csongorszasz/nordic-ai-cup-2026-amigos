"""Parent-side deadlines terminate owned work and recover without queueing."""

import time

from inference_worker import InferenceWorker


def fake_worker(connection):
    connection.send({"status": "ready"})
    try:
        while True:
            message = connection.recv()
            if message["request"].get("slow"):
                time.sleep(10)
            connection.send({
                "status": "result", "id": message["id"],
                "response": {"answers": [False], "evidence_start": [None], "evidence_end": [None]},
            })
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


def failed_worker(connection):
    connection.send({"status": "startup_failed"})
    connection.close()


def test_worker_returns_ready_result():
    worker = InferenceWorker(target=fake_worker, startup_timeout=10.0)
    try:
        worker.warm_up()
        result = worker.predict({}, deadline=time.monotonic() + 2.0)
        assert result["answers"] == [False]
    finally:
        worker.close()


def test_worker_kills_timed_out_request_and_recovers():
    worker = InferenceWorker(target=fake_worker, startup_timeout=10.0)
    try:
        worker.warm_up()
        started = time.monotonic()
        assert worker.predict({"slow": True}, deadline=started + 0.1) is None
        assert time.monotonic() - started < 2.0
        ready_by = time.monotonic() + 10
        while not worker.ready and time.monotonic() < ready_by:
            time.sleep(0.02)
        assert worker.ready
        result = worker.predict({}, deadline=time.monotonic() + 2.0)
        assert result["answers"] == [False]
    finally:
        worker.close()


def test_worker_requires_successful_warmup():
    import pytest

    worker = InferenceWorker(target=failed_worker, startup_timeout=10.0)
    try:
        with pytest.raises(RuntimeError, match="failed startup"):
            worker.warm_up()
        assert not worker.ready
    finally:
        worker.close()
