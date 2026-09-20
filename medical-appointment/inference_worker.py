"""A preloaded, owned inference process with a hard parent-side deadline."""

import logging
import multiprocessing
import threading
import time

logger = logging.getLogger(__name__)


def serve(connection):
    import os

    os.environ["MEDAPP_SKIP_WARMUP"] = "1"
    os.environ["MEDAPP_INFERENCE_WORKER"] = "0"
    from dtos import ASRQuestionRequestDto
    from example import _fallback, _predict, warm_models
    from utils import validate_response

    try:
        warm_models()
    except Exception:
        logger.exception("Inference worker warm-up failed.")
        connection.send({"status": "startup_failed"})
        connection.close()
        return
    connection.send({"status": "ready"})
    try:
        while True:
            message = connection.recv()
            request = ASRQuestionRequestDto.model_validate(message["request"])
            try:
                response = _predict(request, deadline=message["deadline"])
                validate_response(response, len(request.questions))
            except Exception:
                logger.exception("Inference worker request failed; returning no/null guesses.")
                response = _fallback(len(request.questions))
            connection.send({
                "status": "result", "id": message["id"],
                "response": response.model_dump(),
            })
    except (EOFError, BrokenPipeError):
        logger.info("Inference worker's owner disconnected.")
    finally:
        connection.close()


class InferenceWorker:
    def __init__(self, target=serve, startup_timeout=900.0):
        self._context = multiprocessing.get_context("spawn")
        self._target = target
        self._startup_timeout = startup_timeout
        self._lock = threading.Lock()
        self._process = None
        self._connection = None
        self._ready = False
        self._closed = False
        self._recovery = None
        self._request_id = 0

    @property
    def ready(self):
        process = self._process
        if not self._ready or process is None:
            return False
        try:
            return process.is_alive()
        except ValueError:
            return False

    def _stop(self):
        self._ready = False
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=0.2)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=0.2)
            if self._process.is_alive():
                raise RuntimeError("Owned inference process could not be stopped.")
            self._process.close()
            self._process = None

    def _start(self):
        if self._closed:
            raise RuntimeError("Inference worker is closed.")
        self._stop()
        self._connection, child = self._context.Pipe()
        self._process = self._context.Process(
            target=self._target, args=(child,), name="medapp-inference", daemon=True
        )
        self._process.start()
        child.close()
        deadline = time.monotonic() + self._startup_timeout
        try:
            while not self._closed and time.monotonic() < deadline:
                if self._connection.poll(min(0.2, max(0.0, deadline - time.monotonic()))):
                    message = self._connection.recv()
                    if message.get("status") != "ready":
                        raise RuntimeError(f"Inference worker failed startup: {message!r}")
                    self._ready = True
                    return
                if not self._process.is_alive():
                    raise RuntimeError("Inference worker exited during startup.")
            raise TimeoutError("Inference worker did not become ready.")
        except Exception:
            self._stop()
            raise

    def warm_up(self):
        with self._lock:
            if not self.ready:
                self._start()

    def _recover(self):
        try:
            self.warm_up()
        except Exception:
            logger.exception("Inference worker recovery failed; later requests will retry.")

    def _recover_async(self):
        if self._closed:
            return
        if self._recovery is None or not self._recovery.is_alive():
            self._recovery = threading.Thread(
                target=self._recover, name="medapp-recovery", daemon=True
            )
            self._recovery.start()

    def predict(self, request, *, deadline):
        if time.monotonic() >= deadline or self._closed:
            logger.warning("Request has no remaining inference budget.")
            return None
        if not self._lock.acquire(blocking=False):
            logger.warning("Inference worker is busy or recovering; returning guesses.")
            return None
        recover = False
        try:
            if not self.ready:
                logger.warning("Inference worker is unavailable; returning guesses.")
                recover = True
                return None
            self._request_id += 1
            self._connection.send({
                "id": self._request_id, "request": request, "deadline": deadline
            })
            remaining = max(0.0, deadline - time.monotonic())
            if not self._connection.poll(remaining):
                logger.error("Hard inference deadline exceeded; terminating the owned worker.")
                self._stop()
                recover = True
                return None
            message = self._connection.recv()
            if message.get("status") != "result" or message.get("id") != self._request_id:
                raise ValueError("Unexpected inference worker response.")
            return message["response"]
        except (EOFError, BrokenPipeError, OSError, ValueError, KeyError):
            logger.exception("Inference worker communication failed.")
            self._stop()
            recover = True
            return None
        finally:
            self._lock.release()
            if recover:
                self._recover_async()

    def close(self):
        self._closed = True
        with self._lock:
            self._stop()
