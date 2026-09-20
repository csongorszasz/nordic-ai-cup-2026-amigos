"""Ordered, synchronous collection from persistent CPU-only spawn workers."""

from __future__ import annotations

import math
import logging
import multiprocessing
import os
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from multiprocessing.process import BaseProcess

    from src.benchmarking.config import SimulationSettings
    from src.training.env import EnvironmentAdapter, EnvTransition
    from src.utils.DTOs import ActionRequest, StepResponse


_CHILD_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMEXPR_MAX_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}
_SPAWN_LOCK = threading.Lock()


@contextmanager
def _cpu_spawn_environment():
    # Spawn may re-import __main__ before reaching our target. Inheritance sets
    # native thread limits early enough even if that module imports NumPy.
    with _SPAWN_LOCK:
        original = {name: os.environ.get(name) for name in _CHILD_ENVIRONMENT}
        os.environ.update(_CHILD_ENVIRONMENT)
        try:
            yield
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


@dataclass(frozen=True)
class WorkerFailure:
    operation: str
    error_type: str
    message: str
    traceback: str


class RemoteWorkerError(RuntimeError):
    def __init__(self, worker_index: int, failure: WorkerFailure):
        self.worker_index = worker_index
        self.failure = failure
        self.operation = failure.operation
        self.error_type = failure.error_type
        self.remote_traceback = failure.traceback
        super().__init__(
            f"Environment worker {worker_index} failed during {failure.operation}: "
            f"{failure.error_type}: {failure.message}\n{failure.traceback}"
        )


class WorkerTimeoutError(TimeoutError):
    def __init__(self, worker_index: int, operation: str, phase: str, seconds: float):
        self.worker_index = worker_index
        self.operation = operation
        self.phase = phase
        super().__init__(
            f"Environment worker {worker_index} timed out {phase} {operation} "
            f"(batch timeout {seconds:g}s)."
        )


@dataclass(frozen=True)
class _Reply:
    operation: str
    value: object = None
    failure: WorkerFailure | None = None


@dataclass
class _Worker:
    connection: Connection
    process: BaseProcess
    expected_ids: tuple[int, ...] | None = None
    done: bool = False


def _send_reply(connection: Connection, reply: _Reply) -> bool:
    try:
        connection.send(reply)
    except (BrokenPipeError, ConnectionResetError, EOFError):
        # A collector abort closes every pipe, including workers still replying.
        return False
    return True


def _worker_main(
    connection: Connection, settings: dict, adapter_factory: Callable | None,
    action_repeat: int = 1,
):
    os.environ.update(_CHILD_ENVIRONMENT)
    operation = "initialize"
    try:
        from src.benchmarking.config import SimulationSettings

        if adapter_factory is None:
            from src.training.env import EnvironmentAdapter

            adapter_factory = EnvironmentAdapter
        resolved = SimulationSettings.model_validate(settings, strict=True)
        adapter = (
            adapter_factory(resolved)
            if action_repeat == 1
            else adapter_factory(resolved, action_repeat=action_repeat)
        )
        if not _send_reply(connection, _Reply("ready")):
            return
        while True:
            try:
                operation, payload = connection.recv()
            except EOFError:
                return
            if operation == "reset":
                value = adapter.reset(payload)
            elif operation == "step":
                value = adapter.step(payload)
            else:
                raise ValueError(f"Unknown environment worker operation: {operation!r}.")
            if not _send_reply(connection, _Reply(operation, value)):
                return
    except BaseException as exc:
        # The process boundary must report arbitrary native failures, including
        # SystemExit, rather than turn them into a successful or empty transition.
        failure = WorkerFailure(
            operation, type(exc).__name__, str(exc), traceback.format_exc(),
        )
        if not _send_reply(connection, _Reply(operation, failure=failure)):
            logging.getLogger(__name__).error(
                "Worker %s could not deliver %s failure after collector disconnected: %s: %s",
                os.getpid(), operation, failure.error_type, failure.message,
            )
            raise SystemExit(1) from None
    finally:
        connection.close()


class EnvironmentPool:
    """A blocking batch is one policy version; replies retain worker-index order.

    Use as a context manager, or call close explicitly. A remote/transport failure
    closes the entire pool: partial batches are never returned or retried.
    _adapter_factory is a spawn-picklable, module-level test seam, not a live core.
    """

    def __init__(
        self,
        workers: int,
        settings: SimulationSettings | None = None,
        *,
        timeout_seconds: float = 180,
        action_repeat: int = 1,
        _adapter_factory: Callable[[SimulationSettings], EnvironmentAdapter] | None = None,
    ):
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("Workers must be a positive integer.")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("Worker timeout must be a positive finite number of seconds.")
        if _adapter_factory is not None and not callable(_adapter_factory):
            raise TypeError("Adapter factory must be callable.")
        if (
            isinstance(action_repeat, bool) or not isinstance(action_repeat, int)
            or not 1 <= action_repeat <= 10
        ):
            raise ValueError("action_repeat must be an integer in [1, 10].")
        from src.benchmarking.config import SimulationSettings

        if settings is not None and not isinstance(settings, SimulationSettings):
            raise TypeError("Settings must be SimulationSettings or None.")
        settings = (
            SimulationSettings() if settings is None
            else SimulationSettings.model_validate(settings.model_dump(), strict=True)
        )
        self.workers = workers
        self.timeout_seconds = float(timeout_seconds)
        self.action_repeat = action_repeat
        self._workers: list[_Worker] = []
        self._closed = False
        self._io_thread: threading.Thread | None = None
        context = multiprocessing.get_context("spawn")
        try:
            with _cpu_spawn_environment():
                for index in range(workers):
                    parent, child = context.Pipe()
                    try:
                        process = context.Process(
                            target=_worker_main,
                            args=(
                                child, settings.model_dump(), _adapter_factory,
                                self.action_repeat,
                            ),
                            name=f"survival-environment-{index}",
                            daemon=False,
                        )
                        self._workers.append(_Worker(parent, process))
                        process.start()
                    except BaseException:
                        parent.close()
                        raise
                    finally:
                        child.close()
            self._exchange(tuple(range(workers)), "ready")
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> EnvironmentPool:
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Environment pool is closed.")
        if self._io_thread is not None:
            raise RuntimeError("Environment pool already has a batch in progress.")

    def _batch(self, values: Sequence, name: str) -> list:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            raise ValueError(f"{name} must be a sequence with one entry per worker.")
        batch = list(values)
        if len(batch) != self.workers:
            raise ValueError(
                f"{name} must have one entry per worker: expected {self.workers}, "
                f"received {len(batch)}."
            )
        return batch

    def reset(self, seeds: Sequence[int]) -> list[StepResponse]:
        self._ensure_open()
        from src.training.seeds import _validate_seed
        from src.utils.DTOs import StepResponse

        batch = [_validate_seed(seed) for seed in self._batch(seeds, "Seeds")]
        indices = tuple(range(self.workers))
        responses = self._exchange(indices, "reset", batch)
        if not all(isinstance(response, StepResponse) for response in responses):
            self.close()
            raise RuntimeError("Environment worker reset did not return a StepResponse.")
        for index, response in zip(indices, responses, strict=True):
            self._remember(index, response)
        return responses

    def reset_at(self, index: int, seed: int) -> StepResponse:
        self._ensure_open()
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.workers:
            raise ValueError("Worker index must be an integer in the pool's range.")
        from src.training.seeds import _validate_seed
        from src.utils.DTOs import StepResponse

        seed = _validate_seed(seed)
        response = self._exchange((index,), "reset", [seed])[0]
        if not isinstance(response, StepResponse):
            self.close()
            raise RuntimeError("Environment worker reset did not return a StepResponse.")
        self._remember(index, response)
        return response

    def step(self, actions: Sequence[Sequence[ActionRequest]]) -> list[EnvTransition]:
        self._ensure_open()
        batch = self._batch(actions, "Actions")
        from src.benchmarking.policies import validate_actions
        from src.training.env import EnvTransition

        validated = []
        for index, (worker, proposed) in enumerate(zip(self._workers, batch, strict=True)):
            if worker.expected_ids is None:
                raise RuntimeError(f"Reset environment worker {index} before stepping it.")
            if worker.done:
                raise RuntimeError(f"Environment worker {index} is terminal; reset it first.")
            validated.append(validate_actions(proposed, worker.expected_ids))
        responses = self._exchange(tuple(range(self.workers)), "step", validated)
        if not all(isinstance(response, EnvTransition) for response in responses):
            self.close()
            raise RuntimeError("Environment worker step did not return an EnvTransition.")
        for index, transition in enumerate(responses):
            self._remember(index, transition.observation)
        return responses

    def _remember(self, index: int, observation: StepResponse) -> None:
        worker = self._workers[index]
        worker.expected_ids = tuple(agent.agent_id for agent in observation.agent_status)
        worker.done = observation.game_status == "game_over"

    def _exchange(
        self, indices: tuple[int, ...], operation: str, payloads: list | None = None,
    ) -> list:
        self._ensure_open()
        results = []
        errors = []
        progress = [(indices[0], "receiving")]

        def exchange():
            try:
                if payloads is not None:
                    for index, payload in zip(indices, payloads, strict=True):
                        progress[0] = (index, "sending")
                        self._workers[index].connection.send((operation, payload))
                responses = []
                for index in indices:
                    progress[0] = (index, "receiving")
                    reply = self._workers[index].connection.recv()
                    if not isinstance(reply, _Reply):
                        raise RuntimeError(f"Environment worker {index} sent an invalid reply.")
                    if reply.failure is not None:
                        raise RemoteWorkerError(index, reply.failure)
                    if reply.operation != operation:
                        raise RuntimeError(
                            f"Environment worker {index} replied to {reply.operation}, "
                            f"not {operation}."
                        )
                    responses.append(reply.value)
                results.append(responses)
            except BaseException as exc:
                errors.append(exc)

        # Pipe send and recv can both block mid-message (poll alone is not a
        # complete-message timeout). A bounded I/O thread also covers large DTOs.
        thread = threading.Thread(target=exchange, name="survival-pool-io", daemon=True)
        self._io_thread = thread
        try:
            thread.start()
            thread.join(self.timeout_seconds)
            index, phase = progress[0]
            if thread.is_alive():
                raise WorkerTimeoutError(index, operation, phase, self.timeout_seconds)
            if errors:
                if isinstance(errors[0], (EOFError, OSError)):
                    raise RuntimeError(
                        f"Environment worker {index} disconnected while {phase} {operation}."
                    ) from errors[0]
                raise errors[0]
            return results[0]
        except BaseException:
            self.close()
            raise
        finally:
            if not thread.is_alive():
                self._io_thread = None

    def _join_workers(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        for worker in self._workers:
            if worker.process.pid is not None:
                worker.process.join(max(0.0, deadline - time.monotonic()))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # EOF is the graceful shutdown signal; sending "close" could itself
        # block behind a stuck simulation or a full pipe.
        for worker in self._workers:
            worker.connection.close()
        self._join_workers(min(self.timeout_seconds, 1.0))
        for worker in self._workers:
            if worker.process.is_alive():
                worker.process.terminate()
        self._join_workers(1.0)
        for worker in self._workers:
            if worker.process.is_alive():
                worker.process.kill()
        self._join_workers(1.0)
        if self._io_thread is not None and self._io_thread.ident is not None:
            self._io_thread.join(1.0)
        remaining = [
            worker.process.pid for worker in self._workers if worker.process.is_alive()
        ]
        for worker in self._workers:
            if not worker.process.is_alive():
                worker.process.close()
        if remaining or (self._io_thread is not None and self._io_thread.is_alive()):
            raise RuntimeError(f"Environment pool cleanup did not finish; live worker PIDs: {remaining}.")
