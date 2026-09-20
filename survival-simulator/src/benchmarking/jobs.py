"""Bounded, CPU-only episode processes with owned-process watchdogs."""

from __future__ import annotations

import math
import multiprocessing
import time
import traceback
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from typing import Generic, TypeVar


J = TypeVar("J")
R = TypeVar("R")


class JobExecutionError(RuntimeError):
    def __init__(self, job, message: str, *, result=None):
        self.job = job
        self.result = result
        super().__init__(message)


@dataclass
class _Running(Generic[J]):
    job: J
    process: BaseProcess
    connection: Connection
    started: float


def _worker(connection: Connection, function: Callable, job) -> None:
    try:
        result = function(job)
        connection.send(("ok", result))
    except BaseException as error:
        connection.send((
            "error", f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
            getattr(error, "result", None),
        ))
    finally:
        connection.close()


def _stop(process: BaseProcess) -> None:
    if process.is_alive():
        process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)
    if process.is_alive():
        raise RuntimeError(f"Owned episode process {process.pid} did not stop.")
    process.close()


def run_jobs(
    jobs: Sequence[J], function: Callable[[J], R], *, workers: int,
    timeout_seconds: float, progress: Callable[[], None] | None = None,
) -> Iterator[tuple[J, R]]:
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("Episode workers must be between 1 and 32.")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Episode watchdog must be positive and finite.")
    from src.training.workers import _cpu_spawn_environment

    context = multiprocessing.get_context("spawn")
    pending = iter(jobs)
    active: dict[Connection, _Running[J]] = {}
    exhausted = False
    try:
        while active or not exhausted:
            while len(active) < workers and not exhausted:
                try:
                    job = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                receiver, sender = context.Pipe(duplex=False)
                process = context.Process(target=_worker, args=(sender, function, job))
                try:
                    with _cpu_spawn_environment():
                        process.start()
                except BaseException:
                    receiver.close()
                    sender.close()
                    if process.pid is not None:
                        _stop(process)
                    raise
                sender.close()
                active[receiver] = _Running(job, process, receiver, time.monotonic())
            if not active:
                break
            for connection in wait(list(active), timeout=0.5):
                running = active.pop(connection)
                try:
                    try:
                        reply = connection.recv()
                    except EOFError as error:
                        raise JobExecutionError(
                            running.job, f"Episode process {running.process.pid} exited without a result.",
                        ) from error
                    if reply[0] == "error":
                        raise JobExecutionError(running.job, reply[1], result=reply[2])
                    if reply[0] != "ok":
                        raise JobExecutionError(running.job, "Invalid episode worker reply.")
                    yield running.job, reply[1]
                finally:
                    connection.close()
                    running.process.join(timeout=1)
                    _stop(running.process)
            for running in active.values():
                if time.monotonic() - running.started > timeout_seconds:
                    raise JobExecutionError(
                        running.job,
                        f"Episode watchdog exceeded {timeout_seconds:g}s in owned process {running.process.pid}.",
                    )
            if progress is not None:
                progress()
    finally:
        for running in active.values():
            running.connection.close()
            _stop(running.process)
