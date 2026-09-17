import multiprocessing
import os
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.benchmarking.config import SimulationSettings
from src.training.workers import (
    EnvironmentPool, RemoteWorkerError, WorkerTimeoutError, _CHILD_ENVIRONMENT, _worker_main,
)
from src.utils.DTOs import ActionRequest, ObservationResponse, StepResponse


RESET_ERROR = 4_294_967_290
STEP_ERROR = 4_294_967_289
STEP_EXIT = 4_294_967_288
STEP_SLEEP = 4_294_967_287
TERMINAL = 777
SLOW = 100


def actions(order=(7, 2), distance=0.0):
    return [
        ActionRequest(agent_id=agent_id, move_distance=distance + offset,
                      move_direction=0.0, turn_angle=0.0, spawn_agent=False)
        for offset, agent_id in enumerate(order)
    ]


class FakeAdapter:
    """Importable under Windows spawn; no native world is ever constructed."""

    def __init__(self, settings):
        self.numpy_before_adapter = "numpy" in sys.modules
        from src.training.env import EnvTransition

        self.transition_type = EnvTransition
        self.settings = settings
        self.seed = None
        self.ticks = 0
        self.resets = 0
        self.order = []
        self.distances = []
        self.received_at = self.finished_at = 0.0

    def reset(self, seed):
        if seed == RESET_ERROR:
            raise ValueError("fixture reset failed")
        self.seed = seed
        self.ticks = 0
        self.resets += 1
        self.order = []
        self.distances = []
        return self._observation()

    def step(self, proposed):
        if self.seed == STEP_ERROR:
            raise RuntimeError("fixture step failed")
        if self.seed == STEP_EXIT:
            os._exit(23)
        if self.seed == STEP_SLEEP:
            time.sleep(30)
        self.received_at = time.monotonic()
        if self.seed == SLOW:
            time.sleep(0.4)
        self.ticks += 1
        self.order = [action.agent_id for action in proposed]
        self.distances = [action.move_distance for action in proposed]
        self.finished_at = time.monotonic()
        observation = self._observation()
        return self.transition_type(
            observation, 0.25, observation.game_status == "game_over",
        )

    def _observation(self):
        terminal = self.seed == TERMINAL and self.ticks > 0
        metadata = {
            "type": "Fixture",
            "pid": os.getpid(),
            "start_method": multiprocessing.get_start_method(),
            "seed": self.seed,
            "ticks": self.ticks,
            "resets": self.resets,
            "order": self.order,
            "distances": self.distances,
            "received_at": self.received_at,
            "finished_at": self.finished_at,
            "numpy_before_adapter": self.numpy_before_adapter,
            "torch_loaded": "torch" in sys.modules,
            "child_environment": {name: os.environ.get(name) for name in _CHILD_ENVIRONMENT},
            "dt": self.settings.dt,
            "time_limit": self.settings.time_limit,
        }
        return StepResponse(
            game_status="game_over" if terminal else "ok",
            score=float(self.seed) + self.ticks * 0.25,
            sim_time=(self.ticks + 1) * self.settings.dt,
            n_agents=0 if terminal else 2,
            agent_status=[] if terminal else [
                ObservationResponse(
                    agent_id=agent_id, energy=150, age=0.1, biome="forest",
                    speed=10, sprint_speed=20, hearing_radius=50, vision_angle=1,
                    vision_range=200, max_energy=500, observations=[metadata],
                )
                for agent_id in (7, 2)
            ],
        )


class FailingInitializationAdapter:
    def __init__(self, settings):
        raise ValueError("fixture initialization failed")


class SleepingInitializationAdapter:
    def __init__(self, settings):
        time.sleep(30)


class BlockedSend:
    def __init__(self, connection):
        self.connection = connection
        self.released = threading.Event()

    @property
    def closed(self):
        return self.connection.closed

    def send(self, value):
        self.released.wait(30)
        raise OSError("closed blocked write")

    def close(self):
        self.connection.close()
        self.released.set()


def metadata(observation):
    return observation.agent_status[0].observations[0]


class EnvironmentPoolTests(unittest.TestCase):
    def pool(self, workers=1, **kwargs):
        return EnvironmentPool(workers, timeout_seconds=15, _adapter_factory=FakeAdapter, **kwargs)

    def assert_closed(self, pool, pids=()):
        self.assertTrue(pool._closed)
        for worker in pool._workers:
            self.assertTrue(worker.connection.closed)
            self.assertTrue(worker.process._closed)
        self.assertTrue(pool._io_thread is None or not pool._io_thread.is_alive())
        active_pids = {process.pid for process in multiprocessing.active_children()}
        self.assertFalse(set(pids) & active_pids)

    def test_persistent_spawn_cpu_workers_and_parent_environment_restoration(self):
        with patch.dict(os.environ, {
            "OMP_NUM_THREADS": "7", "OPENBLAS_NUM_THREADS": "9", "CUDA_VISIBLE_DEVICES": "5",
        }):
            previous = {name: os.environ.get(name) for name in _CHILD_ENVIRONMENT}
            settings = SimulationSettings(dt=0.2, time_limit=12)
            with self.pool(2, settings=settings) as pool:
                responses = pool.reset([11, 22])
                records = [metadata(response) for response in responses]
                pids = [record["pid"] for record in records]
                self.assertEqual(len(set(pids)), 2)
                self.assertNotIn(os.getpid(), pids)
                for record in records:
                    self.assertEqual(record["start_method"], "spawn")
                    self.assertEqual(record["child_environment"], _CHILD_ENVIRONMENT)
                    self.assertFalse(record["numpy_before_adapter"])
                    self.assertFalse(record["torch_loaded"])
                    self.assertEqual((record["dt"], record["time_limit"]), (0.2, 12.0))
                self.assertEqual(
                    {name: os.environ.get(name) for name in _CHILD_ENVIRONMENT}, previous,
                )
                transitions = pool.step([actions((2, 7), 10), actions((7, 2), 20)])
                self.assertEqual(
                    [metadata(value.observation)["pid"] for value in transitions], pids,
                )
                self.assertEqual(
                    [metadata(value.observation)["order"] for value in transitions],
                    [[2, 7], [7, 2]],
                )
                self.assertEqual(
                    [metadata(value.observation)["distances"] for value in transitions],
                    [[10.0, 11.0], [20.0, 21.0]],
                )
                self.assertEqual([value.reward for value in transitions], [0.25, 0.25])
                reset = pool.reset_at(0, 33)
                self.assertEqual(metadata(reset)["pid"], pids[0])
                self.assertEqual(metadata(reset)["resets"], 2)
                values = pool.step([actions(), actions()])
                self.assertEqual(
                    [metadata(value.observation)["ticks"] for value in values], [1, 2],
                )
                self.assertEqual(
                    [metadata(value.observation)["resets"] for value in values], [2, 1],
                )
            self.assert_closed(pool, pids)
            self.assertEqual(
                {name: os.environ.get(name) for name in _CHILD_ENVIRONMENT}, previous,
            )
        pool.close()

    def test_entire_batch_is_sent_before_receiving_in_worker_index_order(self):
        with self.pool(2) as pool:
            pool.reset([SLOW, 200])
            values = pool.step([actions(), actions()])
            slow, fast = [metadata(value.observation) for value in values]
            self.assertEqual([slow["seed"], fast["seed"]], [SLOW, 200])
            self.assertLess(fast["received_at"], slow["finished_at"])
            self.assertLess(fast["finished_at"], slow["finished_at"])

    def test_invalid_batches_are_rejected_before_any_partial_send(self):
        with self.pool(2) as pool:
            pool.reset([11, 22])
            for seeds in (None, "12", iter([1, 2]), [1], [1, 2, 3], [1, True], [1, -1]):
                with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                    pool.reset(seeds)
            for batch in (
                None, "actions", iter([actions(), actions()]), [actions()],
                [actions(), actions(), actions()], [actions(), []],
                [actions(), actions((7, 7))], [actions(), actions((7, 99))],
                [actions(), actions(distance=float("nan"))],
                [actions(), [value.model_dump() for value in actions()]],
            ):
                with self.subTest(batch=batch), self.assertRaises(ValueError):
                    pool.step(batch)
            for index, seed in ((True, 1), (-1, 1), (2, 1), (0.5, 1), (0, True), (0, 2**32)):
                with self.subTest(index=index, seed=seed), self.assertRaises(ValueError):
                    pool.reset_at(index, seed)
            values = pool.step([actions(), actions()])
            self.assertEqual([metadata(value.observation)["ticks"] for value in values], [1, 1])
            self.assertEqual([metadata(value.observation)["resets"] for value in values], [1, 1])
            self.assertEqual([metadata(value.observation)["seed"] for value in values], [11, 22])

    def test_every_worker_must_be_reset_before_stepping(self):
        with self.pool(2) as pool:
            with self.assertRaisesRegex(RuntimeError, "Reset"):
                pool.step([actions(), actions()])
            pool.reset_at(0, 11)
            with self.assertRaisesRegex(RuntimeError, "worker 1"):
                pool.step([actions(), actions()])
            pool.reset_at(1, 22)
            values = pool.step([actions(), actions()])
            self.assertEqual([metadata(value.observation)["ticks"] for value in values], [1, 1])

    def test_terminal_workers_require_explicit_reset_without_advancing_other_workers(self):
        with self.pool(2) as pool:
            pool.reset([22, TERMINAL])
            values = pool.step([actions(), actions()])
            self.assertEqual([value.terminated for value in values], [False, True])
            with self.assertRaisesRegex(RuntimeError, "worker 1 is terminal"):
                pool.step([actions(), []])
            self.assertEqual(pool.reset_at(1, 44).game_status, "ok")
            values = pool.step([actions(), actions()])
            self.assertEqual([metadata(value.observation)["ticks"] for value in values], [2, 1])

    def test_returned_dto_mutation_cannot_change_worker_or_parent_action_identity(self):
        with self.pool() as pool:
            response = pool.reset([11])[0]
            response.game_status = "game_over"
            response.score = -100
            response.agent_status[0].agent_id = 999
            response.agent_status[0].observations.clear()
            response.agent_status.clear()
            value = pool.step([actions()])[0]
            self.assertFalse(value.terminated)
            self.assertEqual(value.observation.score, 11.25)
            self.assertEqual([agent.agent_id for agent in value.observation.agent_status], [7, 2])

    def test_remote_reset_error_is_structured_and_closes_all_workers(self):
        pool = self.pool(2)
        pids = [worker.process.pid for worker in pool._workers]
        with self.assertRaises(RemoteWorkerError) as caught:
            pool.reset([11, RESET_ERROR])
        error = caught.exception
        self.assertEqual(error.worker_index, 1)
        self.assertEqual(error.operation, "reset")
        self.assertEqual(error.error_type, "ValueError")
        self.assertEqual(error.failure.message, "fixture reset failed")
        self.assertIn("Traceback (most recent call last)", error.remote_traceback)
        self.assertIn("fixture reset failed", str(error))
        self.assert_closed(pool, pids)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            pool.reset([11, 22])

    def test_remote_step_error_is_not_a_zero_reward_or_partial_success(self):
        with self.pool(2) as pool:
            pool.reset([11, STEP_ERROR])
            pids = [worker.process.pid for worker in pool._workers]
            with self.assertRaises(RemoteWorkerError) as caught:
                pool.step([actions(), actions()])
            self.assertEqual(caught.exception.worker_index, 1)
            self.assertEqual(caught.exception.operation, "step")
            self.assertEqual(caught.exception.error_type, "RuntimeError")
            self.assertIn("fixture step failed", caught.exception.remote_traceback)
        self.assert_closed(pool, pids)

    def test_worker_crash_surfaces_disconnection_and_cleans_up(self):
        with self.pool() as pool:
            pool.reset([STEP_EXIT])
            pids = [worker.process.pid for worker in pool._workers]
            with self.assertRaisesRegex(RuntimeError, "worker 0 disconnected"):
                pool.step([actions()])
        self.assert_closed(pool, pids)

    def test_receive_timeout_terminates_only_owned_workers_and_joins_io(self):
        with self.pool() as pool:
            pool.reset([STEP_SLEEP])
            pids = [worker.process.pid for worker in pool._workers]
            started = time.monotonic()
            with patch.object(pool, "timeout_seconds", 0.1):
                with self.assertRaises(WorkerTimeoutError) as caught:
                    pool.step([actions()])
            self.assertEqual(caught.exception.worker_index, 0)
            self.assertEqual(caught.exception.operation, "step")
            self.assertEqual(caught.exception.phase, "receiving")
            self.assertLess(time.monotonic() - started, 5.0)
        self.assert_closed(pool, pids)

    def test_timeout_also_covers_a_blocked_pipe_write(self):
        with self.pool() as pool:
            pool.reset([11])
            pids = [worker.process.pid for worker in pool._workers]
            pool._workers[0].connection = BlockedSend(pool._workers[0].connection)
            with patch.object(pool, "timeout_seconds", 0.1):
                with self.assertRaises(WorkerTimeoutError) as caught:
                    pool.step([actions()])
            self.assertEqual(caught.exception.phase, "sending")
        self.assert_closed(pool, pids)

    def test_initialization_failure_is_reported_and_parent_environment_is_restored(self):
        previous = {name: os.environ.get(name) for name in _CHILD_ENVIRONMENT}
        pool = EnvironmentPool.__new__(EnvironmentPool)
        with self.assertRaises(RemoteWorkerError) as caught:
            pool.__init__(2, timeout_seconds=15, _adapter_factory=FailingInitializationAdapter)
        self.assertEqual(caught.exception.operation, "initialize")
        self.assertEqual(caught.exception.error_type, "ValueError")
        self.assertIn("fixture initialization failed", caught.exception.remote_traceback)
        self.assert_closed(pool)
        self.assertEqual({name: os.environ.get(name) for name in _CHILD_ENVIRONMENT}, previous)

    def test_initialization_timeout_cleans_up_spawned_processes(self):
        pool = EnvironmentPool.__new__(EnvironmentPool)
        started = time.monotonic()
        with self.assertRaises(WorkerTimeoutError):
            pool.__init__(1, timeout_seconds=0.2, _adapter_factory=SleepingInitializationAdapter)
        self.assertLess(time.monotonic() - started, 5.0)
        self.assert_closed(pool)

    def test_disconnected_failure_receiver_logs_the_original_error_and_exits_nonzero(self):
        connection = Mock()
        connection.send.side_effect = BrokenPipeError("collector disconnected")
        with patch.dict(os.environ, {}):
            with self.assertLogs("src.training.workers", level="ERROR") as logged:
                with self.assertRaises(SystemExit) as caught:
                    _worker_main(
                        connection, SimulationSettings().model_dump(), FailingInitializationAdapter,
                    )
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("ValueError: fixture initialization failed", logged.output[0])
        connection.close.assert_called_once_with()

    def test_context_does_not_swallow_the_callers_exception_and_close_is_idempotent(self):
        pool = self.pool()
        pids = [worker.process.pid for worker in pool._workers]
        with self.assertRaisesRegex(ValueError, "collector failed"):
            with pool:
                raise ValueError("collector failed")
        self.assert_closed(pool, pids)
        pool.close()
        for operation in (lambda: pool.reset([1]), lambda: pool.reset_at(0, 1),
                          lambda: pool.step([actions()]), pool.__enter__):
            with self.assertRaisesRegex(RuntimeError, "closed"):
                operation()

    def test_invalid_constructor_arguments_do_not_launch_workers(self):
        with patch("src.training.workers.multiprocessing.get_context") as context:
            for workers in (True, False, 0, -1, 1.5, "1", None):
                with self.subTest(workers=workers), self.assertRaises(ValueError):
                    EnvironmentPool(workers)
            for timeout in (True, False, 0, -1, float("inf"), float("nan"), "1", None):
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    EnvironmentPool(1, timeout_seconds=timeout)
            with self.assertRaises(TypeError):
                EnvironmentPool(1, settings={})
            with self.assertRaises(TypeError):
                EnvironmentPool(1, _adapter_factory=1)
            with self.assertRaises(ValueError):
                EnvironmentPool(1, SimulationSettings().model_copy(update={"starting_agents": True}))
        context.assert_not_called()

    def test_worker_module_import_is_lazy_for_numpy_engine_and_torch(self):
        code = (
            "import sys; import src.training.workers; "
            "assert 'numpy' not in sys.modules; "
            "assert 'src.core' not in sys.modules; "
            "assert 'torch' not in sys.modules"
        )
        subprocess.run(
            [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
            check=True, capture_output=True, text=True, timeout=15,
        )


if __name__ == "__main__":
    unittest.main()
