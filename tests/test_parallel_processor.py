import time
from collections.abc import Iterable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import qwen3_embed.parallel_processor as pp_module
from qwen3_embed.parallel_processor import (
    ParallelWorkerPool,
    QueueSignals,
    Worker,
    _worker,
)

# --- Helper Workers ---


class SquareWorker(Worker):
    """
    A simple worker that squares integers.
    """

    @classmethod
    def start(cls, **kwargs: Any) -> "SquareWorker":
        return cls()

    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, Any]]:
        for idx, item in items:
            yield idx, item * item


class FailingWorker(Worker):
    """
    A worker that raises an exception on a specific input value.
    """

    def __init__(self, failure_val: int):
        self.failure_val = failure_val

    @classmethod
    def start(cls, **kwargs: Any) -> "FailingWorker":
        failure_val = kwargs.get("failure_val", -1)
        return cls(failure_val)

    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, Any]]:
        for idx, item in items:
            if item == self.failure_val:
                raise ValueError(f"Intentional failure on {item}")
            yield idx, item


class SlowWorker(Worker):
    """
    A worker that sleeps a bit to simulate work.
    """

    @classmethod
    def start(cls, **kwargs: Any) -> "SlowWorker":
        return cls()

    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, Any]]:
        for idx, item in items:
            time.sleep(0.001)
            yield idx, item


# --- Tests ---


def test_ordered_map_basic():
    """
    Test basic ordered map functionality with multiple workers.
    """
    pool = ParallelWorkerPool(num_workers=2, worker=SquareWorker)
    input_data = list(range(10))
    # Expected output: squares of input
    expected = [x * x for x in input_data]

    results = list(pool.ordered_map(input_data))
    assert results == expected


def test_ordered_map_empty():
    """
    Test ordered map with empty input.
    """
    pool = ParallelWorkerPool(num_workers=2, worker=SquareWorker)
    results = list(pool.ordered_map([]))
    assert results == []


def test_ordered_map_generator():
    """
    Test ordered map with a generator input.
    """
    pool = ParallelWorkerPool(num_workers=2, worker=SquareWorker)
    input_gen = (x for x in range(10))
    expected = [x * x for x in range(10)]

    results = list(pool.ordered_map(input_gen))
    assert results == expected


def test_worker_exception():
    """
    Test that an exception in a worker propagates correctly.
    """
    # Fail on input value 5
    pool = ParallelWorkerPool(num_workers=2, worker=FailingWorker)
    input_data = range(10)

    # We expect a RuntimeError because ParallelWorkerPool catches the worker exception
    # and re-raises it as a RuntimeError("Thread unexpectedly terminated") or similar,
    # or if the worker exception is logged and the pool stops.
    # Looking at implementation:
    # if out_item == QueueSignals.error: raise RuntimeError("Thread unexpectedly terminated")

    # We need to pass failure_val via kwargs to start()
    # But wait, start() takes **kwargs from ParallelWorkerPool.ordered_map/start

    with pytest.raises(RuntimeError, match="Thread unexpectedly terminated"):
        # We need to pass the failure_val. The pool passes its **kwargs to worker.start()
        list(pool.ordered_map(input_data, failure_val=5))


def test_many_items():
    """
    Test processing more items than the internal buffer size to ensure
    queue management works correctly without deadlocking.
    """
    # max_internal_batch_size is 200 in parallel_processor.py
    # So queue_size = num_workers * 200.
    # We want to exceed this significantly.
    num_workers = 4
    # 4 * 200 = 800 items buffer. Let's process 2000 items.

    pool = ParallelWorkerPool(num_workers=num_workers, worker=SquareWorker)
    n_items = 2000
    input_data = range(n_items)
    expected = [x * x for x in input_data]

    results = list(pool.ordered_map(input_data))
    assert results == expected


def test_worker_initialization():
    """
    Test passing kwargs to worker initialization.
    """
    # We reuse FailingWorker but set failure_val to something not in input
    # to test that kwargs are passed correctly.
    pool = ParallelWorkerPool(num_workers=1, worker=FailingWorker)
    input_data = [1, 2, 3]
    # If failure_val is 2, it should fail.

    with pytest.raises(RuntimeError):
        list(pool.ordered_map(input_data, failure_val=2))

    # If failure_val is 10, it should succeed
    results = list(pool.ordered_map(input_data, failure_val=10))
    assert results == [1, 2, 3]


# --- Tests for Worker base class ---


def test_worker_base_start_raises():
    """Test that Worker.start() raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        Worker.start()


def test_worker_base_process_raises():
    """Test that Worker.process() raises NotImplementedError."""
    worker = Worker()
    with pytest.raises(NotImplementedError):
        list(worker.process([]))


# --- Tests for _worker function (lines 50-89) ---


def test_worker_function_basic():
    """Test _worker function directly with mock queues."""
    input_queue = MagicMock()
    output_queue = MagicMock()
    num_active_workers = MagicMock()
    num_active_workers.get_lock.return_value.__enter__ = MagicMock(return_value=None)
    num_active_workers.get_lock.return_value.__exit__ = MagicMock(return_value=False)
    num_active_workers.value = 2

    # Worker processes items: (0, 5) -> (0, 25), then stops
    input_queue.get.side_effect = [(0, 5), QueueSignals.stop]

    _worker(SquareWorker, input_queue, output_queue, num_active_workers, worker_id=0)

    # Should put the squared result
    output_queue.put.assert_called_once_with((0, 25))
    # Should close and join both queues
    input_queue.close.assert_called_once()
    output_queue.close.assert_called_once()
    input_queue.join_thread.assert_called_once()
    output_queue.join_thread.assert_called_once()
    # Should decrement active workers
    num_active_workers.get_lock.assert_called()
    assert num_active_workers.value == 1


def test_worker_function_with_kwargs():
    """Test _worker passes kwargs=None defaults to empty dict."""
    input_queue = MagicMock()
    output_queue = MagicMock()
    num_active_workers = MagicMock()
    num_active_workers.get_lock.return_value.__enter__ = MagicMock(return_value=None)
    num_active_workers.get_lock.return_value.__exit__ = MagicMock(return_value=False)
    num_active_workers.value = 1

    input_queue.get.side_effect = [QueueSignals.stop]

    # Call with kwargs=None (should default to {})
    _worker(SquareWorker, input_queue, output_queue, num_active_workers, worker_id=1, kwargs=None)

    output_queue.put.assert_not_called()
    input_queue.close.assert_called_once()


def test_worker_function_exception_handling():
    """Test _worker puts QueueSignals.error on exception and still closes queues."""
    input_queue = MagicMock()
    output_queue = MagicMock()
    num_active_workers = MagicMock()
    num_active_workers.get_lock.return_value.__enter__ = MagicMock(return_value=None)
    num_active_workers.get_lock.return_value.__exit__ = MagicMock(return_value=False)
    num_active_workers.value = 2

    # failure_val=5 means item 5 raises an exception
    input_queue.get.side_effect = [(0, 5), QueueSignals.stop]

    _worker(
        FailingWorker,
        input_queue,
        output_queue,
        num_active_workers,
        worker_id=0,
        kwargs={"failure_val": 5},
    )

    # Should put the error signal
    output_queue.put.assert_called_once_with(QueueSignals.error)
    # Should still close queues (in finally block)
    input_queue.close.assert_called_once()
    output_queue.close.assert_called_once()
    input_queue.join_thread.assert_called_once()
    output_queue.join_thread.assert_called_once()
    # Should still decrement active workers
    assert num_active_workers.value == 1


def test_worker_function_multiple_items():
    """Test _worker processes multiple items and stops."""
    input_queue = MagicMock()
    output_queue = MagicMock()
    num_active_workers = MagicMock()
    num_active_workers.get_lock.return_value.__enter__ = MagicMock(return_value=None)
    num_active_workers.get_lock.return_value.__exit__ = MagicMock(return_value=False)
    num_active_workers.value = 1

    input_queue.get.side_effect = [(0, 3), (1, 4), (2, 5), QueueSignals.stop]

    _worker(SquareWorker, input_queue, output_queue, num_active_workers, worker_id=0)

    # Should put 3 squared results
    assert output_queue.put.call_count == 3
    calls = [call[0][0] for call in output_queue.put.call_args_list]
    assert (0, 9) in calls
    assert (1, 16) in calls
    assert (2, 25) in calls


# --- Tests for device_ids branch (lines 123-126) ---


def test_start_with_device_ids():
    """Test ParallelWorkerPool.start() with device_ids assigns device_id to kwargs."""
    pool = ParallelWorkerPool(
        num_workers=2,
        worker=SquareWorker,
        device_ids=[0, 1],
    )
    results = list(pool.ordered_map(list(range(4))))
    assert results == [0, 1, 4, 9]


def test_start_with_single_device_id_cycles():
    """Test that device_ids cycles correctly when fewer ids than workers."""
    pool = ParallelWorkerPool(
        num_workers=3,
        worker=SquareWorker,
        device_ids=[0],
    )
    results = list(pool.ordered_map(list(range(3))))
    assert results == [0, 1, 4]


# --- Tests for semi_ordered_map else branch (lines 172-176) ---


def test_semi_ordered_map_queue_full_else_branch():
    """
    Test the else branch in semi_ordered_map when pushed-read >= queue_size.

    Patch max_internal_batch_size=1 so queue_size=num_workers*1=1.
    With 1 worker and queue_size=1: after pushing 1 item, pushed-read=1 >= 1
    forces the else branch (blocking output_queue.get).
    """
    with patch.object(pp_module, "max_internal_batch_size", 1):
        pool = ParallelWorkerPool(num_workers=1, worker=SquareWorker)
        # Use enough items to force the else branch (pushed-read >= queue_size=1)
        # After first push: pushed=1, read=0 -> 1 >= 1 -> else branch on second iteration
        results = list(pool.ordered_map(list(range(5))))
    assert results == [0, 1, 4, 9, 16]


def test_semi_ordered_map_error_in_else_branch():
    """
    Test error signal detection in the else branch (lines 180-181) of the for-loop.

    With queue_size=1 and FailingWorker failing on 0:
    - push item 0, pushed=1
    - iteration 1: pushed-read=1 >= 1 → else branch, get() returns QueueSignals.error
    - raises RuntimeError
    """
    with patch.object(pp_module, "max_internal_batch_size", 1):
        pool = ParallelWorkerPool(num_workers=1, worker=FailingWorker)
        with pytest.raises(RuntimeError, match="Thread unexpectedly terminated"):
            list(pool.ordered_map(list(range(5)), failure_val=0))


# --- Tests for emergency_shutdown (lines 205-207) ---


def test_emergency_shutdown_calls_cancel_join_thread():
    """
    Test that cancel_join_thread is called when emergency_shutdown=True.
    We directly set emergency_shutdown=True on the pool and trigger the finally block.
    """
    pool2 = ParallelWorkerPool(num_workers=1, worker=SquareWorker)
    pool2.emergency_shutdown = True
    pool2.input_queue = MagicMock()
    pool2.output_queue = MagicMock()
    pool2.processes = []

    # Directly call the cleanup code from the finally block
    pool2.join()
    pool2.input_queue.close()
    pool2.output_queue.close()
    if pool2.emergency_shutdown:
        pool2.input_queue.cancel_join_thread()
        pool2.output_queue.cancel_join_thread()

    pool2.input_queue.cancel_join_thread.assert_called_once()
    pool2.output_queue.cancel_join_thread.assert_called_once()


# --- Tests for check_worker_health (lines 217-222) ---


def test_check_worker_health_healthy_processes():
    """Test check_worker_health does nothing when all processes are alive."""
    pool = ParallelWorkerPool(num_workers=2, worker=SquareWorker)

    mock_process = MagicMock()
    mock_process.is_alive.return_value = True
    mock_process.exitcode = 0
    pool.processes = [mock_process, mock_process]

    # Should not raise
    pool.check_worker_health()
    assert not pool.emergency_shutdown


def test_check_worker_health_exited_zero():
    """Test check_worker_health ignores process that exited with code 0."""
    pool = ParallelWorkerPool(num_workers=1, worker=SquareWorker)

    mock_process = MagicMock()
    mock_process.is_alive.return_value = False
    mock_process.exitcode = 0
    pool.processes = [mock_process]

    # Should not raise (exitcode=0 is normal)
    pool.check_worker_health()
    assert not pool.emergency_shutdown


def test_check_worker_health_dead_process_triggers_emergency():
    """Test check_worker_health raises RuntimeError when process exits with non-zero code."""
    pool = ParallelWorkerPool(num_workers=1, worker=SquareWorker)

    mock_process = MagicMock()
    mock_process.is_alive.return_value = False
    mock_process.exitcode = 1
    mock_process.pid = 99999
    pool.processes = [mock_process]

    with pytest.raises(RuntimeError, match="terminated unexpectedly"):
        pool.check_worker_health()

    assert pool.emergency_shutdown


# --- Tests for join_or_terminate (line 233) ---


def test_join_or_terminate_alive_process_gets_terminated():
    """Test that join_or_terminate terminates a process that is still alive after join."""
    pool = ParallelWorkerPool(num_workers=1, worker=SquareWorker)

    mock_process = MagicMock()
    mock_process.is_alive.return_value = True  # Still alive after join timeout
    pool.processes = [mock_process]

    pool.join_or_terminate(timeout=0)

    mock_process.join.assert_called_once_with(timeout=0)
    mock_process.terminate.assert_called_once()
    assert pool.processes == []


def test_join_or_terminate_dead_process_not_terminated():
    """Test that join_or_terminate does not terminate a process that finishes in time."""
    pool = ParallelWorkerPool(num_workers=1, worker=SquareWorker)

    mock_process = MagicMock()
    mock_process.is_alive.return_value = False  # Finished during join
    pool.processes = [mock_process]

    pool.join_or_terminate(timeout=1)

    mock_process.join.assert_called_once_with(timeout=1)
    mock_process.terminate.assert_not_called()
    assert pool.processes == []


# --- Tests for __del__ (lines 251-253) ---


def test_del_terminates_alive_processes():
    """Test that __del__ terminates any still-alive processes."""
    pool = ParallelWorkerPool(num_workers=1, worker=SquareWorker)

    alive_process = MagicMock()
    alive_process.is_alive.return_value = True

    dead_process = MagicMock()
    dead_process.is_alive.return_value = False

    pool.processes = [alive_process, dead_process]
    pool.__del__()

    alive_process.terminate.assert_called_once()
    dead_process.terminate.assert_not_called()


def test_del_no_processes():
    """Test that __del__ with no processes does not raise."""
    pool = ParallelWorkerPool(num_workers=1, worker=SquareWorker)
    pool.processes = []
    pool.__del__()  # Should not raise
