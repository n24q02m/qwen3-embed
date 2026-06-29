from collections.abc import Iterable
from multiprocessing.sharedctypes import Synchronized as BaseValue
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from qwen3_embed.parallel_processor import (
    ParallelWorkerPool,
    PoolConfig,
    QueueSignals,
    Worker,
    _cleanup_worker,
    _get_items_from_queue,
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
                raise RuntimeError(f"Simulated failure on item {item}")
            yield idx, item * item


# --- Unit Tests ---


def test_parallel_worker_pool_basic():
    """Test ParallelWorkerPool with a simple SquareWorker."""
    config = PoolConfig(num_workers=2)
    pool = ParallelWorkerPool(worker=SquareWorker, config=config)

    items = [1, 2, 3, 4, 5]
    results = list(pool.ordered_map(items))

    assert sorted(results) == [1, 4, 9, 16, 25]


def test_parallel_worker_pool_semi_ordered():
    """Test semi_ordered_map returns items as they are processed."""
    config = PoolConfig(num_workers=2)
    pool = ParallelWorkerPool(worker=SquareWorker, config=config)

    items = [1, 2, 3, 4, 5]
    results = list(pool.semi_ordered_map(items))

    # semi_ordered_map yields (idx, result) tuples
    assert len(results) == 5
    result_values = [res for idx, res in results]
    assert sorted(result_values) == [1, 4, 9, 16, 25]


def test_parallel_worker_pool_error_handling():
    """
    Test that a failing worker puts QueueSignals.error on the queue
    and the main process subsequently raises a RuntimeError.
    """
    # Using FailingWorker with failure_val=2.
    # When item '2' is processed by the worker subprocess, it raises an exception.
    pool = ParallelWorkerPool(worker=FailingWorker, config=PoolConfig(num_workers=2))

    with pytest.raises(RuntimeError, match="Thread unexpectedly terminated"):
        # The list consumption forces the stream through the pool
        list(pool.ordered_map([1, 2, 3], failure_val=2))


class StartFailingWorker(Worker):
    @classmethod
    def start(cls, **kwargs: Any) -> "StartFailingWorker":
        raise RuntimeError("Initialization failed")

    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, Any]]:
        yield from items


def test_worker_start_exception_handling():
    """Test that _worker handles exceptions during Worker.start() gracefully."""
    input_queue = MagicMock()
    output_queue = MagicMock()
    num_active_workers = MagicMock()
    num_active_workers.get_lock.return_value.__enter__ = MagicMock()
    num_active_workers.get_lock.return_value.__exit__ = MagicMock()
    num_active_workers.value = 1
    worker_id = 0

    _worker(
        worker_class=StartFailingWorker,
        input_queue=input_queue,
        output_queue=output_queue,
        num_active_workers=num_active_workers,
        worker_id=worker_id,
        kwargs={},
    )

    # Verify that QueueSignals.error was put in the output queue
    output_queue.put.assert_called_with(QueueSignals.error)
    # Verify that the worker was cleaned up (num_active_workers decremented)
    assert num_active_workers.value == 0


class ProcessFailingWorker(Worker):
    @classmethod
    def start(cls, **kwargs: Any) -> "ProcessFailingWorker":
        return cls()

    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, Any]]:
        raise RuntimeError("Processing failed")


def test_worker_processing_exception_handling():
    """Test that _worker handles exceptions during worker.process() gracefully."""
    input_queue = MagicMock()
    output_queue = MagicMock()
    num_active_workers = MagicMock()
    num_active_workers.get_lock.return_value.__enter__ = MagicMock()
    num_active_workers.get_lock.return_value.__exit__ = MagicMock()
    num_active_workers.value = 1
    worker_id = 0

    # Mock _get_items_from_queue to return a single item then stop
    # This triggers worker.process() and then the exception.
    with patch("qwen3_embed.parallel_processor._get_items_from_queue", return_value=[(0, "item")]):
        _worker(
            worker_class=ProcessFailingWorker,
            input_queue=input_queue,
            output_queue=output_queue,
            num_active_workers=num_active_workers,
            worker_id=worker_id,
            kwargs={},
        )

    # Verify that QueueSignals.error was put in the output queue
    output_queue.put.assert_called_with(QueueSignals.error)
    # Verify that the worker was cleaned up (num_active_workers decremented)
    assert num_active_workers.value == 0


# --- Coverage Enhancement Tests ---


def test_worker_success_path():
    """Test that _worker successfully processes items in a normal flow."""
    input_queue = MagicMock()
    output_queue = MagicMock()
    num_active_workers = MagicMock()
    num_active_workers.get_lock.return_value.__enter__ = MagicMock()
    num_active_workers.get_lock.return_value.__exit__ = MagicMock()
    num_active_workers.value = 1
    worker_id = 0

    # Mock _get_items_from_queue to return items then stop
    # We yield (0, 10) then stop.
    with patch("qwen3_embed.parallel_processor._get_items_from_queue", return_value=[(0, 10)]):
        _worker(
            worker_class=SquareWorker,
            input_queue=input_queue,
            output_queue=output_queue,
            num_active_workers=num_active_workers,
            worker_id=worker_id,
            kwargs={},
        )

    # SquareWorker should have yielded (0, 100)
    output_queue.put.assert_called_with((0, 100))
    # Verify cleanup happened
    assert num_active_workers.value == 0


def test_worker_none_kwargs():
    """Test _worker handles kwargs=None (line 82)."""
    input_queue = MagicMock()
    output_queue = MagicMock()
    num_active_workers = MagicMock()
    num_active_workers.get_lock.return_value.__enter__ = MagicMock()
    num_active_workers.get_lock.return_value.__exit__ = MagicMock()
    num_active_workers.value = 1
    worker_id = 0

    with patch("qwen3_embed.parallel_processor._get_items_from_queue", return_value=[]):
        _worker(
            worker_class=SquareWorker,
            input_queue=input_queue,
            output_queue=output_queue,
            num_active_workers=num_active_workers,
            worker_id=worker_id,
            kwargs=None,
        )
    assert num_active_workers.value == 0


def test_get_items_from_queue_logic():
    """Test _get_items_from_queue generator and stop signal."""
    mock_queue = MagicMock()
    mock_queue.get.side_effect = [1, 2, QueueSignals.stop]

    items = list(_get_items_from_queue(mock_queue))
    assert items == [1, 2]
    assert mock_queue.get.call_count == 3


def test_cleanup_worker_logic():
    """Test _cleanup_worker functionality."""
    input_queue = MagicMock()
    output_queue = MagicMock()
    num_active_workers = MagicMock()
    num_active_workers.get_lock.return_value.__enter__ = MagicMock()
    num_active_workers.get_lock.return_value.__exit__ = MagicMock()
    num_active_workers.value = 5
    worker_id = 1

    _cleanup_worker(input_queue, output_queue, num_active_workers, worker_id)

    input_queue.close.assert_called_once()
    output_queue.close.assert_called_once()
    input_queue.join_thread.assert_called_once()
    output_queue.join_thread.assert_called_once()
    assert num_active_workers.value == 4


def test_parallel_worker_pool_default_config():
    """Test ParallelWorkerPool handles None config by creating a default one."""
    pool = ParallelWorkerPool(worker=SquareWorker, config=None)
    assert pool.num_workers == 1
    assert pool.worker_class == SquareWorker


def test_pool_with_device_ids():
    """Test ParallelWorkerPool.start with device_ids (lines 146-148)."""
    config = PoolConfig(num_workers=2, device_ids=[0, 1])
    pool = ParallelWorkerPool(worker=SquareWorker, config=config)
    pool.ctx = MagicMock()
    pool.ctx.Value.return_value = MagicMock(spec=BaseValue)
    pool.ctx.Process.return_value = MagicMock()

    pool.start(extra_param="value")

    assert pool.ctx.Process.call_count == 2
    # Check device_id propagation
    args1 = pool.ctx.Process.call_args_list[0][1]["args"]
    args2 = pool.ctx.Process.call_args_list[1][1]["args"]
    assert args1[5]["device_id"] == 0
    assert args2[5]["device_id"] == 1


def test_process_stream_error_signal():
    """Test that QueueSignals.error in _process_stream raises RuntimeError."""
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=1))
    pool.input_queue = MagicMock()
    pool.output_queue = MagicMock()
    pool.output_queue.get_nowait.return_value = QueueSignals.error
    pool.join_or_terminate = MagicMock()  # type: ignore[assignment]
    pool.check_worker_health = MagicMock()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="Thread unexpectedly terminated"):
        list(pool._process_stream([1]))

    pool.join_or_terminate.assert_called_once()  # type: ignore[attr-defined]


def test_join_or_terminate_mixed_states():
    """Test join_or_terminate with multiple processes in different states."""
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=3))

    # P1: Finishes immediately
    p1 = MagicMock()
    p1.is_alive.return_value = False

    # P2: Hangs and needs termination
    p2 = MagicMock()
    p2.is_alive.return_value = True

    # P3: Also hangs
    p3 = MagicMock()
    p3.is_alive.return_value = True

    pool.processes = [p1, p2, p3]

    pool.join_or_terminate(timeout=1)

    # P1 should have been joined but not terminated
    p1.join.assert_called_once_with(timeout=1)
    p1.terminate.assert_not_called()

    # P2 and P3 should have been joined AND terminated
    p2.join.assert_called_once_with(timeout=1)
    p2.terminate.assert_called_once()
    p3.join.assert_called_once_with(timeout=1)
    p3.terminate.assert_called_once()

    # Processes list should be cleared
    assert pool.processes == []


def test_join_or_terminate_empty():
    """Test join_or_terminate with no processes."""
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=0))
    pool.processes = []
    pool.join_or_terminate()
    assert pool.processes == []


def test_cleanup_worker_error_handling():
    """Test that _cleanup_worker handles exceptions during queue cleanup gracefully."""
    input_queue = MagicMock()
    output_queue = MagicMock()
    num_active_workers = MagicMock()
    num_active_workers.get_lock.return_value.__enter__ = MagicMock()
    num_active_workers.get_lock.return_value.__exit__ = MagicMock()
    num_active_workers.value = 5
    worker_id = 1

    # Mock input_queue.close to raise an exception
    input_queue.close.side_effect = Exception("Cleanup failed")

    with patch("logging.exception") as mock_log_exception:
        _cleanup_worker(input_queue, output_queue, num_active_workers, worker_id)

    # Verify that the exception was logged
    mock_log_exception.assert_called_once()
    assert "failed to cleanup queues" in mock_log_exception.call_args[0][0]

    # Verify that num_active_workers was still decremented despite the error
    assert num_active_workers.value == 4


def test_get_items_from_queue_error_handling():
    """Test that _get_items_from_queue propagates exceptions from queue.get()."""
    mock_queue = MagicMock()
    mock_queue.get.side_effect = Exception("Queue error")

    with pytest.raises(Exception, match="Queue error"):
        list(_get_items_from_queue(mock_queue))


def test_ordered_map_failing_stream():
    """Test that ordered_map handles a stream that raises an exception."""

    def failing_stream():
        yield 1
        yield 2
        raise ValueError("Stream failed")

    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=1))

    with pytest.raises(ValueError, match="Stream failed"):
        list(pool.ordered_map(failing_stream()))

    # Verify pool is cleaned up (processes list cleared)
    assert len(pool.processes) == 0
