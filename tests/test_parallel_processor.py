import time
from collections.abc import Iterable
from typing import Any

import pytest

from qwen3_embed.parallel_processor import ParallelWorkerPool, Worker

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
