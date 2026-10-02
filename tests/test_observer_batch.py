from __future__ import annotations

import contextlib
import threading
import time
from typing import TYPE_CHECKING

import pytest

from watchdog.events import FileModifiedEvent, FileSystemEventHandler
from watchdog.observers.api import BaseObserver, BatchErrorPolicy, EventBatch, EventEmitter

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@pytest.fixture
def observer_factory() -> Iterator:
    observers = []

    def factory(**kwargs) -> BaseObserver:
        observer = BaseObserver(EventEmitter, **kwargs)
        observers.append(observer)
        return observer

    yield factory

    for observer in observers:
        observer.stop()
        with contextlib.suppress(RuntimeError):
            observer.join()


def wait_for(condition: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def test_invalid_batch_configuration() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        BaseObserver(EventEmitter, batch_size=0)
    with pytest.raises(ValueError, match="batch_max_wait"):
        BaseObserver(EventEmitter, batch_size=2, batch_max_wait=0)


def test_subscription_change_applies_at_next_batch(observer_factory) -> None:
    """A handler replacing another handler mid-batch: the current batch still
    uses the old subscription, the next batch uses the new one."""
    observer = observer_factory(batch_size=2, batch_max_wait=5.0)
    replaced = threading.Event()
    new_handler_done = threading.Event()
    old_seen: list[str] = []
    new_seen: list[str] = []

    class ReplacingHandler(FileSystemEventHandler):
        def on_modified(self, event) -> None:
            if not replaced.is_set():
                observer.remove_handler_for_watch(old_handler, watch)
                observer.add_handler_for_watch(new_handler, watch)
                replaced.set()

    class RecordingHandler(FileSystemEventHandler):
        def __init__(self, seen: list[str], done: threading.Event | None = None) -> None:
            self._seen = seen
            self._done = done

        def on_modified(self, event) -> None:
            self._seen.append(event.src_path)
            if self._done is not None and len(self._seen) == 2:
                self._done.set()

    old_handler = RecordingHandler(old_seen)
    new_handler = RecordingHandler(new_seen, done=new_handler_done)

    watch = observer.schedule(ReplacingHandler(), "")
    observer.add_handler_for_watch(old_handler, watch)

    # Enqueue two full batches up front so batching is deterministic.
    for src_path in ("a", "b", "c", "d"):
        observer.event_queue.put((FileModifiedEvent(src_path), watch))

    observer.start()
    try:
        assert new_handler_done.wait(timeout=5)
        assert replaced.is_set()
        # The first batch was dispatched with the subscription snapshot taken
        # at its start: the old handler still received it.
        assert old_seen == ["a", "b"]
        # The replacement took effect at the start of the next batch.
        assert new_seen == ["c", "d"]
    finally:
        observer.stop()
        observer.join()


def test_handler_receives_read_only_batch(observer_factory) -> None:
    observer = observer_factory(batch_size=2, batch_max_wait=5.0)
    received: list[EventBatch] = []
    done = threading.Event()

    class BatchHandler(FileSystemEventHandler):
        def dispatch_batch(self, batch) -> None:
            received.append(batch)
            done.set()

    handler = BatchHandler()
    watch = observer.schedule(handler, "")
    observer.event_queue.put((FileModifiedEvent("a"), watch))
    observer.event_queue.put((FileModifiedEvent("b"), watch))

    observer.start()
    try:
        assert done.wait(timeout=5)
        (batch,) = received
        assert isinstance(batch, EventBatch)
        assert len(batch) == 2
        # Enqueue order is preserved.
        assert [event.src_path for event in batch.events] == ["a", "b"]
        assert [event.src_path for event, _ in batch] == ["a", "b"]
        assert batch.watches == (watch, watch)
        # The batch is read-only.
        with pytest.raises(AttributeError):
            batch.append((FileModifiedEvent("c"), watch))
        with pytest.raises(TypeError):
            batch[0] = (FileModifiedEvent("c"), watch)
    finally:
        observer.stop()
        observer.join()


def test_partial_batch_dispatched_after_max_wait(observer_factory) -> None:
    observer = observer_factory(batch_size=10, batch_max_wait=0.2)
    seen: list[str] = []
    done = threading.Event()

    class RecordingHandler(FileSystemEventHandler):
        def on_modified(self, event) -> None:
            seen.append(event.src_path)
            if len(seen) == 2:
                done.set()

    watch = observer.schedule(RecordingHandler(), "")
    observer.event_queue.put((FileModifiedEvent("a"), watch))
    observer.event_queue.put((FileModifiedEvent("b"), watch))

    observer.start()
    try:
        # The batch never fills up; max_wait flushes it.
        assert done.wait(timeout=5)
        assert seen == ["a", "b"]
    finally:
        observer.stop()
        observer.join()


def test_handler_exception_continue_policy_records_error_and_join_completes(observer_factory) -> None:
    observer = observer_factory(batch_size=2, batch_max_wait=5.0, batch_error_policy=BatchErrorPolicy.CONTINUE)
    good_seen: list[str] = []
    good_done = threading.Event()

    class FailingHandler(FileSystemEventHandler):
        def dispatch_batch(self, batch) -> None:
            msg = "boom"
            raise RuntimeError(msg)

    class GoodHandler(FileSystemEventHandler):
        def on_modified(self, event) -> None:
            good_seen.append(event.src_path)
            if len(good_seen) == 2:
                good_done.set()

    failing_handler = FailingHandler()
    watch = observer.schedule(failing_handler, "")
    observer.add_handler_for_watch(GoodHandler(), watch)
    observer.event_queue.put((FileModifiedEvent("a"), watch))
    observer.event_queue.put((FileModifiedEvent("b"), watch))

    observer.start()
    try:
        assert good_done.wait(timeout=5)
        # CONTINUE policy: the failure was recorded and the remaining
        # handlers of the batch were still dispatched.
        assert good_seen == ["a", "b"]
        assert wait_for(lambda: len(observer.batch_errors) == 1)
        (error,) = observer.batch_errors
        assert error.handler is failing_handler
        assert isinstance(error.exception, RuntimeError)
        assert [event.src_path for event in error.batch.events] == ["a", "b"]

        # Every dequeued entry was acknowledged with task_done: the queue
        # bookkeeping is balanced and join() completes.
        joiner = threading.Thread(target=observer.event_queue.join, daemon=True)
        joiner.start()
        joiner.join(timeout=5)
        assert not joiner.is_alive()
    finally:
        observer.stop()
        # The observer thread ends even though a handler raised.
        observer.join()
    assert not observer.is_alive()


def test_handler_exception_abort_policy_stops_batch_dispatch(observer_factory) -> None:
    observer = observer_factory(batch_size=2, batch_max_wait=5.0, batch_error_policy=BatchErrorPolicy.ABORT)
    good_seen: list[str] = []

    class FailingHandler(FileSystemEventHandler):
        def dispatch_batch(self, batch) -> None:
            msg = "boom"
            raise RuntimeError(msg)

    class GoodHandler(FileSystemEventHandler):
        def on_modified(self, event) -> None:
            good_seen.append(event.src_path)

    failing_watch = observer.schedule(FailingHandler(), "failing")
    good_watch = observer.schedule(GoodHandler(), "good")
    # The failing watch appears first in the batch.
    observer.event_queue.put((FileModifiedEvent("a"), failing_watch))
    observer.event_queue.put((FileModifiedEvent("b"), good_watch))

    observer.start()
    try:
        assert wait_for(lambda: len(observer.batch_errors) == 1)
        # ABORT policy: dispatching the rest of the batch was skipped.
        assert good_seen == []
        (error,) = observer.batch_errors
        assert [event.src_path for event in error.batch.events] == ["a"]
    finally:
        observer.stop()
        observer.join()
    assert not observer.is_alive()


def test_stop_respects_batch_boundary(observer_factory) -> None:
    """A stop requested while a batch is being dispatched does not interrupt
    the batch: it completes with the subscriptions from its start."""
    observer = observer_factory(batch_size=2, batch_max_wait=5.0)
    dispatch_started = threading.Event()
    finish_dispatch = threading.Event()
    seen: list[str] = []

    class SlowHandler(FileSystemEventHandler):
        def on_modified(self, event) -> None:
            dispatch_started.set()
            assert finish_dispatch.wait(timeout=5)
            seen.append(event.src_path)

    watch = observer.schedule(SlowHandler(), "")
    observer.event_queue.put((FileModifiedEvent("a"), watch))
    observer.event_queue.put((FileModifiedEvent("b"), watch))

    observer.start()
    assert dispatch_started.wait(timeout=5)
    # stop() clears all subscriptions immediately, but the in-flight batch
    # still completes with the snapshot taken at its start.
    observer.stop()
    finish_dispatch.set()
    observer.join()
    assert seen == ["a", "b"]
    assert not observer.is_alive()
