from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import pytest

from watchdog.events import FileModifiedEvent, FileOpenedEvent, FileSystemEventHandler, LoggingEventHandler
from watchdog.observers.api import (
    BaseObserver,
    BatchErrorPolicy,
    EventDispatcher,
    EventEmitter,
    EventQueue,
    ObservedWatch,
)


def test_observer_constructor():
    ObservedWatch(Path("/foobar"), recursive=True)


def test_observer__eq__():
    watch1 = ObservedWatch("/foobar", recursive=True)
    watch2 = ObservedWatch("/foobar", recursive=True)
    watch_ne1 = ObservedWatch("/foo", recursive=True)
    watch_ne2 = ObservedWatch("/foobar", recursive=False)

    assert watch1 == watch2
    assert watch1.__eq__(watch2)
    assert not watch1.__eq__(watch_ne1)
    assert not watch1.__eq__(watch_ne2)


def test_observer__ne__():
    watch1 = ObservedWatch("/foobar", recursive=True)
    watch2 = ObservedWatch("/foobar", recursive=True)
    watch_ne1 = ObservedWatch("/foo", recursive=True)
    watch_ne2 = ObservedWatch("/foobar", recursive=False)

    assert not watch1.__ne__(watch2)
    assert watch1.__ne__(watch_ne1)
    assert watch1.__ne__(watch_ne2)


def test_observer__eq__ignores_follow_symlink():
    """Two watches that differ only in ``follow_symlink`` must not compare equal.

    ``ObservedWatch.key`` used to build its tuple from ``path``, ``is_recursive`` and
    ``event_filter`` only, omitting ``follow_symlink`` even though the latter is a
    real constructor parameter stored on the instance. ``event_filter`` was
    deliberately added to the key so that two watches on the same path with
    different filters stay independent; ``follow_symlink`` should get the same
    treatment, otherwise scheduling the same path twice with different
    ``follow_symlink`` values collapses onto a single watch (see
    ``test_observer_schedule_respects_second_follow_symlink`` below for the
    resulting observer-level bug).
    """
    watch_no_follow = ObservedWatch("/foobar", recursive=True, follow_symlink=False)
    watch_follow = ObservedWatch("/foobar", recursive=True, follow_symlink=True)

    assert watch_no_follow != watch_follow
    assert hash(watch_no_follow) != hash(watch_follow)


def test_observer_schedule_respects_second_follow_symlink():
    """A second ``schedule()`` call for the same path must honor its own follow_symlink.

    Because ``ObservedWatch`` equality used to ignore ``follow_symlink``, scheduling
    "/foobar" with ``follow_symlink=False`` and then again with
    ``follow_symlink=True`` makes the second watch compare equal to the first.
    ``BaseObserver.schedule()`` then treats an emitter as already existing for it
    and never creates one for the ``follow_symlink=True`` request: the emitter
    actually reachable from the returned watch still reports
    ``follow_symlink=False``, silently dropping the caller's request.
    """
    observer = BaseObserver(EventEmitter)
    handler = LoggingEventHandler()

    watch_no_follow = observer.schedule(handler, "/foobar", recursive=True, follow_symlink=False)
    watch_follow = observer.schedule(handler, "/foobar", recursive=True, follow_symlink=True)

    assert len(observer.emitters) == 2
    assert not observer._emitter_for_watch[watch_no_follow].watch.follow_symlink  # noqa: SLF001
    assert observer._emitter_for_watch[watch_follow].watch.follow_symlink  # noqa: SLF001


def test_observer__repr__():
    observed_watch = ObservedWatch("/foobar", recursive=True)
    repr_str = "<ObservedWatch: path='/foobar', is_recursive=True>"
    assert observed_watch.__repr__() == repr(observed_watch)
    assert repr(observed_watch) == repr_str

    observed_watch = ObservedWatch("/foobar", recursive=False, event_filter=[FileOpenedEvent, FileModifiedEvent])
    repr_str = "<ObservedWatch: path='/foobar', is_recursive=False, event_filter=FileModifiedEvent|FileOpenedEvent>"
    assert observed_watch.__repr__() == repr(observed_watch)
    assert repr(observed_watch) == repr_str


def test_event_emitter():
    event_queue = EventQueue()
    watch = ObservedWatch("/foobar", recursive=True)
    event_emitter = EventEmitter(event_queue, watch, timeout=1)
    event_emitter.queue_event(FileModifiedEvent("/foobar/blah"))


def test_event_dispatcher():
    event = FileModifiedEvent("/foobar")
    watch = ObservedWatch("/path", recursive=True)

    class TestableEventDispatcher(EventDispatcher):
        def dispatch_event(self, event, watch):
            assert True

    event_dispatcher = TestableEventDispatcher()
    event_dispatcher.event_queue.put((event, watch))
    event_dispatcher.start()
    time.sleep(1)
    event_dispatcher.stop()
    event_dispatcher.join()


def test_observer_basic():
    observer = BaseObserver(EventEmitter)
    handler = LoggingEventHandler()

    watch = observer.schedule(handler, "/foobar", recursive=True)
    observer.add_handler_for_watch(handler, watch)
    observer.add_handler_for_watch(handler, watch)
    observer.remove_handler_for_watch(handler, watch)
    with pytest.raises(KeyError):
        observer.remove_handler_for_watch(handler, watch)
    observer.unschedule(watch)
    with pytest.raises(KeyError):
        observer.unschedule(watch)

    watch = observer.schedule(handler, "/foobar", recursive=True)
    observer.event_queue.put((FileModifiedEvent("/foobar"), watch))
    observer.start()
    time.sleep(1)
    observer.unschedule_all()
    observer.stop()
    observer.join()


def test_observer_marks_every_queued_item_done():
    """The observer must mark both the dispatched event and the stop sentinel done.

    ``BaseObserver.dispatch_events()`` calls ``event_queue.task_done()`` on the
    stop-sentinel path as well as after dispatching. Without the sentinel call a
    caller's ``event_queue.join()`` never returns, and nothing else here notices:
    the other tests only join the *thread*, which stops either way.
    """
    observer = BaseObserver(EventEmitter)
    handler = LoggingEventHandler()

    watch = observer.schedule(handler, "/foobar", recursive=True)
    observer.event_queue.put((FileModifiedEvent("/foobar"), watch))
    observer.start()
    time.sleep(1)
    observer.unschedule_all()
    observer.stop()
    observer.join()

    # Asserted rather than calling event_queue.join(), which would hang the run
    # instead of failing if either task_done() goes missing.
    assert observer.event_queue.unfinished_tasks == 0


def test_stale_event_does_not_resurrect_an_unscheduled_watch():
    """A watch removed by unschedule() must stay removed when a queued event lands.

    ``BaseObserver._handlers`` is a ``defaultdict``, so indexing it in
    ``dispatch_events()`` recreates whatever key it is given. An event queued
    before ``unschedule()`` is still dispatched afterwards, which put the removed
    watch back as an empty mapping that nothing ever cleans up.
    """
    observer = BaseObserver(EventEmitter)
    handler = LoggingEventHandler()

    watch = observer.schedule(handler, "/foobar", recursive=True)
    observer.event_queue.put((FileModifiedEvent("/foobar"), watch))
    observer.unschedule(watch)
    assert watch not in observer._handlers  # noqa: SLF001

    observer.dispatch_events(observer.event_queue)

    assert watch not in observer._handlers  # noqa: SLF001


class BatchRecordingHandler:
    def __init__(self, name, batches):
        self.name = name
        self.batches = batches

    def dispatch_batch(self, batch):
        self.batches.append((self.name, tuple(event.src_path for event in batch)))
        assert isinstance(batch, tuple)


def test_batch_dispatch_uses_snapshot_until_next_batch():
    observer = BaseObserver(EventEmitter, batch_size=2, batch_timeout=None)
    watch = ObservedWatch("/foobar", recursive=True)
    batches = []

    class ReplacingHandler(BatchRecordingHandler):
        def dispatch_batch(self, batch):
            super().dispatch_batch(batch)
            observer.remove_handler_for_watch(old_second, watch)
            observer.add_handler_for_watch(new_second, watch)

    first = ReplacingHandler("first", batches)
    old_second = BatchRecordingHandler("old", batches)
    new_second = BatchRecordingHandler("new", batches)

    observer.add_handler_for_watch(first, watch)
    observer.add_handler_for_watch(old_second, watch)
    observer.event_queue.put((FileModifiedEvent("/one"), watch))
    observer.event_queue.put((FileModifiedEvent("/two"), watch))
    observer.event_queue.put((FileModifiedEvent("/three"), watch))
    observer.event_queue.put((FileModifiedEvent("/four"), watch))

    observer.dispatch_events(observer.event_queue)
    assert batches == [
        ("first", ("/one", "/two")),
        ("old", ("/one", "/two")),
    ]

    observer.dispatch_events(observer.event_queue)
    assert batches == [
        ("first", ("/one", "/two")),
        ("old", ("/one", "/two")),
        ("first", ("/three", "/four")),
        ("new", ("/three", "/four")),
    ]
    assert observer.event_queue.unfinished_tasks == 0


def test_batch_dispatch_preserves_order_and_read_only_batch():
    observer = BaseObserver(EventEmitter, batch_size=3, batch_timeout=None)
    watch = ObservedWatch("/foobar", recursive=True)

    class OrderHandler:
        def __init__(self):
            self.batches = []

        def dispatch_batch(self, batch):
            with pytest.raises(TypeError):
                batch[0] = FileModifiedEvent("/changed")
            self.batches.append(tuple(event.src_path for event in batch))

    handler = OrderHandler()
    observer.add_handler_for_watch(handler, watch)
    for index in range(3):
        observer.event_queue.put((FileModifiedEvent(f"/{index}"), watch))

    observer.dispatch_events(observer.event_queue)

    assert handler.batches == [("/0", "/1", "/2")]
    assert observer.event_queue.unfinished_tasks == 0


def test_batch_dispatch_continue_policy_logs_and_continues(caplog):
    observer = BaseObserver(EventEmitter, batch_size=2, batch_timeout=None)
    watch = ObservedWatch("/foobar", recursive=True)
    good_batches = []

    class FailingHandler:
        def dispatch_batch(self, batch):
            raise RuntimeError("boom")

    good = BatchRecordingHandler("good", good_batches)
    observer.add_handler_for_watch(FailingHandler(), watch)
    observer.add_handler_for_watch(good, watch)
    observer.event_queue.put((FileModifiedEvent("/one"), watch))
    observer.event_queue.put((FileModifiedEvent("/two"), watch))

    with caplog.at_level(logging.ERROR):
        observer.dispatch_events(observer.event_queue)

    assert good_batches == [("good", ("/one", "/two"))]
    assert "Unhandled exception while dispatching an event batch" in caplog.text
    assert observer.event_queue.unfinished_tasks == 0


def test_batch_dispatch_stop_policy_finishes_batch_and_join_returns():
    observer = BaseObserver(
        EventEmitter,
        batch_size=2,
        batch_timeout=None,
        batch_error_policy=BatchErrorPolicy.STOP,
    )
    watch = ObservedWatch("/foobar", recursive=True)
    good_batches = []

    class FailingHandler:
        def dispatch_batch(self, batch):
            raise RuntimeError("boom")

    good = BatchRecordingHandler("good", good_batches)
    observer.add_handler_for_watch(FailingHandler(), watch)
    observer.add_handler_for_watch(good, watch)
    observer.event_queue.put((FileModifiedEvent("/one"), watch))
    observer.event_queue.put((FileModifiedEvent("/two"), watch))

    observer.start()
    observer.event_queue.join()
    observer.join(timeout=2)

    assert not observer.is_alive()
    assert good_batches == [("good", ("/one", "/two"))]
    assert observer.event_queue.unfinished_tasks == 0


def test_batch_configuration_validation():
    with pytest.raises(TypeError, match="positive integer"):
        BaseObserver(EventEmitter, batch_size=True)
    with pytest.raises(ValueError, match="positive integer"):
        BaseObserver(EventEmitter, batch_size=0)
    with pytest.raises(ValueError, match="non-negative"):
        BaseObserver(EventEmitter, batch_size=1, batch_timeout=-0.1)


def test_batch_dispatch_finishes_current_batch_when_watch_is_unscheduled():
    observer = BaseObserver(EventEmitter, batch_size=2, batch_timeout=None)
    watch = observer.schedule(FileSystemEventHandler(), "/foobar", recursive=True)
    batches = []
    remaining = BatchRecordingHandler("remaining", batches)

    class UnschedulingHandler:
        def dispatch_batch(self, batch):
            observer.unschedule(watch)
            batches.append(("unscheduling", tuple(event.src_path for event in batch)))

    observer.add_handler_for_watch(UnschedulingHandler(), watch)
    observer.add_handler_for_watch(remaining, watch)
    observer.event_queue.put((FileModifiedEvent("/one"), watch))
    observer.event_queue.put((FileModifiedEvent("/two"), watch))

    observer.dispatch_events(observer.event_queue)

    assert batches == [
        ("unscheduling", ("/one", "/two")),
        ("remaining", ("/one", "/two")),
    ]
    assert not observer.emitters
    assert observer.event_queue.unfinished_tasks == 0


def test_batch_dispatch_waits_for_current_batch_on_stop():
    observer = BaseObserver(EventEmitter, batch_size=2, batch_timeout=None)
    watch = ObservedWatch("/foobar", recursive=True)
    callback_started = threading.Event()
    release_callback = threading.Event()
    batches = []

    class SlowHandler:
        def dispatch_batch(self, batch):
            callback_started.set()
            assert release_callback.wait(2)
            batches.append(tuple(event.src_path for event in batch))

    observer.add_handler_for_watch(SlowHandler(), watch)
    observer.event_queue.put((FileModifiedEvent("/one"), watch))
    observer.event_queue.put((FileModifiedEvent("/two"), watch))

    observer.start()
    assert callback_started.wait(2)
    observer.stop()
    release_callback.set()
    observer.join(timeout=2)

    assert not observer.is_alive()
    assert batches == [("/one", "/two")]
    assert observer.event_queue.unfinished_tasks == 0


def test_batch_timeout_requires_batch_size():
    with pytest.raises(ValueError, match="batch_timeout requires batch_size"):
        BaseObserver(EventEmitter, batch_timeout=0.1)


def test_unschedule_does_not_hold_subscription_lock_while_emitter_stops():
    class BlockingEmitter(EventEmitter):
        stop_started = threading.Event()
        release_stop = threading.Event()

        def stop(self):
            type(self).stop_started.set()
            assert type(self).release_stop.wait(2)
            super().stop()

    observer = BaseObserver(BlockingEmitter)
    watch = observer.schedule(FileSystemEventHandler(), "/foobar", recursive=True)
    unscheduled = threading.Event()

    def unschedule():
        observer.unschedule(watch)
        unscheduled.set()

    thread = threading.Thread(target=unschedule)
    observer.start()
    thread.start()
    assert BlockingEmitter.stop_started.wait(2)
    assert observer._lock.acquire(timeout=0.1)  # noqa: SLF001
    observer._lock.release()  # noqa: SLF001
    BlockingEmitter.release_stop.set()
    assert unscheduled.wait(2)
    thread.join()
    observer.stop()
    observer.join(timeout=2)

    assert not observer.is_alive()


def test_batch_dispatch_flushes_after_timeout():
    observer = BaseObserver(EventEmitter, batch_size=3, batch_timeout=0.01)
    watch = ObservedWatch("/foobar", recursive=True)
    batches = []
    handler = BatchRecordingHandler("handler", batches)

    observer.add_handler_for_watch(handler, watch)
    observer.event_queue.put((FileModifiedEvent("/one"), watch))

    observer.dispatch_events(observer.event_queue)

    assert batches == [("handler", ("/one",))]
    assert observer.event_queue.unfinished_tasks == 0
