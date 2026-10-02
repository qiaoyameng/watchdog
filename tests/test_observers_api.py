from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import pytest

from watchdog.events import (
    EventQueueOverflowedEvent,
    FileCreatedEvent,
    FileModifiedEvent,
    FileOpenedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
    LoggingEventHandler,
)
from watchdog.observers.api import BaseObserver, EventDispatcher, EventEmitter, EventQueue, ObservedWatch


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


def test_default_event_queue_remains_unbounded():
    event_queue = EventQueue()
    watch = ObservedWatch("/foobar", recursive=False)
    emitter = EventEmitter(event_queue, watch)

    for index in range(100):
        emitter.queue_event(FileModifiedEvent(f"/foobar/{index}"))

    assert event_queue.maxsize == 0
    assert event_queue.qsize() == 100
    assert not event_queue._active_overflowed_watches  # noqa: SLF001


def test_bounded_queue_coalesces_consecutive_overflows_for_a_watch():
    event_queue = EventQueue(maxsize=2)
    watch = ObservedWatch("/foobar", recursive=False)
    emitter = EventEmitter(event_queue, watch)

    emitter.queue_event(FileCreatedEvent("/foobar/a"))
    emitter.queue_event(FileModifiedEvent("/foobar/b"))
    emitter.queue_event(FileModifiedEvent("/foobar/c"))
    emitter.queue_event(FileModifiedEvent("/foobar/d"))
    event_queue.mark_overflow(watch)

    entries = [event_queue.get_nowait() for _ in range(3)]

    assert [event for event, _ in entries] == [
        FileCreatedEvent("/foobar/a"),
        FileModifiedEvent("/foobar/b"),
        EventQueueOverflowedEvent("/foobar", is_synthetic=True),
    ]
    assert {watch for _, watch in entries} == {watch}
    with pytest.raises(queue.Empty):
        event_queue.get_nowait()


def test_bounded_queue_deduplicates_without_reporting_overflow():
    event_queue = EventQueue(maxsize=2)
    watch = ObservedWatch("/foobar", recursive=False)
    emitter = EventEmitter(event_queue, watch)
    repeated_event = FileModifiedEvent("/foobar/a")

    emitter.queue_event(repeated_event)
    emitter.queue_event(repeated_event)
    emitter.queue_event(repeated_event)
    emitter.queue_event(repeated_event)

    entries = [event_queue.get_nowait()]

    assert [event for event, _ in entries] == [FileModifiedEvent("/foobar/a")]
    assert not event_queue._active_overflowed_watches  # noqa: SLF001
    with pytest.raises(queue.Empty):
        event_queue.get_nowait()


def test_bounded_queue_delivers_overflow_after_existing_events_then_accepts_more():
    event_queue = EventQueue(maxsize=2)
    watch = ObservedWatch("/foobar", recursive=False)
    emitter = EventEmitter(event_queue, watch)

    emitter.queue_event(FileCreatedEvent("/foobar/a"))
    emitter.queue_event(FileCreatedEvent("/foobar/b"))
    emitter.queue_event(FileCreatedEvent("/foobar/lost"))

    first_event, first_watch = event_queue.get_nowait()
    second_event, second_watch = event_queue.get_nowait()
    overflow_event, overflow_watch = event_queue.get_nowait()

    assert first_event == FileCreatedEvent("/foobar/a")
    assert second_event == FileCreatedEvent("/foobar/b")
    assert isinstance(overflow_event, EventQueueOverflowedEvent)
    assert overflow_event.is_synthetic
    assert overflow_event.src_path == watch.path
    assert (first_watch, second_watch, overflow_watch) == (watch, watch, watch)

    emitter.queue_event(FileModifiedEvent("/foobar/c"))
    following_event, following_watch = event_queue.get_nowait()
    assert following_event == FileModifiedEvent("/foobar/c")
    assert following_watch == watch


def test_bounded_queue_overflow_bypasses_event_filter():
    event_queue = EventQueue(maxsize=1)
    watch = ObservedWatch("/foobar", recursive=False)
    emitter = EventEmitter(event_queue, watch, event_filter=[FileCreatedEvent])

    emitter.queue_event(FileCreatedEvent("/foobar/a"))
    emitter.queue_event(FileCreatedEvent("/foobar/b"))

    event, overflow_watch = event_queue.get_nowait()
    assert event == FileCreatedEvent("/foobar/a")
    overflow_event, _ = event_queue.get_nowait()
    assert isinstance(overflow_event, EventQueueOverflowedEvent)
    assert overflow_watch == watch


def test_discarding_a_watch_releases_its_queued_overflow_for_another_watch():
    event_queue = EventQueue(maxsize=2)
    first_watch = ObservedWatch("/first", recursive=False)
    second_watch = ObservedWatch("/second", recursive=False)
    first_emitter = EventEmitter(event_queue, first_watch)
    second_emitter = EventEmitter(event_queue, second_watch)

    first_emitter.queue_event(FileModifiedEvent("/first/a"))
    first_emitter.queue_event(FileModifiedEvent("/first/b"))
    first_emitter.queue_event(FileModifiedEvent("/first/lost"))
    second_emitter.queue_event(FileModifiedEvent("/second/lost"))

    event_queue.get_nowait()
    event_queue.task_done()

    event_queue.discard_watch(first_watch)

    retained_event, retained_watch = event_queue.get_nowait()
    overflow_event, overflow_watch = event_queue.get_nowait()
    assert retained_event == FileModifiedEvent("/first/b")
    assert retained_watch == first_watch
    assert isinstance(overflow_event, EventQueueOverflowedEvent)
    assert overflow_watch == second_watch
    with pytest.raises(queue.Empty):
        event_queue.get_nowait()


def test_bounded_observer_overflows_recovers_and_stops_without_real_disk_load():
    class ControlledEmitter(EventEmitter):
        def queue_events(self, timeout: float) -> None:
            self.stopped_event.wait(0.01)

    class SlowHandler(FileSystemEventHandler):
        def __init__(self) -> None:
            self.reached_second_event = threading.Event()
            self.release_second_event = threading.Event()
            self.events: list[FileSystemEvent] = []

        def on_any_event(self, event: FileSystemEvent) -> None:
            self.events.append(event)
            if len(self.events) == 2:
                self.reached_second_event.set()
                self.release_second_event.wait(5)

    observer = BaseObserver(ControlledEmitter, event_queue_maxsize=2)
    handler = SlowHandler()
    watch = observer.schedule(handler, "/foobar", recursive=True)
    emitter = next(iter(observer.emitters))

    observer.start()
    try:
        emitter.queue_event(FileCreatedEvent("/foobar/a"))
        emitter.queue_event(FileCreatedEvent("/foobar/b"))
        emitter.queue_event(FileCreatedEvent("/foobar/lost"))
        emitter.queue_event(FileCreatedEvent("/foobar/lost-too"))

        assert handler.reached_second_event.wait(5)
        emitter.queue_event(FileModifiedEvent("/foobar/after-gap"))
        handler.release_second_event.set()

        deadline = time.monotonic() + 5
        while len(handler.events) < 4 and time.monotonic() < deadline:
            time.sleep(0.01)

        assert [type(event) for event in handler.events] == [
            FileCreatedEvent,
            FileCreatedEvent,
            EventQueueOverflowedEvent,
            FileModifiedEvent,
        ]
        assert handler.events[2].src_path == watch.path
    finally:
        observer.stop()
        observer.join(5)

    assert not observer.is_alive()
    assert observer.event_queue.empty()


def test_bounded_observer_stop_wakes_when_queue_is_full():
    class ControlledEmitter(EventEmitter):
        def queue_events(self, timeout: float) -> None:
            self.stopped_event.wait(0.01)

    observer = BaseObserver(ControlledEmitter, event_queue_maxsize=2)
    watch = observer.schedule(LoggingEventHandler(), "/foobar", recursive=True)
    emitter = next(iter(observer.emitters))

    observer.start()
    emitter.queue_event(FileModifiedEvent("/foobar/a"))
    emitter.queue_event(FileModifiedEvent("/foobar/b"))
    emitter.queue_event(FileModifiedEvent("/foobar/lost"))

    observer.stop()
    observer.join(5)

    assert not observer.is_alive()
    assert watch not in observer._emitter_for_watch  # noqa: SLF001


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

    ``BaseObserver._handlers`` is a ``defaultdict(set)``, so indexing it in
    ``dispatch_events()`` recreates whatever key it is given. An event queued
    before ``unschedule()`` is still dispatched afterwards, which put the removed
    watch back as an empty set that nothing ever cleans up.
    """
    observer = BaseObserver(EventEmitter)
    handler = LoggingEventHandler()

    watch = observer.schedule(handler, "/foobar", recursive=True)
    observer.event_queue.put((FileModifiedEvent("/foobar"), watch))
    observer.unschedule(watch)
    assert watch not in observer._handlers  # noqa: SLF001

    observer.dispatch_events(observer.event_queue)

    assert watch not in observer._handlers  # noqa: SLF001
