from __future__ import annotations

import threading

from watchdog.events import (
    EVENT_TYPE_OVERFLOW,
    FileModifiedEvent,
    FileSystemEventHandler,
    OverflowEvent,
)
from watchdog.observers.api import BaseObserver, EventEmitter, EventQueue, ObservedWatch


def make_emitter(event_queue: EventQueue, path: str = "/foobar") -> tuple[EventEmitter, ObservedWatch]:
    watch = ObservedWatch(path, recursive=True)
    return EventEmitter(event_queue, watch, timeout=1), watch


class RecordingHandler(FileSystemEventHandler):
    def __init__(self) -> None:
        self.events = []
        self.overflows = []

    def on_any_event(self, event):
        self.events.append(event)

    def on_overflow(self, event):
        self.overflows.append(event)


def test_bounded_queue_merges_consecutive_losses_into_one_gap():
    event_queue = EventQueue(2)
    emitter, watch = make_emitter(event_queue)

    emitter.queue_event(FileModifiedEvent("/foobar/a"))
    emitter.queue_event(FileModifiedEvent("/foobar/b"))
    # The queue is now full; both of these are dropped, not blocked on.
    emitter.queue_event(FileModifiedEvent("/foobar/c"))
    emitter.queue_event(FileModifiedEvent("/foobar/d"))

    assert event_queue.qsize() == 2
    assert event_queue.has_pending_overflows

    # The notification itself is held outside the bounded queue: it cannot be
    # dropped by the full queue, and promotion while full is a no-op.
    event_queue.promote_pending_overflows()
    assert event_queue.qsize() == 2
    assert event_queue.has_pending_overflows

    assert event_queue.get_nowait()[0].src_path == "/foobar/a"
    assert event_queue.get_nowait()[0].src_path == "/foobar/b"

    # Once the consumer frees slots, a single gap covering both losses
    # becomes visible, tied to the affected watch.
    event_queue.promote_pending_overflows()
    assert not event_queue.has_pending_overflows
    gap_event, gap_watch = event_queue.get_nowait()
    assert isinstance(gap_event, OverflowEvent)
    assert gap_event.event_type == EVENT_TYPE_OVERFLOW
    assert gap_event.is_synthetic
    assert gap_event.dropped_count == 2
    assert gap_event.src_path == watch.path
    assert gap_watch == watch


def test_events_queued_after_overflow_still_arrive():
    event_queue = EventQueue(2)
    emitter, _watch = make_emitter(event_queue)

    emitter.queue_event(FileModifiedEvent("/foobar/a"))
    emitter.queue_event(FileModifiedEvent("/foobar/b"))
    emitter.queue_event(FileModifiedEvent("/foobar/lost"))

    assert event_queue.get_nowait()[0].src_path == "/foobar/a"
    assert event_queue.get_nowait()[0].src_path == "/foobar/b"

    event_queue.promote_pending_overflows()
    emitter.queue_event(FileModifiedEvent("/foobar/c"))

    entries = [event_queue.get_nowait() for _ in range(2)]
    assert [type(event) for event, _ in entries] == [OverflowEvent, FileModifiedEvent]
    assert entries[-1][0].src_path == "/foobar/c"


def test_unbounded_queue_keeps_default_behavior():
    event_queue = EventQueue()
    emitter, _watch = make_emitter(event_queue)

    for i in range(100):
        emitter.queue_event(FileModifiedEvent(f"/foobar/{i}"))

    assert not event_queue.has_pending_overflows
    assert event_queue.qsize() == 100


def test_emitter_does_not_block_on_full_queue():
    event_queue = EventQueue(1)
    emitter, _watch = make_emitter(event_queue)
    emitter.queue_event(FileModifiedEvent("/foobar/a"))

    produced = threading.Event()

    def produce() -> None:
        emitter.queue_event(FileModifiedEvent("/foobar/b"))
        produced.set()

    thread = threading.Thread(target=produce)
    thread.start()
    thread.join(timeout=5)

    assert produced.is_set()
    assert event_queue.has_pending_overflows


def test_observer_dispatches_gap_then_subsequent_events():
    observer = BaseObserver(EventEmitter, maxsize=2)
    handler = RecordingHandler()
    watch = observer.schedule(handler, "/foobar", recursive=True)
    emitter = observer._emitter_for_watch[watch]  # noqa: SLF001

    emitter.queue_event(FileModifiedEvent("/foobar/a"))
    emitter.queue_event(FileModifiedEvent("/foobar/b"))
    # Dropped while the (not yet started) consumer cannot keep up.
    emitter.queue_event(FileModifiedEvent("/foobar/c"))
    emitter.queue_event(FileModifiedEvent("/foobar/d"))

    # Drive the consumer manually.
    observer.dispatch_events(observer.event_queue)
    observer.dispatch_events(observer.event_queue)

    # New events are accepted again once the queue has room.
    emitter.queue_event(FileModifiedEvent("/foobar/e"))
    observer.dispatch_events(observer.event_queue)
    observer.dispatch_events(observer.event_queue)

    assert [event.src_path for event in handler.events] == ["/foobar/a", "/foobar/b", "/foobar", "/foobar/e"]
    gap = handler.events[2]
    assert isinstance(gap, OverflowEvent)
    assert handler.overflows == [gap]
    assert gap.dropped_count == 2


def test_unschedule_discards_pending_overflow():
    observer = BaseObserver(EventEmitter, maxsize=1)
    handler = RecordingHandler()
    watch = observer.schedule(handler, "/foobar", recursive=True)
    emitter = observer._emitter_for_watch[watch]  # noqa: SLF001

    emitter.queue_event(FileModifiedEvent("/foobar/a"))
    emitter.queue_event(FileModifiedEvent("/foobar/lost"))
    assert observer.event_queue.has_pending_overflows

    observer.unschedule(watch)
    assert not observer.event_queue.has_pending_overflows


def test_observer_stop_with_full_bounded_queue():
    observer = BaseObserver(EventEmitter, maxsize=2)
    handler = RecordingHandler()
    watch = observer.schedule(handler, "/foobar", recursive=True)
    emitter = observer._emitter_for_watch[watch]  # noqa: SLF001

    # Fill the bounded queue and overflow it before the consumer starts.
    for i in range(4):
        emitter.queue_event(FileModifiedEvent(f"/foobar/{i}"))
    assert observer.event_queue.has_pending_overflows

    observer.start()
    observer.stop()
    observer.join(timeout=5)

    # The stop sentinel must reach the dispatcher even though the queue was
    # full, and no overflow notification may be left waiting forever.
    assert not observer.is_alive()
    assert not observer.event_queue.has_pending_overflows
