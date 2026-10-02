from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.events import EventQueueOverflowedEvent, FileSystemEvent
from watchdog.utils import BaseThread
from watchdog.utils.bricks import SkipRepeatsQueue

if TYPE_CHECKING:
    import types

    from typing_extensions import Self

    from watchdog.events import FileSystemEventHandler

DEFAULT_EMITTER_TIMEOUT = 1.0  # in seconds
DEFAULT_OBSERVER_TIMEOUT = 1.0  # in seconds


class EventQueue(SkipRepeatsQueue):
    """Thread-safe event queue based on a special queue that skips adding
    the same event (:class:`watchdog.events.FileSystemEvent`) multiple times
    consecutively. Thus avoiding dispatching multiple event handling
    calls when multiple identical events are produced quicker than an observer
    can consume them.

    :param maxsize:
        Maximum number of items allowed in the queue. If ``<= 0`` (the
        default), the queue size is infinite.
    :type maxsize:
        ``int``
    """

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize)
        self._active_overflowed_watches: set[ObservedWatch] = set()
        self._queued_overflowed_watches: set[ObservedWatch] = set()
        self._shutdown = False

    @staticmethod
    def _entry_watch(entry: object) -> ObservedWatch | None:
        if isinstance(entry, tuple) and len(entry) == 2:
            watch = entry[1]
            if isinstance(watch, ObservedWatch):
                return watch
        return None

    def _is_overflow_entry(self, entry: object) -> bool:
        return isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], EventQueueOverflowedEvent)

    def put_nowait(self, item: object) -> None:
        with self.mutex:
            if self._last_item is not None and item == self._last_item:
                return
            if self.maxsize > 0 and self._qsize() >= self.maxsize:
                raise queue.Full
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()
            self.all_tasks_done.notify_all()

    def mark_overflow(self, watch: ObservedWatch) -> None:
        """Record that events were dropped and ensure a gap notification is delivered."""
        with self.mutex:
            if watch in self._active_overflowed_watches:
                return
            self._active_overflowed_watches.add(watch)
            if not self._shutdown and (self.maxsize <= 0 or self._qsize() < self.maxsize):
                self._put_overflow(watch)

    def _put_overflow(self, watch: ObservedWatch) -> None:
        self._put((EventQueueOverflowedEvent(watch.path, is_synthetic=True), watch))
        self.unfinished_tasks += 1
        self._queued_overflowed_watches.add(watch)
        self.not_empty.notify()
        self.all_tasks_done.notify_all()

    def _put_pending_overflows(self) -> None:
        if self._shutdown:
            return
        while self.maxsize <= 0 or self._qsize() < self.maxsize:
            pending_watch = next(iter(self._active_overflowed_watches - self._queued_overflowed_watches), None)
            if pending_watch is None:
                break
            self._put_overflow(pending_watch)

    def _after_get(self, entry: object) -> None:
        watch = self._entry_watch(entry)
        if (
            watch is not None
            and isinstance(entry, tuple)
            and isinstance(entry[0], EventQueueOverflowedEvent)
        ):
            self._queued_overflowed_watches.discard(watch)
            self._active_overflowed_watches.discard(watch)

        self._put_pending_overflows()

    def _get(self) -> object:
        entry = super()._get()
        self._after_get(entry)
        return entry

    def _remove_overflow_entries(self, watches: set[ObservedWatch] | None = None) -> int:
        watches_to_remove = self._queued_overflowed_watches if watches is None else watches
        removed_count = 0
        for entry in list(self.queue):
            entry_watch = self._entry_watch(entry)
            if entry_watch is not None and entry_watch in watches_to_remove and self._is_overflow_entry(entry):
                self._queued_overflowed_watches.discard(entry_watch)
                removed_count += 1
                self.queue.remove(entry)

        if not self.queue:
            self._last_item = None
        else:
            self._last_item = self.queue[-1]
        self.unfinished_tasks -= removed_count
        if removed_count:
            self.not_full.notify_all()
            self.all_tasks_done.notify_all()
        return removed_count

    def discard_watch(self, watch: ObservedWatch) -> None:
        """Remove gap state and queued gap notifications for a cancelled watch."""
        with self.mutex:
            self._active_overflowed_watches.discard(watch)
            if watch in self._queued_overflowed_watches:
                self._remove_overflow_entries({watch})
                self._put_pending_overflows()

    def discard_all_overflows(self) -> None:
        """Remove all gap notifications during observer shutdown."""
        with self.mutex:
            if self._active_overflowed_watches:
                self._active_overflowed_watches.clear()
                self._remove_overflow_entries()

    def shutdown(self) -> None:
        """Wake consumers blocked on an empty or full bounded queue."""
        with self.mutex:
            self._shutdown = True
            self.not_empty.notify_all()

    def get(self, block: bool = True, timeout: float | None = None) -> object:
        with self.not_empty:
            if not block:
                if not self._qsize() and not self._shutdown:
                    raise queue.Empty
            elif timeout is None:
                while not self._qsize() and not self._shutdown:
                    self.not_empty.wait()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                end_time = time.monotonic() + timeout
                while not self._qsize() and not self._shutdown:
                    remaining = end_time - time.monotonic()
                    if remaining <= 0:
                        raise queue.Empty
                    self.not_empty.wait(remaining)

            if not self._qsize():
                self.unfinished_tasks += 1
                self.all_tasks_done.notify_all()
                return EventDispatcher.stop_event

            entry = self._get()
            self.not_full.notify()
            return entry


class ObservedWatch:
    """An scheduled watch.

    :param path:
        Path string.
    :type path:
        ``str`` or :class:`pathlib.Path`
    :param recursive:
        ``True`` if watch is recursive; ``False`` otherwise.
    :type recursive:
        ``bool``
    :param event_filter:
        Optional collection of :class:`watchdog.events.FileSystemEvent` to watch.
    :type event_filter:
        Iterable[:class:`watchdog.events.FileSystemEvent`] | None
    :param follow_symlink:
        ``True`` if symlinks are followed; ``False`` otherwise.
    :type follow_symlink:
        ``bool``
    """

    def __init__(
        self,
        path: str | Path,
        *,
        recursive: bool,
        event_filter: list[type[FileSystemEvent]] | None = None,
        follow_symlink: bool = False,
    ):
        self._path = str(path) if isinstance(path, Path) else path
        self._is_recursive = recursive
        self._follow_symlink = follow_symlink
        self._event_filter = frozenset(event_filter) if event_filter is not None else None

    @property
    def path(self) -> str:
        """The path that this watch monitors."""
        return self._path

    @property
    def is_recursive(self) -> bool:
        """Determines whether subdirectories are watched for the path."""
        return self._is_recursive

    @property
    def follow_symlink(self) -> bool:
        """Determines whether symlink are followed."""
        return self._follow_symlink

    @property
    def event_filter(self) -> frozenset[type[FileSystemEvent]] | None:
        """Collection of event types watched for the path"""
        return self._event_filter

    @property
    def key(self) -> tuple[str, bool, frozenset[type[FileSystemEvent]] | None, bool]:
        """A tuple key identifying the watch (path, recursive, event_filter, follow_symlink)."""
        return self.path, self.is_recursive, self.event_filter, self.follow_symlink

    def __eq__(self, watch: object) -> bool:
        if not isinstance(watch, ObservedWatch):
            return NotImplemented
        return self.key == watch.key

    def __ne__(self, watch: object) -> bool:
        if not isinstance(watch, ObservedWatch):
            return NotImplemented
        return self.key != watch.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        if self.event_filter is not None:
            event_filter_str = "|".join(sorted(_cls.__name__ for _cls in self.event_filter))
            event_filter_str = f", event_filter={event_filter_str}"
        else:
            event_filter_str = ""
        return f"<{type(self).__name__}: path={self.path!r}, is_recursive={self.is_recursive}{event_filter_str}>"


# Observer classes
class EventEmitter(BaseThread):
    """Producer thread base class subclassed by event emitters
    that generate events and populate a queue with them.

    :param event_queue:
        The event queue to populate with generated events.
    :type event_queue:
        :class:`EventQueue`
    :param watch:
        The watch to observe and produce events for.
    :type watch:
        :class:`ObservedWatch`
    :param timeout:
        Timeout (in seconds) between successive attempts at reading events.
    :type timeout:
        ``float``
    :param event_filter:
        Collection of event types to emit, or None for no filtering (default).
    :type event_filter:
        Iterable[:class:`watchdog.events.FileSystemEvent`] | None
    """

    def __init__(
        self,
        event_queue: EventQueue,
        watch: ObservedWatch,
        *,
        timeout: float = DEFAULT_EMITTER_TIMEOUT,
        event_filter: list[type[FileSystemEvent]] | None = None,
    ) -> None:
        super().__init__()
        self._event_queue = event_queue
        self._watch = watch
        self._timeout = timeout
        self._event_filter = frozenset(event_filter) if event_filter is not None else None

    @property
    def timeout(self) -> float:
        """Blocking timeout for reading events."""
        return self._timeout

    @property
    def watch(self) -> ObservedWatch:
        """The watch associated with this emitter."""
        return self._watch

    def queue_event(self, event: FileSystemEvent) -> None:
        """Queues a single event.

        :param event:
            Event to be queued.
        :type event:
            An instance of :class:`watchdog.events.FileSystemEvent`
            or a subclass.
        """
        if self._event_filter is None or any(isinstance(event, cls) for cls in self._event_filter):
            try:
                self._event_queue.put_nowait((event, self.watch))
            except queue.Full:
                self._event_queue.mark_overflow(self.watch)

    def queue_events(self, timeout: float) -> None:
        """Override this method to populate the event queue with events
        per interval period.

        :param timeout:
            Timeout (in seconds) between successive attempts at
            reading events.
        :type timeout:
            ``float``
        """

    def run(self) -> None:
        while self.should_keep_running():
            self.queue_events(self.timeout)


class EventDispatcher(BaseThread):
    """Consumer thread base class subclassed by event observer threads
    that dispatch events from an event queue to appropriate event handlers.

    :param timeout:
        Timeout value (in seconds) passed to emitters
        constructions in the child class BaseObserver.
    :type timeout:
        ``float``
    :param event_queue_maxsize:
        Maximum number of events buffered by the observer. ``<= 0`` (the
        default) means an unbounded queue.
    :type event_queue_maxsize:
        ``int``
    """

    stop_event = object()
    """Sentinel returned by a shut down queue to signal a requested stop."""

    def __init__(self, *, timeout: float = DEFAULT_OBSERVER_TIMEOUT, event_queue_maxsize: int = 0) -> None:
        super().__init__()
        self._event_queue = EventQueue(event_queue_maxsize)
        self._timeout = timeout

    @property
    def timeout(self) -> float:
        """Timeout value to construct emitters with."""
        return self._timeout

    def stop(self) -> None:
        BaseThread.stop(self)
        self.event_queue.discard_all_overflows()
        self.event_queue.shutdown()

    @property
    def event_queue(self) -> EventQueue:
        """The event queue which is populated with file system events
        by emitters and from which events are dispatched by a dispatcher
        thread.
        """
        return self._event_queue

    def dispatch_events(self, event_queue: EventQueue) -> None:
        """Override this method to consume events from an event queue, blocking
        on the queue for the specified timeout before raising :class:`queue.Empty`.

        :param event_queue:
            Event queue to populate with one set of events.
        :type event_queue:
            :class:`EventQueue`
        :raises:
            :class:`queue.Empty`
        """

    def run(self) -> None:
        while self.should_keep_running():
            try:
                self.dispatch_events(self.event_queue)
            except queue.Empty:
                continue


class BaseObserver(EventDispatcher):
    """Base observer.

    :param emitter_class:
        The class of the emitter to use (e.g., a subclass of :class:`EventEmitter`).
    :type emitter_class:
        ``type[EventEmitter]``
    :param timeout:
        The timeout (in seconds) for the observer to wait on events.
    :type timeout:
        ``float``
    :param event_queue_maxsize:
        Maximum number of events buffered by the observer. ``<= 0`` (the
        default) means an unbounded queue.
    :type event_queue_maxsize:
        ``int``
    """

    def __init__(
        self,
        emitter_class: type[EventEmitter],
        *,
        timeout: float = DEFAULT_OBSERVER_TIMEOUT,
        event_queue_maxsize: int = 0,
    ) -> None:
        super().__init__(timeout=timeout, event_queue_maxsize=event_queue_maxsize)
        self._emitter_class = emitter_class
        self._lock = threading.RLock()
        self._watches: set[ObservedWatch] = set()
        self._handlers: defaultdict[ObservedWatch, set[FileSystemEventHandler]] = defaultdict(set)
        self._emitters: set[EventEmitter] = set()
        self._emitter_for_watch: dict[ObservedWatch, EventEmitter] = {}

    def _add_emitter(self, emitter: EventEmitter) -> None:
        self._emitter_for_watch[emitter.watch] = emitter
        self._emitters.add(emitter)

    def _remove_emitter(self, emitter: EventEmitter) -> None:
        del self._emitter_for_watch[emitter.watch]
        self._emitters.remove(emitter)
        emitter.stop()
        try:
            emitter.join()
        except RuntimeError:
            pass
        self._event_queue.discard_watch(emitter.watch)

    def _clear_emitters(self) -> None:
        for emitter in self._emitters:
            emitter.stop()
        for emitter in self._emitters:
            try:
                emitter.join()
            except RuntimeError:
                pass
        self._event_queue.discard_all_overflows()
        self._emitters.clear()
        self._emitter_for_watch.clear()

    def _add_handler_for_watch(self, event_handler: FileSystemEventHandler, watch: ObservedWatch) -> None:
        self._handlers[watch].add(event_handler)

    def _remove_handlers_for_watch(self, watch: ObservedWatch) -> None:
        del self._handlers[watch]

    @property
    def emitters(self) -> set[EventEmitter]:
        """Returns event emitter created by this observer."""
        return self._emitters

    def start(self) -> None:
        for emitter in self._emitters.copy():
            try:
                emitter.start()
            except Exception:
                self._remove_emitter(emitter)
                raise
        super().start()

    def __enter__(self) -> Self:
        """Context manager entry. Starts the observer if not already running."""
        if not self.is_alive():
            self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Context manager exit. Stops the observer and waits for it to finish."""
        if self.is_alive():
            self.stop()
            self.join()

    def schedule(
        self,
        event_handler: FileSystemEventHandler,
        path: str | Path,
        *,
        recursive: bool = False,
        event_filter: list[type[FileSystemEvent]] | None = None,
        follow_symlink: bool = False,
    ) -> ObservedWatch:
        """Schedules watching a path and calls appropriate methods specified
        in the given event handler in response to file system events.

        :param event_handler:
            An event handler instance that has appropriate event handling
            methods which will be called by the observer in response to
            file system events.
        :type event_handler:
            :class:`watchdog.events.FileSystemEventHandler` or a subclass
        :param path:
            Directory path that will be monitored.
        :type path:
            ``str`` or :class:`pathlib.Path`
        :param recursive:
            ``True`` if events will be emitted for sub-directories
            traversed recursively; ``False`` otherwise.
        :type recursive:
            ``bool``
        :param event_filter:
            Collection of event types to emit, or None for no filtering (default).
        :type event_filter:
            Iterable[:class:`watchdog.events.FileSystemEvent`] | None
        :param follow_symlink:
            ``True`` if symlinks should be followed; ``False`` otherwise.
        :type follow_symlink:
            ``bool``
        :return:
            An :class:`ObservedWatch` object instance representing
            a watch.
        """
        with self._lock:
            watch = ObservedWatch(path, recursive=recursive, event_filter=event_filter, follow_symlink=follow_symlink)
            self._add_handler_for_watch(event_handler, watch)

            # If we don't have an emitter for this watch already, create it.
            if watch not in self._emitter_for_watch:
                emitter = self._emitter_class(self.event_queue, watch, timeout=self.timeout, event_filter=event_filter)
                if self.is_alive():
                    emitter.start()
                self._add_emitter(emitter)
            self._watches.add(watch)
        return watch

    def add_handler_for_watch(self, event_handler: FileSystemEventHandler, watch: ObservedWatch) -> None:
        """Adds a handler for the given watch.

        :param event_handler:
            An event handler instance that has appropriate event handling
            methods which will be called by the observer in response to
            file system events.
        :type event_handler:
            :class:`watchdog.events.FileSystemEventHandler` or a subclass
        :param watch:
            The watch to add a handler for.
        :type watch:
            An instance of :class:`ObservedWatch` or a subclass of
            :class:`ObservedWatch`
        """
        with self._lock:
            self._add_handler_for_watch(event_handler, watch)

    def remove_handler_for_watch(self, event_handler: FileSystemEventHandler, watch: ObservedWatch) -> None:
        """Removes a handler for the given watch.

        :param event_handler:
            An event handler instance that has appropriate event handling
            methods which will be called by the observer in response to
            file system events.
        :type event_handler:
            :class:`watchdog.events.FileSystemEventHandler` or a subclass
        :param watch:
            The watch to remove a handler for.
        :type watch:
            An instance of :class:`ObservedWatch` or a subclass of
            :class:`ObservedWatch`
        """
        with self._lock:
            self._handlers[watch].remove(event_handler)

    def unschedule(self, watch: ObservedWatch) -> None:
        """Unschedules a watch.

        :param watch:
            The watch to unschedule.
        :type watch:
            An instance of :class:`ObservedWatch` or a subclass of
            :class:`ObservedWatch`
        """
        with self._lock:
            emitter = self._emitter_for_watch[watch]
            del self._handlers[watch]
            self._remove_emitter(emitter)
            self._watches.remove(watch)

    def unschedule_all(self) -> None:
        """Unschedules all watches and detaches all associated event handlers."""
        with self._lock:
            self._handlers.clear()
            self._clear_emitters()
            self._watches.clear()

    def on_thread_stop(self) -> None:
        self.unschedule_all()

    def dispatch_events(self, event_queue: EventQueue) -> None:
        entry = event_queue.get(block=True)
        if entry is EventDispatcher.stop_event:
            event_queue.task_done()
            return

        event, watch = entry

        with self._lock:
            # To allow unschedule/stop and safe removal of event handlers
            # within event handlers itself, check if the handler is still
            # registered after every dispatch.
            # Use .get() rather than indexing: _handlers is a defaultdict, so
            # indexing it here would resurrect a watch that unschedule() removed
            # but whose events were already queued.
            for handler in list(self._handlers.get(watch, ())):
                if handler in self._handlers.get(watch, ()):
                    handler.dispatch(event)
        event_queue.task_done()
