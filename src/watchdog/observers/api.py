from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from watchdog.utils import BaseThread
from watchdog.utils.bricks import SkipRepeatsQueue

if TYPE_CHECKING:
    import types
    from collections.abc import Iterable, Mapping

    from typing_extensions import Self

    from watchdog.events import FileSystemEvent, FileSystemEventHandler

DEFAULT_EMITTER_TIMEOUT = 1.0  # in seconds
DEFAULT_OBSERVER_TIMEOUT = 1.0  # in seconds
DEFAULT_BATCH_MAX_WAIT = 1.0  # in seconds


class EventQueue(SkipRepeatsQueue):
    """Thread-safe event queue based on a special queue that skips adding
    the same event (:class:`watchdog.events.FileSystemEvent`) multiple times
    consecutively. Thus avoiding dispatching multiple event handling
    calls when multiple identical events are produced quicker than an observer
    can consume them.

    :param maxsize:
        Maximum number of items allowed in the queue. If <= 0, the queue size is infinite.
    :type maxsize:
        ``int``
    """


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


class EventBatch(Sequence):
    """A read-only, ordered sequence of ``(event, watch)`` entries dispatched
    together as a single batch by :meth:`BaseObserver.dispatch_event_batch`.

    Entries keep the order in which they were dequeued (which is the order in
    which they were enqueued by emitters). The batch is immutable: handlers
    receiving it cannot modify its content.
    """

    def __init__(self, entries: Iterable[tuple[FileSystemEvent, ObservedWatch]]) -> None:
        self._entries = tuple(entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index):
        return self._entries[index]

    def __iter__(self):
        return iter(self._entries)

    @property
    def events(self) -> tuple[FileSystemEvent, ...]:
        """The events of this batch, in enqueue order."""
        return tuple(event for event, _ in self._entries)

    @property
    def watches(self) -> tuple[ObservedWatch, ...]:
        """The watches of this batch, in enqueue order."""
        return tuple(watch for _, watch in self._entries)

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: entries={self._entries!r}>"


class BatchErrorPolicy(Enum):
    """Policy applied by :meth:`BaseObserver.dispatch_event_batch` when an
    event handler raises an exception while processing a batch.
    """

    CONTINUE = "continue"
    """Record the failure in :attr:`BaseObserver.batch_errors` and keep
    dispatching the remaining handlers of the batch."""

    ABORT = "abort"
    """Record the failure in :attr:`BaseObserver.batch_errors` and stop
    dispatching the rest of the batch."""


class BatchDispatchError(NamedTuple):
    """Record of an event handler failure during batch dispatch."""

    handler: FileSystemEventHandler
    """The handler that raised."""

    batch: EventBatch
    """The (sub-)batch the handler was called with."""

    exception: Exception
    """The exception raised by the handler."""


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
            self._event_queue.put((event, self.watch))

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
    """

    stop_event = object()
    """Event inserted into the queue to signal a requested stop."""

    def __init__(self, *, timeout: float = DEFAULT_OBSERVER_TIMEOUT) -> None:
        super().__init__()
        self._event_queue = EventQueue()
        self._timeout = timeout

    @property
    def timeout(self) -> float:
        """Timeout value to construct emitters with."""
        return self._timeout

    def stop(self) -> None:
        BaseThread.stop(self)
        with contextlib.suppress(queue.Full):
            self.event_queue.put_nowait(EventDispatcher.stop_event)

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
    :param batch_size:
        Optional maximum number of events dispatched together as a single
        read-only :class:`EventBatch`. When ``None`` (default), events are
        dispatched one at a time, preserving the historical behavior.
    :type batch_size:
        ``int`` | ``None``
    :param batch_max_wait:
        Maximum time (in seconds) spent collecting a batch before dispatching
        whatever has been collected. Only used when ``batch_size`` is set.
    :type batch_max_wait:
        ``float``
    :param batch_error_policy:
        Policy applied when an event handler raises during batch dispatch.
        Only used when ``batch_size`` is set.
    :type batch_error_policy:
        :class:`BatchErrorPolicy`
    """

    def __init__(
        self,
        emitter_class: type[EventEmitter],
        *,
        timeout: float = DEFAULT_OBSERVER_TIMEOUT,
        batch_size: int | None = None,
        batch_max_wait: float = DEFAULT_BATCH_MAX_WAIT,
        batch_error_policy: BatchErrorPolicy = BatchErrorPolicy.CONTINUE,
    ) -> None:
        super().__init__(timeout=timeout)
        if batch_size is not None and batch_size < 1:
            error_message = "batch_size must be a positive integer or None"
            raise ValueError(error_message)
        if batch_size is not None and batch_max_wait <= 0:
            error_message = "batch_max_wait must be greater than zero"
            raise ValueError(error_message)
        self._emitter_class = emitter_class
        self._lock = threading.RLock()
        self._watches: set[ObservedWatch] = set()
        self._handlers: defaultdict[ObservedWatch, set[FileSystemEventHandler]] = defaultdict(set)
        self._emitters: set[EventEmitter] = set()
        self._emitter_for_watch: dict[ObservedWatch, EventEmitter] = {}
        self._batch_size = batch_size
        self._batch_max_wait = batch_max_wait
        self._batch_error_policy = batch_error_policy
        self._batch_errors: list[BatchDispatchError] = []

    def _add_emitter(self, emitter: EventEmitter) -> None:
        self._emitter_for_watch[emitter.watch] = emitter
        self._emitters.add(emitter)

    def _remove_emitter(self, emitter: EventEmitter) -> None:
        del self._emitter_for_watch[emitter.watch]
        self._emitters.remove(emitter)
        emitter.stop()
        with contextlib.suppress(RuntimeError):
            emitter.join()

    def _clear_emitters(self) -> None:
        for emitter in self._emitters:
            emitter.stop()
        for emitter in self._emitters:
            with contextlib.suppress(RuntimeError):
                emitter.join()
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

    @property
    def batch_size(self) -> int | None:
        """Maximum number of events dispatched per batch, or ``None`` when
        batched dispatching is disabled."""
        return self._batch_size

    @property
    def batch_max_wait(self) -> float:
        """Maximum time (in seconds) spent collecting a single batch."""
        return self._batch_max_wait

    @property
    def batch_error_policy(self) -> BatchErrorPolicy:
        """Policy applied when an event handler raises during batch dispatch."""
        return self._batch_error_policy

    @property
    def batch_errors(self) -> tuple[BatchDispatchError, ...]:
        """Failures recorded while dispatching batches, in recording order."""
        return tuple(self._batch_errors)

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
        if self._batch_size is not None:
            self._dispatch_events_batched(event_queue)
            return

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

    def _dispatch_events_batched(self, event_queue: EventQueue) -> None:
        # Snapshot the subscriptions *before* collecting the batch: the
        # snapshot defines the batch boundaries. Any subscription change
        # (schedule/unschedule/stop) made while the batch is being collected
        # or dispatched only takes effect at the start of the next batch.
        # The lock is only held for the duration of the copy, never while
        # blocking on the queue or running handler callbacks.
        with self._lock:
            handlers = {watch: tuple(watch_handlers) for watch, watch_handlers in self._handlers.items()}

        entries = self._collect_batch(event_queue)
        if not entries:
            return

        batch = EventBatch(entries)
        try:
            self.dispatch_event_batch(batch, handlers)
        finally:
            # Acknowledge every dequeued entry exactly once, even when a
            # handler fails, so that ``event_queue.join()`` can complete.
            for _ in batch:
                event_queue.task_done()

    def _collect_batch(self, event_queue: EventQueue) -> list[tuple[FileSystemEvent, ObservedWatch]]:
        """Collects up to ``batch_size`` entries from the queue, waiting at
        most ``batch_max_wait`` seconds in total for the batch to fill.

        A requested stop truncates the collection: the entries gathered so
        far still form the final batch, and the stop itself is acknowledged
        so that queue bookkeeping stays balanced.
        """
        entries: list[tuple[FileSystemEvent, ObservedWatch]] = []
        deadline = time.monotonic() + self._batch_max_wait
        while len(entries) < self._batch_size:
            remaining = deadline - time.monotonic()
            if entries and remaining <= 0:
                break
            try:
                entry = event_queue.get(block=True, timeout=max(remaining, 0.0))
            except queue.Empty:
                break
            if entry is EventDispatcher.stop_event:
                event_queue.task_done()
                break
            entries.append(entry)
        return entries

    def dispatch_event_batch(
        self,
        batch: EventBatch,
        handlers: Mapping[ObservedWatch, tuple[FileSystemEventHandler, ...]],
    ) -> None:
        """Dispatches a read-only batch of events to the subscribed handlers.

        Events are grouped per watch (preserving enqueue order within each
        watch) and every subscribed handler is called once per watch with a
        read-only :class:`EventBatch` via its ``dispatch_batch`` method.
        Handlers that do not implement ``dispatch_batch`` receive each event
        individually through ``dispatch``, in enqueue order.

        :param batch:
            The read-only batch of events to dispatch.
        :type batch:
            :class:`EventBatch`
        :param handlers:
            Read-only snapshot of the subscriptions taken at the start of the
            batch. Subscription changes made while dispatching only apply to
            the next batch.
        :type handlers:
            Mapping[:class:`ObservedWatch`, tuple of handlers]
        """
        entries_by_watch: dict[ObservedWatch, list[tuple[FileSystemEvent, ObservedWatch]]] = {}
        for entry in batch:
            entries_by_watch.setdefault(entry[1], []).append(entry)

        for watch, watch_entries in entries_by_watch.items():
            watch_batch = EventBatch(watch_entries)
            for handler in handlers.get(watch, ()):
                try:
                    self._dispatch_batch_to_handler(handler, watch_batch)
                except Exception as exc:
                    self._batch_errors.append(BatchDispatchError(handler=handler, batch=watch_batch, exception=exc))
                    if self._batch_error_policy is BatchErrorPolicy.ABORT:
                        return

    @staticmethod
    def _dispatch_batch_to_handler(handler: FileSystemEventHandler, batch: EventBatch) -> None:
        dispatch_batch = getattr(handler, "dispatch_batch", None)
        if dispatch_batch is not None:
            dispatch_batch(batch)
        else:
            for event in batch.events:
                handler.dispatch(event)
