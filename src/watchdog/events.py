""":module: watchdog.events
:synopsis: File system events and event handlers.
:author: yesudeep@google.com (Yesudeep Mangalapilly)
:author: Mickaël Schoentgen <contact@tiger-222.fr>

Event Classes
-------------
.. autoclass:: FileState
   :members:
   :show-inheritance:

.. autoclass:: FileSystemEvent
   :members:
   :show-inheritance:
   :inherited-members:

.. autoclass:: FileSystemMovedEvent
   :members:
   :show-inheritance:

.. autoclass:: FileMovedEvent
   :members:
   :show-inheritance:

.. autoclass:: DirMovedEvent
   :members:
   :show-inheritance:

.. autoclass:: FileModifiedEvent
   :members:
   :show-inheritance:

.. autoclass:: DirModifiedEvent
   :members:
   :show-inheritance:

.. autoclass:: FileCreatedEvent
   :members:
   :show-inheritance:

.. autoclass:: FileClosedEvent
   :members:
   :show-inheritance:

.. autoclass:: FileClosedNoWriteEvent
   :members:
   :show-inheritance:

.. autoclass:: FileOpenedEvent
   :members:
   :show-inheritance:

.. autoclass:: FileStableEvent
   :members:
   :show-inheritance:

.. autoclass:: DirCreatedEvent
   :members:
   :show-inheritance:

.. autoclass:: FileDeletedEvent
   :members:
   :show-inheritance:

.. autoclass:: DirDeletedEvent
   :members:
   :show-inheritance:


Event Handler Classes
---------------------
.. autoclass:: FileSystemEventHandler
   :members:
   :show-inheritance:

.. autoclass:: PatternMatchingEventHandler
   :members:
   :show-inheritance:

.. autoclass:: RegexMatchingEventHandler
   :members:
   :show-inheritance:

.. autoclass:: LoggingEventHandler
   :members:
   :show-inheritance:

.. autoclass:: StableFileEventHandler
   :members:
   :show-inheritance:

"""

from __future__ import annotations

import logging
import os
import os.path
import re
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from watchdog.utils.patterns import match_any_paths

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator

EVENT_TYPE_MOVED = "moved"
EVENT_TYPE_DELETED = "deleted"
EVENT_TYPE_CREATED = "created"
EVENT_TYPE_MODIFIED = "modified"
EVENT_TYPE_CLOSED = "closed"
EVENT_TYPE_CLOSED_NO_WRITE = "closed_no_write"
EVENT_TYPE_OPENED = "opened"
EVENT_TYPE_STABLE = "stable"


@dataclass(frozen=True)
class FileState:
    """Snapshot of the state used to decide whether a file is stable."""

    path: bytes | str
    size: int
    mtime: float | int
    identity: tuple[int, int]
    forced: bool = False

    @classmethod
    def from_stat_result(cls, path: bytes | str, stat_result: os.stat_result, *, forced: bool = False) -> FileState:
        return cls(
            path=path,
            size=stat_result.st_size,
            mtime=stat_result.st_mtime,
            identity=(stat_result.st_dev, stat_result.st_ino),
            forced=forced,
        )


@dataclass(unsafe_hash=True)
class FileSystemEvent:
    """Represents a file system event that is triggered when a change
    occurs on the monitored file system.

    FileSystemEvent instances are mutable, so avoid using them as
    dictionary keys or set members: mutating an instance after
    insertion can silently break lookups.
    """

    src_path: bytes | str
    dest_path: bytes | str = ""
    event_type: str = field(default="", init=False)
    is_directory: bool = field(default=False, init=False)
    """True if event was triggered for a directory; False otherwise."""

    is_synthetic: bool = field(default=False)
    """True if event was synthesized; False otherwise.
    These are events that weren't actually broadcast by the OS, but
    are presumed to have happened based on other, actual events.
    """


class FileSystemMovedEvent(FileSystemEvent):
    """File system event representing any kind of file system movement."""

    event_type = EVENT_TYPE_MOVED


# File events.


class FileDeletedEvent(FileSystemEvent):
    """File system event representing file deletion on the file system."""

    event_type = EVENT_TYPE_DELETED


class FileModifiedEvent(FileSystemEvent):
    """File system event representing file modification on the file system."""

    event_type = EVENT_TYPE_MODIFIED


class FileCreatedEvent(FileSystemEvent):
    """File system event representing file creation on the file system."""

    event_type = EVENT_TYPE_CREATED


class FileMovedEvent(FileSystemMovedEvent):
    """File system event representing file movement on the file system."""


class FileClosedEvent(FileSystemEvent):
    """File system event representing file close on the file system.

    Emitted only by ``InotifyObserver`` on Linux.
    """

    event_type = EVENT_TYPE_CLOSED


class FileClosedNoWriteEvent(FileSystemEvent):
    """File system event representing an unmodified file close on the file system.

    Emitted only by ``InotifyObserver`` on Linux.
    """

    event_type = EVENT_TYPE_CLOSED_NO_WRITE


class FileOpenedEvent(FileSystemEvent):
    """File system event representing file open on the file system.

    Emitted only by ``InotifyObserver`` on Linux.
    """

    event_type = EVENT_TYPE_OPENED


class FileStableEvent(FileSystemEvent):
    """File system event representing a file that has finished changing."""

    event_type = EVENT_TYPE_STABLE

    def __init__(self, state: FileState) -> None:
        super().__init__(state.path)
        self.state = state

    @property
    def size(self) -> int:
        """Size of the stable file."""
        return self.state.size

    @property
    def mtime(self) -> float | int:
        """Modification time of the stable file."""
        return self.state.mtime

    @property
    def identity(self) -> tuple[int, int]:
        """``(st_dev, st_ino)`` identity of the stable file."""
        return self.state.identity

    @property
    def forced(self) -> bool:
        """Whether the maximum waiting time was reached."""
        return self.state.forced


# Directory events.


class DirDeletedEvent(FileSystemEvent):
    """File system event representing directory deletion on the file system."""

    event_type = EVENT_TYPE_DELETED
    is_directory = True


class DirModifiedEvent(FileSystemEvent):
    """File system event representing directory modification on the file system."""

    event_type = EVENT_TYPE_MODIFIED
    is_directory = True


class DirCreatedEvent(FileSystemEvent):
    """File system event representing directory creation on the file system."""

    event_type = EVENT_TYPE_CREATED
    is_directory = True


class DirMovedEvent(FileSystemMovedEvent):
    """File system event representing directory movement on the file system."""

    is_directory = True


class FileSystemEventHandler:
    """Base file system event handler that you can override methods from."""

    def dispatch(self, event: FileSystemEvent) -> None:
        """Dispatches events to the appropriate methods.

        :param event:
            The event object representing the file system event.
        :type event:
            :class:`watchdog.events.FileSystemEvent`
        """
        self.on_any_event(event)
        getattr(self, f"on_{event.event_type}")(event)

    def on_any_event(self, event: FileSystemEvent) -> None:
        """Catch-all event handler.

        :param event:
            The event object representing the file system event.
        :type event:
            :class:`watchdog.events.FileSystemEvent`
        """

    def on_moved(self, event: DirMovedEvent | FileMovedEvent) -> None:
        """Called when a file or a directory is moved or renamed.

        :param event:
            Event representing file/directory movement.
        :type event:
            :class:`watchdog.events.DirMovedEvent` or :class:`watchdog.events.FileMovedEvent`
        """

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        """Called when a file or directory is created.

        :param event:
            Event representing file/directory creation.
        :type event:
            :class:`watchdog.events.DirCreatedEvent` or :class:`watchdog.events.FileCreatedEvent`
        """

    def on_deleted(self, event: DirDeletedEvent | FileDeletedEvent) -> None:
        """Called when a file or directory is deleted.

        :param event:
            Event representing file/directory deletion.
        :type event:
            :class:`watchdog.events.DirDeletedEvent` or :class:`watchdog.events.FileDeletedEvent`
        """

    def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
        """Called when a file or directory is modified.

        :param event:
            Event representing file/directory modification.
        :type event:
            :class:`watchdog.events.DirModifiedEvent` or :class:`watchdog.events.FileModifiedEvent`
        """

    def on_closed(self, event: FileClosedEvent) -> None:
        """Called when a file opened for writing is closed.

        Emitted only by ``InotifyObserver`` on Linux.

        :param event:
            Event representing file closing.
        :type event:
            :class:`watchdog.events.FileClosedEvent`
        """

    def on_closed_no_write(self, event: FileClosedNoWriteEvent) -> None:
        """Called when a file opened for reading is closed.

        Emitted only by ``InotifyObserver`` on Linux.

        :param event:
            Event representing file closing.
        :type event:
            :class:`watchdog.events.FileClosedNoWriteEvent`
        """

    def on_opened(self, event: FileOpenedEvent) -> None:
        """Called when a file is opened.

        Emitted only by ``InotifyObserver`` on Linux.

        :param event:
            Event representing file opening.
        :type event:
            :class:`watchdog.events.FileOpenedEvent`
        """

    def on_stable(self, event: FileStableEvent) -> None:
        """Called when a file has remained unchanged for the configured interval.

        :param event:
            Event containing the final stat-derived file state.
        :type event:
            :class:`watchdog.events.FileStableEvent`
        """


class PatternMatchingEventHandler(FileSystemEventHandler):
    """Matches given patterns with file paths associated with occurring events.
    Uses pathlib's `PurePath.match()` method. `patterns` and `ignore_patterns`
    are expected to be a list of strings.
    """

    def __init__(
        self,
        *,
        patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        ignore_directories: bool = False,
        case_sensitive: bool = False,
    ):
        super().__init__()

        self._patterns = patterns
        self._ignore_patterns = ignore_patterns
        self._ignore_directories = ignore_directories
        self._case_sensitive = case_sensitive

    @property
    def patterns(self) -> list[str] | None:
        """(Read-only)
        Patterns to allow matching event paths.
        """
        return self._patterns

    @property
    def ignore_patterns(self) -> list[str] | None:
        """(Read-only)
        Patterns to ignore matching event paths.
        """
        return self._ignore_patterns

    @property
    def ignore_directories(self) -> bool:
        """(Read-only)
        ``True`` if directories should be ignored; ``False`` otherwise.
        """
        return self._ignore_directories

    @property
    def case_sensitive(self) -> bool:
        """(Read-only)
        ``True`` if path names should be matched sensitive to case; ``False``
        otherwise.
        """
        return self._case_sensitive

    def dispatch(self, event: FileSystemEvent) -> None:
        """Dispatches events to the appropriate methods.

        :param event:
            The event object representing the file system event.
        :type event:
            :class:`watchdog.events.FileSystemEvent`
        """
        if self.ignore_directories and event.is_directory:
            return

        paths = []
        if hasattr(event, "dest_path"):
            paths.append(os.fsdecode(event.dest_path))
        if event.src_path:
            paths.append(os.fsdecode(event.src_path))

        if match_any_paths(
            paths,
            included_patterns=self.patterns,
            excluded_patterns=self.ignore_patterns,
            case_sensitive=self.case_sensitive,
        ):
            super().dispatch(event)


class RegexMatchingEventHandler(FileSystemEventHandler):
    """Matches given regexes with file paths associated with occurring events.
    Uses the `re` module.
    """

    def __init__(
        self,
        *,
        regexes: list[str] | None = None,
        ignore_regexes: list[str] | None = None,
        ignore_directories: bool = False,
        case_sensitive: bool = False,
    ):
        super().__init__()

        if regexes is None:
            regexes = [r".*"]
        elif isinstance(regexes, str):
            regexes = [regexes]
        if ignore_regexes is None:
            ignore_regexes = []
        if case_sensitive:
            self._regexes = [re.compile(r) for r in regexes]
            self._ignore_regexes = [re.compile(r) for r in ignore_regexes]
        else:
            self._regexes = [re.compile(r, re.IGNORECASE) for r in regexes]
            self._ignore_regexes = [re.compile(r, re.IGNORECASE) for r in ignore_regexes]
        self._ignore_directories = ignore_directories
        self._case_sensitive = case_sensitive

    @property
    def regexes(self) -> list[re.Pattern[str]]:
        """(Read-only)
        Regexes to allow matching event paths.
        """
        return self._regexes

    @property
    def ignore_regexes(self) -> list[re.Pattern[str]]:
        """(Read-only)
        Regexes to ignore matching event paths.
        """
        return self._ignore_regexes

    @property
    def ignore_directories(self) -> bool:
        """(Read-only)
        ``True`` if directories should be ignored; ``False`` otherwise.
        """
        return self._ignore_directories

    @property
    def case_sensitive(self) -> bool:
        """(Read-only)
        ``True`` if path names should be matched sensitive to case; ``False``
        otherwise.
        """
        return self._case_sensitive

    def dispatch(self, event: FileSystemEvent) -> None:
        """Dispatches events to the appropriate methods.

        :param event:
            The event object representing the file system event.
        :type event:
            :class:`watchdog.events.FileSystemEvent`
        """
        if self.ignore_directories and event.is_directory:
            return

        paths = []
        if hasattr(event, "dest_path"):
            paths.append(os.fsdecode(event.dest_path))
        if event.src_path:
            paths.append(os.fsdecode(event.src_path))

        if any(r.match(p) for r in self.ignore_regexes for p in paths):
            return

        if any(r.match(p) for r in self.regexes for p in paths):
            super().dispatch(event)


class LoggingEventHandler(FileSystemEventHandler):
    """Logs all the events captured."""

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        super().__init__()
        self.logger = logger or logging.root

    def on_moved(self, event: DirMovedEvent | FileMovedEvent) -> None:
        super().on_moved(event)

        what = "directory" if event.is_directory else "file"
        self.logger.info("Moved %s: from %s to %s", what, event.src_path, event.dest_path)

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        super().on_created(event)

        what = "directory" if event.is_directory else "file"
        self.logger.info("Created %s: %s", what, event.src_path)

    def on_deleted(self, event: DirDeletedEvent | FileDeletedEvent) -> None:
        super().on_deleted(event)

        what = "directory" if event.is_directory else "file"
        self.logger.info("Deleted %s: %s", what, event.src_path)

    def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
        super().on_modified(event)

        what = "directory" if event.is_directory else "file"
        self.logger.info("Modified %s: %s", what, event.src_path)

    def on_closed(self, event: FileClosedEvent) -> None:
        super().on_closed(event)

        self.logger.info("Closed modified file: %s", event.src_path)

    def on_closed_no_write(self, event: FileClosedNoWriteEvent) -> None:
        super().on_closed_no_write(event)

        self.logger.info("Closed read file: %s", event.src_path)

    def on_opened(self, event: FileOpenedEvent) -> None:
        super().on_opened(event)

        self.logger.info("Opened file: %s", event.src_path)


def _default_timer(delay: float, callback: Callable[[], None]) -> object:
    timer = threading.Timer(delay, callback)
    timer.daemon = True
    timer.start()
    return timer


class StableFileEventHandler(FileSystemEventHandler):
    """Emit one event after a file's size and modification time have stopped changing.

    The handler is composable: original filesystem events can be subscribed to
    independently, while ``downstream`` receives only :class:`FileStableEvent`
    instances. Clock, stat and timer creation are injectable, which makes the
    handler deterministic in tests.

    :param downstream:
        Handler or callable receiving stable-file events.
    :param stable_interval_seconds:
        Time that size and modification time must remain unchanged.
    :param max_wait_seconds:
        Upper bound for waiting on one continuously changing file. The final
        event is marked as forced when this bound is reached.
    :param stat:
        Callable returning an object compatible with :func:`os.stat`.
    :param clock:
        Monotonic clock callable, injectable for deterministic tests.
    :param timer:
        Callable creating a delayed callback, injectable for deterministic
        tests.
    """

    def __init__(
        self,
        downstream: FileSystemEventHandler | Callable[[FileStableEvent], None] | None = None,
        *,
        stable_interval_seconds: float,
        max_wait_seconds: float | None = None,
        stat: Callable[[bytes | str], os.stat_result] = os.stat,
        clock: Callable[[], float] = time.monotonic,
        timer: Callable[[float, Callable[[], None]], object] = _default_timer,
    ) -> None:
        super().__init__()
        if stable_interval_seconds < 0:
            raise ValueError("stable_interval_seconds must be non-negative")
        if max_wait_seconds is None:
            max_wait_seconds = max(stable_interval_seconds, 60.0)
        if max_wait_seconds < stable_interval_seconds:
            raise ValueError("max_wait_seconds must be >= stable_interval_seconds")

        self.downstream = downstream
        self.stable_interval_seconds = stable_interval_seconds
        self.max_wait_seconds = max_wait_seconds
        self._stat = stat
        self._clock = clock
        self._timer = timer
        self._raw_subscribers: list[FileSystemEventHandler | Callable[[FileSystemEvent], None]] = []
        self._pending: dict[bytes | str, _PendingFile] = {}
        self._condition = threading.Condition(threading.RLock())
        self._in_flight = 0
        self._in_flight_threads: set[int] = set()
        self._closed = False

    def subscribe_raw(
        self,
        subscriber: FileSystemEventHandler | Callable[[FileSystemEvent], None],
    ) -> FileSystemEventHandler | Callable[[FileSystemEvent], None]:
        """Subscribe to the unmodified filesystem event stream."""
        with self._condition:
            self._raw_subscribers.append(subscriber)
        return subscriber

    def unsubscribe_raw(
        self,
        subscriber: FileSystemEventHandler | Callable[[FileSystemEvent], None],
    ) -> None:
        """Remove a subscriber from the unmodified filesystem event stream."""
        with self._condition:
            self._raw_subscribers.remove(subscriber)

    def dispatch(self, event: FileSystemEvent) -> None:
        """Forward raw events, then update stability tracking for file events."""
        with self._condition:
            subscribers = tuple(self._raw_subscribers)
            closed = self._closed

        for subscriber in subscribers:
            if isinstance(subscriber, FileSystemEventHandler):
                subscriber.dispatch(event)
            else:
                subscriber(event)

        if closed or event.is_directory:
            return

        if event.event_type == EVENT_TYPE_MOVED:
            self.on_moved(event)
        else:
            getattr(self, f"on_{event.event_type}", lambda _event: None)(event)

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        self._arm_from_event(event.src_path)

    def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
        self._arm_from_event(event.src_path)

    def on_closed(self, event: FileClosedEvent) -> None:
        self._arm_from_event(event.src_path)

    def on_deleted(self, event: DirDeletedEvent | FileDeletedEvent) -> None:
        with self._condition:
            pending = self._pending.pop(event.src_path, None)
        if pending is not None:
            pending.cancel()

    def on_moved(self, event: DirMovedEvent | FileMovedEvent) -> None:
        with self._condition:
            source = self._pending.pop(event.src_path, None)
        if source is not None:
            source.cancel()
        self._arm_from_event(event.dest_path, moved_from=source)

    def stop(self, *, drain: bool = False) -> None:
        """Stop tracking and explicitly drain or discard all pending files."""
        with self._condition:
            self._closed = True
            while self._in_flight and threading.get_ident() not in self._in_flight_threads:
                self._condition.wait()
            pending = self._pending
            self._pending = {}

        for item in pending.values():
            item.cancel()

        if drain:
            for item in pending.values():
                self._drain_pending(item)

    def _arm_from_event(self, path: bytes | str, moved_from: _PendingFile | None = None) -> None:
        with self._condition:
            if self._closed:
                return
            now = self._clock()
            try:
                state = FileState.from_stat_result(path, self._stat(path))
            except FileNotFoundError:
                if moved_from is not None:
                    self._pending[path] = _PendingFile.unresolved(
                        path=path,
                        expected_identity=moved_from.identity,
                        first_seen_at=moved_from.first_seen_at,
                    )
                    self._schedule(self._pending[path], now)
                    return
                existing = self._pending.pop(path, None)
                if existing is not None:
                    existing.cancel()
                return
            except OSError:
                existing = self._pending.get(path)
                if existing is None and moved_from is not None:
                    self._pending[path] = _PendingFile.unresolved(
                        path=path,
                        expected_identity=moved_from.identity,
                        first_seen_at=moved_from.first_seen_at,
                    )
                    self._schedule(self._pending[path], now)
                    return
                if existing is None:
                    return
                existing.last_error_at = now
                existing.stable_since = now
                self._schedule(existing, now)
                return

            existing = self._pending.get(path)

            if existing is not None and existing.identity != state.identity:
                existing.cancel()
                existing = None

            if existing is None:
                if moved_from is not None and moved_from.identity == state.identity:
                    moved_from.path = path
                    moved_from.timer = None
                    moved_from.expected_identity = None
                    moved_from.last_error_at = None
                    if moved_from.state is None or (
                        moved_from.state.size,
                        moved_from.state.mtime,
                    ) != (state.size, state.mtime):
                        moved_from.state = state
                        moved_from.stable_since = now
                    self._pending[path] = moved_from
                    self._schedule(moved_from, now)
                else:
                    self._pending[path] = _PendingFile(
                        path=path,
                        state=state,
                        first_seen_at=now,
                        stable_since=now,
                    )
                    self._schedule(self._pending[path], now)
                return

            existing.last_error_at = None
            existing.expected_identity = None
            if existing.state is None or (existing.state.size, existing.state.mtime) != (
                state.size,
                state.mtime,
            ):
                existing.state = state
                existing.stable_since = now
            self._schedule(existing, now)

    def _drain_pending(self, pending: _PendingFile) -> None:
        with self._condition:
            try:
                state = FileState.from_stat_result(
                    pending.path,
                    self._stat(pending.path),
                    forced=True,
                )
            except OSError:
                return
            if state.identity != pending.identity:
                return
            pending.state = state
            self._emit(pending, state)

    def _schedule(self, pending: _PendingFile, now: float) -> None:
        stable_at = pending.stable_since + self.stable_interval_seconds
        max_wait_at = pending.first_seen_at + self.max_wait_seconds
        due_at = min(stable_at, max_wait_at)
        if due_at < now:
            due_at = now

        if pending.timer is not None:
            pending.cancel()

        token = pending.token + 1
        pending.token = token
        pending.timer = self._timer(max(0.0, due_at - now), lambda: self._due(pending.path, token))

    def _due(self, path: bytes | str, token: int) -> None:
        with self._condition:
            if self._closed:
                return
            pending = self._pending.get(path)
            if pending is None or pending.token != token:
                return
            self._in_flight += 1
            thread_ident = threading.get_ident()
            self._in_flight_threads.add(thread_ident)

            try:
                now = self._clock()
                try:
                    current = FileState.from_stat_result(path, self._stat(path))
                except FileNotFoundError:
                    if pending.state is None:
                        self._schedule(pending, now)
                    else:
                        self._pending.pop(path, None)
                        pending.cancel()
                    return
                except OSError:
                    if now - pending.first_seen_at >= self.max_wait_seconds:
                        self._pending.pop(path, None)
                        pending.cancel()
                        return
                    pending.last_error_at = now
                    pending.stable_since = now
                    self._schedule(pending, now)
                    return

                if current.identity != pending.identity:
                    self._pending.pop(path, None)
                    pending.cancel()
                    return

                max_wait_reached = now - pending.first_seen_at >= self.max_wait_seconds
                signature_changed = pending.state is None or (
                    pending.state.size,
                    pending.state.mtime,
                ) != (current.size, current.mtime)
                if signature_changed:
                    pending.state = current
                    pending.stable_since = now
                    pending.expected_identity = None
                    pending.last_error_at = None

                if max_wait_reached:
                    self._pending.pop(path, None)
                    pending.cancel()
                    forced_state = FileState(
                        path=path,
                        size=current.size,
                        mtime=current.mtime,
                        identity=current.identity,
                        forced=True,
                    )
                    self._emit(pending, forced_state)
                    return

                if not signature_changed and now - pending.stable_since >= self.stable_interval_seconds:
                    self._pending.pop(path, None)
                    pending.cancel()
                    self._emit(pending, current)
                    return

                self._schedule(pending, now)
            finally:
                self._in_flight -= 1
                self._in_flight_threads.discard(thread_ident)
                if self._closed:
                    self._condition.notify_all()

    def _emit(self, pending: _PendingFile, state: FileState) -> None:
        event = FileStableEvent(state)
        self.on_stable(event)
        if self.downstream is None:
            return
        if isinstance(self.downstream, FileSystemEventHandler):
            self.downstream.dispatch(event)
        else:
            self.downstream(event)


@dataclass
class _PendingFile:
    path: bytes | str
    state: FileState | None
    first_seen_at: float
    stable_since: float
    expected_identity: tuple[int, int] | None = None
    last_error_at: float | None = None
    token: int = 0
    timer: object | None = None

    @property
    def identity(self) -> tuple[int, int]:
        return self.state.identity if self.state is not None else self.expected_identity  # type: ignore[return-value]

    @classmethod
    def unresolved(
        cls,
        *,
        path: bytes | str,
        expected_identity: tuple[int, int],
        first_seen_at: float,
    ) -> _PendingFile:
        return cls(
            path=path,
            state=None,
            first_seen_at=first_seen_at,
            stable_since=first_seen_at,
            expected_identity=expected_identity,
            last_error_at=first_seen_at,
        )

    def cancel(self) -> None:
        if self.timer is not None:
            cancel = getattr(self.timer, "cancel", None)
            if callable(cancel):
                cancel()
            self.timer = None


def generate_sub_moved_events(
    src_dir_path: bytes | str,
    dest_dir_path: bytes | str,
) -> Generator[DirMovedEvent | FileMovedEvent]:
    """Generates an event list of :class:`DirMovedEvent` and
    :class:`FileMovedEvent` objects for all the files and directories within
    the given moved directory that were moved along with the directory.

    :param src_dir_path:
        The source path of the moved directory.
    :param dest_dir_path:
        The destination path of the moved directory.
    :returns:
        An iterable of file system events of type :class:`DirMovedEvent` and
        :class:`FileMovedEvent`.
    """
    for root, directories, filenames in os.walk(dest_dir_path):  # type: ignore[type-var]
        for directory in directories:
            full_path = os.path.join(root, directory)  # type: ignore[call-overload]
            renamed_path = src_dir_path + full_path[len(dest_dir_path) :] if src_dir_path else ""
            yield DirMovedEvent(renamed_path, full_path, is_synthetic=True)
        for filename in filenames:
            full_path = os.path.join(root, filename)  # type: ignore[call-overload]
            renamed_path = src_dir_path + full_path[len(dest_dir_path) :] if src_dir_path else ""
            yield FileMovedEvent(renamed_path, full_path, is_synthetic=True)


def generate_sub_created_events(src_dir_path: bytes | str) -> Generator[DirCreatedEvent | FileCreatedEvent]:
    """Generates an event list of :class:`DirCreatedEvent` and
    :class:`FileCreatedEvent` objects for all the files and directories within
    the given moved directory that were moved along with the directory.

    :param src_dir_path:
        The source path of the created directory.
    :returns:
        An iterable of file system events of type :class:`DirCreatedEvent` and
        :class:`FileCreatedEvent`.
    """
    for root, directories, filenames in os.walk(src_dir_path):  # type: ignore[type-var]
        for directory in directories:
            full_path = os.path.join(root, directory)  # type: ignore[call-overload]
            yield DirCreatedEvent(full_path, is_synthetic=True)
        for filename in filenames:
            full_path = os.path.join(root, filename)  # type: ignore[call-overload]
            yield FileCreatedEvent(full_path, is_synthetic=True)
