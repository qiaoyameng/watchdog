""":module: watchdog.events
:synopsis: File system events and event handlers.
:author: yesudeep@google.com (Yesudeep Mangalapilly)
:author: Mickaël Schoentgen <contact@tiger-222.fr>

Event Classes
-------------
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
import os.path
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from watchdog.utils.patterns import match_any_paths

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

EVENT_TYPE_MOVED = "moved"
EVENT_TYPE_DELETED = "deleted"
EVENT_TYPE_CREATED = "created"
EVENT_TYPE_MODIFIED = "modified"
EVENT_TYPE_CLOSED = "closed"
EVENT_TYPE_CLOSED_NO_WRITE = "closed_no_write"
EVENT_TYPE_OPENED = "opened"
EVENT_TYPE_STABLE = "stable"


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
    """File system event emitted when a file's size and modification time
    have remained unchanged for a configured stability window.

    Emitted only by :class:`StableFileEventHandler`, never directly by an
    observer. Carries the file state observed when the file was declared
    stable.
    """

    event_type = EVENT_TYPE_STABLE

    def __init__(self, src_path: bytes | str, *, size: int, mtime: float) -> None:
        super().__init__(src_path)
        self.size = size
        """Size of the file, in bytes, when it was declared stable."""
        self.mtime = mtime
        """Modification time of the file when it was declared stable."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: src_path={self.src_path!r}, size={self.size!r}, mtime={self.mtime!r}>"


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
        """Called when a file's size and modification time have remained
        unchanged for a configured stability window.

        Emitted only by :class:`StableFileEventHandler`, never directly by
        an observer.

        :param event:
            Event representing the stabilized file.
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


@dataclass
class _FileState:
    """Snapshot of the file state used to detect changes."""

    size: int
    mtime: float
    inode: int


@dataclass
class _PendingFile:
    """Bookkeeping for a file whose stability is being monitored."""

    state: _FileState
    first_seen: float
    last_change: float


class StableFileEventHandler(FileSystemEventHandler):
    """Emits :class:`FileStableEvent` once a file stops changing.

    Watches the raw event stream for file creations, modifications, moves
    and deletions, and declares a file *stable* when its size and mtime
    have not changed for ``stability_window`` seconds. This is useful for
    reacting to uploads or downloads that are written in several chunks:
    instead of acting on every intermediate write, consumers act on the
    single stable notification.

    The handler is passive: it records state from dispatched events and
    re-checks pending files when :meth:`check_pending` is called. Call
    :meth:`check_pending` periodically (for example from a timer thread)
    to drive stable-event emission. Both the clock and the stat function
    are injectable, which keeps the handler fully testable without
    sleeping.

    Repeated modifications reset the stability window. Files that keep
    changing are bounded by ``max_wait``: once a file has been pending for
    that long, the ``on_max_wait`` policy decides whether a stable event
    is emitted with the current state (``"emit"``) or the file is dropped
    silently (``"drop"``).

    Renames move the pending entry to the destination path. Deletions and
    stat failures drop the pending entry, and a file recreated at the same
    path is detected through its inode and never mistaken for the previous
    file.

    The raw event stream remains available: pass a ``delegate`` handler to
    receive every dispatched event untouched, or schedule additional
    handlers on the same watch.

    :param stability_window:
        Number of seconds size and mtime must stay unchanged before a
        file is declared stable.
    :type stability_window:
        ``float``
    :param max_wait:
        Maximum number of seconds a file may stay pending before the
        ``on_max_wait`` policy is applied. ``None`` waits forever.
    :type max_wait:
        ``float`` or ``None``
    :param on_max_wait:
        ``"emit"`` to emit a stable event with the current file state when
        ``max_wait`` elapses, ``"drop"`` to discard the file silently.
    :type on_max_wait:
        ``str``
    :param clock:
        Zero-argument callable returning the current time in seconds.
        Defaults to :func:`time.monotonic`.
    :param stat:
        Callable returning an :class:`os.stat_result` for a path. Defaults
        to :func:`os.stat`.
    :param delegate:
        Optional handler that receives every raw event dispatched to this
        handler, before any stability tracking happens.
    :type delegate:
        :class:`watchdog.events.FileSystemEventHandler` or ``None``
    """

    MAX_WAIT_EMIT = "emit"
    MAX_WAIT_DROP = "drop"

    def __init__(
        self,
        *,
        stability_window: float,
        max_wait: float | None = None,
        on_max_wait: str = MAX_WAIT_EMIT,
        clock: Callable[[], float] = time.monotonic,
        stat: Callable[[bytes | str], os.stat_result] = os.stat,
        delegate: FileSystemEventHandler | None = None,
    ) -> None:
        super().__init__()
        if stability_window <= 0:
            error_msg = f"stability_window must be positive, got {stability_window!r}"
            raise ValueError(error_msg)
        if max_wait is not None and max_wait <= 0:
            error_msg = f"max_wait must be positive or None, got {max_wait!r}"
            raise ValueError(error_msg)
        if on_max_wait not in (self.MAX_WAIT_EMIT, self.MAX_WAIT_DROP):
            error_msg = f"on_max_wait must be {self.MAX_WAIT_EMIT!r} or {self.MAX_WAIT_DROP!r}, got {on_max_wait!r}"
            raise ValueError(error_msg)

        self.stability_window = stability_window
        self.max_wait = max_wait
        self.on_max_wait = on_max_wait
        self._clock = clock
        self._stat = stat
        self._delegate = delegate
        self._pending: dict[bytes | str, _PendingFile] = {}

    @property
    def pending_paths(self) -> tuple[bytes | str, ...]:
        """(Read-only) Paths currently waiting to become stable."""
        return tuple(self._pending)

    def dispatch(self, event: FileSystemEvent) -> None:
        """Forwards the raw event to the delegate, then tracks it."""
        if self._delegate is not None:
            self._delegate.dispatch(event)
        super().dispatch(event)

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        if not event.is_directory:
            self._track(event.src_path)

    def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
        if not event.is_directory:
            self._track(event.src_path)

    def on_moved(self, event: DirMovedEvent | FileMovedEvent) -> None:
        if event.is_directory:
            return
        pending = self._pending.pop(event.src_path, None)
        if pending is None:
            # The moved-in file was not tracked yet; start tracking it.
            self._track(event.dest_path)
        else:
            # Same file at a new path: keep its history. Any entry already
            # pending at the destination is replaced, since that file was
            # just overwritten by the rename.
            self._pending[event.dest_path] = pending

    def on_deleted(self, event: DirDeletedEvent | FileDeletedEvent) -> None:
        if not event.is_directory:
            # Forget the file so a later creation at the same path is
            # treated as a brand new file, never as the deleted one.
            self._pending.pop(event.src_path, None)

    def check_pending(self, now: float | None = None) -> None:
        """Re-stats every pending file and emits events for stabilized ones.

        :param now:
            Timestamp to evaluate stability against. Defaults to the
            configured clock, which makes the method easy to drive from
            tests with a fake clock.
        :type now:
            ``float`` or ``None``
        """
        if now is None:
            now = self._clock()
        for path, pending in list(self._pending.items()):
            state = self._stat_state(path)
            if state is None:
                # Stat failed (file gone or unreadable): cannot confirm
                # stability, drop the entry.
                del self._pending[path]
                continue
            if state.inode != pending.state.inode:
                # Deleted and recreated without (or before) a delete event:
                # this is a different file, restart its window.
                self._pending[path] = _PendingFile(state=state, first_seen=now, last_change=now)
                continue
            if (state.size, state.mtime) != (pending.state.size, pending.state.mtime):
                pending.state = state
                pending.last_change = now
                continue
            if now - pending.last_change >= self.stability_window:
                del self._pending[path]
                self._emit_stable(path, state)
            elif self.max_wait is not None and now - pending.first_seen >= self.max_wait:
                del self._pending[path]
                if self.on_max_wait == self.MAX_WAIT_EMIT:
                    self._emit_stable(path, state)

    def stop(self, *, drain: bool = False) -> None:
        """Stops tracking and settles all pending files.

        :param drain:
            When ``False`` (the default) every pending file is discarded
            and no further events are emitted. When ``True`` each pending
            file is stat'ed one last time and a stable event is emitted
            with its current state; files that can no longer be stat'ed
            are skipped.
        :type drain:
            ``bool``
        """
        pending = list(self._pending.items())
        self._pending.clear()
        if drain:
            for path, _ in pending:
                state = self._stat_state(path)
                if state is not None:
                    self._emit_stable(path, state)

    def _track(self, path: bytes | str) -> None:
        now = self._clock()
        state = self._stat_state(path)
        if state is None:
            # The file vanished between the event and the stat call.
            self._pending.pop(path, None)
            return
        pending = self._pending.get(path)
        if pending is None or pending.state.inode != state.inode:
            self._pending[path] = _PendingFile(state=state, first_seen=now, last_change=now)
        elif (state.size, state.mtime) != (pending.state.size, pending.state.mtime):
            pending.state = state
            pending.last_change = now

    def _stat_state(self, path: bytes | str) -> _FileState | None:
        try:
            result = self._stat(path)
        except OSError:
            return None
        return _FileState(size=result.st_size, mtime=result.st_mtime, inode=result.st_ino)

    def _emit_stable(self, path: bytes | str, state: _FileState) -> None:
        self.on_stable(FileStableEvent(path, size=state.size, mtime=state.mtime))
