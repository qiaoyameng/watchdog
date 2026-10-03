""":module: watchdog.observers.inotify_buffer
:synopsis: queue-like class for ``Inotify`` to group move events.
:author: thomas.amland@gmail.com (Thomas Amland)
:author: Mickaël Schoentgen <contact@tiger-222.fr>
:author: Joachim Coenen <joachimcoenen@icloud.com>
:platforms: linux 2.6.13+
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, NamedTuple, Optional, Union, cast

from watchdog.utils import UnsupportedLibcError, platform
from watchdog.utils.delayed_queue import DelayedQueue

if TYPE_CHECKING:
    from typing import TypeAlias

    from watchdog.observers.inotify import InotifyWatchGroup
    from watchdog.observers.inotify_c import InotifyEvent, WatchDescriptor

logger = logging.getLogger(__name__)


class PathedInotifyEvent(NamedTuple):
    """An InotifyEvent and its full source path"""

    ev: InotifyEvent
    path: bytes


GroupedInotifyEvent: TypeAlias = Union[PathedInotifyEvent, tuple[PathedInotifyEvent, PathedInotifyEvent]]


class InotifyMoveEventGrouper:
    """A queue-like class for `Inotify` that holds IN_MOVE_FROM events for
    `delay` seconds. During this time, IN_MOVED_FROM and IN_MOVED_TO events are
    paired.
    """

    delay = 0.5

    def __init__(self) -> None:
        self._queue: DelayedQueue[GroupedInotifyEvent] = DelayedQueue(self.delay)

    def read_event(self) -> GroupedInotifyEvent | None:
        """Returns a single event or a tuple of from/to events in case of a
        paired move event. If this buffer has been closed, raise the Closed
        exception.
        """
        return self._queue.get()

    def put_event(self, event: PathedInotifyEvent) -> None:
        """Add an event to the `queue`. When adding an IN_MOVE_TO event, remove
        the previous added matching IN_MOVE_FROM event and add them back to the
        queue as a tuple.
        """
        logger.debug("in-event %s", event)
        # Only add delay for unmatched move_from events
        should_delay = event.ev.is_moved_from

        grouped_event = self._group_moved_to_event(event) if event.ev.is_moved_to else event

        self._queue.put(grouped_event, delay=should_delay)

    def put_grouped(self, grouped_event: GroupedInotifyEvent) -> None:
        """Add an already-paired move event tuple (or a plain event) to the
        queue without any delay. Used by `InotifyPairingGroup` members to
        publish moves that were paired across watch groups.
        """
        logger.debug("in-event %s", grouped_event)
        self._queue.put(grouped_event, delay=False)

    def remove_pending(self, event: PathedInotifyEvent) -> PathedInotifyEvent | None:
        """Remove a specific queued event, returning it if it was still queued.
        Used to consume a pending IN_MOVED_FROM event when it gets paired by
        another member of an `InotifyPairingGroup`. Returns None if the event
        was already released to a reader (e.g. its delay expired).
        """

        def matching(queued: GroupedInotifyEvent) -> bool:
            return isinstance(queued, PathedInotifyEvent) and queued == event

        return cast(Optional[PathedInotifyEvent], self._queue.remove(matching))

    def _group_moved_to_event(self, to_event: PathedInotifyEvent) -> GroupedInotifyEvent:
        """Group any matching move events by checking if a matching move_from is
        in delay queue already and removing it"""
        cookie = to_event.ev.cookie

        def matching_from_event(event: GroupedInotifyEvent) -> bool:
            return isinstance(event, PathedInotifyEvent) and event.ev.is_moved_from and event.ev.cookie == cookie

        # Check if move_from is in delayqueue already
        from_event = cast(PathedInotifyEvent, self._queue.remove(matching_from_event))
        if from_event is None:
            logger.debug("could not find matching move_from event")

        return (from_event, to_event) if from_event is not None else to_event

    def get_queued_moved_from_event(self, cookie: int) -> PathedInotifyEvent | None:
        """Finds a queued IN_MOVED_FROM event with the give cookie, but does not
        remove it."""

        def matching_from_event(event: GroupedInotifyEvent) -> bool:
            return isinstance(event, PathedInotifyEvent) and event.ev.is_moved_from and event.ev.cookie == cookie

        return cast(PathedInotifyEvent, self._queue.find(matching_from_event))

    def close(self) -> None:
        """closes the queue"""
        self._queue.close()


class PendingMovedFrom(NamedTuple):
    """A pending IN_MOVED_FROM event waiting for its IN_MOVED_TO counterpart."""

    owner: InotifyWatchGroup
    """The member watch group that received the IN_MOVED_FROM event."""
    event: PathedInotifyEvent
    """The event, with its path expressed in the owner's path space."""
    generation: int
    """The generation of the inotify instance the event was received from."""
    inserted_at: float
    """`time.monotonic()` timestamp of insertion, for staleness checks."""


class InotifyPairingGroup:
    """An explicit group of `InotifyWatchGroup` instances that share one
    pairable event source.

    Watches (typically one per monitored root directory) join the group by
    passing the same `InotifyPairingGroup` instance to their
    `InotifyWatchGroup`. Within the group, an IN_MOVED_FROM event received by
    one member is paired with the IN_MOVED_TO event received by another
    member, so a file moved between two separately watched roots is reported
    as a single move to both ends.

    Pairing is based exclusively on the kernel-provided event cookie together
    with the generation of the inotify instance the events were received
    from. File names and event arrival times are never used for pairing.
    The generation is bumped whenever the inotify instance is recreated or
    its event queue overflows, so stale cookies from a previous generation
    can never pair with fresh events.

    Dispatch and de-duplication rules
    ---------------------------------
    - A raw inotify event is delivered only to the member that owns the watch
      it arrived on.
    - A paired move is delivered as a ``(moved_from, moved_to)`` tuple to BOTH
      the source member and the target member, exactly once each. If both
      ends belong to the same member, the tuple is delivered only once.
    - The from-event path is expressed in the source member's path space and
      the to-event path in the target member's path space; both members
      receive the same tuple.
    - A pending IN_MOVED_FROM pairs with at most one IN_MOVED_TO (the table
      entry is consumed on pairing), and vice versa.

    Degradation rules (events remain explainable)
    ---------------------------------------------
    - If one end of a move is not covered by any member of the group, the
      other end is delivered unpaired (surfacing as a deletion or creation).
    - If no IN_MOVED_TO arrives within the pairing delay, the pending
      IN_MOVED_FROM is released to its owner unpaired.
    - If a watch is removed while it has a pending IN_MOVED_FROM, the pairing
      entry is cancelled; the event itself is still delivered unpaired.
    - If the kernel event queue overflows (IN_Q_OVERFLOW), all pending
      pairings are discarded and affected events are delivered unpaired.
    - The pending-pair table is bounded by ``max_pending``; when it is full,
      the oldest entries are evicted and their events degrade to unpaired.

    This mode is only supported on Linux.
    """

    def __init__(self, *, max_pending: int = 1024, delay: float | None = None) -> None:
        if not platform.is_linux():
            msg = f"{type(self).__name__} is only supported on Linux."
            raise UnsupportedLibcError(msg)
        self.max_pending = max_pending
        self.delay = InotifyMoveEventGrouper.delay if delay is None else delay
        self._lock = threading.Lock()
        # Maps (inotify instance generation, cookie) -> pending event.
        # Insertion order is used for bounded FIFO eviction.
        self._pending: dict[tuple[int, int], PendingMovedFrom] = {}

    def __len__(self) -> int:
        """The number of currently pending (unpaired) IN_MOVED_FROM events."""
        with self._lock:
            return len(self._pending)

    def put_moved_from(self, owner: InotifyWatchGroup, event: PathedInotifyEvent, generation: int) -> None:
        """Register a pending IN_MOVED_FROM event for pairing. The event must
        also be queued (delayed) in the owner's own event grouper, so that it
        is released unpaired if no IN_MOVED_TO pairs with it in time.
        """
        with self._lock:
            self._purge_stale_locked()
            while len(self._pending) >= self.max_pending:
                evicted_key = next(iter(self._pending))
                logger.debug("pending move table full, evicting %s", self._pending[evicted_key].event)
                del self._pending[evicted_key]
            key = (generation, event.ev.cookie)
            self._pending[key] = PendingMovedFrom(owner, event, generation, time.monotonic())

    def take_moved_from(self, generation: int, cookie: int) -> PendingMovedFrom | None:
        """Remove and return the pending IN_MOVED_FROM event matching the
        given inotify instance generation and cookie, or None if there is no
        (fresh) matching event. Only events from the same generation of the
        same inotify instance can pair.
        """
        with self._lock:
            entry = self._pending.pop((generation, cookie), None)
            if entry is None:
                return None
            if self._is_stale(entry):
                logger.debug("discarding stale pending move event %s", entry.event)
                return None
            return entry

    def cancel_pending(self, owner: InotifyWatchGroup, wd: WatchDescriptor | None = None) -> None:
        """Cancel pending pairings of a member, e.g. because a watch was
        removed or the member was deactivated. If ``wd`` is given, only
        pairings for events of that watch are cancelled. The affected events
        are still delivered to the owner unpaired by its own event grouper.
        """
        with self._lock:
            keys = [
                key
                for key, entry in self._pending.items()
                if entry.owner is owner and (wd is None or entry.event.ev.wd == wd)
            ]
            for key in keys:
                del self._pending[key]

    def on_queue_overflow(self) -> None:
        """Discard all pending pairings after the kernel reported
        IN_Q_OVERFLOW. Events may have been lost and cookies may be reused,
        so no pending event can be trusted to pair anymore.
        """
        with self._lock:
            if self._pending:
                logger.debug("inotify queue overflow, discarding %d pending move events", len(self._pending))
                self._pending.clear()

    def _is_stale(self, entry: PendingMovedFrom) -> bool:
        return time.monotonic() - entry.inserted_at > self.delay

    def _purge_stale_locked(self) -> None:
        stale_keys = [key for key, entry in self._pending.items() if self._is_stale(entry)]
        for key in stale_keys:
            del self._pending[key]
