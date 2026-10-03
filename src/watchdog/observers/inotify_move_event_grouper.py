""":module: watchdog.observers.inotify_move_event_grouper
:synopsis: Pair Linux inotify move events within an explicit watch group.
"""

from __future__ import annotations

import enum
import threading
import time
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Union

if TYPE_CHECKING:
    from watchdog.observers.inotify_c import InotifyEvent

MemberId = int
Notification = tuple["MovePairMember", "PathedInotifyEvent", "PathedInotifyEvent", bool, bool]


class MovePairReason(enum.Enum):
    PAIRING_TIMEOUT = "pairing_timeout"
    STALE_COOKIE = "stale_cookie"
    WATCH_REMOVED = "watch_removed"
    QUEUE_OVERFLOW = "queue_overflow"
    PENDING_TABLE_OVERFLOW = "pending_table_overflow"


class PathedInotifyEvent(NamedTuple):
    ev: InotifyEvent
    path: bytes
    source_generation: int = 0


class ExplainedInotifyEvent(NamedTuple):
    event: PathedInotifyEvent | None
    reason: MovePairReason

GroupedInotifyEvent = Union[
    PathedInotifyEvent,
    tuple[PathedInotifyEvent, PathedInotifyEvent],
    ExplainedInotifyEvent,
]

class MovePairMember:
    _id: MemberId

    def on_paired_move(
        self,
        source: PathedInotifyEvent,
        destination: PathedInotifyEvent,
        *,
        is_source: bool,
        is_destination: bool,
    ) -> None:
        """Called after two halves have been authenticated by the kernel."""


@dataclass(frozen=True)
class _MoveKey:
    source_generation: int
    cookie: int


@dataclass
class _PendingMove:
    deadline: float
    source: PathedInotifyEvent | None = None
    source_member: MemberId | None = None
    destination: PathedInotifyEvent | None = None
    destination_member: MemberId | None = None
    expired: bool = False


@dataclass
class _PendingQueueEntry:
    key: _MoveKey
    member_id: int


class InotifyMoveEventGrouper:
    """Pair move events from watches that explicitly belong to one group."""

    def __init__(
        self,
        *,
        delay: float = 0.5,
        max_pending: int = 1024,
        close_when_empty: bool = True,
        delay_destinations: bool = False,
    ) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.delay = delay
        self.max_pending = max_pending
        self.close_when_empty = close_when_empty
        self.delay_destinations = delay_destinations
        self.preserve_arrival_order = not delay_destinations
        self._condition = threading.Condition(threading.RLock())
        self._members: weakref.WeakValueDictionary[int, MovePairMember] = weakref.WeakValueDictionary()
        self._active_members: set[int] = set()
        self._queues: dict[int, list[GroupedInotifyEvent]] = {}
        self._pending: dict[_MoveKey, _PendingMove] = {}
        self._overflowed_generations: set[int] = set()
        self._closed = False

    def register_member(self, member: MovePairMember) -> None:
        with self._condition:
            self._members[member._id] = member
            self._active_members.add(member._id)
            self._queues.setdefault(member._id, [])
            self._closed = False
            self._condition.notify_all()

    def unregister_member(self, member_id: MemberId) -> None:
        with self._condition:
            self._active_members.discard(member_id)
            self._members.pop(member_id, None)
            self._queues.pop(member_id, None)
            self._drop_member_pending_locked(member_id)
            if self.close_when_empty and not self._active_members:
                self._closed = True
            self._condition.notify_all()

    def put_event(
        self,
        event: PathedInotifyEvent,
        member_id: MemberId,
        *,
        source_generation: int | None = None,
    ) -> None:
        if source_generation is not None:
            event = PathedInotifyEvent(event.ev, event.path, source_generation)
        with self._condition:
            self._expire_pending_locked()
            if self._closed or member_id not in self._active_members:
                return
            if not event.ev.is_move:
                self._enqueue_locked(member_id, event)
                self._condition.notify_all()
                return

            key = _MoveKey(event.source_generation, event.ev.cookie)
            pending = self._pending.get(key)
            if event.ev.is_moved_from:
                notifications = self._put_half_locked(key, pending, event, member_id, is_source=True)
            else:
                notifications = self._put_half_locked(key, pending, event, member_id, is_source=False)
            self._condition.notify_all()

        for member, source, destination, is_source, is_destination in notifications:
            member.on_paired_move(source, destination, is_source=is_source, is_destination=is_destination)

    def read_event(self, member_id: MemberId, *, block: bool = True) -> GroupedInotifyEvent | None:
        with self._condition:
            while True:
                self._expire_pending_locked()
                queue = self._queues.get(member_id)
                if queue:
                    head = queue[0]
                    if not isinstance(head, _PendingQueueEntry):
                        return queue.pop(0)
                    pending = self._pending.get(head.key)
                    if pending is None:
                        queue.pop(0)
                        continue
                    timeout = pending.deadline - time.monotonic()
                    if timeout > 0:
                        self._condition.wait(timeout)
                    else:
                        self._expire_pending_locked()
                    continue
                if self._closed or member_id not in self._active_members or not block:
                    return None
                timeout = self._next_deadline_locked(member_id)
                if timeout is not None and timeout <= 0:
                    continue
                self._condition.wait(timeout)

    def cancel_watch(self, member_id: MemberId, watch_descriptor: int) -> None:
        with self._condition:
            for key, pending in list(self._pending.items()):
                if self._is_half_for_watch(pending.source_member, pending.source, member_id, watch_descriptor):
                    self._finish_cancelled_half_locked(key, pending, pending.source, member_id, True)
                elif self._is_half_for_watch(
                    pending.destination_member,
                    pending.destination,
                    member_id,
                    watch_descriptor,
                ):
                    self._finish_cancelled_half_locked(key, pending, pending.destination, member_id, False)
            self._condition.notify_all()

    def queue_overflow(self, source_generation: int) -> None:
        if source_generation in self._overflowed_generations:
            return
        self._overflowed_generations.add(source_generation)
        with self._condition:
            marker = ExplainedInotifyEvent(None, MovePairReason.QUEUE_OVERFLOW)
            for member_id in list(self._active_members):
                for pending in list(self._pending.values()):
                    self._enqueue_pending_halves_locked(pending, member_id, MovePairReason.QUEUE_OVERFLOW)
                self._enqueue_locked(member_id, marker)
            self._pending.clear()
            self._condition.notify_all()

    def get_queued_moved_from_event(self, cookie: int) -> PathedInotifyEvent | None:
        with self._condition:
            for pending in self._pending.values():
                if not pending.expired and pending.source is not None and pending.source.ev.cookie == cookie:
                    return pending.source
        return None

    def close(self) -> None:
        with self._condition:
            self._active_members.clear()
            self._members.clear()
            self._queues.clear()
            self._pending.clear()
            self._closed = True
            self._condition.notify_all()

    @staticmethod
    def _is_half_for_watch(
        owner: int | None,
        event: PathedInotifyEvent | None,
        member_id: int,
        watch_descriptor: int,
    ) -> bool:
        return owner == member_id and event is not None and event.ev.wd == watch_descriptor

    def _put_half_locked(
        self,
        key: _MoveKey,
        pending: _PendingMove | None,
        event: PathedInotifyEvent,
        member_id: int,
        *,
        is_source: bool,
    ) -> list[Notification]:
        if pending is None:
            if len(self._pending) >= self.max_pending:
                self._enqueue_locked(member_id, ExplainedInotifyEvent(event, MovePairReason.PENDING_TABLE_OVERFLOW))
            elif not is_source and not self.delay_destinations:
                self._enqueue_locked(member_id, event)
            else:
                self._pending[key] = _PendingMove(
                    deadline=time.monotonic() + self.delay,
                    source=event if is_source else None,
                    source_member=member_id if is_source else None,
                    destination=None if is_source else event,
                    destination_member=None if is_source else member_id,
                )
                if self.preserve_arrival_order:
                    self._enqueue_locked(member_id, _PendingQueueEntry(key, member_id))
            return []

        counterpart = pending.destination if is_source else pending.source
        counterpart_member = pending.destination_member if is_source else pending.source_member
        if not is_source and not self.delay_destinations:
            if pending.expired or counterpart is None or counterpart_member is None:
                self._pending.pop(key, None)
                self._enqueue_locked(member_id, event)
                return []
            source, destination = counterpart, event
            source_member, destination_member = counterpart_member, member_id
        else:
            if (
                pending.expired
                or counterpart is None
                or counterpart_member is None
                or pending.deadline <= time.monotonic()
            ):
                self._enqueue_locked(member_id, ExplainedInotifyEvent(event, MovePairReason.STALE_COOKIE))
                return []
            source = event if is_source else counterpart
            destination = counterpart if is_source else event
            source_member = member_id if is_source else counterpart_member
            destination_member = counterpart_member if is_source else member_id

        self._pending.pop(key, None)
        pair = (source, destination)
        self._replace_pending_queue_entry_locked(key, source_member, pair)
        if destination_member != source_member:
            self._replace_pending_queue_entry_locked(key, destination_member, pair)
        return self._pair_notifications_locked(source, destination, source_member, destination_member)

    def _pair_notifications_locked(
        self,
        source: PathedInotifyEvent,
        destination: PathedInotifyEvent,
        source_member: int,
        destination_member: int,
    ) -> list[Notification]:
        member_ids = {source_member, destination_member} & self._active_members
        ordered_ids = sorted(member_ids, key=lambda member_id: (member_id != destination_member, member_id))
        return [
            (
                self._members[member_id],
                source,
                destination,
                member_id == source_member,
                member_id == destination_member,
            )
            for member_id in ordered_ids
            if member_id in self._members
        ]

    def _expire_pending_locked(self) -> None:
        now = time.monotonic()
        for key, pending in list(self._pending.items()):
            if pending.expired:
                if pending.deadline <= now:
                    self._pending.pop(key, None)
                continue
            if pending.deadline > now:
                continue
            pending.expired = True
            pending.deadline = now + self.delay
            self._release_pending_queue_entry_locked(pending, pending.source_member, key, pending.source)
            self._release_pending_queue_entry_locked(pending, pending.destination_member, key, pending.destination)
            return

    def _finish_cancelled_half_locked(
        self,
        key: _MoveKey,
        pending: _PendingMove,
        event: PathedInotifyEvent | None,
        member_id: int,
        is_source: bool,
    ) -> None:
        if event is None:
            return
        if is_source:
            pending.source = None
            pending.source_member = None
        else:
            pending.destination = None
            pending.destination_member = None
        pending.expired = True
        pending.deadline = time.monotonic() + self.delay
        self._enqueue_locked(member_id, ExplainedInotifyEvent(event, MovePairReason.WATCH_REMOVED))

    def _replace_pending_queue_entry_locked(self, key: _MoveKey, member_id: int, event: GroupedInotifyEvent) -> None:
        queue = self._queues.setdefault(member_id, [])
        for index, queued in enumerate(queue):
            if isinstance(queued, _PendingQueueEntry) and queued.key == key and queued.member_id == member_id:
                queue[index] = event
                return
        queue.append(event)

    def _release_pending_queue_entry_locked(
        self,
        pending: _PendingMove,
        member_id: int | None,
        key: _MoveKey,
        event: PathedInotifyEvent | None,
    ) -> None:
        if member_id is None or event is None:
            return
        self._replace_pending_queue_entry_locked(
            key,
            member_id,
            ExplainedInotifyEvent(event, MovePairReason.PAIRING_TIMEOUT),
        )

    def _enqueue_pending_halves_locked(
        self,
        pending: _PendingMove,
        member_id: int | None,
        reason: MovePairReason,
    ) -> None:
        if member_id is None:
            return
        if member_id == pending.source_member:
            event = pending.source
        elif member_id == pending.destination_member:
            event = pending.destination
        else:
            event = None
        if event is not None:
            self._enqueue_locked(member_id, ExplainedInotifyEvent(event, reason))

    def _enqueue_locked(self, member_id: int, event: GroupedInotifyEvent) -> None:
        self._queues.setdefault(member_id, []).append(event)

    def _next_deadline_locked(self, member_id: int) -> float | None:
        now = time.monotonic()
        deadlines = [
            pending.deadline - now
            for pending in self._pending.values()
            if not pending.expired
            if (
                (pending.source_member == member_id and pending.source is not None)
                or (pending.destination_member == member_id and pending.destination is not None)
            )
        ]
        return min(deadlines) if deadlines else None

    def _drop_member_pending_locked(self, member_id: int) -> None:
        for key, pending in list(self._pending.items()):
            if pending.source_member == member_id:
                pending.source = None
                pending.source_member = None
            if pending.destination_member == member_id:
                pending.destination = None
                pending.destination_member = None
            if pending.source is None and pending.destination is None:
                self._pending.pop(key, None)


class InotifyMoveEventGroup(InotifyMoveEventGrouper):
    """Explicit group shared by watches that may see opposite rename halves."""

    def __init__(self, *, delay: float = 0.5, max_pending: int = 1024) -> None:
        super().__init__(delay=delay, max_pending=max_pending, close_when_empty=False, delay_destinations=True)
