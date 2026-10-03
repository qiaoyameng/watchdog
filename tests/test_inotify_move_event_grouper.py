from __future__ import annotations

from types import SimpleNamespace

from watchdog.observers.inotify_c import InotifyConstants, WatchDescriptor
from watchdog.observers.inotify_move_event_grouper import (
    ExplainedInotifyEvent,
    InotifyMoveEventGroup, InotifyMoveEventGrouper,
    MovePairReason,
    PathedInotifyEvent,
)


class DummyMember:
    def __init__(self, member_id: int) -> None:
        self._id = member_id
        self.paired_moves: list[tuple[bytes, bytes]] = []

    def on_paired_move(self, source, destination, *, is_source, is_destination) -> None:
        self.paired_moves.append((source.path, destination.path))


def event(
    mask: int,
    cookie: int = 0,
    *,
    wd: int = 1,
    name: bytes = b"name",
    path: bytes = b"/root/name",
    generation: int = 1,
):
    return PathedInotifyEvent(
        SimpleNamespace(
            wd=WatchDescriptor(wd),
            mask=mask,
            cookie=cookie,
            name=name,
            is_move=bool(mask & InotifyConstants.IN_MOVE),
            is_moved_from=bool(mask & InotifyConstants.IN_MOVED_FROM),
            is_moved_to=bool(mask & InotifyConstants.IN_MOVED_TO),
        ),
        path,
        generation,
    )


def moved(cookie: int, *, to: bool = False, **kwargs):
    return event(InotifyConstants.IN_MOVED_TO if to else InotifyConstants.IN_MOVED_FROM, cookie, **kwargs)


def test_move_halves_pair_in_reverse_order():
    group = InotifyMoveEventGroup(delay=0.1)
    source = DummyMember(1)
    destination = DummyMember(2)
    group.register_member(source)
    group.register_member(destination)

    group.put_event(moved(42, to=True, wd=2, path=b"/destination/name"), 2)
    group.put_event(moved(42, wd=1, path=b"/source/name"), 1)

    assert group.read_event(1) == (
        moved(42, wd=1, path=b"/source/name"),
        moved(42, to=True, wd=2, path=b"/destination/name"),
    )
    assert group.read_event(2, block=False) is not None
    assert source.paired_moves == [(b"/source/name", b"/destination/name")]
    assert destination.paired_moves == [(b"/source/name", b"/destination/name")]
    group.close()


def test_expired_cookie_is_stale_for_late_counterpart():
    group = InotifyMoveEventGroup(delay=0.01)
    member = DummyMember(1)
    group.register_member(member)

    group.put_event(moved(7), 1)
    expired = group.read_event(1)
    assert isinstance(expired, ExplainedInotifyEvent)
    assert expired.reason is MovePairReason.PAIRING_TIMEOUT

    group.put_event(moved(7, to=True), 1)
    stale = group.read_event(1)
    assert isinstance(stale, ExplainedInotifyEvent)
    assert stale.reason is MovePairReason.STALE_COOKIE
    group.close()


def test_cookies_from_different_generations_do_not_pair():
    group = InotifyMoveEventGroup(delay=0.01)
    group.register_member(DummyMember(1))
    group.register_member(DummyMember(2))

    group.put_event(moved(9, generation=1), 1)
    group.put_event(moved(9, to=True, generation=2), 2)

    source_event = group.read_event(1)
    destination_event = group.read_event(2)
    assert isinstance(source_event, ExplainedInotifyEvent)
    assert isinstance(destination_event, ExplainedInotifyEvent)
    assert source_event.reason is MovePairReason.PAIRING_TIMEOUT
    assert destination_event.reason is MovePairReason.PAIRING_TIMEOUT
    group.close()


def test_cancelled_watch_releases_pending_half():
    group = InotifyMoveEventGroup(delay=1)
    group.register_member(DummyMember(1))

    group.put_event(moved(11, wd=3), 1)
    group.cancel_watch(1, 3)
    cancelled = group.read_event(1, block=False)
    assert isinstance(cancelled, ExplainedInotifyEvent)
    assert cancelled.reason is MovePairReason.WATCH_REMOVED

    group.put_event(moved(11, to=True), 1)
    stale = group.read_event(1, block=False)
    assert isinstance(stale, ExplainedInotifyEvent)
    assert stale.reason is MovePairReason.STALE_COOKIE
    group.close()


def test_pending_table_is_bounded():
    group = InotifyMoveEventGroup(delay=1, max_pending=1)
    member = DummyMember(1)
    group.register_member(member)

    group.put_event(moved(1), 1)
    group.put_event(moved(2), 1)
    overflow = group.read_event(1, block=False)
    assert isinstance(overflow, ExplainedInotifyEvent)
    assert overflow.reason is MovePairReason.PENDING_TABLE_OVERFLOW
    group.close()


def test_queue_overflow_degrades_pending_and_marks_member():
    group = InotifyMoveEventGroup(delay=1)
    group.register_member(DummyMember(1))

    group.put_event(moved(5), 1)
    group.queue_overflow(1)

    pending = group.read_event(1, block=False)
    marker = group.read_event(1, block=False)
    assert isinstance(pending, ExplainedInotifyEvent)
    assert pending.reason is MovePairReason.QUEUE_OVERFLOW
    assert marker == ExplainedInotifyEvent(None, MovePairReason.QUEUE_OVERFLOW)
    group.close()


def test_queue_overflow_is_deduplicated_per_generation():
    group = InotifyMoveEventGroup(delay=1)
    group.register_member(DummyMember(1))
    group.register_member(DummyMember(2))

    group.queue_overflow(3)
    group.queue_overflow(3)

    assert group.read_event(1, block=False) == ExplainedInotifyEvent(None, MovePairReason.QUEUE_OVERFLOW)
    assert group.read_event(2, block=False) == ExplainedInotifyEvent(None, MovePairReason.QUEUE_OVERFLOW)
    assert group.read_event(1, block=False) is None
    group.close()
