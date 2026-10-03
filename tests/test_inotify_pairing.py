from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import pytest

from watchdog.utils import UnsupportedLibcError, platform

if not platform.is_linux():
    pytest.skip("GNU/Linux only.", allow_module_level=True)

from watchdog.events import FileMovedEvent, FileSystemEventHandler
from watchdog.observers.inotify import InotifyObserver, InotifyWatchGroup
from watchdog.observers.inotify_c import (
    WATCHDOG_ALL_EVENTS,
    CallbackId,
    InotifyConstants,
    InotifyEvent,
    InotifyFD,
    Mask,
    WatchDescriptor,
)
from watchdog.observers.inotify_move_event_grouper import (
    GroupedInotifyEvent,
    InotifyMoveEventGrouper,
    InotifyPairingGroup,
    PathedInotifyEvent,
)

from .shell import mkdir, mv, touch

if TYPE_CHECKING:
    from collections.abc import Callable

    from watchdog.observers.inotify_c import WatchCallback


@pytest.fixture(autouse=True)
def _short_grouper_delay(monkeypatch):
    """Speed up tests: release unpaired IN_MOVED_FROM events after 0.2s."""
    monkeypatch.setattr(InotifyMoveEventGrouper, "delay", 0.2)


class FakeInotifyFD:
    """Stand-in for InotifyFD that lets tests drive raw inotify event
    sequences (out-of-order delivery, stale cookies, cancellations) without
    touching the kernel.
    """

    def __init__(self) -> None:
        self.generation = 1
        self._wd_counter = 0
        self._wd_for_path: dict[bytes, WatchDescriptor] = {}
        self.registered_groups: list[InotifyPairingGroup] = []

    def add_callback(self, path: bytes, mask: Mask, callback: WatchCallback, id_: CallbackId) -> WatchDescriptor:
        wd = self._wd_for_path.get(path)
        if wd is None:
            self._wd_counter += 1
            wd = WatchDescriptor(self._wd_counter)
            self._wd_for_path[path] = wd
        return wd

    def remove_callbacks(self, callbacks: list[tuple[WatchDescriptor, CallbackId]]) -> None:
        pass

    def register_pairing_group(self, group: InotifyPairingGroup) -> None:
        self.registered_groups.append(group)

    def unregister_pairing_group(self, group: InotifyPairingGroup) -> None:
        self.registered_groups.remove(group)


def make_watch_group(
    inotify_fd: FakeInotifyFD | InotifyFD,
    path: bytes,
    pairing_group: InotifyPairingGroup | None = None,
    **kwargs,
) -> InotifyWatchGroup:
    return InotifyWatchGroup(
        inotify_fd,  # type: ignore[arg-type]
        path,
        event_mask=WATCHDOG_ALL_EVENTS,
        pairing_group=pairing_group,
        **kwargs,
    )


def wd_of(watch_group: InotifyWatchGroup, path: bytes) -> WatchDescriptor:
    return watch_group._active_callbacks_by_path[path]  # noqa: SLF001


def moved_from(watch_group: InotifyWatchGroup, dir_path: bytes, cookie: int, name: bytes = b"f") -> None:
    wd = wd_of(watch_group, dir_path)
    watch_group.on_event(InotifyEvent(wd, Mask(InotifyConstants.IN_MOVED_FROM), cookie, name))


def moved_to(watch_group: InotifyWatchGroup, dir_path: bytes, cookie: int, name: bytes = b"f") -> None:
    wd = wd_of(watch_group, dir_path)
    watch_group.on_event(InotifyEvent(wd, Mask(InotifyConstants.IN_MOVED_TO), cookie, name))


def wait_for_move_event(read_event: Callable[[], GroupedInotifyEvent]) -> GroupedInotifyEvent:
    while True:
        event = read_event()
        if not isinstance(event, PathedInotifyEvent) or event.ev.is_move:
            return event


@pytest.mark.timeout(5)
def test_out_of_order_events_do_not_pair(tmp_path):
    """A MOVED_TO arriving before its MOVED_FROM must not pair retroactively."""
    src_dir = str(tmp_path / "src").encode()
    dst_dir = str(tmp_path / "dst").encode()
    mkdir(src_dir)
    mkdir(dst_dir)
    pairing = InotifyPairingGroup()
    fake_fd = FakeInotifyFD()
    group_src = make_watch_group(fake_fd, src_dir, pairing)
    group_dst = make_watch_group(fake_fd, dst_dir, pairing)

    moved_to(group_dst, dst_dir, cookie=7)
    event = group_dst.read_event()
    assert isinstance(event, PathedInotifyEvent)
    assert event.ev.is_moved_to
    assert event.path == dst_dir + b"/f"

    moved_from(group_src, src_dir, cookie=7)
    event = group_src.read_event()
    assert isinstance(event, PathedInotifyEvent)
    assert event.ev.is_moved_from
    assert event.path == src_dir + b"/f"

    group_src.deactivate()
    group_dst.deactivate()


@pytest.mark.timeout(5)
def test_stale_cookie_from_previous_generation_does_not_pair(tmp_path):
    """Cookies are only valid within one generation of an inotify instance."""
    src_dir = str(tmp_path / "src").encode()
    dst_dir = str(tmp_path / "dst").encode()
    mkdir(src_dir)
    mkdir(dst_dir)
    pairing = InotifyPairingGroup()
    fake_fd = FakeInotifyFD()
    group_src = make_watch_group(fake_fd, src_dir, pairing)
    group_dst = make_watch_group(fake_fd, dst_dir, pairing)

    fake_fd.generation = 1
    moved_from(group_src, src_dir, cookie=9)

    # The inotify instance was recreated (or its queue overflowed):
    fake_fd.generation = 2
    moved_to(group_dst, dst_dir, cookie=9)

    event = group_dst.read_event()
    assert isinstance(event, PathedInotifyEvent)
    assert event.ev.is_moved_to

    event = group_src.read_event()
    assert isinstance(event, PathedInotifyEvent)
    assert event.ev.is_moved_from

    group_src.deactivate()
    group_dst.deactivate()


@pytest.mark.timeout(5)
def test_same_generation_cookie_pairs_across_groups(tmp_path):
    src_dir = str(tmp_path / "src").encode()
    dst_dir = str(tmp_path / "dst").encode()
    mkdir(src_dir)
    mkdir(dst_dir)
    pairing = InotifyPairingGroup()
    fake_fd = FakeInotifyFD()
    group_src = make_watch_group(fake_fd, src_dir, pairing)
    group_dst = make_watch_group(fake_fd, dst_dir, pairing)

    moved_from(group_src, src_dir, cookie=3)
    moved_to(group_dst, dst_dir, cookie=3, name=b"g")

    # Both ends receive the same paired event exactly once.
    for group in (group_src, group_dst):
        frm, to = wait_for_move_event(group.read_event)
        assert frm.path == src_dir + b"/f"
        assert to.path == dst_dir + b"/g"

    group_src.deactivate()
    group_dst.deactivate()


@pytest.mark.timeout(5)
def test_cancelled_watch_does_not_pair(tmp_path):
    """Removing a watch cancels its pending pairings; events stay explainable."""
    src_dir = str(tmp_path / "src").encode()
    dst_dir = str(tmp_path / "dst").encode()
    mkdir(src_dir)
    mkdir(dst_dir)
    pairing = InotifyPairingGroup()
    fake_fd = FakeInotifyFD()
    group_src = make_watch_group(fake_fd, src_dir, pairing)
    group_dst = make_watch_group(fake_fd, dst_dir, pairing)

    moved_from(group_src, src_dir, cookie=5)
    group_src.on_watch_deleted(wd_of(group_src, src_dir))

    moved_to(group_dst, dst_dir, cookie=5)
    event = group_dst.read_event()
    assert isinstance(event, PathedInotifyEvent)
    assert event.ev.is_moved_to

    # The pending MOVED_FROM is still delivered to its owner, unpaired.
    event = group_src.read_event()
    assert isinstance(event, PathedInotifyEvent)
    assert event.ev.is_moved_from

    group_src.deactivate()
    group_dst.deactivate()


@pytest.mark.timeout(5)
def test_pending_table_is_bounded(tmp_path):
    """When the pending table is full, the oldest entry is evicted and its
    event degrades to an unpaired one."""
    src_dir = str(tmp_path / "src").encode()
    dst_dir = str(tmp_path / "dst").encode()
    mkdir(src_dir)
    mkdir(dst_dir)
    pairing = InotifyPairingGroup(max_pending=2)
    fake_fd = FakeInotifyFD()
    group_src = make_watch_group(fake_fd, src_dir, pairing)
    group_dst = make_watch_group(fake_fd, dst_dir, pairing)

    for cookie in (1, 2, 3):
        moved_from(group_src, src_dir, cookie=cookie)
    assert len(pairing) == 2

    # cookie 1 was evicted: no pairing.
    moved_to(group_dst, dst_dir, cookie=1)
    event = group_dst.read_event()
    assert isinstance(event, PathedInotifyEvent)
    assert event.ev.is_moved_to

    # cookie 3 is still pending: pairs.
    moved_to(group_dst, dst_dir, cookie=3)
    frm, to = wait_for_move_event(group_dst.read_event)
    assert frm.path == src_dir + b"/f"
    assert to.path == dst_dir + b"/f"

    group_src.deactivate()
    group_dst.deactivate()


@pytest.mark.timeout(5)
def test_queue_overflow_discards_pending_pairings(tmp_path):
    src_dir = str(tmp_path / "src").encode()
    dst_dir = str(tmp_path / "dst").encode()
    mkdir(src_dir)
    mkdir(dst_dir)
    pairing = InotifyPairingGroup()
    fake_fd = FakeInotifyFD()
    group_src = make_watch_group(fake_fd, src_dir, pairing)
    group_dst = make_watch_group(fake_fd, dst_dir, pairing)

    moved_from(group_src, src_dir, cookie=4)
    assert len(pairing) == 1
    pairing.on_queue_overflow()
    assert len(pairing) == 0

    moved_to(group_dst, dst_dir, cookie=4)
    event = group_dst.read_event()
    assert isinstance(event, PathedInotifyEvent)
    assert event.ev.is_moved_to

    group_src.deactivate()
    group_dst.deactivate()


@pytest.mark.timeout(5)
def test_inotify_fd_queue_overflow_bumps_generation_and_notifies_groups():
    inotify_fd = InotifyFD.get_instance()
    pairing = InotifyPairingGroup()
    inotify_fd.register_pairing_group(pairing)
    inotify_fd.register_pairing_group(pairing)  # ref-counted
    try:
        generation = inotify_fd.generation
        overflow = InotifyEvent(WatchDescriptor(-1), Mask(InotifyConstants.IN_Q_OVERFLOW), 0, b"")
        inotify_fd.handle_event(overflow)
        assert inotify_fd.generation > generation
        inotify_fd.unregister_pairing_group(pairing)
        assert pairing in inotify_fd._pairing_groups  # noqa: SLF001
    finally:
        inotify_fd.unregister_pairing_group(pairing)
        inotify_fd.unregister_pairing_group(pairing)


def test_pairing_group_is_linux_only(monkeypatch):
    monkeypatch.setattr(platform, "is_linux", lambda: False)
    with pytest.raises(UnsupportedLibcError):
        InotifyPairingGroup()


@pytest.mark.timeout(5)
def test_cross_root_file_move_is_paired(p):
    """A real rename between two watched roots is reported as one move to
    both watch groups."""
    mkdir(p("root1"))
    mkdir(p("root2"))
    touch(p("root1", "a"))

    pairing = InotifyPairingGroup()
    inotify_fd = InotifyFD.get_instance()
    group1 = make_watch_group(inotify_fd, p("root1").encode(), pairing)
    group2 = make_watch_group(inotify_fd, p("root2").encode(), pairing)

    mv(p("root1", "a"), p("root2", "b"))

    for group in (group1, group2):
        frm, to = wait_for_move_event(group.read_event)
        assert frm.path == p("root1", "a").encode()
        assert to.path == p("root2", "b").encode()

    group1.deactivate()
    group2.deactivate()


@pytest.mark.timeout(5)
def test_cross_root_directory_move_updates_descendant_watches(p):
    """After a directory moves across roots, descendant watches follow it:
    the source group forgets the old paths, the target group watches the
    tree at its new location."""
    mkdir(p("root1", "sub", "inner"), parents=True)
    mkdir(p("root2"))
    touch(p("root1", "sub", "inner", "f"))

    pairing = InotifyPairingGroup()
    inotify_fd = InotifyFD.get_instance()
    group1 = make_watch_group(inotify_fd, p("root1").encode(), pairing, is_recursive=True)
    group2 = make_watch_group(inotify_fd, p("root2").encode(), pairing, is_recursive=True)

    mv(p("root1", "sub"), p("root2", "sub"))

    for group in (group1, group2):
        frm, to = wait_for_move_event(group.read_event)
        assert frm.path == p("root1", "sub").encode()
        assert to.path == p("root2", "sub").encode()

    # The source group detached the moved subtree...
    moved_prefix = p("root1", "sub").encode()
    assert not any(path.startswith(moved_prefix) for path in group1._active_callbacks_by_path)  # noqa: SLF001
    # ...and the target group watches it at its new location.
    assert p("root2", "sub").encode() in group2._active_callbacks_by_path  # noqa: SLF001
    assert p("root2", "sub", "inner").encode() in group2._active_callbacks_by_path  # noqa: SLF001

    # New events inside the moved tree are reported by the target group.
    touch(p("root2", "sub", "inner", "g"))
    expected = p("root2", "sub", "inner", "g").encode()
    event = group2.read_event()
    while not (isinstance(event, PathedInotifyEvent) and event.path == expected):
        event = group2.read_event()
    assert event.ev.mask & (InotifyConstants.IN_CREATE | InotifyConstants.IN_OPEN)

    group1.deactivate()
    group2.deactivate()


@pytest.mark.timeout(5)
def test_move_out_of_the_group_degrades_to_unpaired(p):
    """Moving a file to a directory no member watches degrades to an
    unpaired MOVED_FROM (surfacing as a deletion)."""
    mkdir(p("root1"))
    mkdir(p("root2"))
    mkdir(p("outside"))
    touch(p("root1", "a"))

    pairing = InotifyPairingGroup()
    inotify_fd = InotifyFD.get_instance()
    group1 = make_watch_group(inotify_fd, p("root1").encode(), pairing)
    group2 = make_watch_group(inotify_fd, p("root2").encode(), pairing)

    mv(p("root1", "a"), p("outside", "b"))

    event = wait_for_move_event(group1.read_event)
    assert isinstance(event, PathedInotifyEvent)
    assert event.ev.is_moved_from
    assert event.path == p("root1", "a").encode()

    group1.deactivate()
    group2.deactivate()


@pytest.mark.timeout(5)
def test_move_into_the_group_degrades_to_unpaired(p):
    """Moving a file from a directory no member watches degrades to an
    unpaired MOVED_TO (surfacing as a creation)."""
    mkdir(p("root1"))
    mkdir(p("root2"))
    mkdir(p("outside"))
    touch(p("outside", "a"))

    pairing = InotifyPairingGroup()
    inotify_fd = InotifyFD.get_instance()
    group1 = make_watch_group(inotify_fd, p("root1").encode(), pairing)
    group2 = make_watch_group(inotify_fd, p("root2").encode(), pairing)

    mv(p("outside", "a"), p("root2", "b"))

    event = wait_for_move_event(group2.read_event)
    assert isinstance(event, PathedInotifyEvent)
    assert event.ev.is_moved_to
    assert event.path == p("root2", "b").encode()

    group1.deactivate()
    group2.deactivate()


@pytest.mark.timeout(10)
def test_observer_reports_cross_root_move_to_both_handlers(p):
    """End to end: an observer watching two roots with one pairing group
    delivers one full move event per watched root."""
    mkdir(p("root1"))
    mkdir(p("root2"))
    touch(p("root1", "a"))

    events: list = []
    got_move = threading.Event()

    class Handler(FileSystemEventHandler):
        def on_any_event(self, event):
            events.append(event)
            if isinstance(event, FileMovedEvent):
                got_move.set()

    pairing = InotifyPairingGroup()
    observer = InotifyObserver(pairing_group=pairing)
    observer.schedule(Handler(), p("root1"))
    observer.schedule(Handler(), p("root2"))
    observer.start()
    try:
        mv(p("root1", "a"), p("root2", "b"))
        assert got_move.wait(timeout=5)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and len([e for e in events if isinstance(e, FileMovedEvent)]) < 2:
            time.sleep(0.05)
    finally:
        observer.stop()
        observer.join()

    move_events = [e for e in events if isinstance(e, FileMovedEvent)]
    # De-duplication: exactly one paired move event per watched root.
    assert len(move_events) == 2
    for event in move_events:
        assert event.src_path == p("root1", "a")
        assert event.dest_path == p("root2", "b")
