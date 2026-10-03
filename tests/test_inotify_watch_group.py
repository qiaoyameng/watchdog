from __future__ import annotations

from typing import Callable

import pytest

from watchdog.utils import platform

if not platform.is_linux():
    pytest.skip("GNU/Linux only.", allow_module_level=True)

import os
import random

from watchdog.observers.inotify import InotifyWatchGroup
from watchdog.observers.inotify_c import WATCHDOG_ALL_EVENTS, InotifyConstants, InotifyFD, Mask
from watchdog.observers.inotify_move_event_grouper import (
    ExplainedInotifyEvent,
    GroupedInotifyEvent,
    InotifyMoveEventGroup,
    InotifyMoveEventGrouper,
    MovePairReason,
    PathedInotifyEvent,
)

from .shell import mkdir, mount_tmpfs, mv, rm, symlink, touch, unmount


def wait_for_move_event(read_event: Callable[[], GroupedInotifyEvent]) -> GroupedInotifyEvent:
    while True:
        event = read_event()
        if not isinstance(event, PathedInotifyEvent) or event.ev.is_move:
            return event


def create_inotify_watch(
    path: bytes,
    *,
    recursive: bool = False,
    follow_symlink: bool = False,
    move_event_grouper: InotifyMoveEventGroup | None = None,
) -> InotifyWatchGroup:
    return InotifyWatchGroup(
        InotifyFD.get_instance(),
        path,
        is_recursive=recursive,
        follow_symlink=follow_symlink,
        event_mask=WATCHDOG_ALL_EVENTS,
        move_event_grouper=move_event_grouper,
    )


@pytest.mark.timeout(5)
def test_move_from(p):
    mkdir(p("dir1"))
    mkdir(p("dir2"))
    touch(p("dir1", "a"))

    inotify = create_inotify_watch(p("dir1").encode())
    mv(p("dir1", "a"), p("dir2", "b"))
    assert_event(inotify, p("dir1", "a"), InotifyConstants.IN_MOVED_FROM)
    inotify.deactivate()


@pytest.mark.timeout(5)
def test_move_to(p):
    mkdir(p("dir1"))
    mkdir(p("dir2"))
    touch(p("dir1", "a"))

    inotify = create_inotify_watch(p("dir2").encode())
    mv(p("dir1", "a"), p("dir2", "b"))
    assert_event(inotify, p("dir2", "b"), InotifyConstants.IN_MOVED_TO)
    inotify.deactivate()


@pytest.mark.timeout(5)
def test_move_internal(p):
    mkdir(p("dir1"))
    mkdir(p("dir2"))
    touch(p("dir1", "a"))

    inotify = create_inotify_watch(p("").encode(), recursive=True)
    mv(p("dir1", "a"), p("dir2", "b"))
    frm, to = wait_for_move_event(inotify.read_event)
    assert frm.path == p("dir1", "a").encode()
    assert to.path == p("dir2", "b").encode()
    inotify.deactivate()


@pytest.mark.timeout(5)
def test_move_internal_symlink_followed(p):
    mkdir(p("dir", "dir1"), parents=True)
    mkdir(p("dir", "dir2"))
    touch(p("dir", "dir1", "a"))
    symlink(p("dir"), p("symdir"), target_is_directory=True)

    inotify = create_inotify_watch(p("symdir").encode(), recursive=True, follow_symlink=True)
    mv(p("dir", "dir1", "a"), p("dir", "dir2", "b"))
    frm, to = wait_for_move_event(inotify.read_event)
    assert frm.path == p("symdir", "dir1", "a").encode()
    assert to.path == p("symdir", "dir2", "b").encode()
    inotify.deactivate()


@pytest.mark.timeout(10)
def test_move_internal_batch(p):
    n = 100
    mkdir(p("dir1"))
    mkdir(p("dir2"))
    files = [str(i) for i in range(n)]
    for f in files:
        touch(p("dir1", f))

    inotify = create_inotify_watch(p("").encode(), recursive=True)

    random.shuffle(files)
    for f in files:
        mv(p("dir1", f), p("dir2", f))

    # Check that all n events are paired
    for _ in range(n):
        frm, to = wait_for_move_event(inotify.read_event)
        assert os.path.dirname(frm.path).endswith(b"/dir1")
        assert os.path.dirname(to.path).endswith(b"/dir2")
        assert frm.ev.name == to.ev.name
    inotify.deactivate()


@pytest.mark.timeout(5)
def test_delete_watched_directory(p):
    mkdir(p("dir"))
    inotify = create_inotify_watch(p("dir").encode())
    rm(p("dir"), recursive=True)

    # Wait for the event to be picked up
    inotify.read_event()

    # Ensure InotifyBuffer shuts down cleanly without raising an exception
    inotify.deactivate()


@pytest.mark.timeout(5)
def test_delete_watched_directory_symlink_followed(p):
    mkdir(p("dir", "dir2"), parents=True)
    symlink(p("dir"), p("symdir"), target_is_directory=True)

    inotify = create_inotify_watch(p("symdir").encode(), follow_symlink=True)
    rm(p("dir", "dir2"), recursive=True)

    # Wait for the event to be picked up
    event = inotify.read_event()
    while not isinstance(event, PathedInotifyEvent) or (
        event.ev.mask != (InotifyConstants.IN_DELETE | InotifyConstants.IN_ISDIR)
    ):
        event = inotify.read_event()

    # Ensure InotifyBuffer shuts down cleanly without raising an exception
    inotify.deactivate()


@pytest.mark.timeout(5)
def test_delete_watched_directory_symlink_followed_recursive(p):
    mkdir(p("dir"), parents=True)
    mkdir(p("dir2", "dir3", "dir4"), parents=True)
    symlink(p("dir2"), p("dir", "symdir"), target_is_directory=True)

    inotify = create_inotify_watch(p("dir").encode(), follow_symlink=True, recursive=True)
    rm(p("dir2", "dir3", "dir4"), recursive=True)

    # Wait for the event to be picked up
    event = inotify.read_event()
    while not isinstance(event, PathedInotifyEvent) or (
        event.ev.mask != (InotifyConstants.IN_DELETE | InotifyConstants.IN_ISDIR)
    ):
        event = inotify.read_event()

    # Ensure InotifyBuffer shuts down cleanly without raising an exception
    inotify.deactivate()


@pytest.mark.timeout(5)
@pytest.mark.skipif("GITHUB_REF" not in os.environ, reason="sudo password prompt")
def test_unmount_watched_directory_filesystem(p):
    mkdir(p("dir1"))
    mount_tmpfs(p("dir1"))
    mkdir(p("dir1/dir2"))
    inotify = create_inotify_watch(p("dir1/dir2").encode())
    unmount(p("dir1"))

    # Wait for the event to be picked up
    inotify.read_event()

    # Ensure InotifyBuffer shuts down cleanly without raising an exception
    inotify.deactivate()
    assert not inotify.is_active
    assert not inotify._active_callbacks_by_watch  # noqa: SLF001
    assert not inotify._active_callbacks_by_watch  # noqa: SLF001


def assert_event(inotify: InotifyWatchGroup, expected_path: str, expected_kind: Mask):
    event = inotify.read_event()
    if isinstance(event, ExplainedInotifyEvent):
        assert event.event is not None
        event = event.event
    assert event.path == expected_path.encode()
    assert event.ev.mask & expected_kind


def assert_touch_events(inotify: InotifyWatchGroup, expected_path: str):
    assert_event(inotify, expected_path, InotifyConstants.IN_OPEN)
    assert_event(inotify, expected_path, InotifyConstants.IN_ATTRIB)
    assert_event(inotify, expected_path, InotifyConstants.IN_CLOSE_WRITE)


@pytest.mark.timeout(5)
def test_watch_groups_are_independent(p):
    original_path = p("rootdir", "dir1", "a")
    destination_path = p("rootdir", "dir2", "b")

    def setup() -> None:
        mkdir(p("rootdir"))
        mkdir(p("rootdir", "dir1"))
        mkdir(p("rootdir", "dir2"))
        touch(original_path)

    def run() -> None:
        mv(original_path, destination_path)
        touch(destination_path)  # generates events after the move.
        rm(destination_path)  # generates delete event after the move.

    def cleanup() -> None:
        rm(p("rootdir"), recursive=True)

    def assert_inotify_a_events(inotify_a: InotifyWatchGroup) -> None:
        # check inotify_a uses the original path of the file.
        assert_touch_events(inotify_a, original_path)
        assert_event(inotify_a, original_path, InotifyConstants.IN_ATTRIB)
        assert_event(inotify_a, original_path, InotifyConstants.IN_DELETE_SELF)

    def assert_inotify_root_events(inotify_root: InotifyWatchGroup) -> None:
        # check inotify_root tracks the new path of the file.
        ev1_move = wait_for_move_event(inotify_root.read_event)
        assert not isinstance(ev1_move, PathedInotifyEvent)
        assert_touch_events(inotify_root, destination_path)
        assert_event(inotify_root, destination_path, InotifyConstants.IN_DELETE)

    # inotify_a works alone:
    setup()
    inotify_a = create_inotify_watch(original_path.encode())
    run()
    assert_inotify_a_events(inotify_a)
    inotify_a.deactivate()
    cleanup()

    # inotify_a is not affected by inotify_root:
    setup()
    inotify_a = create_inotify_watch(original_path.encode())
    inotify_root = create_inotify_watch(p("rootdir").encode(), recursive=True)
    run()
    assert_inotify_a_events(inotify_a)
    assert_inotify_root_events(inotify_root)
    inotify_root.deactivate()
    inotify_a.deactivate()
    cleanup()


@pytest.mark.timeout(10)
def test_explicit_group_pairs_real_move_across_roots(p):
    mkdir(p("source_root"))
    mkdir(p("destination_root"))
    mkdir(p("source_root", "tree"))
    mkdir(p("source_root", "tree", "child"))
    touch(p("source_root", "tree", "child", "a"))

    move_group = InotifyMoveEventGroup(delay=0.2)
    source = create_inotify_watch(p("source_root").encode(), recursive=True, move_event_grouper=move_group)
    destination = create_inotify_watch(
        p("destination_root").encode(),
        recursive=True,
        move_event_grouper=move_group,
    )

    try:
        mv(p("source_root", "tree"), p("destination_root", "tree"))

        source_event = wait_for_move_event(source.read_event)
        destination_event = wait_for_move_event(destination.read_event)
        assert source_event == destination_event
        frm, to = source_event
        assert frm.path == p("source_root", "tree").encode()
        assert to.path == p("destination_root", "tree").encode()

        touch(p("destination_root", "tree", "child", "b"))
        event = destination.read_event()
        found_paths = []
        for _ in range(8):
            if isinstance(event, PathedInotifyEvent):
                found_paths.append(event.path)
                if event.path.endswith(b"/b"):
                    break
            event = destination.read_event()
        assert any(path.endswith(b"/b") for path in found_paths)
    finally:
        destination.deactivate()
        source.deactivate()
        move_group.close()


@pytest.mark.timeout(10)
def test_observer_schedule_group_pairs_move_across_roots(p):
    from queue import Queue

    from watchdog.events import FileSystemEventHandler
    from watchdog.observers.inotify import InotifyObserver

    class Recorder(FileSystemEventHandler):
        def __init__(self) -> None:
            self.events: Queue = Queue()

        def on_moved(self, event):
            self.events.put(event)

    mkdir(p("source_root"))
    mkdir(p("destination_root"))
    touch(p("source_root", "a"))

    group = InotifyMoveEventGroup(delay=0.2)
    source_handler = Recorder()
    destination_handler = Recorder()
    observer = InotifyObserver()
    observer.start()
    source_watch = observer.schedule(source_handler, p("source_root"), move_event_group=group)
    destination_watch = observer.schedule(destination_handler, p("destination_root"), move_event_group=group)

    try:
        mv(p("source_root", "a"), p("destination_root", "b"))
        source_event = source_handler.events.get(timeout=5)
        destination_event = destination_handler.events.get(timeout=5)
        assert source_event == destination_event
        assert source_event.src_path == p("source_root", "a")
        assert source_event.dest_path == p("destination_root", "b")
    finally:
        observer.unschedule(destination_watch)
        observer.unschedule(source_watch)
        observer.stop()
        observer.join()
        group.close()


@pytest.mark.timeout(10)
def test_explicit_group_degrades_when_source_is_not_a_member(p):
    mkdir(p("outside"))
    mkdir(p("destination_root"))
    touch(p("outside", "a"))

    group = InotifyMoveEventGroup(delay=0.05)
    destination = create_inotify_watch(p("destination_root").encode(), move_event_grouper=group)

    try:
        mv(p("outside", "a"), p("destination_root", "b"))
        event = destination.read_event()
        assert isinstance(event, ExplainedInotifyEvent)
        assert event.reason is MovePairReason.PAIRING_TIMEOUT
        assert event.event is not None
        assert event.event.path == p("destination_root", "b").encode()
    finally:
        destination.deactivate()
        group.close()


@pytest.mark.timeout(10)
def test_pending_move_is_cancelled_when_watch_removed(p):
    mkdir(p("dir"))
    touch(p("dir", "a"))

    group = InotifyMoveEventGroup(delay=1)
    watch_group = create_inotify_watch(p("dir").encode(), move_event_grouper=group)
    try:
        mv(p("dir", "a"), p("gone"))
        rm(p("dir"), recursive=True)
        reasons = []
        while True:
            event = watch_group.read_event()
            if isinstance(event, ExplainedInotifyEvent):
                reasons.append(event.reason)
                if event.reason is MovePairReason.WATCH_REMOVED:
                    break
        assert reasons.count(MovePairReason.WATCH_REMOVED) == 1
    finally:
        watch_group.deactivate()
        group.close()
