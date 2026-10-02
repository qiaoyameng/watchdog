from __future__ import annotations

import os
from queue import Empty, Queue
from time import sleep

import pytest

from watchdog.events import (
    DirCreatedEvent,
    DirDeletedEvent,
    DirModifiedEvent,
    DirMovedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
)
from watchdog.observers.api import ObservedWatch
from watchdog.observers.polling import PollingEmitter as Emitter

from .shell import mkdir, mkdtemp, msize, mv, rm, touch

SLEEP_TIME = 0.4
TEMP_DIR = mkdtemp()


def test_set_prune_dirs_rebuilds_baseline_without_spurious_deletes(tmp_path):
    # Switching the prune rules at runtime must rebuild the comparison
    # baseline explicitly so files hidden by the new rules are not emitted as
    # a batch of deletions on the next poll.
    mkdir(str(tmp_path / "cache"))
    touch(str(tmp_path / "cache" / "a"))
    touch(str(tmp_path / "keep"))

    watch = ObservedWatch(str(tmp_path), recursive=True)
    queue = Queue()
    emitter = Emitter(queue, watch, timeout=0)

    emitter.on_thread_start()
    assert str(tmp_path / "cache" / "a") in emitter._snapshot.paths

    # Change the rules and explicitly rebuild the baseline.
    emitter.set_prune_dirs(["cache"])
    assert str(tmp_path / "cache" / "a") not in emitter._snapshot.paths

    # The next poll compares two snapshots built under the same rules.
    emitter.queue_events(0)

    paths = set()
    while True:
        try:
            event, _watch = queue.get_nowait()
            paths.add(event.src_path)
        except Empty:
            break
    assert str(tmp_path / "cache") not in paths
    assert str(tmp_path / "cache" / "a") not in paths

    # A real change outside the pruned subtree is still detected.
    touch(str(tmp_path / "new"))
    emitter.queue_events(0)
    events = []
    while True:
        try:
            event, _watch = queue.get_nowait()
            events.append(event)
        except Empty:
            break
    assert any(event.src_path == str(tmp_path / "new") for event in events)


def test_emitter_accepts_initial_prune_dirs(tmp_path):
    mkdir(str(tmp_path / "cache"))
    touch(str(tmp_path / "cache" / "a"))
    touch(str(tmp_path / "keep"))

    watch = ObservedWatch(str(tmp_path), recursive=True)
    emitter = Emitter(Queue(), watch, timeout=0, prune_dirs=["cache"])
    emitter.on_thread_start()

    assert str(tmp_path / "cache" / "a") not in emitter._snapshot.paths
    assert str(tmp_path / "keep") in emitter._snapshot.paths


def p(*args):
    """
    Convenience function to join the temporary directory path
    with the provided arguments.
    """
    return os.path.join(TEMP_DIR, *args)


@pytest.fixture
def event_queue():
    return Queue()


@pytest.fixture
def emitter(event_queue):
    watch = ObservedWatch(TEMP_DIR, recursive=True)
    em = Emitter(event_queue, watch, timeout=0.2)
    em.start()
    yield em
    em.stop()
    em.join(5)


def test___init__(event_queue, emitter):
    sleep(SLEEP_TIME)
    mkdir(p("project"))

    sleep(SLEEP_TIME)
    mkdir(p("project", "blah"))

    sleep(SLEEP_TIME)
    touch(p("afile"))

    sleep(SLEEP_TIME)
    touch(p("fromfile"))

    sleep(SLEEP_TIME)
    mv(p("fromfile"), p("project", "tofile"))

    sleep(SLEEP_TIME)
    touch(p("afile"))

    sleep(SLEEP_TIME)
    mv(p("project", "blah"), p("project", "boo"))

    sleep(SLEEP_TIME)
    rm(p("project"), recursive=True)

    sleep(SLEEP_TIME)
    rm(p("afile"))

    sleep(SLEEP_TIME)
    msize(p("bfile"))

    sleep(SLEEP_TIME)
    rm(p("bfile"))

    sleep(SLEEP_TIME)
    emitter.stop()

    # What we need here for the tests to pass is a collection type
    # that is:
    #   * unordered
    #   * non-unique
    # A multiset! Python's collections.Counter class seems appropriate.
    expected = {
        DirModifiedEvent(p()),
        DirCreatedEvent(p("project")),
        DirModifiedEvent(p("project")),
        DirCreatedEvent(p("project", "blah")),
        FileCreatedEvent(p("afile")),
        DirModifiedEvent(p()),
        FileCreatedEvent(p("fromfile")),
        DirModifiedEvent(p()),
        DirModifiedEvent(p()),
        FileModifiedEvent(p("afile")),
        DirModifiedEvent(p("project")),
        DirModifiedEvent(p()),
        FileDeletedEvent(p("project", "tofile")),
        DirDeletedEvent(p("project", "boo")),
        DirDeletedEvent(p("project")),
        DirModifiedEvent(p()),
        FileDeletedEvent(p("afile")),
        DirModifiedEvent(p()),
        FileCreatedEvent(p("bfile")),
        FileModifiedEvent(p("bfile")),
        DirModifiedEvent(p()),
        FileDeletedEvent(p("bfile")),
    }

    expected.add(FileMovedEvent(p("fromfile"), p("project", "tofile")))
    expected.add(DirMovedEvent(p("project", "blah"), p("project", "boo")))

    got = set()

    while True:
        try:
            event, _ = event_queue.get_nowait()
            got.add(event)
        except Empty:
            break

    assert expected == got


def test_delete_watched_dir(event_queue, emitter):
    rm(p(""), recursive=True)

    sleep(SLEEP_TIME)
    emitter.stop()

    # What we need here for the tests to pass is a collection type
    # that is:
    #   * unordered
    #   * non-unique
    # A multiset! Python's collections.Counter class seems appropriate.
    expected = {
        DirDeletedEvent(os.path.dirname(p(""))),
    }

    got = set()

    while True:
        try:
            event, _ = event_queue.get_nowait()
            got.add(event)
        except Empty:
            break

    assert expected == got
