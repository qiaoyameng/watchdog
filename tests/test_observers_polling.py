from __future__ import annotations

import os
import shutil
import tempfile
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


def _drain(event_queue):
    events = []
    while True:
        try:
            events.append(event_queue.get_nowait()[0])
        except Empty:
            break
    return events


def test_polling_emitter_excludes_directory_subtree():
    tmp = tempfile.mkdtemp()
    try:
        os.mkdir(os.path.join(tmp, "cache"))
        open(os.path.join(tmp, "cache", "artifact"), "w").close()
        open(os.path.join(tmp, "tracked"), "w").close()

        visited = []

        def recording_listdir(path):
            visited.append(path)
            return os.scandir(path)

        event_queue = Queue()
        watch = ObservedWatch(tmp, recursive=True)
        emitter = Emitter(
            event_queue, watch, timeout=0.05, listdir=recording_listdir, exclude_dirs=["cache"]
        )
        emitter.start()
        try:
            sleep(SLEEP_TIME)
            assert os.path.join(tmp, "cache") not in visited

            open(os.path.join(tmp, "cache", "later"), "w").close()
            sleep(SLEEP_TIME)
            events = _drain(event_queue)
            assert all("cache" not in event.src_path for event in events)
        finally:
            emitter.stop()
            emitter.join(5)
    finally:
        shutil.rmtree(tmp)


def test_polling_emitter_set_exclude_dirs_rebuilds_baseline():
    tmp = tempfile.mkdtemp()
    try:
        os.mkdir(os.path.join(tmp, "cache"))
        open(os.path.join(tmp, "cache", "artifact"), "w").close()
        os.mkdir(os.path.join(tmp, "src"))
        open(os.path.join(tmp, "src", "module.py"), "w").close()

        event_queue = Queue()
        watch = ObservedWatch(tmp, recursive=True)
        emitter = Emitter(event_queue, watch, timeout=0.05)
        emitter.start()
        try:
            sleep(SLEEP_TIME)
            _drain(event_queue)

            # Switching the rules rebuilds the baseline: no batch of spurious
            # deletion events for the newly hidden subtree.
            emitter.set_exclude_dirs(["cache"])
            sleep(SLEEP_TIME)
            events = _drain(event_queue)
            assert all(
                not (isinstance(event, FileDeletedEvent) and "cache" in event.src_path)
                for event in events
            )

            open(os.path.join(tmp, "cache", "later"), "w").close()
            open(os.path.join(tmp, "src", "other.py"), "w").close()
            sleep(SLEEP_TIME)
            events = _drain(event_queue)
            paths = {event.src_path for event in events}
            assert os.path.join(tmp, "cache", "later") not in paths
            assert os.path.join(tmp, "src", "other.py") in paths

            # Relaxing the rules (again with an explicit rebuild) surfaces the
            # subtree as created instead of reporting anything deleted.
            emitter.set_exclude_dirs([])
            open(os.path.join(tmp, "cache", "newly_exposed"), "w").close()
            sleep(SLEEP_TIME)
            events = _drain(event_queue)
            assert not any(isinstance(event, FileDeletedEvent) for event in events)
            assert os.path.join(tmp, "cache", "newly_exposed") in {
                event.src_path for event in events if isinstance(event, FileCreatedEvent)
            }
        finally:
            emitter.stop()
            emitter.join(5)
    finally:
        shutil.rmtree(tmp)


def test_polling_emitter_set_exclude_dirs_failure_keeps_previous_rules():
    tmp = tempfile.mkdtemp()
    try:
        open(os.path.join(tmp, "file"), "w").close()

        event_queue = Queue()
        watch = ObservedWatch(tmp, recursive=True)
        emitter = Emitter(event_queue, watch, timeout=0.05, exclude_dirs=["cache"])
        emitter.start()
        try:
            sleep(SLEEP_TIME)
            _drain(event_queue)

            shutil.rmtree(tmp)

            with pytest.raises(OSError):
                emitter.set_exclude_dirs([])

            # Rules and baseline were left untouched.
            assert emitter._exclude_dirs == ["cache"]
            sleep(SLEEP_TIME)
            events = _drain(event_queue)
            # The watched directory disappeared; the next poll reports it in
            # the usual way rather than emitting per-file deletions.
            assert not any(isinstance(event, FileDeletedEvent) for event in events)
        finally:
            emitter.stop()
            emitter.join(5)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_polling_emitter_set_exclude_dirs_invalid_patterns_keeps_previous_rules():
    tmp = tempfile.mkdtemp()
    try:
        open(os.path.join(tmp, "file"), "w").close()
        os.mkdir(os.path.join(tmp, "dir"))

        event_queue = Queue()
        watch = ObservedWatch(tmp, recursive=True)
        emitter = Emitter(event_queue, watch, timeout=0.05, exclude_dirs=["cache"])
        emitter.start()
        try:
            sleep(SLEEP_TIME)
            _drain(event_queue)

            # "**" conflicts with the implicit include-all pattern.
            with pytest.raises(ValueError):
                emitter.set_exclude_dirs(["**"])
            assert emitter._exclude_dirs == ["cache"]

            # The emitter still works with the previous rules.
            open(os.path.join(tmp, "other"), "w").close()
            sleep(SLEEP_TIME)
            events = _drain(event_queue)
            assert os.path.join(tmp, "other") in {event.src_path for event in events}
        finally:
            emitter.stop()
            emitter.join(5)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
