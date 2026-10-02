from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import pytest

from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileStableEvent,
    FileSystemEvent,
    FileSystemEventHandler,
    StableFileEventHandler,
)


@dataclass(frozen=True)
class FakeStat:
    st_size: int
    st_mtime: int
    st_dev: int = 1
    st_ino: int = 1


@dataclass
class FakeFile:
    size: int
    mtime: int
    inode: int


class FakeFileSystem:
    def __init__(self) -> None:
        self.files: dict[str, FakeFile] = {}
        self.stat_failures: dict[str, OSError] = {}
        self.next_inode = 10

    def write(self, path: str, size: int, mtime: int) -> None:
        file = self.files.get(path)
        if file is None:
            file = FakeFile(size=size, mtime=mtime, inode=self.next_inode)
            self.next_inode += 1
            self.files[path] = file
        else:
            file.size = size
            file.mtime = mtime

    def rename(self, source: str, destination: str) -> None:
        self.files[destination] = self.files.pop(source)

    def delete(self, path: str) -> None:
        self.files.pop(path, None)

    def recreate(self, path: str, size: int, mtime: int) -> None:
        self.delete(path)
        self.write(path, size=size, mtime=mtime)

    def stat(self, path: str) -> FakeStat:
        if path in self.stat_failures:
            raise self.stat_failures[path]
        try:
            file = self.files[path]
        except KeyError as error:
            raise FileNotFoundError(path) from error
        return FakeStat(st_size=file.size, st_mtime=file.mtime, st_ino=file.inode)


@dataclass(order=True)
class _TimerJob:
    due_at: float
    sequence: int
    callback: object = field(compare=False)
    cancelled: bool = field(default=False, compare=False)


@dataclass
class _FakeTimer:
    job: _TimerJob

    def cancel(self) -> None:
        self.job.cancelled = True


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sequence = 0
        self.jobs: list[_TimerJob] = []

    def time(self) -> float:
        return self.now

    def timer(self, delay: float, callback: object) -> _FakeTimer:
        job = _TimerJob(self.now + delay, self.sequence, callback)
        self.sequence += 1
        heapq.heappush(self.jobs, job)
        return _FakeTimer(job)

    def advance(self, seconds: float) -> None:
        deadline = self.now + seconds
        while self.jobs:
            job = self.jobs[0]
            if job.cancelled:
                heapq.heappop(self.jobs)
                continue
            if job.due_at > deadline:
                break
            heapq.heappop(self.jobs)
            if job.cancelled:
                continue
            self.now = job.due_at
            job.callback()
        self.now = deadline
        self.jobs = [job for job in self.jobs if not job.cancelled]
        heapq.heapify(self.jobs)


class RecordingStableHandler(FileSystemEventHandler):
    def __init__(self) -> None:
        self.events: list[FileStableEvent] = []

    def on_stable(self, event: FileStableEvent) -> None:
        self.events.append(event)


class RecordingRawHandler(FileSystemEventHandler):
    def __init__(self) -> None:
        self.events: list[FileSystemEvent] = []

    def on_any_event(self, event: FileSystemEvent) -> None:
        self.events.append(event)


def make_handler(
    file_system: FakeFileSystem,
    clock: FakeClock,
    downstream: FileSystemEventHandler | None = None,
) -> StableFileEventHandler:
    return StableFileEventHandler(
        downstream if downstream is not None else RecordingStableHandler(),
        stable_interval_seconds=5,
        stat=file_system.stat,
        clock=clock.time,
        timer=clock.timer,
    )


def test_callable_downstream_receives_unforced_event_after_quiet_interval():
    clock = FakeClock()
    file_system = FakeFileSystem()
    events: list[FileStableEvent] = []
    handler = StableFileEventHandler(
        events.append,
        stable_interval_seconds=2,
        stat=file_system.stat,
        clock=clock.time,
        timer=clock.timer,
    )
    path = "/tmp/callback.txt"

    file_system.write(path, size=1, mtime=1)
    handler.dispatch(FileCreatedEvent(path))
    clock.advance(2)

    assert len(events) == 1
    assert events[0].src_path == path
    assert events[0].size == 1
    assert not events[0].forced


def test_two_writes_rename_and_same_path_rebuild_notify_final_state():
    clock = FakeClock()
    file_system = FakeFileSystem()
    stable_handler = RecordingStableHandler()
    raw_handler = RecordingRawHandler()
    raw_callbacks: list[FileSystemEvent] = []
    handler = StableFileEventHandler(
        stable_handler,
        stable_interval_seconds=5,
        stat=file_system.stat,
        clock=clock.time,
        timer=clock.timer,
    )
    handler.subscribe_raw(raw_handler)
    handler.subscribe_raw(raw_callbacks.append)

    upload = "/tmp/upload.txt"
    renamed = "/tmp/final.txt"

    file_system.write(upload, size=3, mtime=1)
    handler.dispatch(FileCreatedEvent(upload))

    clock.advance(2)
    file_system.write(upload, size=7, mtime=2)
    handler.dispatch(FileModifiedEvent(upload))

    clock.advance(2)
    file_system.rename(upload, renamed)
    handler.dispatch(FileMovedEvent(upload, renamed))

    clock.advance(2)
    file_system.delete(renamed)
    handler.dispatch(FileDeletedEvent(renamed))
    file_system.write(renamed, size=11, mtime=5)
    handler.dispatch(FileCreatedEvent(renamed))

    clock.advance(5)

    assert [event.src_path for event in raw_handler.events] == [
        upload,
        upload,
        upload,
        renamed,
        renamed,
    ]
    assert raw_callbacks == raw_handler.events
    assert len(stable_handler.events) == 1
    event = stable_handler.events[0]
    assert event.src_path == renamed
    assert event.size == 11
    assert event.mtime == 5
    assert event.identity == (1, 11)
    assert not event.forced


def test_stat_failure_on_rebuild_does_not_identify_new_file_as_old_file():
    clock = FakeClock()
    file_system = FakeFileSystem()
    handler = make_handler(file_system, clock)
    path = "/tmp/upload.txt"

    file_system.write(path, size=3, mtime=1)
    handler.dispatch(FileCreatedEvent(path))
    clock.advance(2)
    file_system.stat_failures[path] = PermissionError("temporary failure")
    handler.dispatch(FileModifiedEvent(path))
    clock.advance(3)
    file_system.stat_failures.clear()
    file_system.recreate(path, size=9, mtime=9)
    handler.dispatch(FileDeletedEvent(path))
    handler.dispatch(FileCreatedEvent(path))
    clock.advance(5)

    assert handler.downstream is not None
    assert isinstance(handler.downstream, RecordingStableHandler)
    assert len(handler.downstream.events) == 1
    assert handler.downstream.events[0].size == 9
    assert handler.downstream.events[0].identity == (1, file_system.files[path].inode)


def test_stat_failure_after_move_is_rebound_by_identity():
    clock = FakeClock()
    file_system = FakeFileSystem()
    downstream = RecordingStableHandler()
    handler = StableFileEventHandler(
        downstream,
        stable_interval_seconds=5,
        stat=file_system.stat,
        clock=clock.time,
        timer=clock.timer,
    )
    source = "/tmp/moving.txt"
    destination = "/tmp/moved.txt"

    file_system.write(source, size=4, mtime=1)
    handler.dispatch(FileCreatedEvent(source))
    clock.advance(1)
    file_system.rename(source, destination)
    file_system.stat_failures[destination] = PermissionError("temporary failure")
    handler.dispatch(FileMovedEvent(source, destination))

    clock.advance(1)
    file_system.stat_failures.clear()
    file_system.recreate(destination, size=8, mtime=3)
    clock.advance(4)
    assert downstream.events == []

    handler.dispatch(FileCreatedEvent(destination))
    clock.advance(5)
    assert len(downstream.events) == 1
    assert downstream.events[0].size == 8
    assert downstream.events[0].identity == (1, file_system.files[destination].inode)


def test_max_wait_seconds_forces_notification_for_continuous_writes():
    clock = FakeClock()
    file_system = FakeFileSystem()
    downstream = RecordingStableHandler()
    handler = StableFileEventHandler(
        downstream,
        stable_interval_seconds=5,
        max_wait_seconds=12,
        stat=file_system.stat,
        clock=clock.time,
        timer=clock.timer,
    )
    path = "/tmp/stream.txt"

    file_system.write(path, size=1, mtime=0)
    handler.dispatch(FileCreatedEvent(path))
    for mtime in range(1, 13):
        file_system.write(path, size=mtime + 1, mtime=mtime)
        handler.dispatch(FileModifiedEvent(path))
        clock.advance(1)

    assert len(downstream.events) == 1
    event = downstream.events[0]
    assert event.forced
    assert event.size == 13
    assert event.mtime == 12
    clock.advance(10)
    assert len(downstream.events) == 1


def test_stop_can_discard_or_drain_pending_files():
    clock = FakeClock()
    file_system = FakeFileSystem()
    path = "/tmp/short-lived.txt"
    file_system.write(path, size=2, mtime=1)

    discard_sink = RecordingStableHandler()
    discard_handler = make_handler(file_system, clock, downstream=discard_sink)
    discard_handler.dispatch(FileCreatedEvent(path))
    discard_handler.stop(drain=False)
    clock.advance(10)
    assert discard_sink.events == []

    drain_clock = FakeClock()
    drain_sink = RecordingStableHandler()
    drain_handler = StableFileEventHandler(
        drain_sink,
        stable_interval_seconds=5,
        stat=file_system.stat,
        clock=drain_clock.time,
        timer=drain_clock.timer,
    )
    drain_handler.dispatch(FileCreatedEvent(path))
    drain_handler.stop(drain=True)
    assert len(drain_sink.events) == 1
    assert drain_sink.events[0].forced
    assert drain_sink.events[0].size == 2


def test_invalid_max_wait_seconds():
    with pytest.raises(ValueError):
        StableFileEventHandler(stable_interval_seconds=5, max_wait_seconds=4)
