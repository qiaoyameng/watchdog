from __future__ import annotations

import pytest

from watchdog.events import (
    EVENT_TYPE_STABLE,
    DirCreatedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileStableEvent,
    FileSystemEventHandler,
    StableFileEventHandler,
)

WINDOW = 5.0


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeStat:
    def __init__(self, size: int, mtime: float, ino: int) -> None:
        self.st_size = size
        self.st_mtime = mtime
        self.st_ino = ino


class FakeFS:
    """Minimal fake file system keyed by path."""

    def __init__(self) -> None:
        self._files: dict[str, FakeStat] = {}
        self._next_ino = 1

    def write(self, path: str, *, size: int, mtime: float) -> None:
        current = self._files.get(path)
        if current is None:
            ino = self._next_ino
            self._next_ino += 1
        else:
            ino = current.st_ino
        self._files[path] = FakeStat(size, mtime, ino)

    def rename(self, src: str, dest: str) -> None:
        self._files[dest] = self._files.pop(src)

    def delete(self, path: str) -> None:
        del self._files[path]

    def stat(self, path: str) -> FakeStat:
        try:
            return self._files[path]
        except KeyError:
            raise FileNotFoundError(path) from None


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def fs():
    return FakeFS()


def make_handler(clock, fs, **kwargs):
    stable_events = []
    handler = StableFileEventHandler(
        stability_window=kwargs.pop("stability_window", WINDOW),
        clock=clock,
        stat=fs.stat,
        **kwargs,
    )
    handler.on_stable = stable_events.append
    return handler, stable_events


def test_file_stable_event_attributes():
    event = FileStableEvent("/w/a.txt", size=42, mtime=12.5)
    assert event.src_path == "/w/a.txt"
    assert event.event_type == EVENT_TYPE_STABLE
    assert not event.is_directory
    assert not event.is_synthetic
    assert event.size == 42
    assert event.mtime == 12.5
    assert "42" in repr(event)


def test_stable_after_window_with_two_writes(clock, fs):
    handler, stable_events = make_handler(clock, fs)

    fs.write("/w/a.txt", size=10, mtime=1.0)
    handler.dispatch(FileCreatedEvent("/w/a.txt"))

    # Second write before the window elapses resets it.
    clock.advance(3.0)
    fs.write("/w/a.txt", size=20, mtime=2.0)
    handler.dispatch(FileModifiedEvent("/w/a.txt"))

    clock.advance(4.0)
    handler.check_pending()
    assert stable_events == []

    clock.advance(1.0)
    handler.check_pending()
    assert len(stable_events) == 1
    event = stable_events[0]
    assert event.src_path == "/w/a.txt"
    assert event.size == 20
    assert event.mtime == 2.0

    # Already emitted: no duplicate notifications.
    clock.advance(WINDOW)
    handler.check_pending()
    assert len(stable_events) == 1


def test_modification_detected_by_check_pending_without_event(clock, fs):
    handler, stable_events = make_handler(clock, fs)

    fs.write("/w/a.txt", size=10, mtime=1.0)
    handler.dispatch(FileCreatedEvent("/w/a.txt"))

    # Writer keeps appending; the event stream is silent but stat changes.
    clock.advance(4.0)
    fs.write("/w/a.txt", size=99, mtime=4.0)
    handler.check_pending()
    assert stable_events == []

    clock.advance(WINDOW)
    handler.check_pending()
    assert len(stable_events) == 1
    assert stable_events[0].size == 99
    assert stable_events[0].mtime == 4.0


def test_rename_moves_pending_entry(clock, fs):
    handler, stable_events = make_handler(clock, fs)

    fs.write("/w/upload.part", size=10, mtime=1.0)
    handler.dispatch(FileCreatedEvent("/w/upload.part"))

    clock.advance(2.0)
    fs.rename("/w/upload.part", "/w/upload.txt")
    handler.dispatch(FileMovedEvent("/w/upload.part", "/w/upload.txt"))

    # The remaining window (3s) carries over to the new path.
    clock.advance(3.0)
    handler.check_pending()
    assert len(stable_events) == 1
    event = stable_events[0]
    assert event.src_path == "/w/upload.txt"
    assert event.size == 10
    assert event.mtime == 1.0
    assert handler.pending_paths == ()


def test_recreate_same_path_is_a_new_file(clock, fs):
    handler, stable_events = make_handler(clock, fs)

    fs.write("/w/a.txt", size=10, mtime=1.0)
    handler.dispatch(FileCreatedEvent("/w/a.txt"))

    clock.advance(4.0)
    fs.delete("/w/a.txt")
    handler.dispatch(FileDeletedEvent("/w/a.txt"))
    fs.write("/w/a.txt", size=50, mtime=9.0)
    handler.dispatch(FileCreatedEvent("/w/a.txt"))

    # The old file would have been stable 1s from now; the new one must not be.
    clock.advance(4.0)
    handler.check_pending()
    assert stable_events == []

    clock.advance(1.0)
    handler.check_pending()
    assert len(stable_events) == 1
    event = stable_events[0]
    assert event.size == 50
    assert event.mtime == 9.0


def test_recreate_detected_via_inode_without_delete_event(clock, fs):
    handler, stable_events = make_handler(clock, fs)

    fs.write("/w/a.txt", size=10, mtime=1.0)
    handler.dispatch(FileCreatedEvent("/w/a.txt"))

    # Delete + recreate with no events in between: the inode change must
    # still prevent the new file from inheriting the old one's window.
    clock.advance(4.0)
    fs.delete("/w/a.txt")
    fs.write("/w/a.txt", size=50, mtime=9.0)
    handler.check_pending()
    assert stable_events == []

    clock.advance(4.0)
    handler.check_pending()
    assert stable_events == []

    clock.advance(1.0)
    handler.check_pending()
    assert len(stable_events) == 1
    assert stable_events[0].size == 50
    assert stable_events[0].mtime == 9.0


def test_stat_failure_drops_pending(clock, fs):
    handler, stable_events = make_handler(clock, fs)

    fs.write("/w/a.txt", size=10, mtime=1.0)
    handler.dispatch(FileCreatedEvent("/w/a.txt"))

    # File disappears without a delete event before the window elapses.
    clock.advance(2.0)
    fs.delete("/w/a.txt")
    handler.check_pending()
    assert handler.pending_paths == ()

    clock.advance(WINDOW * 2)
    handler.check_pending()
    assert stable_events == []


def test_max_wait_emit_policy(clock, fs):
    handler, stable_events = make_handler(clock, fs, max_wait=8.0, on_max_wait="emit")

    fs.write("/w/a.txt", size=10, mtime=1.0)
    handler.dispatch(FileCreatedEvent("/w/a.txt"))

    # Continuous writes keep resetting the 5s stability window.
    for step in range(1, 5):
        clock.advance(2.0)
        fs.write("/w/a.txt", size=10 + step, mtime=1.0 + step)
        handler.dispatch(FileModifiedEvent("/w/a.txt"))
        handler.check_pending()

    # 10s > max_wait=8s: emitted once, with the latest observed state.
    assert len(stable_events) == 1
    event = stable_events[0]
    assert event.src_path == "/w/a.txt"
    assert event.size == 14
    assert event.mtime == 5.0
    assert handler.pending_paths == ()


def test_max_wait_drop_policy(clock, fs):
    handler, stable_events = make_handler(clock, fs, max_wait=8.0, on_max_wait="drop")

    fs.write("/w/a.txt", size=10, mtime=1.0)
    handler.dispatch(FileCreatedEvent("/w/a.txt"))

    for step in range(1, 5):
        clock.advance(2.0)
        fs.write("/w/a.txt", size=10 + step, mtime=1.0 + step)
        handler.dispatch(FileModifiedEvent("/w/a.txt"))
        handler.check_pending()

    assert stable_events == []
    assert handler.pending_paths == ()

    # Dropped for good: even after the writes stop, nothing is emitted.
    clock.advance(WINDOW * 2)
    handler.check_pending()
    assert stable_events == []


def test_stop_discards_pending_by_default(clock, fs):
    handler, stable_events = make_handler(clock, fs)

    fs.write("/w/a.txt", size=10, mtime=1.0)
    handler.dispatch(FileCreatedEvent("/w/a.txt"))

    handler.stop()
    assert handler.pending_paths == ()

    clock.advance(WINDOW * 2)
    handler.check_pending()
    assert stable_events == []


def test_stop_drain_flushes_pending(clock, fs):
    handler, stable_events = make_handler(clock, fs)

    fs.write("/w/a.txt", size=10, mtime=1.0)
    handler.dispatch(FileCreatedEvent("/w/a.txt"))
    fs.write("/w/gone.txt", size=1, mtime=1.0)
    handler.dispatch(FileCreatedEvent("/w/gone.txt"))
    fs.delete("/w/gone.txt")

    handler.stop(drain=True)
    assert handler.pending_paths == ()
    # Only the file that still exists is flushed.
    assert len(stable_events) == 1
    assert stable_events[0].src_path == "/w/a.txt"
    assert stable_events[0].size == 10


def test_delegate_receives_raw_events(clock, fs):
    raw_events = []

    class Recorder(FileSystemEventHandler):
        def on_any_event(self, event):
            raw_events.append(event)

    handler, stable_events = make_handler(clock, fs, delegate=Recorder())

    fs.write("/w/a.txt", size=10, mtime=1.0)
    handler.dispatch(FileCreatedEvent("/w/a.txt"))
    fs.write("/w/a.txt", size=20, mtime=2.0)
    handler.dispatch(FileModifiedEvent("/w/a.txt"))
    fs.rename("/w/a.txt", "/w/b.txt")
    handler.dispatch(FileMovedEvent("/w/a.txt", "/w/b.txt"))

    assert [type(event) for event in raw_events] == [FileCreatedEvent, FileModifiedEvent, FileMovedEvent]

    clock.advance(WINDOW)
    handler.check_pending()
    assert len(stable_events) == 1
    assert stable_events[0].src_path == "/w/b.txt"
    # The derived stable event is not part of the raw stream.
    assert not any(isinstance(event, FileStableEvent) for event in raw_events)


def test_directory_events_are_ignored(clock, fs):
    handler, stable_events = make_handler(clock, fs)

    handler.dispatch(DirCreatedEvent("/w/sub"))
    assert handler.pending_paths == ()

    clock.advance(WINDOW)
    handler.check_pending()
    assert stable_events == []


def test_invalid_configuration_rejected(clock, fs):
    with pytest.raises(ValueError, match="stability_window"):
        StableFileEventHandler(stability_window=0, clock=clock, stat=fs.stat)
    with pytest.raises(ValueError, match="max_wait"):
        StableFileEventHandler(stability_window=1, max_wait=-1, clock=clock, stat=fs.stat)
    with pytest.raises(ValueError, match="on_max_wait"):
        StableFileEventHandler(stability_window=1, on_max_wait="explode", clock=clock, stat=fs.stat)
