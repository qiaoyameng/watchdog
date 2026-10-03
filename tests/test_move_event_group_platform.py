from __future__ import annotations

import pytest

from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver
from watchdog.observers.inotify_move_event_grouper import InotifyMoveEventGroup

def test_polling_observer_rejects_explicit_move_group():
    observer = PollingObserver()
    with pytest.raises(NotImplementedError, match="Linux inotify"):
        observer.schedule(FileSystemEventHandler(), "/", move_event_group=InotifyMoveEventGroup())
