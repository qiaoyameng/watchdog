from __future__ import annotations

import errno
import os
import pickle
import time
from itertools import permutations
from types import SimpleNamespace
from unittest.mock import patch

from watchdog.utils import platform
from watchdog.utils.dirsnapshot import DirectorySnapshot, DirectorySnapshotDiff, EmptyDirectorySnapshot

from .shell import mkdir, mv, rm, touch


def make_snapshot(paths):
    snapshot = DirectorySnapshot.__new__(DirectorySnapshot)
    snapshot._stat_info = {}
    snapshot._inode_to_paths = {}
    for path, (inode, device) in paths.items():
        stat = SimpleNamespace(
            st_ino=inode,
            st_dev=device,
            st_mode=0o100000,
            st_mtime=1,
            st_size=1,
        )
        snapshot._stat_info[path] = stat
        snapshot._inode_to_paths.setdefault((inode, device), set()).add(path)
    return snapshot


def wait():
    """
    Wait long enough for file/folder mtime to change. This is needed
    to be able to detected modifications.
    """
    if platform.is_darwin() or platform.is_windows():
        # on macOS resolution of stat.mtime is only 1 second
        time.sleep(1.5)
    else:
        time.sleep(0.5)


def test_pickle(p):
    """It should be possible to pickle a snapshot."""
    mkdir(p("dir1"))
    snasphot = DirectorySnapshot(p("dir1"))
    pickle.dumps(snasphot)


def test_move_to(p):
    mkdir(p("dir1"))
    mkdir(p("dir2"))
    touch(p("dir1", "a"))
    ref = DirectorySnapshot(p("dir2"))
    mv(p("dir1", "a"), p("dir2", "b"))
    diff = DirectorySnapshotDiff(ref, DirectorySnapshot(p("dir2")))
    assert diff.files_created == [p("dir2", "b")]


def test_move_to_with_context_manager(p):
    mkdir(p("dir1"))
    touch(p("dir1", "a"))
    mkdir(p("dir2"))

    dir1_cm = DirectorySnapshotDiff.ContextManager(p("dir1"))
    dir2_cm = DirectorySnapshotDiff.ContextManager(p("dir2"))
    with dir1_cm, dir2_cm:
        mv(p("dir1", "a"), p("dir2", "b"))

    assert dir1_cm.diff.files_deleted == [p("dir1", "a")]
    assert dir2_cm.diff.files_created == [p("dir2", "b")]


def test_move_from(p):
    mkdir(p("dir1"))
    mkdir(p("dir2"))
    touch(p("dir1", "a"))
    ref = DirectorySnapshot(p("dir1"))
    mv(p("dir1", "a"), p("dir2", "b"))
    diff = DirectorySnapshotDiff(ref, DirectorySnapshot(p("dir1")))
    assert diff.files_deleted == [p("dir1", "a")]


def test_move_internal(p):
    mkdir(p("dir1"))
    mkdir(p("dir2"))
    touch(p("dir1", "a"))
    ref = DirectorySnapshot(p(""))
    mv(p("dir1", "a"), p("dir2", "b"))
    diff = DirectorySnapshotDiff(ref, DirectorySnapshot(p("")))
    assert diff.files_moved == [(p("dir1", "a"), p("dir2", "b"))]
    assert diff.files_created == []
    assert diff.files_deleted == []


def test_move_replace(p):
    mkdir(p("dir1"))
    mkdir(p("dir2"))
    touch(p("dir1", "a"))
    touch(p("dir2", "b"))
    ref = DirectorySnapshot(p(""))
    mv(p("dir1", "a"), p("dir2", "b"))
    diff = DirectorySnapshotDiff(ref, DirectorySnapshot(p("")))
    assert diff.files_moved == [(p("dir1", "a"), p("dir2", "b"))]
    assert diff.files_deleted == [p("dir2", "b")]
    assert diff.files_created == []


def test_dir_modify_on_create(p):
    ref = DirectorySnapshot(p(""))
    wait()
    touch(p("a"))
    diff = DirectorySnapshotDiff(ref, DirectorySnapshot(p("")))
    assert diff.dirs_modified == [p("")]


def test_dir_modify_on_move(p):
    mkdir(p("dir1"))
    mkdir(p("dir2"))
    touch(p("dir1", "a"))
    ref = DirectorySnapshot(p(""))
    wait()
    mv(p("dir1", "a"), p("dir2", "b"))
    diff = DirectorySnapshotDiff(ref, DirectorySnapshot(p("")))
    assert set(diff.dirs_modified) == {p("dir1"), p("dir2")}


def test_detect_modify_for_moved_files(p):
    touch(p("a"))
    ref = DirectorySnapshot(p(""))
    wait()
    touch(p("a"))
    mv(p("a"), p("b"))
    diff = DirectorySnapshotDiff(ref, DirectorySnapshot(p("")))
    assert diff.files_moved == [(p("a"), p("b"))]
    assert diff.files_modified == [p("a")]


def test_replace_dir_with_file(p):
    # Replace a dir with a file of the same name just before the normal listdir
    # call and ensure it doesn't cause an exception

    def listdir_fcn(path):
        if path == p("root", "dir"):
            rm(path, recursive=True)
            touch(path)
        return os.scandir(path)

    mkdir(p("root"))
    mkdir(p("root", "dir"))

    # Should NOT raise an OSError (ENOTDIR)
    DirectorySnapshot(p("root"), listdir=listdir_fcn)


def test_permission_error(p):
    # Test that unreadable folders are not raising exceptions
    mkdir(p("a", "b", "c"), parents=True)

    ref = DirectorySnapshot(p(""))
    walk_orig = DirectorySnapshot.walk

    def walk(self, root):
        """Generate a permission error on folder "a/b"."""
        # Generate the permission error
        if root.startswith(p("a", "b")):
            raise OSError(errno.EACCES, os.strerror(errno.EACCES))

        # Mimic the original method
        yield from walk_orig(self, root)

    with patch.object(DirectorySnapshot, "walk", new=walk):
        # Should NOT raise an OSError (EACCES)
        new_snapshot = DirectorySnapshot(p(""))

    diff = DirectorySnapshotDiff(ref, new_snapshot)
    assert repr(diff)
    assert len(diff) == 1

    # Children of a/b/ are no more accessible and so removed in the new snapshot
    assert diff.dirs_deleted == [(p("a", "b", "c"))]


def test_ignore_device(p):
    # Create a file and take a snapshot.
    touch(p("file"))
    ref = DirectorySnapshot(p(""))
    wait()

    inode_orig = DirectorySnapshot.inode

    inode_times = 0

    def inode(self, path):
        # This function will always return a different device_id,
        # even for the same file.
        nonlocal inode_times
        result = inode_orig(self, path)
        inode_times += 1
        return result[0], result[1] + inode_times

    # Set the custom inode function.
    with patch.object(DirectorySnapshot, "inode", new=inode):
        # If we make the diff of the same directory, since by default the
        # DirectorySnapshotDiff compares the snapshots using the device_id (and it will
        # be different), it thinks that the same file has been deleted and created again.
        snapshot = DirectorySnapshot(p(""))
        diff_with_device = DirectorySnapshotDiff(ref, snapshot)
        assert diff_with_device.files_deleted == [(p("file"))]
        assert diff_with_device.files_created == [(p("file"))]

        # Otherwise, if we choose to ignore the device, the file will not be detected as
        # deleted and re-created.
        snapshot = DirectorySnapshot(p(""))
        diff_without_device = DirectorySnapshotDiff(ref, snapshot, ignore_device=True)
        assert not len(diff_without_device)
        assert diff_without_device.files_deleted == []
        assert diff_without_device.files_created == []


def test_link_aware_hardlinks_stable_with_shuffled_snapshots(p):
    touch(p("a"))
    os.link(p("a"), p("b"))
    os.link(p("a"), p("c"))

    def listdir_in_order(order):
        def listdir(path):
            if path != p(""):
                yield from os.scandir(path)
                return

            entries = list(os.scandir(path))
            entries.sort(key=lambda entry: order.index(entry.name) if entry.name in order else len(order))
            yield from entries

        return listdir

    expected = None
    names = ("a", "b", "c")
    for ref_order in permutations(names):
        ref = DirectorySnapshot(p(""), listdir=listdir_in_order(ref_order))

        rm(p("c"))
        mv(p("b"), p("d"))

        try:
            for snapshot_order in permutations(("a", "c", "d")):
                snapshot = DirectorySnapshot(p(""), listdir=listdir_in_order(snapshot_order))
                diff = DirectorySnapshotDiff(ref, snapshot, link_aware=True)
                result = (
                    diff.files_created,
                    diff.files_deleted,
                    diff.files_modified,
                    diff.files_moved,
                )
                if expected is None:
                    expected = result
                else:
                    assert result == expected
        finally:
            mv(p("d"), p("b"))
            os.link(p("a"), p("c"))

    assert expected == (
        [p("d")],
        [p("b"), p("c")],
        [],
        [],
    )


def test_link_aware_stable_move_pairing():
    ref = make_snapshot({"/a": (1, 1), "/b": (1, 1), "/c": (1, 1)})
    snapshot = make_snapshot({"/d": (1, 1), "/e": (1, 1), "/f": (1, 1)})

    diff = DirectorySnapshotDiff(ref, snapshot, link_aware=True)

    assert diff.files_moved == [("/a", "/d"), ("/b", "/e"), ("/c", "/f")]
    assert diff.files_created == []
    assert diff.files_deleted == []


def test_link_aware_reports_link_create_and_delete_when_move_is_ambiguous():
    ref = make_snapshot({"/a": (1, 1), "/b": (1, 1), "/c": (1, 1)})
    snapshot = make_snapshot({"/a": (1, 1), "/d": (1, 1)})

    diff = DirectorySnapshotDiff(ref, snapshot, link_aware=True)

    assert diff.files_created == ["/d"]
    assert diff.files_deleted == ["/b", "/c"]
    assert diff.files_moved == []


def test_link_aware_does_not_match_inodes_on_different_devices():
    ref = make_snapshot({"/a": (1, 1)})
    snapshot = make_snapshot({"/b": (1, 2)})

    diff = DirectorySnapshotDiff(ref, snapshot, link_aware=True)
    assert diff.files_created == ["/b"]
    assert diff.files_deleted == ["/a"]
    assert diff.files_moved == []

    diff_ignore_device = DirectorySnapshotDiff(ref, snapshot, link_aware=True, ignore_device=True)
    assert diff_ignore_device.files_moved == [("/a", "/b")]


def test_ignore_device_matches_moved_inodes():
    ref = make_snapshot({"/a": (1, 1)})
    snapshot = make_snapshot({"/b": (1, 2)})

    diff = DirectorySnapshotDiff(ref, snapshot, ignore_device=True)

    assert diff.files_moved == [("/a", "/b")]
    assert diff.files_created == []
    assert diff.files_deleted == []


def test_link_aware_content_changes_apply_to_remaining_paths(p):
    touch(p("a"))
    os.link(p("a"), p("b"))
    ref = DirectorySnapshot(p(""))
    wait()

    with open(p("a"), "w", encoding="utf-8") as file:
        file.write("changed")
    mv(p("a"), p("c"))

    diff = DirectorySnapshotDiff(ref, DirectorySnapshot(p("")), link_aware=True)

    assert diff.files_moved == [(p("a"), p("c"))]
    assert diff.files_modified == [p("b"), p("c")]
    assert diff.files_created == []
    assert diff.files_deleted == []


def test_link_aware_content_changes_apply_to_all_links(p):
    touch(p("a"))
    os.link(p("a"), p("b"))
    ref = DirectorySnapshot(p(""))
    wait()

    with open(p("a"), "w", encoding="utf-8") as file:
        file.write("changed")

    diff = DirectorySnapshotDiff(ref, DirectorySnapshot(p("")), link_aware=True)

    assert diff.files_modified == [p("a"), p("b")]
    assert diff.files_created == []
    assert diff.files_deleted == []
    assert diff.files_moved == []


def test_empty_snapshot(p):
    # Create a file and declare a DirectorySnapshot and a EmptyDirectorySnapshot.
    # When we make the diff, although both objects were declared with the same items on
    # the directory, the file and directories created BEFORE the DirectorySnapshot will
    # be detected as newly created.

    touch(p("a"))
    mkdir(p("b", "c"), parents=True)
    ref = DirectorySnapshot(p(""))
    empty = EmptyDirectorySnapshot()
    assert repr(empty) == "{}"

    diff = DirectorySnapshotDiff(empty, ref)
    assert len(diff) == 4
    assert diff.files_created == [p("a")]
    assert sorted(diff.dirs_created) == sorted([p(""), p("b"), p("b", "c")])
