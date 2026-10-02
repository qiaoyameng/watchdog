from __future__ import annotations

import errno
import os
import pickle
import time
from unittest.mock import patch

from watchdog.utils import platform
from watchdog.utils.dirsnapshot import DirectorySnapshot, DirectorySnapshotDiff, EmptyDirectorySnapshot

from .shell import mkdir, mv, rm, touch


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


def test_prune_dirs_skips_whole_subtree(p):
    # A pruned directory is never entered: its listdir is not called, it is
    # not stat-ed and neither it nor its contents appear in the snapshot.
    mkdir(p("cache", "sub"), parents=True)
    touch(p("cache", "a"))
    touch(p("cache", "sub", "b"))
    mkdir(p("keep"))
    touch(p("keep", "c"))

    entered = []

    def listdir_fcn(path):
        entered.append(path)
        return os.scandir(path)

    snapshot = DirectorySnapshot(p(""), listdir=listdir_fcn, prune_dirs=["cache"])

    assert p("cache") not in snapshot.paths
    assert p("cache", "a") not in snapshot.paths
    assert p("cache", "sub") not in snapshot.paths
    assert p("cache", "sub", "b") not in snapshot.paths
    assert p("keep") in snapshot.paths
    assert p("keep", "c") in snapshot.paths
    # The root and "keep" were listed, but "cache" was never entered.
    assert p("cache") not in entered
    assert p("") in entered
    assert p("keep") in entered


def test_prune_dirs_relative_path_with_separators(p):
    # Rules are relative to the snapshot root and accept POSIX separators
    # even when expressed as nested paths.
    mkdir(p("build", "cache", "x"), parents=True)
    touch(p("build", "cache", "x", "f"))
    touch(p("build", "out"))

    entered = []

    def listdir_fcn(path):
        entered.append(path)
        return os.scandir(path)

    snapshot = DirectorySnapshot(p(""), listdir=listdir_fcn, prune_dirs=["build/cache"])

    assert p("build", "cache") not in snapshot.paths
    assert p("build", "cache", "x", "f") not in snapshot.paths
    assert p("build", "out") in snapshot.paths
    assert p("build", "cache") not in entered


def test_prune_dirs_evaluated_before_entering_on_error(p):
    # Pruning is decided before the directory is entered, so an unreadable
    # pruned directory does not surface any error during the scan.
    mkdir(p("cache"))
    touch(p("cache", "a"))

    def listdir_fcn(path):
        if path == p("cache"):
            raise OSError(errno.EACCES, os.strerror(errno.EACCES))
        return os.scandir(path)

    snapshot = DirectorySnapshot(p(""), listdir=listdir_fcn, prune_dirs=["cache"])
    assert p("cache") not in snapshot.paths
    assert p("cache", "a") not in snapshot.paths


def test_prune_dirs_do_not_affect_sibling_files(p):
    # File filtering and directory pruning are separate concerns: pruning a
    # directory never excludes unrelated files.
    mkdir(p("cache"))
    touch(p("cache", "a"))
    touch(p("cache.txt"))

    snapshot = DirectorySnapshot(p(""), prune_dirs=["cache"])
    assert p("cache") not in snapshot.paths
    assert p("cache.txt") in snapshot.paths


def test_prune_dirs_switch_requires_baseline_rebuild(p):
    # Comparing snapshots taken under different prune rules naively reports the
    # newly hidden files as deletions; rebuilding the baseline with the new
    # rules produces no spurious deletions.
    mkdir(p("cache"))
    touch(p("cache", "a"))
    touch(p("keep"))

    ref = DirectorySnapshot(p(""))
    pruned = DirectorySnapshot(p(""), prune_dirs=["cache"])

    naive_diff = DirectorySnapshotDiff(ref, pruned)
    assert naive_diff.files_deleted == [p("cache", "a")]
    assert naive_diff.dirs_deleted == [p("cache")]

    # Explicitly rebuild the comparison baseline using the new rules.
    rebuilt = DirectorySnapshot(p(""), prune_dirs=["cache"])
    diff = DirectorySnapshotDiff(pruned, rebuilt)
    assert diff.files_deleted == []
    assert diff.dirs_deleted == []
    assert diff.files_created == []
    assert diff.dirs_created == []
