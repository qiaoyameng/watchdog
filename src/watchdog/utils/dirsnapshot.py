""":module: watchdog.utils.dirsnapshot
:synopsis: Directory snapshots and comparison.
:author: yesudeep@google.com (Yesudeep Mangalapilly)
:author: Mickaël Schoentgen <contact@tiger-222.fr>

.. ADMONITION:: Where are the moved events? They "disappeared"

    By default, file identity is the ``(st_ino, st_dev)`` pair, so equal
    inode numbers on different devices are not conflated. Use
    ``ignore_device=True`` only when a device id may change for the same
    physical file system, such as between boots; doing so compares inode
    numbers alone and is therefore unsafe for snapshots spanning different
    devices. Enable ``link_aware=True`` to use all hard-linked paths when
    pairing deleted and created snapshot entries.

Classes
-------
.. autoclass:: DirectorySnapshot
   :members:
   :show-inheritance:

.. autoclass:: DirectorySnapshotDiff
   :members:
   :show-inheritance:

.. autoclass:: EmptyDirectorySnapshot
   :members:
   :show-inheritance:

"""

from __future__ import annotations

import contextlib
import errno
import os
from collections import defaultdict
from stat import S_ISDIR
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any, Callable


InodeKey = int | tuple[int, int]
Path = bytes | str


def _path_sort_key(path: bytes | str) -> tuple[int, bytes | str]:
    if isinstance(path, bytes):
        return 1, path
    return 0, path


def _move_sort_key(move: tuple[bytes | str, bytes | str]) -> tuple[tuple[int, bytes | str], tuple[int, bytes | str]]:
    return _path_sort_key(move[0]), _path_sort_key(move[1])


class DirectorySnapshotDiff:
    """Compares two directory snapshots and creates an object that represents
    the difference between the two snapshots.

    :param ref:
        The reference directory snapshot.
    :type ref:
        :class:`DirectorySnapshot`
    :param snapshot:
        The directory snapshot which will be compared
        with the reference snapshot.
    :type snapshot:
        :class:`DirectorySnapshot`
    :param ignore_device:
        A boolean indicating whether to ignore the device id or not.
        By default, a file may be uniquely identified by a combination of its first
        inode and its device id. The problem is that the device id may (or may not)
        change between system boots. This problem would cause the DirectorySnapshotDiff
        to think a file has been deleted and created again but it would be the
        exact same file.
        Set to True only if you are sure you will always use the same device.
    :type ignore_device:
        :class:`bool`
    :param link_aware:
        Track every path sharing the same ``(inode, device)`` identity. Unchanged
        paths are retained first, and remaining deleted/created paths are paired
        as moves using a stable ordering. Equal added and removed counts can be
        explained by link deletion and creation whenever another path for the
        inode remains unchanged, so they are reported as creation and deletion
        instead of moves.
    :type link_aware:
        :class:`bool`
    """

    def __init__(
        self,
        ref: DirectorySnapshot,
        snapshot: DirectorySnapshot,
        *,
        ignore_device: bool = False,
        link_aware: bool = False,
    ) -> None:
        created = snapshot.paths - ref.paths
        deleted = ref.paths - snapshot.paths

        if ignore_device:

            def get_inode(directory: DirectorySnapshot, full_path: bytes | str) -> int | tuple[int, int]:
                return directory.inode(full_path)[0]

        else:

            def get_inode(directory: DirectorySnapshot, full_path: bytes | str) -> int | tuple[int, int]:
                return directory.inode(full_path)

        unchanged_paths: set[bytes | str] = set()

        # check that all unchanged paths have the same inode
        for path in ref.paths & snapshot.paths:
            if get_inode(ref, path) != get_inode(snapshot, path):
                created.add(path)
                deleted.add(path)
            else:
                unchanged_paths.add(path)

        # find moved paths
        moved: set[tuple[bytes | str, bytes | str]] = set()
        if link_aware:
            deleted_paths_by_inode: defaultdict[InodeKey, list[Path]] = defaultdict(list)
            created_paths_by_inode: defaultdict[InodeKey, list[Path]] = defaultdict(list)
            unchanged_paths_by_inode: defaultdict[InodeKey, list[Path]] = defaultdict(list)

            for path in deleted:
                deleted_paths_by_inode[get_inode(ref, path)].append(path)
            for path in created:
                created_paths_by_inode[get_inode(snapshot, path)].append(path)
            for path in unchanged_paths:
                unchanged_paths_by_inode[get_inode(ref, path)].append(path)

            for inode in set(deleted_paths_by_inode) | set(created_paths_by_inode):
                old_paths = sorted(deleted_paths_by_inode.get(inode, ()), key=_path_sort_key)
                new_paths = sorted(created_paths_by_inode.get(inode, ()), key=_path_sort_key)

                if unchanged_paths_by_inode.get(inode) and len(old_paths) == len(new_paths):
                    continue

                for old_path, new_path in zip(old_paths, new_paths):
                    deleted.remove(old_path)
                    created.remove(new_path)
                    moved.add((old_path, new_path))
        else:
            for path in set(deleted):
                inode = get_inode(ref, path)
                new_path = snapshot.path(inode)
                if new_path:
                    # file is not deleted but moved
                    deleted.remove(path)
                    moved.add((path, new_path))

            for path in set(created):
                inode = get_inode(snapshot, path)
                old_path = ref.path(inode)
                if old_path:
                    created.remove(path)
                    moved.add((old_path, path))

        # find modified paths
        # first check paths that have not moved
        modified: set[bytes | str] = set()
        for path in unchanged_paths if link_aware else ref.paths & snapshot.paths:
            if get_inode(ref, path) == get_inode(snapshot, path) and (
                ref.mtime(path) != snapshot.mtime(path) or ref.size(path) != snapshot.size(path)
            ):
                modified.add(path)

        for old_path, new_path in moved:
            if ref.mtime(old_path) != snapshot.mtime(new_path) or ref.size(old_path) != snapshot.size(new_path):
                modified.add(old_path)

        self._dirs_created = sorted((path for path in created if snapshot.isdir(path)), key=_path_sort_key)
        self._dirs_deleted = sorted((path for path in deleted if ref.isdir(path)), key=_path_sort_key)
        self._dirs_modified = sorted((path for path in modified if ref.isdir(path)), key=_path_sort_key)
        self._dirs_moved = sorted(((frm, to) for (frm, to) in moved if ref.isdir(frm)), key=_move_sort_key)

        self._files_created = sorted(created - set(self._dirs_created), key=_path_sort_key)
        self._files_deleted = sorted(deleted - set(self._dirs_deleted), key=_path_sort_key)
        self._files_modified = sorted(modified - set(self._dirs_modified), key=_path_sort_key)
        self._files_moved = sorted(moved - set(self._dirs_moved), key=_move_sort_key)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        fmt = (
            "<{0} files(created={1}, deleted={2}, modified={3}, moved={4}),"
            " folders(created={5}, deleted={6}, modified={7}, moved={8})>"
        )
        return fmt.format(
            type(self).__name__,
            len(self._files_created),
            len(self._files_deleted),
            len(self._files_modified),
            len(self._files_moved),
            len(self._dirs_created),
            len(self._dirs_deleted),
            len(self._dirs_modified),
            len(self._dirs_moved),
        )

    def __len__(self) -> int:
        return sum(len(getattr(self, attr)) for attr in dir(self) if attr.startswith(("_dirs_", "_files_")))

    @property
    def files_created(self) -> list[bytes | str]:
        """List of files that were created."""
        return self._files_created

    @property
    def files_deleted(self) -> list[bytes | str]:
        """List of files that were deleted."""
        return self._files_deleted

    @property
    def files_modified(self) -> list[bytes | str]:
        """List of files that were modified."""
        return self._files_modified

    @property
    def files_moved(self) -> list[tuple[bytes | str, bytes | str]]:
        """List of files that were moved.

        Each event is a two-tuple the first item of which is the path
        that has been renamed to the second item in the tuple.
        """
        return self._files_moved

    @property
    def dirs_modified(self) -> list[bytes | str]:
        """List of directories that were modified."""
        return self._dirs_modified

    @property
    def dirs_moved(self) -> list[tuple[bytes | str, bytes | str]]:
        """List of directories that were moved.

        Each event is a two-tuple the first item of which is the path
        that has been renamed to the second item in the tuple.
        """
        return self._dirs_moved

    @property
    def dirs_deleted(self) -> list[bytes | str]:
        """List of directories that were deleted."""
        return self._dirs_deleted

    @property
    def dirs_created(self) -> list[bytes | str]:
        """List of directories that were created."""
        return self._dirs_created

    class ContextManager:
        """Context manager that creates two directory snapshots and a
        diff object that represents the difference between the two snapshots.

        :param path:
            The directory path for which a snapshot should be taken.
        :type path:
            ``str``
        :param recursive:
            ``True`` if the entire directory tree should be included in the
            snapshot; ``False`` otherwise.
        :type recursive:
            ``bool``
        :param stat:
            Use custom stat function that returns a stat structure for path.
            Currently only st_dev, st_ino, st_mode and st_mtime are needed.

            A function taking a ``path`` as argument which will be called
            for every entry in the directory tree.
        :param listdir:
            Use custom listdir function. For details see ``os.scandir``.
        :param ignore_device:
            A boolean indicating whether to ignore the device id or not.
            By default, a file may be uniquely identified by a combination of its first
            inode and its device id. The problem is that the device id may (or may not)
            change between system boots. This problem would cause the DirectorySnapshotDiff
            to think a file has been deleted and created again but it would be the
            exact same file.
            Set to True only if you are sure you will always use the same device.
        :type ignore_device:
            :class:`bool`
        :param link_aware:
            Track hard links when matching deleted and created paths.
        :type link_aware:
            :class:`bool`
        """

        def __init__(
            self,
            path: str,
            *,
            recursive: bool = True,
            stat: Callable[[str], os.stat_result] = os.stat,
            listdir: Callable[[str | None], Iterator[os.DirEntry]] = os.scandir,
            ignore_device: bool = False,
            link_aware: bool = False,
        ) -> None:
            self.path = path
            self.recursive = recursive
            self.stat = stat
            self.listdir = listdir
            self.ignore_device = ignore_device
            self.link_aware = link_aware

        def __enter__(self) -> None:
            self.pre_snapshot = self.get_snapshot()

        def __exit__(self, *args: object) -> None:
            self.post_snapshot = self.get_snapshot()
            self.diff = DirectorySnapshotDiff(
                self.pre_snapshot,
                self.post_snapshot,
                ignore_device=self.ignore_device,
                link_aware=self.link_aware,
            )

        def get_snapshot(self) -> DirectorySnapshot:
            return DirectorySnapshot(
                path=self.path,
                recursive=self.recursive,
                stat=self.stat,
                listdir=self.listdir,
            )


class DirectorySnapshot:
    """A snapshot of stat information of files in a directory.

    :param path:
        The directory path for which a snapshot should be taken.
    :type path:
        ``str``
    :param recursive:
        ``True`` if the entire directory tree should be included in the
        snapshot; ``False`` otherwise.
    :type recursive:
        ``bool``
    :param stat:
        Use custom stat function that returns a stat structure for path.
        Currently only st_dev, st_ino, st_mode and st_mtime are needed.

        A function taking a ``path`` as argument which will be called
        for every entry in the directory tree.
    :param listdir:
        Use custom listdir function. For details see ``os.scandir``.
    """

    def __init__(
        self,
        path: str,
        *,
        recursive: bool = True,
        stat: Callable[[str], os.stat_result] = os.stat,
        listdir: Callable[[str | None], Iterator[os.DirEntry]] = os.scandir,
    ) -> None:
        self.recursive = recursive
        self.stat = stat
        self.listdir = listdir

        self._stat_info: dict[bytes | str, os.stat_result] = {}
        self._inode_to_paths: dict[tuple[int, int], set[bytes | str]] = {}

        st = self.stat(path)
        self._stat_info[path] = st
        self._inode_to_paths.setdefault((st.st_ino, st.st_dev), set()).add(path)

        for p, st in self.walk(path):
            i = (st.st_ino, st.st_dev)
            self._inode_to_paths.setdefault(i, set()).add(p)
            self._stat_info[p] = st

    def walk(self, root: str) -> Iterator[tuple[str, os.stat_result]]:
        try:
            paths = [os.path.join(root, entry.name) for entry in self.listdir(root)]
        except OSError as e:
            # Directory may have been deleted between finding it in the directory
            # list of its parent and trying to delete its contents. If this
            # happens we treat it as empty. Likewise if the directory was replaced
            # with a file of the same name (less likely, but possible).
            if e.errno in (errno.ENOENT, errno.ENOTDIR, errno.EINVAL):
                return
            else:
                raise

        entries = []
        for p in paths:
            with contextlib.suppress(OSError):
                entry = (p, self.stat(p))
                entries.append(entry)
                yield entry

        if self.recursive:
            for path, st in entries:
                with contextlib.suppress(PermissionError):
                    if S_ISDIR(st.st_mode):
                        yield from self.walk(path)

    @property
    def paths(self) -> set[bytes | str]:
        """Set of file/directory paths in the snapshot."""
        return set(self._stat_info.keys())

    def path(self, uid: tuple[int, int]) -> bytes | str | None:
        """Returns path for id. None if id is unknown to this snapshot."""
        paths = self._inode_to_paths.get(uid)
        return min(paths, key=_path_sort_key) if paths else None

    def paths_for_inode(self, uid: tuple[int, int]) -> list[bytes | str]:
        """Returns all paths for an inode id. The list is empty if the id is unknown."""
        return sorted(self._inode_to_paths.get(uid, ()), key=_path_sort_key)

    def inode(self, path: bytes | str) -> tuple[int, int]:
        """Returns an id for path."""
        st = self._stat_info[path]
        return (st.st_ino, st.st_dev)

    def isdir(self, path: bytes | str) -> bool:
        """Returns ``True`` if the path is a directory."""
        return S_ISDIR(self._stat_info[path].st_mode)

    def mtime(self, path: bytes | str) -> float:
        """Returns the last modification time for path."""
        return self._stat_info[path].st_mtime

    def size(self, path: bytes | str) -> int:
        """Returns the size of the file identified by path."""
        return self._stat_info[path].st_size

    def stat_info(self, path: bytes | str) -> os.stat_result:
        """Returns a stat information object for the specified path from
        the snapshot.

        Attached information is subject to change. Do not use unless
        you specify `stat` in constructor. Use :meth:`inode`, :meth:`mtime`,
        :meth:`isdir` instead.

        :param path:
            The path for which stat information should be obtained
            from a snapshot.
        """
        return self._stat_info[path]

    def __sub__(self, previous_dirsnap: DirectorySnapshot) -> DirectorySnapshotDiff:
        """Allow subtracting a DirectorySnapshot object instance from
        another.

        :returns:
            A :class:`DirectorySnapshotDiff` object.
        """
        return DirectorySnapshotDiff(previous_dirsnap, self)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return str(self._stat_info)


class EmptyDirectorySnapshot(DirectorySnapshot):
    """Class to implement an empty snapshot. This is used together with
    DirectorySnapshot and DirectorySnapshotDiff in order to get all the files/folders
    in the directory as created.
    """

    def __init__(self) -> None:
        self._stat_info: dict[bytes | str, os.stat_result] = {}
        self._inode_to_paths: dict[tuple[int, int], set[bytes | str]] = {}

    @staticmethod
    def path(_: Any) -> None:
        """Mock up method to return the path of the received inode. As the snapshot
        is intended to be empty, it always returns None.

        :returns:
            None.
        """
        return

    @staticmethod
    def paths_for_inode(_: Any) -> list[bytes | str]:
        """Mock up method to return all paths for an inode id. As the snapshot
        is intended to be empty, it always returns an empty list.
        """
        return []

    @property
    def paths(self) -> set:
        """Mock up method to return a set of file/directory paths in the snapshot. As
        the snapshot is intended to be empty, it always returns an empty set.

        :returns:
            An empty set.
        """
        return set()
