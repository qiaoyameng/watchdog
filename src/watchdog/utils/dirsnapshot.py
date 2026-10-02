""":module: watchdog.utils.dirsnapshot
:synopsis: Directory snapshots and comparison.
:author: yesudeep@google.com (Yesudeep Mangalapilly)
:author: Mickaël Schoentgen <contact@tiger-222.fr>

.. ADMONITION:: Where are the moved events? They "disappeared"

    This implementation does not take partition boundaries
    into consideration. It will only work when the directory
    tree is entirely on the same file system. More specifically,
    any part of the code that depends on inode numbers can
    break if partition boundaries are crossed. In these cases,
    the snapshot diff will represent file/directory movement as
    created and deleted events.

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
from stat import S_ISDIR
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any, Callable


def _sort_paths(paths: set[bytes | str]) -> list[bytes | str]:
    try:
        return sorted(paths)
    except TypeError:
        return sorted(paths, key=lambda path: repr(path))


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
        This only removes the device component from identity; it does not make
        inode numbers from different devices safe to compare.
    :type ignore_device:
        :class:`bool`
    :param link_aware:
        Follow all paths for a hard-linked inode instead of retaining only one
        path. Unchanged paths are kept first. Remaining deleted and created paths
        are paired as moves only when both sides have the same number of paths
        for an inode. Pairs are selected by sorting both path sets, so snapshots
        traversed in different orders produce the same result. If a move cannot
        be identified unambiguously, the paths are reported as created and
        deleted. Device ids remain part of the identity unless
        ``ignore_device`` is set.
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

        # Check unchanged paths before looking for moves. In link-aware mode this
        # leaves only paths which actually disappeared or appeared.
        common_paths = ref.paths & snapshot.paths
        unchanged_paths: set[bytes | str] = set()
        for path in common_paths:
            if get_inode(ref, path) != get_inode(snapshot, path):
                created.add(path)
                deleted.add(path)
            else:
                unchanged_paths.add(path)

        # find moved paths
        moved: set[tuple[bytes | str, bytes | str]] = set()
        if link_aware:
            deleted_by_inode: dict[int | tuple[int, int], set[bytes | str]] = {}
            created_by_inode: dict[int | tuple[int, int], set[bytes | str]] = {}
            for path in deleted:
                deleted_by_inode.setdefault(get_inode(ref, path), set()).add(path)
            for path in created:
                created_by_inode.setdefault(get_inode(snapshot, path), set()).add(path)

            for inode in deleted_by_inode.keys() | created_by_inode.keys():
                old_paths = deleted_by_inode.get(inode, set())
                new_paths = created_by_inode.get(inode, set())

                # Only an exact one-to-one correspondence proves a rename at the
                # snapshot level. Different counts can represent link creation or
                # removal, so leave those paths as explicit create/delete events.
                if len(old_paths) == len(new_paths) and old_paths and new_paths:
                    for old_path, new_path in zip(_sort_paths(old_paths), _sort_paths(new_paths)):
                        deleted.discard(old_path)
                        created.discard(new_path)
                        moved.add((old_path, new_path))
        else:
            for path in set(deleted):
                inode = get_inode(ref, path)
                new_path = snapshot.path(inode)
                if new_path:
                    deleted.remove(path)
                    moved.add((path, new_path))

            for path in set(created):
                inode = get_inode(snapshot, path)
                old_path = ref.path(inode)
                if old_path:
                    created.remove(path)
                    moved.add((old_path, path))

        # find modified paths
        # First check paths that did not move. In link-aware mode this records a
        # content change at every remaining path for the inode rather than only
        # the arbitrary path retained by an inode-to-path mapping.
        modified: set[bytes | str] = set()
        for path in unchanged_paths:
            if (
                ref.mtime(path) != snapshot.mtime(path) or ref.size(path) != snapshot.size(path)
            ):
                modified.add(path)

        for old_path, new_path in moved:
            if ref.mtime(old_path) != snapshot.mtime(new_path) or ref.size(old_path) != snapshot.size(new_path):
                modified.add(new_path if link_aware else old_path)

        self._dirs_created = _sort_paths({path for path in created if snapshot.isdir(path)})
        self._dirs_deleted = _sort_paths({path for path in deleted if ref.isdir(path)})
        self._dirs_modified = _sort_paths(
            {path for path in modified if (snapshot if link_aware else ref).isdir(path)}
        )
        self._dirs_moved = sorted(
            ((frm, to) for (frm, to) in moved if ref.isdir(frm)),
            key=repr,
        )

        self._files_created = _sort_paths(created - set(self._dirs_created))
        self._files_deleted = _sort_paths(deleted - set(self._dirs_deleted))
        self._files_modified = _sort_paths(modified - set(self._dirs_modified))
        self._files_moved = sorted(moved - set(self._dirs_moved), key=repr)

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
            Track all paths for hard-linked inodes and use stable move pairing.
            See :class:`DirectorySnapshotDiff` for details.
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

    def path(self, uid: int | tuple[int, int]) -> bytes | str | None:
        """Returns path for id. None if id is unknown to this snapshot."""
        if isinstance(uid, tuple):
            paths = self._inode_to_paths.get(uid)
        else:
            paths = {
                path
                for path, stat_info in self._stat_info.items()
                if stat_info.st_ino == uid
            }
        if not paths:
            return None
        return _sort_paths(paths)[0]

    def paths_for_inode(self, uid: int | tuple[int, int]) -> set[bytes | str]:
        """Returns all paths for an inode id."""
        if isinstance(uid, tuple):
            return set(self._inode_to_paths.get(uid, set()))
        return {
            path
            for path, stat_info in self._stat_info.items()
            if stat_info.st_ino == uid
        }

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

    @property
    def paths(self) -> set:
        """Mock up method to return a set of file/directory paths in the snapshot. As
        the snapshot is intended to be empty, it always returns an empty set.

        :returns:
            An empty set.
        """
        return set()
