from __future__ import annotations

import contextlib
import fcntl
import json
import os
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


class ResourceBusy(RuntimeError):
    """Raised when another process owns a non-blocking experiment lease."""


@dataclass(frozen=True)
class ResourceRequest:
    name: str
    shared: bool = False


def _safe_name(value: str) -> str:
    return "".join(
        character
        if character.isalnum()
        or character in {"-", "_"}
        else "-"
        for character in value
    )


def atomic_write_bytes(
    path: str | Path,
    payload: bytes,
) -> None:
    """Write bytes so a reader never observes a partial file, and survives a crash.

    Every run artifact in this repository -- manifests, run registry entries,
    benchmark state, lease owner records, signed evidence envelopes -- is written
    through here, which makes this the most load-bearing function in the codebase
    and, before this change, one with no test at all.

    Four things make it atomic, and each covers a distinct failure:

    * **The temporary file is in the destination directory.** A temp file elsewhere
      would make the final step a cross-device move, which is not atomic and fails
      outright across filesystems.
    * **Its name is unique and hidden.** The uuid means two writes to the same
      destination never collide -- including two in the same process, which a
      pid-only name does collide on. The leading dot keeps it out of ``*.json``
      globs, so a half-written artifact is never mistaken for a real one by a
      reader or a cleanup script.
    * **The file is fsynced before the rename.** Otherwise the rename can land
      while the contents are still only in the page cache, and a crash after the
      rename yields a correctly named file with no content in it.
    * **The parent directory is fsynced after the rename.** The rename itself is a
      directory mutation. Without this the file's *content* is durable but the
      *entry* may not be, so a power loss can leave the previous version or nothing
      at all. Process death alone does not need this -- page cache survives it --
      which is why the distinction is worth stating rather than assuming.

    The temporary file is removed on any failure, so a failed write leaves the
    previous contents untouched rather than truncating them.
    """
    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = destination.with_name(
        f".{destination.name}."
        f"{os.getpid()}."
        f"{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary,
            destination,
        )
        # Make the rename itself durable. Best-effort: a filesystem that refuses to
        # open a directory for reading is unusual, and failing the write at this
        # point would be worse than the gap it closes.
        _fsync_directory(destination.parent)
    finally:
        with contextlib.suppress(
            FileNotFoundError
        ):
            temporary.unlink()


def _fsync_directory(directory: Path) -> None:
    """Flush a directory entry, so a completed rename survives a power loss.

    Opened read-only and closed immediately. Best-effort by design: this is a
    durability improvement, not a correctness requirement, and a platform that
    cannot do it should not fail the write that already succeeded.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_json(
    path: str | Path,
    payload: object,
) -> None:
    """Write JSON deterministically, atomically.

    ``sort_keys=True`` because these files are evidence: two runs producing the
    same document should produce the same bytes, so a reviewer can diff them.

    Delegates to :func:`atomic_write_bytes` rather than repeating the rename
    sequence, because this function had a weaker twin in
    ``harnessrouter_probe`` that had drifted -- no fsync, a pid-only temporary
    name that collided within a process, and no cleanup on failure -- and two
    copies of a durability primitive is how the signed artifacts ended up written
    by the weaker one.
    """
    atomic_write_bytes(
        path,
        (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )


class Lease:
    """Kernel-backed resource lease with a diagnostic JSON owner record."""

    def __init__(
        self,
        lock_dir: str | Path,
        request: ResourceRequest,
        *,
        command: str = "",
    ) -> None:
        self.lock_dir = Path(lock_dir)
        self.request = request
        self.command = command
        self.lease_id = uuid.uuid4().hex
        self.fd: int | None = None
        self.owner_path: Path | None = None

    def acquire(
        self,
        *,
        blocking: bool = False,
    ) -> Lease:
        self.lock_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        lock_path = (
            self.lock_dir
            / f"{_safe_name(self.request.name)}.lock"
        )
        self.fd = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT,
            0o664,
        )
        mode = (
            fcntl.LOCK_SH
            if self.request.shared
            else fcntl.LOCK_EX
        )
        if not blocking:
            mode |= fcntl.LOCK_NB
        try:
            fcntl.flock(
                self.fd,
                mode,
            )
        except BlockingIOError as exc:
            os.close(self.fd)
            self.fd = None
            raise ResourceBusy(
                f"resource busy: "
                f"{self.request.name}"
            ) from exc

        owners_dir = (
            self.lock_dir
            / "owners"
            / _safe_name(
                self.request.name
            )
        )
        owners_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.owner_path = (
            owners_dir
            / f"{self.lease_id}.json"
        )
        atomic_write_json(
            self.owner_path,
            {
                "lease_id": self.lease_id,
                "resource": (
                    self.request.name
                ),
                "mode": (
                    "shared"
                    if self.request.shared
                    else "exclusive"
                ),
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "acquired_at": time.time(),
                "command": (
                    self.command
                    or " ".join(
                        sys.argv
                    )
                ),
                "cwd": os.getcwd(),
            },
        )
        return self

    def release(self) -> None:
        if self.owner_path is not None:
            with contextlib.suppress(
                FileNotFoundError
            ):
                self.owner_path.unlink()
            self.owner_path = None
        if self.fd is not None:
            with contextlib.suppress(
                OSError
            ):
                fcntl.flock(
                    self.fd,
                    fcntl.LOCK_UN,
                )
            os.close(self.fd)
            self.fd = None

    def __enter__(self) -> Lease:
        return self.acquire()

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ) -> None:
        del exc_type, exc, tb
        self.release()


class LeaseSet:
    def __init__(
        self,
        lock_dir: str | Path,
        requests: list[ResourceRequest],
        *,
        command: str = "",
    ) -> None:
        merged: dict[
            str,
            ResourceRequest,
        ] = {}
        for request in requests:
            previous = merged.get(
                request.name
            )
            if (
                previous is None
                or (
                    previous.shared
                    and not request.shared
                )
            ):
                merged[
                    request.name
                ] = request
        self.leases = [
            Lease(
                lock_dir,
                merged[name],
                command=command,
            )
            for name in sorted(
                merged
            )
        ]

    def acquire(
        self,
        *,
        blocking: bool = False,
    ) -> LeaseSet:
        acquired: list[Lease] = []
        try:
            for lease in self.leases:
                lease.acquire(
                    blocking=blocking
                )
                acquired.append(
                    lease
                )
        except Exception:
            for lease in reversed(
                acquired
            ):
                lease.release()
            raise
        return self

    def release(self) -> None:
        for lease in reversed(
            self.leases
        ):
            lease.release()

    def __enter__(self) -> LeaseSet:
        return self.acquire()

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ) -> None:
        del exc_type, exc, tb
        self.release()
