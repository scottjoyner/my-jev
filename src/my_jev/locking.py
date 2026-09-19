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


def atomic_write_json(
    path: str | Path,
    payload: object,
) -> None:
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
        with temporary.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary,
            destination,
        )
    finally:
        with contextlib.suppress(
            FileNotFoundError
        ):
            temporary.unlink()


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
