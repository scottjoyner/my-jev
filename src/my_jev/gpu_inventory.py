"""Discover GPU availability from the host, and from the lease that guards it.

``gpu_plan`` allocates only from what it is told is free. Until now that meant a
hand-written JSON file, which is an assertion -- it cannot be wrong about which
devices exist, and it cannot notice that somebody started a run five minutes ago.

This module supplies the two facts an inventory needs, and is careful about which
of them it actually knows.

**Which devices exist** comes from PyTorch, or from ``nvidia-smi`` when PyTorch is
unavailable. That is a measurement.

**Whether a GPU is currently in use** comes from :mod:`my_jev.locking`, which is
the repo's existing mutual-exclusion authority. ``experiment.py`` takes a single
exclusive ``ResourceRequest("gpu")`` lease before it runs anything, so
attempting that lease without blocking and releasing it immediately is a real
answer to "is the GPU busy right now", backed by ``flock`` rather than by a flag
somebody remembered to clear.

**How many minutes are free** is neither of those, and this module does not
pretend otherwise. Nobody can know how long a job will take before it runs. Free
minutes are the caller's planning horizon, and the per-device split is a *model*
of how the host's GPUs would be shared, not a measurement of it.

That distinction is why the plan carries the caveat with it. ``gpu_plan`` will
happily place stages across several devices, and this repo's lease model has one
exclusive ``gpu`` resource, so such a plan contends for real. The planner is told
about that rather than left to imply a parallelism the locking system forbids.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path

from .gpu_plan import GpuCandidate
from .locking import LeaseSet, ResourceBusy, ResourceRequest

#: The lease name ``experiment.py`` already takes before touching a GPU. Reused
#: verbatim so the probe asks the same question the real run will ask, rather
#: than a second question that can disagree with it.
GPU_RESOURCE = "gpu"

_DEFAULT_PROBE_SECONDS = 5.0


class InventoryError(RuntimeError):
    """Raised when availability cannot be established at all."""


def detect_local_devices() -> list[str]:
    """Enumerate accelerator devices visible to this host.

    Returns an empty list rather than raising when nothing is detectable: no
    accelerator is a legitimate state for a planning run on a CPU host, and the
    caller decides whether that is fatal.
    """
    devices = _torch_devices()
    if devices:
        return devices
    return _nvidia_smi_devices()


def _torch_devices() -> list[str]:
    try:
        import torch
    except Exception:
        # A torch that cannot even import is an environment problem, not a
        # "no GPU" answer, so fall through to nvidia-smi rather than claiming
        # the host has no accelerator.
        return []
    try:
        if not torch.cuda.is_available():
            return []
        return [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    except Exception:
        return []


def _nvidia_smi_devices() -> list[str]:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return []
    try:
        completed = subprocess.run(  # noqa: S603
            [binary, "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=_DEFAULT_PROBE_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    indices = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return [f"cuda:{index}" for index in indices]


def gpu_lease_is_free(lock_dir: str | Path) -> bool:
    """Whether *no* card is claimed. True only when every detected card is free.

    Kept as a coarse question for callers that do not enumerate devices. A probe
    that takes the pool lock exclusively is blocked by any new-style run, because
    those hold it shared -- so this answers "is the whole pool idle", not "can I
    have a card".

    Prefer :func:`free_devices`, which asks the same question a real run asks and
    can therefore disagree with it less.
    """
    from .gpu_inventory import detect_local_devices

    devices = detect_local_devices()
    if not devices:
        return False
    return len(free_devices(lock_dir, devices)) == len(devices)


def free_devices(
    lock_dir: str | Path,
    devices: Sequence[str],
) -> list[str]:
    """Which of ``devices`` are unclaimed right now.

    Takes the same lease a run would take for that card, then gives it straight
    back. Asking the identical question is the point: a probe that used a coarser
    or differently-named lock could report a card as free while the run it is
    planning for could not actually have it.

    This is a probe with a race in it -- the lease is released before the caller
    acts on the answer, so a run starting in between will contend. That is
    acceptable because the plan is advisory and the real run takes its own lease.
    """
    free: list[str] = []
    for device in devices:
        leases = LeaseSet(
            lock_dir,
            _probe_requests(device),
            command="my-jev-gpu-plan inventory probe",
        )
        try:
            leases.acquire(blocking=False)
        except ResourceBusy:
            continue
        except OSError as exc:
            raise InventoryError(
                f"could not probe the GPU lease in {lock_dir}: {exc}"
            ) from exc
        finally:
            leases.release()
        free.append(device)
    return free


def _probe_requests(device: str) -> list[ResourceRequest]:
    """Mirror of ``experiment.gpu_lease_requests`` for one device.

    Duplicated rather than imported so this module does not depend on the run
    pipeline; the test that pins both to the same names is what stops them drifting.
    """
    from .experiment import gpu_lease_requests

    return gpu_lease_requests(device)


def _split_horizon(
    devices: Sequence[str],
    *,
    horizon_gpu_minutes: float,
    horizon_is_per_device: bool,
) -> list[float]:
    if not devices:
        return []
    if horizon_is_per_device:
        return [horizon_gpu_minutes for _ in devices]
    return [horizon_gpu_minutes / len(devices) for _ in devices]


def local_gpu_inventory(
    *,
    lock_dir: str | Path | None = None,
    horizon_gpu_minutes: float = 0.0,
    horizon_is_per_device: bool = False,
    untrusted: Iterable[str] = (),
    declared_devices: Sequence[str] | None = None,
) -> list[GpuCandidate]:
    """Build a ``gpu_plan`` inventory from this host.

    Args:
        lock_dir: Directory holding the ``gpu`` lease. When supplied, a held
            lease reports zero free minutes on every device -- the GPUs are in
            use by a run, so the plan must not schedule onto them.
        horizon_gpu_minutes: How long the planner may assume a device stays free.
            Zero is the safe default: it plans nothing and says so, rather than
            inventing availability.
        horizon_is_per_device: Whether the horizon applies to each device or is
            divided across them. This repo's lease model has one exclusive
            ``gpu`` resource, so the total is the honest quantity by default.
        untrusted: Device ids to never allocate, regardless of free time.
        declared_devices: Skip detection and use these. For planning against a
            remote fleet, where the local host is not the one that will run.

    Returns:
        One candidate per detected device, carrying the free minutes the
        planner may assume.
    """
    if horizon_gpu_minutes < 0.0:
        raise ValueError("horizon_gpu_minutes must not be negative")

    devices = list(declared_devices) if declared_devices is not None else detect_local_devices()
    if not devices:
        return []

    free = set(devices)
    if lock_dir is not None:
        # Per device, matching how a run would actually claim a card. Probing the
        # pool as a whole would report every card busy as soon as one is taken.
        free = set(free_devices(lock_dir, devices))

    if not free:
        per_device = [0.0 for _ in devices]
    else:
        split = _split_horizon(
            sorted(free),
            horizon_gpu_minutes=horizon_gpu_minutes,
            horizon_is_per_device=horizon_is_per_device,
        )
        per_device = [
            split[sorted(free).index(device)] if device in free else 0.0
            for device in devices
        ]

    untrusted_set = set(untrusted)
    return [
        GpuCandidate(
            gpu_id=device,
            free_gpu_minutes=per_device[index],
            trusted=device not in untrusted_set,
        )
        for index, device in enumerate(devices)
    ]


def inventory_from_file(path: str | Path) -> list[GpuCandidate]:
    """Read an inventory document, as ``gpu_plan_cli`` has always accepted."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload["gpus"] if isinstance(payload, dict) else payload
    return [
        GpuCandidate(
            gpu_id=str(entry["gpu_id"]),
            free_gpu_minutes=float(entry["free_gpu_minutes"]),
            trusted=bool(entry.get("trusted", True)),
        )
        for entry in entries
    ]


def default_lock_dir(experiment_registry_path: str | Path) -> Path:
    """Where ``experiment.py`` keeps its leases, derived from the registry path.

    The registry path is the only thing both share, so the probe asks about the
    same lease directory the run will lock rather than a plausible-looking
    neighbour of it.
    """
    registry = Path(experiment_registry_path)
    if registry.suffix == ".json":
        registry = registry.with_suffix("")
    return Path(f"{registry}.locks") if registry.name != "registry" else registry.parent / ".locks"


def describe_hardware() -> dict[str, object]:
    """A compact, loggable summary of what this host actually has.

    Separate from :func:`local_gpu_inventory` on purpose: this answers "what
    hardware is here", which is a fact worth recording next to a plan, while the
    inventory answers "what may I assume is free", which is an assumption.
    """
    devices = detect_local_devices()
    return {
        "device_count": len(devices),
        "devices": devices,
        "torch_visible": bool(_torch_devices()),
        "nvidia_smi_present": shutil.which("nvidia-smi") is not None,
        "hostname": os.uname().nodename,
    }