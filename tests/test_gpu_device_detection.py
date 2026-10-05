"""Device detection's decision tree, tested without touching real hardware.

`detect_local_devices` runs two probes in order -- torch, then `nvidia-smi` -- and
every branch of that chain was untested. The only direct test was

    assert isinstance(detect_local_devices(), list)

which asserts a return type, runs whatever hardware the machine running it happens
to have, and would pass on a GPU host and a CPU host alike. Its neighbour
`gpu_lease_is_free` carries a docstring recording that the suite once passed locally
and failed in CI for exactly this reason.

These tests pin the chain itself, including the case that matters on a ROCm host:
torch fails to import *and* `nvidia-smi` is absent, so both probes come back empty
on a machine that plainly has a card. That is a lossy fallback and it is
deliberate -- better an honest "I cannot see a GPU" than a false one -- so the test
documents the loss rather than pretending it away.
"""

from __future__ import annotations

import subprocess

import pytest

from my_jev import gpu_inventory
from my_jev.gpu_inventory import detect_local_devices


def test_torch_devices_are_preferred(monkeypatch):
    monkeypatch.setattr(gpu_inventory, "_torch_devices", lambda: ["cuda:0"])
    monkeypatch.setattr(
        gpu_inventory,
        "_nvidia_smi_devices",
        lambda: pytest.fail("nvidia-smi must not be consulted when torch sees a card"),
    )
    assert detect_local_devices() == ["cuda:0"]


def test_nvidia_smi_is_the_fallback_when_torch_sees_nothing(monkeypatch):
    monkeypatch.setattr(gpu_inventory, "_torch_devices", lambda: [])
    monkeypatch.setattr(gpu_inventory, "_nvidia_smi_devices", lambda: ["cuda:3"])
    assert detect_local_devices() == ["cuda:3"]


def test_no_probe_finds_anything_returns_empty(monkeypatch):
    monkeypatch.setattr(gpu_inventory, "_torch_devices", lambda: [])
    monkeypatch.setattr(gpu_inventory, "_nvidia_smi_devices", lambda: [])
    assert detect_local_devices() == []


# --- the torch probe -------------------------------------------------------


def test_torch_probe_reports_one_device_per_index(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    assert gpu_inventory._torch_devices() == ["cuda:0", "cuda:1", "cuda:2"]


def test_torch_probe_returns_empty_when_cuda_is_unavailable(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert gpu_inventory._torch_devices() == []


def test_a_torch_that_cannot_import_is_not_read_as_no_gpu(monkeypatch):
    """A broken install is an environment problem, not an answer about hardware.

    Silently returning "no accelerator" for a torch that will not import would make
    a broken environment look like a CPU-only host, and the run would proceed
    believing there is nothing to plan for.
    """
    import builtins

    real_import = builtins.__import__

    def refuse_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("no torch here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_torch)
    assert gpu_inventory._torch_devices() == []


def test_a_torch_probe_that_raises_is_treated_as_no_answer(monkeypatch):
    """Probing CUDA can fail on a driver that torch otherwise imports cleanly."""
    torch = pytest.importorskip("torch")

    def explode():
        raise RuntimeError("driver mismatch")

    monkeypatch.setattr(torch.cuda, "is_available", explode)
    assert gpu_inventory._torch_devices() == []


# --- the nvidia-smi probe --------------------------------------------------


class _Completed:
    def __init__(self, returncode: int, stdout: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = ""


def test_nvidia_smi_indices_become_cuda_ids(monkeypatch):
    monkeypatch.setattr(gpu_inventory.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        gpu_inventory.subprocess,
        "run",
        lambda *a, **k: _Completed(0, "0\n1\n2\n"),
    )
    assert gpu_inventory._nvidia_smi_devices() == ["cuda:0", "cuda:1", "cuda:2"]


def test_a_missing_nvidia_smi_binary_is_not_an_error(monkeypatch):
    monkeypatch.setattr(gpu_inventory.shutil, "which", lambda name: None)
    assert gpu_inventory._nvidia_smi_devices() == []


@pytest.mark.parametrize("returncode", [1, 2, 127])
def test_a_failing_nvidia_smi_returns_nothing_even_when_it_printed_something(
    monkeypatch, returncode
):
    """Non-empty stdout on a failed probe, which is the case that matters.

    The first version of this test returned an empty string with a non-zero code,
    so it passed for the wrong reason -- the output was already empty and the
    return-code check was never what made it so. Dropping the return-code check
    survived mutation. A failing `nvidia-smi` can easily print a warning or a
    partial index list on stdout, and honouring that output would report devices
    from a probe that failed.
    """
    monkeypatch.setattr(gpu_inventory.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        gpu_inventory.subprocess,
        "run",
        lambda *a, **k: _Completed(returncode, "0\n1\n"),
    )
    assert gpu_inventory._nvidia_smi_devices() == []


def test_a_nvidia_smi_that_raises_is_returned_as_nothing(monkeypatch):
    monkeypatch.setattr(gpu_inventory.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def explode(*args, **kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 5)

    monkeypatch.setattr(gpu_inventory.subprocess, "run", explode)
    assert gpu_inventory._nvidia_smi_devices() == []


def test_the_lossy_case_is_documented_not_hidden():
    """Both probes empty on a host that has a card: empty is returned, honestly.

    Pinned because it is the one case where the answer is wrong rather than
    merely unhelpful, and a future "fix" that guesses a device count would be
    worse than the loss.
    """
    monkeypatch_free = pytest.MonkeyPatch()
    with monkeypatch_free.context() as patch:
        patch.setattr(gpu_inventory, "_torch_devices", lambda: [])
        patch.setattr(gpu_inventory, "_nvidia_smi_devices", lambda: [])
        assert detect_local_devices() == []
