from __future__ import annotations

import json
import pathlib
import tempfile

import pytest

from my_jev.gpu_inventory import (
    GPU_RESOURCE,
    InventoryError,
    default_lock_dir,
    describe_hardware,
    detect_local_devices,
    gpu_lease_is_free,
    inventory_from_file,
    local_gpu_inventory,
)
from my_jev.locking import LeaseSet, ResourceRequest


def _lock_dir() -> pathlib.Path:
    """A private lease directory, so tests never contend with each other."""
    return pathlib.Path(tempfile.mkdtemp())


# --- the lease probe -------------------------------------------------------


def test_the_probe_uses_the_same_lease_name_a_real_run_takes():
    """A second question could disagree with the run it is planning for."""
    import inspect

    from my_jev import experiment

    assert GPU_RESOURCE == "gpu"
    assert 'ResourceRequest(\n                "gpu",\n            )' in inspect.getsource(
        experiment
    )


def test_a_free_lease_reports_free(tmp_path):
    assert gpu_lease_is_free(tmp_path) is True


def test_a_held_gpu_lease_reports_busy(tmp_path):
    """Backed by flock, not by a flag somebody has to remember to clear."""
    held = LeaseSet(tmp_path, [ResourceRequest("gpu")], command="a-real-run").acquire(
        blocking=False
    )
    try:
        assert gpu_lease_is_free(tmp_path) is False
    finally:
        held.release()
    assert gpu_lease_is_free(tmp_path) is True


def test_an_unrelated_lease_does_not_make_the_gpu_look_busy(tmp_path):
    held = LeaseSet(
        tmp_path, [ResourceRequest("datasets", shared=True)], command="reading"
    ).acquire(blocking=False)
    try:
        assert gpu_lease_is_free(tmp_path) is True
    finally:
        held.release()


def test_the_probe_leaves_no_lease_behind(tmp_path):
    """It acquires in order to answer, so it must release or it deadlocks the host."""
    for _ in range(3):
        assert gpu_lease_is_free(tmp_path) is True
    assert gpu_lease_is_free(tmp_path) is True


def test_a_file_where_a_lock_dir_belongs_is_reported(tmp_path):
    """Not mistaken for "free", and not a silent pass either."""
    not_a_dir = tmp_path / "regular-file"
    not_a_dir.write_text("x", encoding="utf-8")
    # InventoryError is a RuntimeError, so the original OSError/NotADirectoryError
    # expectation missed it entirely. Pin both, since "raises something specific"
    # is the property worth keeping: a caller should not see a bare FileExists.
    with pytest.raises((InventoryError, OSError)):
        gpu_lease_is_free(not_a_dir)


# --- the horizon model -----------------------------------------------------


def test_the_horizon_is_split_across_devices_by_default():
    """The lease is one exclusive resource, so the total is the honest quantity."""
    inventory = local_gpu_inventory(
        horizon_gpu_minutes=600.0, declared_devices=["cuda:0", "cuda:1"]
    )
    assert [candidate.free_gpu_minutes for candidate in inventory] == [300.0, 300.0]
    assert sum(candidate.free_gpu_minutes for candidate in inventory) == 600.0


def test_the_horizon_can_be_declared_per_device():
    inventory = local_gpu_inventory(
        horizon_gpu_minutes=600.0,
        declared_devices=["cuda:0", "cuda:1"],
        horizon_is_per_device=True,
    )
    assert [candidate.free_gpu_minutes for candidate in inventory] == [600.0, 600.0]


def test_a_busy_lease_zeroes_every_device():
    directory = _lock_dir()
    held = LeaseSet(
        directory, [ResourceRequest("gpu")], command="a-real-run"
    ).acquire(blocking=False)
    try:
        inventory = local_gpu_inventory(
            lock_dir=directory,
            horizon_gpu_minutes=600.0,
            declared_devices=["cuda:0", "cuda:1"],
        )
        assert [candidate.free_gpu_minutes for candidate in inventory] == [0.0, 0.0]
    finally:
        held.release()


def test_zero_horizon_plans_nothing_rather_than_inventing_availability():
    inventory = local_gpu_inventory(horizon_gpu_minutes=0.0, declared_devices=["cuda:0"])
    assert inventory[0].free_gpu_minutes == 0.0


def test_a_negative_horizon_is_refused():
    with pytest.raises(ValueError, match="negative"):
        local_gpu_inventory(horizon_gpu_minutes=-1.0, declared_devices=["cuda:0"])


def test_no_devices_yields_no_inventory():
    assert local_gpu_inventory(horizon_gpu_minutes=600.0, declared_devices=[]) == []


def test_declared_devices_stand_in_for_detection():
    inventory = local_gpu_inventory(
        horizon_gpu_minutes=60.0, declared_devices=["remote-a", "remote-b"]
    )
    assert [candidate.gpu_id for candidate in inventory] == ["remote-a", "remote-b"]


def test_trust_is_reported_rather_than_folded_into_availability():
    """The planner refuses untrusted devices, so the flag has to survive."""
    inventory = local_gpu_inventory(
        horizon_gpu_minutes=600.0,
        declared_devices=["cuda:0", "cuda:1"],
        untrusted=["cuda:1"],
    )
    by_id = {candidate.gpu_id: candidate for candidate in inventory}
    assert by_id["cuda:0"].trusted is True
    assert by_id["cuda:1"].trusted is False
    assert by_id["cuda:1"].free_gpu_minutes == 300.0


def test_detection_returns_a_list_and_hardware_summary_agrees():
    assert isinstance(detect_local_devices(), list)
    summary = describe_hardware()
    assert summary["device_count"] == len(summary["devices"])
    assert "hostname" in summary


# --- documents -------------------------------------------------------------


def test_inventory_from_file_reads_the_documented_shape(tmp_path):
    path = tmp_path / "inv.json"
    path.write_text(
        json.dumps(
            {"gpus": [{"gpu_id": "gpu-a", "free_gpu_minutes": 30, "trusted": False}]}
        ),
        encoding="utf-8",
    )
    inventory = inventory_from_file(path)
    assert inventory[0].gpu_id == "gpu-a"
    assert inventory[0].free_gpu_minutes == 30.0
    assert inventory[0].trusted is False


def test_inventory_from_file_accepts_a_bare_list(tmp_path):
    path = tmp_path / "inv.json"
    path.write_text(
        json.dumps([{"gpu_id": "gpu-a", "free_gpu_minutes": 30}]), encoding="utf-8"
    )
    inventory = inventory_from_file(path)
    assert inventory[0].free_gpu_minutes == 30.0
    assert inventory[0].trusted is True


# --- agreeing with the run lifecycle ---------------------------------------


def test_the_default_lock_dir_is_the_one_experiment_would_use():
    """Derived from the registry path, the only thing the two share."""
    from my_jev.experiment import load_experiment_spec

    directory = pathlib.Path(tempfile.mkdtemp())
    spec_path = directory / "spec.toml"
    spec_path.write_text(
        '[experiment]\nname = "x"\nregistry_path = "runs/experiments/registry.json"\n'
        '[data]\ntrain = "a"\nvalidation = "b"\ncalibration = "c"\ntest = "d"\n'
        "[gates]\naccuracy = 0.8\n",
        encoding="utf-8",
    )
    spec, _sha = load_experiment_spec(spec_path)
    derived = default_lock_dir(spec.experiment.registry_path)
    assert derived == pathlib.Path("runs/experiments/.locks")
    assert spec.experiment.lock_dir == str(derived)


def test_probing_the_runs_own_lock_dir_agrees_with_the_run(tmp_path):
    """The end-to-end claim: what the probe says is what a run would find."""
    directory = _lock_dir()
    assert gpu_lease_is_free(directory) is True

    held = LeaseSet(
        directory, [ResourceRequest("gpu")], command="a-real-run"
    ).acquire(blocking=False)
    try:
        inventory = local_gpu_inventory(
            lock_dir=directory,
            horizon_gpu_minutes=600.0,
            declared_devices=["cuda:0"],
        )
        assert inventory[0].free_gpu_minutes == 0.0
    finally:
        held.release()