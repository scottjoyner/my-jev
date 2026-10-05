from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from my_jev.experiment import (
    device_assignments_from_plan,
    stage_device_pin,
    _stage,
)


def _plan(tmp_path, assignments):
    """A minimal but valid plan document, so the reader is tested against the contract."""
    plan = {
        "schema_version": "my-jev-gpu-plan-v1",
        "authority": {
            "dispatch_allowed": False,
            "gpu_acquired": False,
            "lease_acquired": False,
            "promotion_granted": False,
            "runtime_authority_changed": False,
        },
        "advisory_only": True,
        "version_id": "plan-abc123",
        "job_sha256": "a" * 64,
        "experiment": "pinning-probe",
        "backbone": "r9700",
        "head_kind": "option_query",
        "data_lineage_sha256": "b" * 64,
        "stages": [
            {
                "kind": kind,
                "picker": picker,
                "selection": "s",
                "rationale": "r",
                "estimated_gpu_minutes": 10.0,
                "measured_gpu_minutes": None,
                "gpu_ids": devices,
            }
            for kind, picker, devices in assignments
        ],
        "budget_gpu_minutes": 1000.0,
        "total_estimated_gpu_minutes": 40.0,
        "within_budget": True,
        "fully_placed": True,
        "deferred_stages": [],
        "budget_reason": None,
        "readiness": None,
        "calibration": None,
        "train_withheld": False,
        "planned_at": "2026-10-04T12:00:00+00:00",
        "notes": [],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


# --- translating a device id ----------------------------------------------


def test_a_cuda_index_becomes_a_visible_devices_value():
    assert stage_device_pin("cuda:1") == "1"
    assert stage_device_pin("cuda:0") == "0"
    assert stage_device_pin("2") == "2"


def test_a_bare_cuda_means_the_first_device():
    assert stage_device_pin("cuda") == "0"


def test_a_device_this_host_does_not_have_is_refused():
    """Running it anyway would put the work somewhere the plan did not say."""
    with pytest.raises(ValueError, match="not visible"):
        stage_device_pin("cuda:3", detected=["cuda:0", "cuda:1"])


def test_an_unparseable_device_id_is_refused():
    with pytest.raises(ValueError, match="unsupported device id"):
        stage_device_pin("gpu-01.internal.lan")


# --- the pin reaches the process -------------------------------------------


def test_the_pin_reaches_the_subprocess(tmp_path):
    _stage(
        "train",
        [
            sys.executable,
            "-c",
            "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES'))",
        ],
        run_dir=tmp_path,
        dry_run=False,
        device="cuda:0",
    )
    assert (tmp_path / "stages" / "train.log").read_text().strip() == "0"


def test_pinning_preserves_the_rest_of_the_environment(tmp_path):
    """subprocess.run replaces the whole env when `env` is given; PATH must survive."""
    _stage(
        "train",
        [
            sys.executable,
            "-c",
            "import os; print(bool(os.environ.get('PATH')))",
        ],
        run_dir=tmp_path,
        dry_run=False,
        device="cuda:0",
    )
    assert (tmp_path / "stages" / "train.log").read_text().strip() == "True"


def test_no_device_means_no_pin(tmp_path):
    _stage(
        "benchmark",
        [
            sys.executable,
            "-c",
            "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES'))",
        ],
        run_dir=tmp_path,
        dry_run=False,
    )
    assert (tmp_path / "stages" / "benchmark.log").read_text().strip() == "None"


def test_the_pin_is_recorded_for_later_reproduction(tmp_path):
    _stage(
        "train",
        [sys.executable, "-c", "pass"],
        run_dir=tmp_path,
        dry_run=False,
        device="cuda:1",
    )
    recorded = json.loads((tmp_path / "stages" / "train.command.json").read_text())
    assert recorded["device"] == "cuda:1"
    assert recorded["CUDA_VISIBLE_DEVICES"] == "cuda:1"


def test_a_dry_run_records_the_intended_pin_without_running(tmp_path):
    _stage(
        "train",
        [sys.executable, "-c", "raise SystemExit(1)"],
        run_dir=tmp_path,
        dry_run=True,
        device="cuda:1",
    )
    recorded = json.loads((tmp_path / "stages" / "train.command.json").read_text())
    assert recorded["device"] == "cuda:1"
    assert not (tmp_path / "stages" / "train.log").exists()


# --- reading a plan --------------------------------------------------------


def test_a_plan_supplies_the_primary_device_per_stage(tmp_path):
    path = _plan(
        tmp_path,
        [
            ("train", "train_config", ["cuda:1", "cuda:0"]),
            ("calibrate", "eval_thresholds", ["cuda:0"]),
        ],
    )
    assert device_assignments_from_plan(path) == {"train": "cuda:1", "calibrate": "cuda:0"}


def test_an_unplaced_stage_is_omitted_rather_than_defaulted(tmp_path):
    """Defaulting would run work on a device the plan never chose."""
    path = _plan(
        tmp_path,
        [
            ("train", "train_config", ["cuda:1"]),
            ("benchmark", "benchmark_suite", []),
        ],
    )
    assert device_assignments_from_plan(path) == {"train": "cuda:1"}


def test_a_plan_document_is_validated_against_the_contract(tmp_path):
    bad = tmp_path / "plan.json"
    bad.write_text(json.dumps({"schema_version": "my-jev-gpu-plan-v1"}))
    with pytest.raises(ValueError):
        device_assignments_from_plan(bad)


def test_plans_agreeing_with_the_pipeline_are_accepted(tmp_path):
    """A stale plan naming a stage the pipeline does not run is the caller's check."""
    from my_jev.experiment import _fleet_benchmark_command  # noqa: F401

    path = _plan(tmp_path, [("fleet_benchmark", "benchmark_suite", ["cuda:0"])])
    assert device_assignments_from_plan(path) == {"fleet_benchmark": "cuda:0"}
# --- the plan and the pipeline must agree ----------------------------------


def _datasets(tmp_path):
    """Real datasets, because a dry run audits them rather than trusting the spec."""
    from my_jev.agentic_synth import generate_agent_policy_records
    from my_jev.data import dump_jsonl
    from my_jev.split import split_records

    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    splits, _ = split_records(
        generate_agent_policy_records(60, seed=11),
        train_fraction=0.7,
        validation_fraction=0.1,
        calibration_fraction=0.1,
        seed=11,
        group_key="auto",
    )
    for name, records in splits.items():
        dump_jsonl(iter(records), data / f"{name}.jsonl")
    return data


def _spec(tmp_path):
    data = _datasets(tmp_path)
    spec = tmp_path / "experiment.toml"
    spec.write_text(
        "[experiment]\n"
        'name = "pin-probe"\n'
        'output_root = "runs"\n'
        'registry_path = "runs/registry.json"\n'
        'lock_dir = "runs/.locks"\n'
        "\n"
        "[model]\n"
        'backend = "encoder_option_query"\n'
        'backbone = "answerdotai/ModernBERT-base"\n'
        'head_kind = "option_query"\n'
        "head_rank = 128\n"
        "\n"
        "[data]\n"
        f'train = "{data}/train.jsonl"\n'
        f'validation = "{data}/validation.jsonl"\n'
        f'calibration = "{data}/calibration.jsonl"\n'
        f'test = "{data}/test.jsonl"\n'
        "\n"
        "[gates]\n"
        "accuracy = 0.8\n",
        encoding="utf-8",
    )
    return spec


def _recorded(run_dir, stage):
    return json.loads(
        (Path(run_dir) / "stages" / f"{stage}.command.json").read_text()
    )


def test_a_plan_naming_a_stage_the_pipeline_never_runs_is_refused(tmp_path):
    """Silently ignoring it would leave the operator believing it applied.

    A plan assigning fleet_benchmark to a job with no fleet_states configured is
    stale, because experiment.py will not add that command.
    """
    from my_jev.experiment import run_experiment

    path = _plan(tmp_path, [("fleet_benchmark", "benchmark_suite", ["cuda:0"])])

    with pytest.raises(ValueError, match="does not run"):
        run_experiment(_spec(tmp_path), dry_run=True, gpu_plan=path)


def test_a_plan_matching_the_pipeline_pins_the_dry_run(tmp_path):
    from my_jev.experiment import run_experiment

    path = _plan(tmp_path, [("train", "train_config", ["cuda:0"])])
    result = run_experiment(_spec(tmp_path), dry_run=True, gpu_plan=path)

    assert result["dry_run"] is True
    assert _recorded(result["run_dir"], "train")["device"] == "cuda:0"


def test_an_unplaced_stage_runs_unpinned_rather_than_defaulted(tmp_path):
    from my_jev.experiment import run_experiment

    path = _plan(
        tmp_path,
        [
            ("train", "train_config", ["cuda:0"]),
            ("benchmark", "benchmark_suite", []),
        ],
    )
    result = run_experiment(_spec(tmp_path), dry_run=True, gpu_plan=path)

    assert _recorded(result["run_dir"], "train")["device"] == "cuda:0"
    # Omitted rather than guessed at: defaulting would put work on a device the
    # plan never chose, which is the failure pinning exists to prevent.
    assert _recorded(result["run_dir"], "benchmark")["device"] is None


def test_an_explicit_assignment_overrides_the_plan(tmp_path):
    """So an operator can correct a plan without editing or regenerating it."""
    from my_jev.experiment import run_experiment

    path = _plan(tmp_path, [("train", "train_config", ["cuda:0"])])
    result = run_experiment(
        _spec(tmp_path),
        dry_run=True,
        gpu_plan=path,
        device_assignments={"train": "cuda:1"},
    )
    assert _recorded(result["run_dir"], "train")["device"] == "cuda:1"


def test_an_unknown_explicit_stage_is_refused(tmp_path):
    from my_jev.experiment import run_experiment

    with pytest.raises(ValueError, match="does not run"):
        run_experiment(
            _spec(tmp_path),
            dry_run=True,
            device_assignments={"not_a_stage": "cuda:0"},
        )