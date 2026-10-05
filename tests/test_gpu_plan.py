from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from my_jev.gpu_plan import (
    STAGE_CHAIN,
    Calibration,
    GpuCandidate,
    ModelVersionPlan,
    PickerKind,
    PlanAuthority,
    PlannedStage,
    ReadinessVerdict,
    StageKind,
    build_model_version_plan,
    estimate_gpu_minutes,
    plan_from_experiment_spec,
    readiness_from_run,
    version_identity,
)

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


def job(**overrides):
    payload = {
        "experiment": {
            "name": "scale-probe",
            "parent": "latest_promoted",
            "backend": "encoder_option_query",
            "backbone": "r9700",
            "head_kind": "option_query",
            "head_rank": 256,
        },
        "model": {
            "backend": "encoder_option_query",
            "backbone": "r9700",
            "head_kind": "option_query",
            "head_rank": 256,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": "all-linear",
        },
        "data": {
            "train": "data/train",
            "validation": "data/val",
            "calibration": "data/cal",
            "test": "data/test",
        },
        "train": {
            "epochs": 3,
            "batch_size": 2,
            "grad_accum": 8,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "max_state_length": 2048,
            "max_candidate_length": 192,
            "max_sequence_length": 1024,
            "seed": 17,
            "freeze_backbone": False,
            "gradient_checkpointing": True,
        },
        "benchmark": {"batch_size": 8, "fleet_states": "states.json"},
        "gates": {"accuracy": 0.8, "ece": 0.15},
    }
    payload.update(overrides)
    return payload


def pool(*minutes):
    return [GpuCandidate(f"gpu-{index}", value) for index, value in enumerate(minutes)]


def plan(**kwargs):
    kwargs.setdefault("pool", pool(4000.0))
    kwargs.setdefault("budget_gpu_minutes", 10_000.0)
    kwargs.setdefault("planned_at", NOW)
    return build_model_version_plan(job(), **kwargs)


# --- version identity ------------------------------------------------------


def test_version_identity_is_stable_across_key_order():
    reordered = {key: job()[key] for key in reversed(list(job()))}
    assert version_identity(job()) == version_identity(reordered)


def test_version_identity_ignores_bookkeeping_fields():
    """Output paths and registry location do not change what gets trained."""
    base = version_identity(job())["version_id"]
    moved = job()
    moved["experiment"]["output_root"] = "/somewhere/else"
    moved["experiment"]["registry_path"] = "/tmp/registry.json"
    assert version_identity(moved)["version_id"] == base


def test_changing_the_training_config_mints_a_new_version():
    base = version_identity(job())["version_id"]
    changed = job()
    changed["train"]["learning_rate"] = 5e-5
    assert version_identity(changed)["version_id"] != base


def test_changing_only_the_dataset_does_not_collide_on_version_id():
    base = version_identity(job())["version_id"]
    swapped = job()
    swapped["data"]["test"] = "data/holdout-v2"
    identity = version_identity(swapped)
    assert identity["version_id"] != base
    assert identity["data_lineage_sha256"] != version_identity(job())["data_lineage_sha256"]


def test_version_id_is_derived_not_invented():
    identity = version_identity(job())
    assert identity["version_id"].startswith("plan-")
    assert identity["job_sha256"].startswith(identity["version_id"].removeprefix("plan-"))


# --- the chain -------------------------------------------------------------


def test_a_fundable_plan_schedules_the_whole_chain_in_dependency_order():
    result = plan()
    assert result.within_budget is True
    assert [stage.kind for stage in result.stages] == list(STAGE_CHAIN)
    assert result.deferred_stages == []


def test_an_unfundable_chain_places_nothing():
    """A partial chain produces no evidence about the version.

    The stages are still listed, unplaced, so the cost breakdown that explains
    the shortfall stays inspectable rather than raising IndexError.
    """
    result = plan(budget_gpu_minutes=10.0)
    assert result.within_budget is False
    assert [stage.kind for stage in result.stages] == list(STAGE_CHAIN)
    assert all(stage.gpu_ids == [] for stage in result.stages)
    assert result.deferred_stages == list(STAGE_CHAIN)
    assert "partial chain" in result.budget_reason


def test_budget_reason_names_the_shortfall():
    result = plan(budget_gpu_minutes=10.0)
    assert "10.0" in result.budget_reason
    assert "892.8" in result.budget_reason


def test_a_zero_budget_funds_nothing():
    result = plan(budget_gpu_minutes=0.0)
    assert result.within_budget is False
    assert result.deferred_stages == list(STAGE_CHAIN)


def test_every_stage_records_which_picker_resolved_it():
    expected = {
        StageKind.train: PickerKind.train_config,
        StageKind.benchmark: PickerKind.benchmark_suite,
        StageKind.evaluate: PickerKind.eval_thresholds,
    }
    for stage in plan().stages:
        assert stage.picker is expected[stage.kind]
        assert stage.selection
        assert stage.rationale


def test_the_train_picker_reflects_the_actual_config():
    stage = plan().stages[0]
    assert "lr=2e-05" in stage.selection
    assert "effective_batch=16" in stage.selection
    assert "seed=17" in stage.selection


def test_the_eval_picker_lists_the_gates_it_will_compare_against():
    stage = plan().stages[2]
    assert stage.selection == "gates=accuracy,ece"


# --- readiness -------------------------------------------------------------


def test_readiness_that_did_not_pass_withholds_all_gpu():
    result = plan(
        readiness=ReadinessVerdict(
            completed=False,
            passed=False,
            reason="baseline run is incomplete; missing: manifest.json",
        )
    )
    assert result.within_budget is False
    assert result.train_withheld is True
    assert result.deferred_stages == list(STAGE_CHAIN)


def test_a_completed_but_failed_readiness_also_withholds():
    result = plan(readiness=ReadinessVerdict(completed=True, passed=False, reason="ece 0.22"))
    assert result.train_withheld is True
    assert result.deferred_stages == list(STAGE_CHAIN)


def test_a_passing_readiness_lets_the_plan_through():
    result = plan(readiness=ReadinessVerdict(completed=True, passed=True))
    assert result.train_withheld is False
    assert result.within_budget is True


def test_readiness_is_consulted_before_budget_arithmetic():
    """Readiness gates first, so a version that is not worth GPU never gets planned."""
    result = plan(budget_gpu_minutes=10_000.0, readiness=ReadinessVerdict(completed=True))
    assert result.train_withheld is True
    assert "readiness" in result.budget_reason.lower()


def test_planning_without_a_verdict_says_so():
    result = plan()
    assert result.readiness is None
    assert any("no readiness verdict" in note for note in result.notes)


def test_readiness_from_run_maps_an_incomplete_baseline_to_not_ready(tmp_path):
    """evaluate_scale_readiness raises here; a planner must not crash on it."""
    verdict = readiness_from_run(str(tmp_path))
    assert verdict.completed is False
    assert verdict.passed is False
    assert verdict.reason


# --- measured minutes ------------------------------------------------------


def test_measured_minutes_replace_the_estimate_for_budgeting():
    result = plan(budget_gpu_minutes=500.0, measured={StageKind.train: 90.0})
    assert result.stages[0].minutes_used_for_budget == 90.0
    assert result.stages[0].calibrated is True


def test_replanning_does_not_keep_budgeting_a_stage_that_already_ran():
    """Otherwise repeated planning inflates the total with no new work justified."""
    estimated = plan()
    recalibrated = plan(measured={StageKind.train: 90.0})
    assert recalibrated.total_estimated_gpu_minutes < estimated.total_estimated_gpu_minutes


def test_an_uncalibrated_stage_is_flagged_in_the_notes():
    result = plan(measured={StageKind.train: 90.0})
    flagged = {note.split()[0] for note in result.notes if "not measured" in note}
    assert flagged == {"benchmark", "evaluate"}


def test_estimate_scales_with_the_shape_of_the_job():
    bigger = job()
    bigger["train"]["epochs"] = 6
    assert estimate_gpu_minutes(StageKind.train, bigger) == pytest.approx(
        2 * estimate_gpu_minutes(StageKind.train, job())
    )


def test_longer_sequences_cost_more():
    longer = job()
    longer["train"]["max_state_length"] = 4096
    assert estimate_gpu_minutes(StageKind.train, longer) > estimate_gpu_minutes(
        StageKind.train, job()
    )


def test_estimates_are_never_negative_for_a_sparse_job():
    sparse = {"train": {}, "model": {}}
    for kind in STAGE_CHAIN:
        assert estimate_gpu_minutes(kind, sparse) > 0.0


# --- gpu placement ---------------------------------------------------------


def test_gpus_are_placed_largest_free_first_with_id_as_tiebreak():
    result = plan(pool=[GpuCandidate("gpu-b", 100.0), GpuCandidate("gpu-a", 100.0)])
    assert result.stages[0].gpu_ids[0] == "gpu-a"


def test_untrusted_devices_are_never_allocated_even_with_free_time():
    result = plan(
        pool=[GpuCandidate("gpu-fast", 5000.0, trusted=False), GpuCandidate("gpu-ok", 100.0)]
    )
    assert "gpu-fast" not in result.gpu_ids_used
    assert result.gpu_ids_used == ["gpu-ok"]


def test_placement_can_span_devices():
    result = plan(pool=pool(10.0, 10.0, 10.0, 10.0))
    assert len(result.gpu_ids_used) >= 1


def test_a_device_with_no_free_time_is_skipped():
    result = plan(pool=[GpuCandidate("idle-0", 0.0), GpuCandidate("busy-1", 5000.0)])
    assert "idle-0" not in result.gpu_ids_used


def test_an_inventory_smaller_than_the_plan_is_called_out():
    result = plan(pool=pool(50.0), budget_gpu_minutes=10_000.0)
    assert result.within_budget is True
    assert any("only reports" in note for note in result.notes)


def test_a_negative_inventory_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        GpuCandidate("gpu-0", -1.0)


def test_an_empty_gpu_id_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        GpuCandidate("  ", 10.0)


# --- determinism -----------------------------------------------------------


def test_the_same_inputs_plan_identically():
    assert plan().model_dump(mode="json") == plan().model_dump(mode="json")


def test_pool_order_does_not_change_the_plan():
    forwards = plan(pool=[GpuCandidate("gpu-a", 100.0), GpuCandidate("gpu-b", 900.0)])
    backwards = plan(pool=[GpuCandidate("gpu-b", 900.0), GpuCandidate("gpu-a", 100.0)])
    assert forwards.gpu_ids_used == backwards.gpu_ids_used
    assert forwards.model_dump(mode="json") == backwards.model_dump(mode="json")


def test_gate_order_does_not_change_the_eval_selection():
    forwards = job()
    forwards["gates"] = {"a": 1.0, "b": 2.0}
    backwards = job()
    backwards["gates"] = {"b": 2.0, "a": 1.0}
    kwargs = {"pool": pool(4000.0), "budget_gpu_minutes": 10_000.0, "planned_at": NOW}
    assert (
        build_model_version_plan(forwards, **kwargs).stages[2].selection
        == build_model_version_plan(backwards, **kwargs).stages[2].selection
    )


def test_the_plan_round_trips_through_json():
    result = plan()
    assert ModelVersionPlan.model_validate_json(result.model_dump_json()).version_id == (
        result.version_id
    )


# --- authority -------------------------------------------------------------


def test_a_plan_grants_no_authority():
    authority = plan().authority
    assert authority.model_dump() == {
        "dispatch_allowed": False,
        "gpu_acquired": False,
        "lease_acquired": False,
        "promotion_granted": False,
        "runtime_authority_changed": False,
    }


@pytest.mark.parametrize("field", list(PlanAuthority.model_fields))
def test_a_cannot_supply_true_for_any_authority_field(field):
    payload = plan().authority.model_dump()
    payload[field] = True
    with pytest.raises(ValueError):
        PlanAuthority.model_validate(payload)


def test_the_plan_is_advisory_only():
    assert plan().advisory_only is True


# --- contract --------------------------------------------------------------


def test_unknown_fields_are_refused():
    payload = plan().model_dump()
    payload["gpu_ids"] = ["gpu-0"]
    with pytest.raises(ValueError):
        ModelVersionPlan.model_validate(payload)


def test_a_stage_refuses_an_unknown_kind():
    payload = PlannedStage(
        kind=StageKind.train,
        picker=PickerKind.train_config,
        selection="x",
        rationale="y",
        estimated_gpu_minutes=1.0,
    ).model_dump()
    payload["kind"] = "deploy"
    with pytest.raises(ValueError):
        PlannedStage.model_validate(payload)


def test_a_negative_estimate_is_refused():
    with pytest.raises(ValueError):
        PlannedStage(
            kind=StageKind.train,
            picker=PickerKind.train_config,
            selection="x",
            rationale="y",
            estimated_gpu_minutes=-1.0,
        )


# --- integration -----------------------------------------------------------


def test_plan_from_experiment_spec_matches_the_raw_mapping():
    from my_jev.experiment import ExperimentSpec

    spec = ExperimentSpec.model_validate(job())
    assert (
        plan_from_experiment_spec(
            spec, pool=pool(4000.0), budget_gpu_minutes=10_000.0, planned_at=NOW
        ).model_dump(mode="json")
        == plan().model_dump(mode="json")
    )


def test_the_plan_serialises_for_an_operator_report():
    document = json.loads(plan().model_dump_json())
    assert document["schema_version"] == "my-jev-gpu-plan-v1"
    assert [stage["kind"] for stage in document["stages"]] == [
        "train",
        "benchmark",
        "evaluate",
    ]
    assert document["authority"]["gpu_acquired"] is False


def test_planned_at_defaults_to_now_and_is_second_resolution():
    result = build_model_version_plan(job(), pool=pool(10.0), budget_gpu_minutes=10_000.0)
    parsed = datetime.fromisoformat(result.planned_at)
    assert parsed.tzinfo is not None
    assert parsed.microsecond == 0
    assert abs(parsed - datetime.now(UTC)) < timedelta(minutes=5)


# --- CLI -------------------------------------------------------------------


def _write_spec(tmp_path, **gates):
    spec = tmp_path / "spec.toml"
    spec.write_text(
        "\n".join(
            [
                "[experiment]",
                'name = "cli-probe"',
                'parent = "latest_promoted"',
                'backend = "encoder_option_query"',
                'backbone = "r9700"',
                'head_kind = "option_query"',
                "head_rank = 256",
                "",
                "[data]",
                'train = "data/train"',
                'validation = "data/val"',
                'calibration = "data/cal"',
                'test = "data/test"',
                "",
                "[train]",
                "epochs = 3",
                "batch_size = 2",
                "grad_accum = 8",
                "learning_rate = 0.00002",
                "max_state_length = 2048",
                "seed = 17",
                "",
                "[gates]",
                *[f"{key} = 0.8" for key in (gates or ("accuracy",))],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return spec


def _write_inventory(tmp_path, gpus):
    path = tmp_path / "inv.json"
    path.write_text(json.dumps({"gpus": gpus}), encoding="utf-8")
    return path


def test_cli_prints_a_fundable_plan_and_exits_zero(tmp_path, capsys):
    from my_jev.gpu_plan_cli import main

    code = main(
        [
            str(_write_spec(tmp_path)),
            "--budget-gpu-minutes",
            "10000",
            "--inventory",
            str(_write_inventory(tmp_path, [{"gpu_id": "gpu-0", "free_gpu_minutes": 4000}])),
        ]
    )
    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["within_budget"] is True
    assert document["stages"][0]["gpu_ids"] == ["gpu-0"]


def test_cli_exits_two_when_nothing_is_fundable(tmp_path, capsys):
    """A shell can gate on the exit code without parsing the document."""
    from my_jev.gpu_plan_cli import main

    code = main(
        [
            str(_write_spec(tmp_path)),
            "--budget-gpu-minutes",
            "1",
            "--inventory",
            str(_write_inventory(tmp_path, [{"gpu_id": "gpu-0", "free_gpu_minutes": 4000}])),
        ]
    )
    assert code == 2
    document = json.loads(capsys.readouterr().out)
    assert document["within_budget"] is False
    assert [stage["gpu_ids"] for stage in document["stages"]] == [[], [], []]


def test_cli_writes_to_output_without_printing(tmp_path, capsys):
    from my_jev.gpu_plan_cli import main

    out = tmp_path / "plan.json"
    code = main(
        [
            str(_write_spec(tmp_path)),
            "--budget-gpu-minutes",
            "10000",
            "--inventory",
            str(_write_inventory(tmp_path, [{"gpu_id": "gpu-0", "free_gpu_minutes": 4000}])),
            "--output",
            str(out),
        ]
    )
    assert code == 0
    assert capsys.readouterr().out == ""
    assert json.loads(out.read_text())["within_budget"] is True


def test_cli_readiness_gate_withholds_gpu(tmp_path, capsys):
    from my_jev.gpu_plan_cli import main

    code = main(
        [
            str(_write_spec(tmp_path)),
            "--budget-gpu-minutes",
            "99999",
            "--readiness-run-dir",
            str(tmp_path / "no-such-run"),
        ]
    )
    assert code == 2
    document = json.loads(capsys.readouterr().out)
    assert document["train_withheld"] is True
    assert document["readiness"]["passed"] is False


def test_cli_asserted_readiness_is_labelled_as_an_assertion(tmp_path, capsys):
    from my_jev.gpu_plan_cli import main

    code = main(
        [
            str(_write_spec(tmp_path)),
            "--budget-gpu-minutes",
            "10000",
            "--readiness-passed",
            "--inventory",
            str(
                _write_inventory(
                    tmp_path, [{"gpu_id": "gpu-0", "free_gpu_minutes": 4000}]
                )
            ),
        ]
    )
    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["readiness"]["source"] == "operator-assertion"
    assert document["train_withheld"] is False


def test_cli_measured_minutes_override_the_estimate(tmp_path, capsys):
    from my_jev.gpu_plan_cli import main

    measured = tmp_path / "measured.json"
    measured.write_text(json.dumps({"train": 60.0, "benchmark": 30.0}), encoding="utf-8")
    code = main(
        [
            str(_write_spec(tmp_path)),
            "--budget-gpu-minutes",
            "10000",
            "--measured",
            str(measured),
            "--inventory",
            str(
                _write_inventory(
                    tmp_path, [{"gpu_id": "gpu-0", "free_gpu_minutes": 4000}]
                )
            ),
        ]
    )
    assert code == 0
    document = json.loads(capsys.readouterr().out)
    train = document["stages"][0]
    assert train["estimated_gpu_minutes"] > train["measured_gpu_minutes"]
    assert train["measured_gpu_minutes"] == 60.0


def test_cli_ignores_unknown_measured_stages(tmp_path, capsys):
    from my_jev.gpu_plan_cli import main

    measured = tmp_path / "measured.json"
    measured.write_text(json.dumps({"deploy": 1.0, "train": 60.0}), encoding="utf-8")
    code = main(
        [
            str(_write_spec(tmp_path)),
            "--budget-gpu-minutes",
            "10000",
            "--measured",
            str(measured),
            "--inventory",
            str(
                _write_inventory(
                    tmp_path, [{"gpu_id": "gpu-0", "free_gpu_minutes": 4000}]
                )
            ),
        ]
    )
    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert all(stage["measured_gpu_minutes"] in (None, 60.0) for stage in document["stages"])


def test_cli_untrusted_gpu_is_skipped(tmp_path, capsys):
    from my_jev.gpu_plan_cli import main

    code = main(
        [
            str(_write_spec(tmp_path)),
            "--budget-gpu-minutes",
            "10000",
            "--inventory",
            str(
                _write_inventory(
                    tmp_path,
                    [
                        {"gpu_id": "gpu-fast", "free_gpu_minutes": 9000, "trusted": False},
                        {"gpu_id": "gpu-ok", "free_gpu_minutes": 9000},
                    ],
                )
            ),
        ]
    )
    assert code == 0
    document = json.loads(capsys.readouterr().out)
    used = {gpu for stage in document["stages"] for gpu in stage["gpu_ids"]}
    assert used == {"gpu-ok"}


def test_cli_rejects_a_spec_missing_promotion_gates(tmp_path):
    """The planner does not relax the spec contract to be more convenient."""
    from my_jev.gpu_plan_cli import main

    spec = tmp_path / "bad.toml"
    spec.write_text(
        '[experiment]\nname = "x"\n\n[data]\ntrain = "a"\nvalidation = "b"\n'
        'calibration = "c"\ntest = "d"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        main([str(spec), "--budget-gpu-minutes", "10000"])


def test_budget_and_placement_are_orthogonal_questions():
    """A plan can fit its budget and still have nowhere to run."""
    result = plan(pool=pool(0.0))
    assert result.within_budget is True
    assert result.fully_placed is False
    assert all(stage.gpu_ids == [] for stage in result.stages)


def test_an_unplaceable_plan_says_why_it_is_not_actionable():
    result = plan(pool=pool(0.0))
    assert any("no GPU has free time" in note for note in result.notes)


def test_a_fully_funded_and_placed_plan_is_both():
    result = plan()
    assert result.within_budget is True
    assert result.fully_placed is True


def test_multi_device_placement_warns_that_the_lease_contends():
    """The GPU lease is one exclusive resource, not one per card."""
    result = plan(pool=pool(100.0, 100.0, 100.0, 100.0))
    assert len(result.gpu_ids_used) >= 1
    assert any("contends rather than running concurrently" in note for note in result.notes)


def test_single_device_placement_does_not_warn_about_contention():
    result = plan(pool=[GpuCandidate("gpu-solo", 10_000.0)])
    assert not any("contends" in note for note in result.notes)


# --- calibration against completed runs ------------------------------------

def _registry(directory, entries):
    from dataclasses import asdict

    from my_jev.registry import ExperimentEntry

    payload = {}
    for run_id, status, created, updated in entries:
        payload[run_id] = asdict(
            ExperimentEntry(
                run_id=run_id,
                experiment="cal",
                created_at=created,
                updated_at=updated,
                status=status,
                spec_path="spec.toml",
                spec_sha256="a" * 64,
                model_backend="encoder_option_query",
                backbone="r9700",
                train_sha256="b" * 64,
                validation_sha256="c" * 64,
                calibration_sha256="d" * 64,
                test_sha256="e" * 64,
                run_dir="runs/x",
            )
        )
    path = directory / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_calibration_uses_a_run_that_completed_the_chain(tmp_path):
    from my_jev.gpu_plan import calibration_from_registry

    path = _registry(
        tmp_path,
        [
            ("good", "candidate", 1_000.0, 1_000.0 + 3_600.0),  # 60 min
            ("bad", "failed", 1_000.0, 1_000.0 + 60.0),  # 1 min, excluded
            ("never", "dry_run", 1_000.0, 1_000.0),  # excluded
        ],
    )
    calibration = calibration_from_registry(path, planned_job=job())

    assert calibration is not None
    assert calibration.samples == 1
    assert calibration.measured_total_gpu_minutes == 60.0
    assert calibration.source_run_ids == ["good"]


def test_a_run_that_died_early_never_lowers_the_calibration(tmp_path):
    """Budgeting from a run that failed in training would understate every stage."""
    from my_jev.gpu_plan import calibration_from_registry

    path = _registry(
        tmp_path,
        [
            ("good", "candidate", 0.0, 6_000.0),  # 100 min
            ("bad", "failed", 0.0, 30.0),  # 0.5 min
        ],
    )
    calibration = calibration_from_registry(path, planned_job=job())
    assert calibration.measured_total_gpu_minutes == 100.0


def test_exclusions_are_named_with_a_reason(tmp_path):
    from my_jev.gpu_plan import calibration_from_registry

    path = _registry(
        tmp_path,
        [
            ("good", "rejected", 0.0, 6_000.0),
            ("bad", "failed", 0.0, 30.0),
            ("never", "dry_run", 0.0, 0.0),
        ],
    )
    excluded = {
        record["run_id"]: record["reason"]
        for record in calibration_from_registry(path, planned_job=job()).excluded
    }
    assert excluded == {
        "bad": "stopped before completing the chain",
        "never": "never started",
    }


def test_a_zero_length_run_is_excluded_as_a_clock_artefact(tmp_path):
    from my_jev.gpu_plan import calibration_from_registry

    path = _registry(tmp_path, [("instant", "candidate", 5_000.0, 5_000.0)])
    calibration = calibration_from_registry(path, planned_job=job())
    assert calibration is None or "instant" not in calibration.source_run_ids


def test_no_completed_run_yields_no_calibration(tmp_path):
    from my_jev.gpu_plan import calibration_from_registry

    path = _registry(tmp_path, [("bad", "failed", 0.0, 30.0)])
    assert calibration_from_registry(path, planned_job=job()) is None


def test_a_missing_registry_yields_no_calibration(tmp_path):
    from my_jev.gpu_plan import calibration_from_registry

    assert calibration_from_registry(tmp_path / "absent.json", planned_job=job()) is None


def test_calibration_rescales_every_stage_onto_the_measured_total(tmp_path):
    from my_jev.gpu_plan import calibration_from_registry

    path = _registry(tmp_path, [("good", "candidate", 0.0, 6_000.0)])
    calibration = calibration_from_registry(path, planned_job=job())
    result = plan(calibration=calibration)

    assert result.total_estimated_gpu_minutes == pytest.approx(100.0)
    assert result.calibration is not None
    assert result.calibration.per_stage_is_inferred is True


def test_the_inference_is_stated_in_the_notes(tmp_path):
    from my_jev.gpu_plan import calibration_from_registry

    path = _registry(tmp_path, [("good", "candidate", 0.0, 6_000.0)])
    calibration = calibration_from_registry(path, planned_job=job())
    notes = " ".join(plan(calibration=calibration).notes)
    assert "remain inferred" in notes
    assert "no per-stage timestamps" in notes


def test_a_calibration_can_turn_an_unaffordable_plan_into_an_affordable_one(tmp_path):
    """The point of calibrating: the heuristic was 9x off."""
    from my_jev.gpu_plan import calibration_from_registry

    path = _registry(tmp_path, [("good", "candidate", 0.0, 6_000.0)])
    calibration = calibration_from_registry(path, planned_job=job())

    assert plan(budget_gpu_minutes=500.0).within_budget is False
    assert plan(budget_gpu_minutes=500.0, calibration=calibration).within_budget is True


def test_per_stage_inference_cannot_be_switched_off():
    """It is always true, so a consumer cannot read these as measurements."""
    payload = Calibration(
        samples=1,
        measured_total_gpu_minutes=1.0,
        estimated_total_gpu_minutes=1.0,
    ).model_dump()
    payload["per_stage_is_inferred"] = False
    with pytest.raises(ValueError):
        Calibration.model_validate(payload)


def test_a_calibration_needs_at_least_one_sample():
    with pytest.raises(ValueError):
        Calibration(
            samples=0,
            measured_total_gpu_minutes=1.0,
            estimated_total_gpu_minutes=1.0,
        )


def test_several_runs_are_averaged(tmp_path):
    from my_jev.gpu_plan import calibration_from_registry

    path = _registry(
        tmp_path,
        [
            ("a", "candidate", 0.0, 6_000.0),  # 100
            ("b", "candidate", 0.0, 3_000.0),  # 50
        ],
    )
    calibration = calibration_from_registry(path, planned_job=job())
    assert calibration.samples == 2
    assert calibration.measured_total_gpu_minutes == 75.0


def test_cli_calibrates_from_a_registry(tmp_path, capsys):
    from my_jev.gpu_plan_cli import main

    code = main(
        [
            str(_write_spec(tmp_path)),
            "--budget-gpu-minutes",
            "10000",
            "--inventory",
            str(
                _write_inventory(
                    tmp_path, [{"gpu_id": "gpu-0", "free_gpu_minutes": 4000}]
                )
            ),
            "--calibration-registry",
            str(_registry(tmp_path, [("good", "candidate", 0.0, 6_000.0)])),
        ]
    )
    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["calibration"]["samples"] == 1
    assert document["calibration"]["per_stage_is_inferred"] is True
    assert any("calibrated against" in note for note in document["notes"])


def test_cli_without_a_registry_reports_no_calibration(tmp_path, capsys):
    from my_jev.gpu_plan_cli import main

    code = main(
        [
            str(_write_spec(tmp_path)),
            "--budget-gpu-minutes",
            "10000",
            "--inventory",
            str(
                _write_inventory(
                    tmp_path, [{"gpu_id": "gpu-0", "free_gpu_minutes": 4000}]
                )
            ),
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["calibration"] is None
