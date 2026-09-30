from __future__ import annotations

import json
from pathlib import Path

import pytest

from my_jev.scale_readiness import (
    evaluate_scale_readiness,
)


def _write_json(
    path: Path,
    payload,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            payload
        )
        + "\n",
        encoding="utf-8",
    )


def _baseline_run(
    tmp_path: Path,
) -> Path:
    root = tmp_path / "run"

    _write_json(
        root / "manifest.json",
        {
            "git": {
                "revision": "a" * 40,
                "dirty": False,
            },
            "spec_sha256": "s" * 64,
            "spec": {
                "experiment": {
                    "name": (
                        "assistx-policy-modernbert-"
                        "r9700-baseline"
                    )
                }
            },
            "datasets": {
                "train": {
                    "sha256": "1" * 64,
                },
                "validation": {
                    "sha256": "2" * 64,
                },
                "calibration": {
                    "sha256": "3" * 64,
                },
                "test": {
                    "sha256": "4" * 64,
                },
            },
            "benchmark_inputs": {
                "fleet_states": {
                    "sha256": "5" * 64,
                }
            },
            "data_audit": {
                "passed": True,
            },
        },
    )

    _write_json(
        root
        / "checkpoints"
        / "best"
        / "my_jev_config.json",
        {
            "extra": {
                "run_config": {
                    "device": "cuda",
                    "epochs": 1,
                    "batch_size": 2,
                    "grad_accum": 8,
                    "max_state_length": 2048,
                    "max_candidate_length": 192,
                    "bf16_requested": True,
                    "bf16_enabled": True,
                    "freeze_backbone": False,
                    "gradient_checkpointing": True,
                }
            }
        },
    )
    (
        root
        / "checkpoints"
        / "best"
        / "model.pt"
    ).write_bytes(
        b"checkpoint"
    )

    _write_json(
        root / "calibration.json",
        {
            "temperature": 1.1,
        },
    )
    _write_json(
        root / "benchmark.json",
        {
            "normal": {
                "accuracy": 0.78,
                "ece": 0.08,
                "nll": 0.62,
                "brier": 0.22,
            },
            "controls": {
                "accuracy_delta_vs_shuffled": 0.18,
                "accuracy_delta_vs_uniform": 0.25,
                "mean_normal_vs_shuffled_kl": 0.09,
                "choice_order_invariance": {
                    "top1_agreement": 0.995,
                    "mean_abs_probability_delta": 0.004,
                },
                "policy_consistency": {
                    "violation_rate": 0.01,
                },
            },
            "latency": {
                "decisions_per_second": 500.0,
                "batch_ms_p95": 50.0,
            },
        },
    )
    _write_json(
        root / "promotion.json",
        {
            "passed": False,
            "failed": [
                "fleet:hard_failure_defer_rate",
            ],
        },
    )
    _write_json(
        root / "fleet-benchmark.json",
        {
            "states": {
                "sha256": "5" * 64,
            },
            "promotion": {
                "passed": False,
                "failed": [
                    "hard_failure_defer_rate",
                ],
            },
        },
    )
    _write_json(
        root / "r9700-doctor.json",
        {
            "hip_version": "7.0",
            "devices": [
                {
                    "name": (
                        "AMD Radeon AI PRO R9700"
                    ),
                    "total_memory_gib": 32.0,
                }
            ],
        },
    )
    (
        root
        / "pip-freeze.txt"
    ).write_text(
        "torch==x\n",
        encoding="utf-8",
    )
    _write_json(
        root
        / "r9700-baseline-receipt.json",
        {
            "git_sha": "a" * 40,
            "spec_sha256": "s" * 64,
            "dataset_sha256": {
                "train": "1" * 64,
                "validation": "2" * 64,
                "calibration": "3" * 64,
                "test": "4" * 64,
            },
            "fleet_benchmark": {
                "sha256": "5" * 64,
            },
            "artifacts": {},
        },
    )

    return root


def test_scale_readiness_can_pass_even_when_exploratory_fleet_gate_fails(
    tmp_path: Path,
):
    root = _baseline_run(
        tmp_path
    )

    result = (
        evaluate_scale_readiness(
            root,
            expected_sha=(
                "a" * 40
            ),
        )
    )

    assert (
        result["passed"]
        is True
    )
    assert (
        result["decision"]
        == "scale"
    )
    assert (
        result[
            "fleet_observer_evidence"
        ]["blocking"]
        is False
    )
    assert (
        result[
            "fleet_observer_evidence"
        ]["passed"]
        is False
    )
    assert (
        result[
            "authority_boundary"
        ]["dispatch_allowed"]
        is False
    )


def test_scale_readiness_holds_state_insensitive_baseline(
    tmp_path: Path,
):
    root = _baseline_run(
        tmp_path
    )
    benchmark = json.loads(
        (
            root
            / "benchmark.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    benchmark[
        "controls"
    ][
        "accuracy_delta_vs_shuffled"
    ] = 0.01
    benchmark[
        "controls"
    ][
        "mean_normal_vs_shuffled_kl"
    ] = 0.001
    _write_json(
        root / "benchmark.json",
        benchmark,
    )

    result = (
        evaluate_scale_readiness(
            root
        )
    )

    assert (
        result["passed"]
        is False
    )
    assert (
        result["decision"]
        == "hold"
    )
    assert (
        "min_accuracy_delta_vs_shuffled"
        in result["failed"]
    )
    assert (
        "min_normal_vs_shuffled_kl"
        in result["failed"]
    )


def test_scale_readiness_holds_failed_data_audit(
    tmp_path: Path,
):
    root = _baseline_run(
        tmp_path
    )
    manifest = json.loads(
        (
            root
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    manifest[
        "data_audit"
    ]["passed"] = False
    _write_json(
        root / "manifest.json",
        manifest,
    )

    result = (
        evaluate_scale_readiness(
            root
        )
    )

    assert (
        result["passed"]
        is False
    )
    assert (
        "pretraining_data_audit"
        in result["failed"]
    )


def test_scale_readiness_rejects_wrong_experiment_identity(
    tmp_path: Path,
):
    root = _baseline_run(
        tmp_path
    )
    manifest = json.loads(
        (
            root
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    manifest[
        "spec"
    ][
        "experiment"
    ][
        "name"
    ] = "different-experiment"
    _write_json(
        root / "manifest.json",
        manifest,
    )

    with pytest.raises(
        ValueError,
        match="pinned R9700",
    ):
        evaluate_scale_readiness(
            root
        )


def test_scale_readiness_reports_valid_shadow_bundle(
    tmp_path: Path,
):
    root = _baseline_run(
        tmp_path
    )
    _write_json(
        root
        / "shadow-evaluation"
        / "shadow-evaluation-receipt.json",
        {
            "candidate": {
                "git_sha": "a" * 40,
            },
            "dispatch_allowed": False,
            "runtime_authority_changed": False,
        },
    )

    result = (
        evaluate_scale_readiness(
            root
        )
    )

    assert (
        result[
            "shadow_evaluation"
        ]["completed"]
        is True
    )
    assert (
        result[
            "shadow_evaluation"
        ]["same_candidate_sha"]
        is True
    )


def test_scale_readiness_holds_mismatched_shadow_candidate(
    tmp_path: Path,
):
    root = _baseline_run(
        tmp_path
    )
    _write_json(
        root
        / "shadow-evaluation"
        / "shadow-evaluation-receipt.json",
        {
            "candidate": {
                "git_sha": "b" * 40,
            },
            "dispatch_allowed": False,
            "runtime_authority_changed": False,
        },
    )

    result = (
        evaluate_scale_readiness(
            root
        )
    )

    assert (
        result["passed"]
        is False
    )
    assert (
        "shadow_same_candidate_sha"
        in result["failed"]
    )


def test_scale_readiness_holds_shadow_authority_violation(
    tmp_path: Path,
):
    root = _baseline_run(
        tmp_path
    )
    _write_json(
        root
        / "shadow-evaluation"
        / "shadow-evaluation-receipt.json",
        {
            "candidate": {
                "git_sha": "a" * 40,
            },
            "dispatch_allowed": True,
            "runtime_authority_changed": True,
        },
    )

    result = (
        evaluate_scale_readiness(
            root
        )
    )

    assert (
        result["passed"]
        is False
    )
    assert (
        "shadow_dispatch_disabled"
        in result["failed"]
    )
    assert (
        "shadow_authority_unchanged"
        in result["failed"]
    )
