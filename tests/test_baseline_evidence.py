import json

import pytest

from my_jev.baseline_evidence import validate_baseline_run


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _run(tmp_path):
    root = tmp_path / "run"
    _write_json(
        root / "manifest.json",
        {"git": {"revision": "a" * 40, "dirty": False}},
    )
    _write_json(
        root / "checkpoints/best/my_jev_config.json",
        {
            "extra": {
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
        },
    )
    (root / "checkpoints/best/model.pt").write_bytes(b"checkpoint")
    _write_json(root / "calibration.json", {"temperature": 1.1})
    _write_json(root / "benchmark.json", {"metrics": {"accuracy": 0.7}})
    _write_json(root / "promotion.json", {"status": "rejected"})
    _write_json(
        root / "r9700-doctor.json",
        {
            "hip_version": "6.x",
            "devices": [
                {"name": "AMD Radeon AI PRO R9700", "memory_gib": 32}
            ],
        },
    )
    (root / "pip-freeze.txt").write_text("torch==x\n", encoding="utf-8")
    _write_json(
        root / "r9700-baseline-receipt.json",
        {"git_sha": "a" * 40, "artifacts": {}},
    )
    return root


def test_validator_accepts_rejected_baseline_as_evidence(tmp_path):
    root = _run(tmp_path)

    report = validate_baseline_run(
        root,
        expected_sha="a" * 40,
    )

    assert report["evidence_only"] is True
    assert report["runtime_authority_changed"] is False
    assert report["promotion_status"] == "rejected"
    assert report["git_sha"] == "a" * 40
    assert report["r9700_devices"][0]["name"].endswith("R9700")
    assert "checkpoints/best/model.pt" in report["artifacts"]


def test_validator_rejects_wrong_exact_sha(tmp_path):
    root = _run(tmp_path)

    with pytest.raises(ValueError, match="SHA mismatch"):
        validate_baseline_run(
            root,
            expected_sha="b" * 40,
        )


def test_validator_rejects_dirty_manifest(tmp_path):
    root = _run(tmp_path)
    _write_json(
        root / "manifest.json",
        {"git": {"revision": "a" * 40, "dirty": True}},
    )

    with pytest.raises(ValueError, match="dirty"):
        validate_baseline_run(root)


def test_validator_rejects_non_r9700_doctor_evidence(tmp_path):
    root = _run(tmp_path)
    _write_json(
        root / "r9700-doctor.json",
        {
            "hip_version": "6.x",
            "devices": [{"name": "Other GPU"}],
        },
    )

    with pytest.raises(ValueError, match="R9700"):
        validate_baseline_run(root)
