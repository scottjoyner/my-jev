from __future__ import annotations

import json

import pytest

from my_jev.assistx_policy_bundle import (
    BUNDLE_SCHEMA,
    RECORD_SCHEMA,
    canonical_sha256,
    import_assistx_policy_bundle,
    validate_assistx_policy_bundle,
)
from my_jev.data import load_jsonl


PRODUCER_SHA = "5caad17f0ab2e83194f944e953db3f9397e17dbd"
AUTHORITY = {
    "dispatch_allowed": False,
    "approval_granted": False,
    "claim_acquired": False,
    "mutation_allowed": False,
    "routing_authority_changed": False,
}


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _record(record_id, group_id, split, *, context_tokens=32768):
    row = {
        "schema": RECORD_SCHEMA,
        "record_id": record_id,
        "group_id": group_id,
        "split": split,
        "state": json.dumps(
            {
                "task_family": "coding",
                "context_tokens": context_tokens,
                "messages": [{"role": "user", "content": "Fix the bug."}],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "questions": {
            "execution_policy": {
                "type": "choice",
                "instructions": "Choose the best observed execution policy.",
                "options": ["exec-a", "exec-b"],
                "metadata": {
                    "descriptors": {
                        "exec-a": {"node_id": "x1-370"},
                        "exec-b": {"node_id": "r9700"},
                    }
                },
            }
        },
        "targets": {
            "execution_policy": {
                "distribution": [0.0, 1.0],
            }
        },
        "metadata": {
            "evidence_only": True,
            "dispatch_allowed": False,
            "routing_authority_changed": False,
            "case_id": group_id,
            "case_sha256": "a" * 64,
            "task_family": "coding",
            "context_tokens": context_tokens,
            "authority": dict(AUTHORITY),
        },
    }
    row["record_sha256"] = canonical_sha256(row)
    return row


def _bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    records = [
        _record("r1", "case-a", "train"),
        _record("r2", "case-a", "train", context_tokens=131072),
        _record("r3", "case-b", "validation"),
        _record("r4", "case-c", "calibration"),
        _record("r5", "case-d", "test"),
    ]
    _write_jsonl(root / "records.jsonl", records)
    for split in ("train", "validation", "calibration", "test"):
        _write_jsonl(
            root / f"{split}.jsonl",
            [row for row in records if row["split"] == split],
        )

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "producer": {
            "repository": "scottjoyner/auto-assist",
            "git_sha": PRODUCER_SHA,
        },
        "record_schema": RECORD_SCHEMA,
        "split": {
            "schema": "sha256-ranked-group-v1",
            "seed": "assistx-policy-training-v1",
            "fractions": {
                "train": 0.625,
                "validation": 0.125,
                "calibration": 0.125,
                "test": 0.125,
            },
            "group_count": 4,
            "record_counts": {
                "train": 2,
                "validation": 1,
                "calibration": 1,
                "test": 1,
            },
            "group_assignments": {
                "case-a": "train",
                "case-b": "validation",
                "case-c": "calibration",
                "case-d": "test",
            },
        },
        "inputs": {},
        "tie_ratio": 1.03,
        "record_count": len(records),
        "excluded": {},
        "records_sha256": canonical_sha256(
            [row["record_sha256"] for row in records]
        ),
        "authority": dict(AUTHORITY),
        "evidence_only": True,
        "dispatch_allowed": False,
        "production_promotion_authorized": False,
        "routing_authority_changed": False,
    }
    manifest["bundle_sha256"] = canonical_sha256(manifest)
    _write_json(root / "manifest.json", manifest)
    return root, manifest


def test_exact_bundle_validates_and_preserves_group_split(tmp_path):
    root, manifest = _bundle(tmp_path)
    result = validate_assistx_policy_bundle(
        root,
        expected_producer_sha=PRODUCER_SHA,
        expected_bundle_sha=manifest["bundle_sha256"],
    )

    assert result["record_count"] == 5
    assert result["split_counts"] == {
        "train": 2,
        "validation": 1,
        "calibration": 1,
        "test": 1,
    }
    assert result["authority"]["dispatch_allowed"] is False


def test_import_materializes_native_decision_records(tmp_path):
    root, manifest = _bundle(tmp_path)
    output = tmp_path / "imported"

    receipt = import_assistx_policy_bundle(
        root,
        output,
        expected_producer_sha=PRODUCER_SHA,
        expected_bundle_sha=manifest["bundle_sha256"],
    )

    train = load_jsonl(output / "train.jsonl")
    assert len(train) == 2
    assert train[0].metadata["domain"] == "assistx_inference_policy"
    assert train[0].metadata["family_id"] == "case-a"
    assert train[0].targets is not None
    assert receipt["bundle_sha256"] == manifest["bundle_sha256"]
    assert receipt["authority"]["runtime_authority_changed"] is False


def test_wrong_producer_sha_is_rejected(tmp_path):
    root, _ = _bundle(tmp_path)
    with pytest.raises(ValueError, match="producer git SHA"):
        validate_assistx_policy_bundle(
            root,
            expected_producer_sha="0" * 40,
        )


def test_record_tamper_is_rejected(tmp_path):
    root, _ = _bundle(tmp_path)
    rows = [
        json.loads(line)
        for line in (root / "records.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    rows[0]["state"] = "tampered"
    _write_jsonl(root / "records.jsonl", rows)

    with pytest.raises(ValueError, match="record hash mismatch"):
        validate_assistx_policy_bundle(
            root,
            expected_producer_sha=PRODUCER_SHA,
        )


def test_group_split_drift_is_rejected(tmp_path):
    root, _ = _bundle(tmp_path)
    rows = [
        json.loads(line)
        for line in (root / "records.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    rows[1]["split"] = "test"
    rows[1]["record_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in rows[1].items()
            if key != "record_sha256"
        }
    )
    _write_jsonl(root / "records.jsonl", rows)

    with pytest.raises(ValueError, match="frozen group assignment"):
        validate_assistx_policy_bundle(
            root,
            expected_producer_sha=PRODUCER_SHA,
        )


def test_widened_record_authority_is_rejected(tmp_path):
    root, _ = _bundle(tmp_path)
    rows = [
        json.loads(line)
        for line in (root / "records.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    rows[0]["metadata"]["authority"]["mutation_allowed"] = True
    rows[0]["record_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in rows[0].items()
            if key != "record_sha256"
        }
    )
    _write_jsonl(root / "records.jsonl", rows)

    with pytest.raises(ValueError, match="authority"):
        validate_assistx_policy_bundle(
            root,
            expected_producer_sha=PRODUCER_SHA,
        )


def test_split_file_drift_is_rejected(tmp_path):
    root, _ = _bundle(tmp_path)
    test_rows = [
        json.loads(line)
        for line in (root / "test.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    test_rows[0]["metadata"]["case_id"] = "other"
    _write_jsonl(root / "test.jsonl", test_rows)

    with pytest.raises(ValueError, match="split row"):
        validate_assistx_policy_bundle(
            root,
            expected_producer_sha=PRODUCER_SHA,
        )
