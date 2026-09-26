from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .data import dump_jsonl
from .manifest import dataset_manifest
from .schema import DecisionRecord

BUNDLE_SCHEMA = "assistx-policy-training-bundle-v1"
RECORD_SCHEMA = "assistx-policy-decision-record-v1"
IMPORT_SCHEMA = "my-jev-assistx-policy-import-v1"
SPLITS = ("train", "validation", "calibration", "test")
AUTHORITY_FIELDS = (
    "dispatch_allowed",
    "approval_granted",
    "claim_acquired",
    "mutation_allowed",
    "routing_authority_changed",
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected JSON object"
                )
            rows.append(value)
    return rows


def _authority_safe(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and all(value.get(field, False) is False for field in AUTHORITY_FIELDS)
    )


def _native_record(row: dict[str, Any]) -> DecisionRecord:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("record metadata must be an object")
    native = {
        "state": row.get("state"),
        "questions": row.get("questions"),
        "targets": row.get("targets"),
        "metadata": {
            **metadata,
            "domain": "assistx_inference_policy",
            "family_id": str(row["group_id"]),
            "generator_version": BUNDLE_SCHEMA,
            "assistx_record_id": str(row["record_id"]),
            "assistx_record_sha256": str(row["record_sha256"]),
            "assistx_split": str(row["split"]),
        },
    }
    return DecisionRecord.model_validate(native)


def validate_assistx_policy_bundle(
    bundle_dir: str | Path,
    *,
    expected_producer_sha: str,
    expected_bundle_sha: str | None = None,
) -> dict[str, Any]:
    root = Path(bundle_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"bundle directory does not exist: {root}")
    if not expected_producer_sha:
        raise ValueError("expected_producer_sha is required")

    manifest = _load_json(root / "manifest.json")
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("AssistX policy bundle schema mismatch")
    producer = manifest.get("producer")
    if not isinstance(producer, dict):
        raise ValueError("bundle producer identity is missing")
    if producer.get("repository") != "scottjoyner/auto-assist":
        raise ValueError("bundle producer repository mismatch")
    if producer.get("git_sha") != expected_producer_sha:
        raise ValueError("bundle producer git SHA mismatch")
    if manifest.get("evidence_only") is not True:
        raise ValueError("bundle must remain evidence-only")
    if manifest.get("dispatch_allowed") is not False:
        raise ValueError("bundle cannot grant dispatch authority")
    if manifest.get("production_promotion_authorized") is not False:
        raise ValueError("bundle cannot authorize production promotion")
    if manifest.get("routing_authority_changed") is not False:
        raise ValueError("bundle cannot change routing authority")
    if not _authority_safe(manifest.get("authority")):
        raise ValueError("bundle authority must remain all-false")

    observed_bundle_sha = str(manifest.get("bundle_sha256") or "")
    unsigned_manifest = {
        key: value
        for key, value in manifest.items()
        if key != "bundle_sha256"
    }
    computed_bundle_sha = canonical_sha256(unsigned_manifest)
    if observed_bundle_sha != computed_bundle_sha:
        raise ValueError("bundle manifest hash mismatch")
    if expected_bundle_sha and observed_bundle_sha != expected_bundle_sha:
        raise ValueError("bundle SHA does not match expected_bundle_sha")

    records = _load_jsonl(root / "records.jsonl")
    if len(records) != int(manifest.get("record_count") or 0):
        raise ValueError("bundle record count mismatch")

    split_config = manifest.get("split")
    if not isinstance(split_config, dict):
        raise ValueError("bundle split configuration is missing")
    assignments = split_config.get("group_assignments")
    if not isinstance(assignments, dict):
        raise ValueError("bundle group split assignments are missing")

    seen_record_ids: set[str] = set()
    native_by_split: dict[str, list[DecisionRecord]] = {
        split: [] for split in SPLITS
    }
    raw_ids_by_split: dict[str, list[str]] = {
        split: [] for split in SPLITS
    }
    record_hashes: list[str] = []

    for row in records:
        if row.get("schema") != RECORD_SCHEMA:
            raise ValueError("bundle record schema mismatch")
        record_id = str(row.get("record_id") or "")
        if not record_id:
            raise ValueError("bundle record_id is required")
        if record_id in seen_record_ids:
            raise ValueError(f"duplicate bundle record_id: {record_id}")
        seen_record_ids.add(record_id)

        split = str(row.get("split") or "")
        if split not in SPLITS:
            raise ValueError(f"unsupported bundle split: {split}")
        group_id = str(row.get("group_id") or "")
        if not group_id:
            raise ValueError(f"{record_id}: group_id is required")
        if assignments.get(group_id) != split:
            raise ValueError(
                f"{record_id}: split does not match frozen group assignment"
            )

        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{record_id}: metadata must be an object")
        if metadata.get("evidence_only") is not True:
            raise ValueError(f"{record_id}: evidence_only must remain true")
        if metadata.get("dispatch_allowed") is not False:
            raise ValueError(f"{record_id}: dispatch_allowed must remain false")
        if metadata.get("routing_authority_changed") is not False:
            raise ValueError(
                f"{record_id}: routing authority must remain unchanged"
            )
        if not _authority_safe(metadata.get("authority")):
            raise ValueError(f"{record_id}: authority must remain all-false")

        observed_record_sha = str(row.get("record_sha256") or "")
        unsigned_record = {
            key: value
            for key, value in row.items()
            if key != "record_sha256"
        }
        computed_record_sha = canonical_sha256(unsigned_record)
        if observed_record_sha != computed_record_sha:
            raise ValueError(f"{record_id}: record hash mismatch")
        record_hashes.append(observed_record_sha)

        native = _native_record(row)
        native_by_split[split].append(native)
        raw_ids_by_split[split].append(record_id)

    if manifest.get("records_sha256") != canonical_sha256(record_hashes):
        raise ValueError("bundle records_sha256 mismatch")

    expected_counts = split_config.get("record_counts")
    if not isinstance(expected_counts, dict):
        raise ValueError("bundle split record counts are missing")
    for split in SPLITS:
        if int(expected_counts.get(split) or 0) != len(native_by_split[split]):
            raise ValueError(f"{split}: split record count mismatch")

        split_rows = _load_jsonl(root / f"{split}.jsonl")
        split_ids = [str(row.get("record_id") or "") for row in split_rows]
        if split_ids != raw_ids_by_split[split]:
            raise ValueError(
                f"{split}: split file does not exactly match records.jsonl"
            )
        for split_row, expected_id in zip(
            split_rows,
            raw_ids_by_split[split],
            strict=True,
        ):
            expected = next(
                row for row in records if row["record_id"] == expected_id
            )
            if canonical_sha256(split_row) != canonical_sha256(expected):
                raise ValueError(
                    f"{split}: split row bytes/content drift for {expected_id}"
                )

    group_splits: dict[str, set[str]] = {}
    for row in records:
        group_splits.setdefault(str(row["group_id"]), set()).add(
            str(row["split"])
        )
    leaking = sorted(
        group_id
        for group_id, values in group_splits.items()
        if len(values) != 1
    )
    if leaking:
        raise ValueError(f"group leakage detected: {leaking[:5]}")

    return {
        "schema": IMPORT_SCHEMA,
        "bundle_dir": str(root),
        "bundle_sha256": observed_bundle_sha,
        "producer_git_sha": expected_producer_sha,
        "record_count": len(records),
        "split_counts": {
            split: len(native_by_split[split])
            for split in SPLITS
        },
        "records": native_by_split,
        "authority": {
            "evidence_only": True,
            "dispatch_allowed": False,
            "runtime_authority_changed": False,
        },
    }


def import_assistx_policy_bundle(
    bundle_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_producer_sha: str,
    expected_bundle_sha: str | None = None,
) -> dict[str, Any]:
    validated = validate_assistx_policy_bundle(
        bundle_dir,
        expected_producer_sha=expected_producer_sha,
        expected_bundle_sha=expected_bundle_sha,
    )
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError(
            f"import output directory must be empty: {destination}"
        )

    split_manifests: dict[str, Any] = {}
    records_by_split = validated.pop("records")
    for split in SPLITS:
        path = destination / f"{split}.jsonl"
        dump_jsonl(iter(records_by_split[split]), path)
        split_manifests[split] = dataset_manifest(path)

    receipt = {
        **validated,
        "output_dir": str(destination),
        "splits": split_manifests,
        "authority": {
            "evidence_only": True,
            "dispatch_allowed": False,
            "runtime_authority_changed": False,
        },
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    (destination / "import-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and import an exact-SHA AssistX policy-training bundle "
            "into native my-jev DecisionRecord splits."
        )
    )
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-producer-sha", required=True)
    parser.add_argument("--expected-bundle-sha")
    args = parser.parse_args()

    receipt = import_assistx_policy_bundle(
        args.bundle_dir,
        args.output_dir,
        expected_producer_sha=args.expected_producer_sha,
        expected_bundle_sha=args.expected_bundle_sha,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
