from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .data import load_jsonl
from .schema import DecisionRecord

SPLIT_NAMES = (
    "train",
    "validation",
    "calibration",
    "test",
)


def _semantic_digest(
    record: DecisionRecord,
) -> str:
    payload = {
        "state": record.state,
        "questions": {
            name: question.model_dump(
                mode="json"
            )
            for name, question in sorted(
                record.questions.items()
            )
        },
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(
        raw
    ).hexdigest()


def _overlaps(
    values: dict[str, set[str]],
) -> list[dict[str, object]]:
    found: list[
        dict[str, object]
    ] = []
    names = list(values)
    for index, left in enumerate(
        names
    ):
        for right in names[
            index + 1 :
        ]:
            overlap = (
                values[left]
                .intersection(
                    values[right]
                )
            )
            if overlap:
                found.append(
                    {
                        "left": left,
                        "right": right,
                        "count": len(
                            overlap
                        ),
                        "examples": sorted(
                            overlap
                        )[:5],
                    }
                )
    return found


def audit_dataset_splits(
    paths: dict[
        str,
        str | Path,
    ],
    *,
    group_key: str = "auto",
    fail_on_error: bool = True,
) -> dict[str, object]:
    missing_splits = (
        set(SPLIT_NAMES)
        - set(paths)
    )
    if missing_splits:
        raise ValueError(
            "missing required dataset splits: "
            f"{sorted(missing_splits)}"
        )

    loaded: dict[
        str,
        list[DecisionRecord],
    ] = {}
    issues: list[str] = []

    for name in SPLIT_NAMES:
        path = Path(paths[name])
        records = load_jsonl(path)
        loaded[name] = records
        if not records:
            issues.append(
                f"{name} split is empty"
            )

    family_presence = [
        "family_id"
        in record.metadata
        for records in loaded.values()
        for record in records
    ]
    if group_key == "auto":
        effective_group_key = (
            "family_id"
            if any(family_presence)
            else None
        )
    elif group_key in {
        "",
        "none",
    }:
        effective_group_key = None
    else:
        effective_group_key = (
            group_key
        )

    semantic_sets: dict[
        str,
        set[str],
    ] = {}
    group_sets: dict[
        str,
        set[str],
    ] = {}
    split_report: dict[
        str,
        dict[str, object],
    ] = {}

    for name in SPLIT_NAMES:
        records = loaded[name]
        semantics = {
            _semantic_digest(record)
            for record in records
        }
        semantic_sets[name] = (
            semantics
        )

        groups: set[str] = set()
        missing_group = 0
        unlabeled = 0
        target_questions = 0
        for record in records:
            if not record.targets:
                unlabeled += 1
            else:
                target_questions += len(
                    record.targets
                )

            if (
                effective_group_key
                is not None
            ):
                value = (
                    record.metadata.get(
                        effective_group_key
                    )
                )
                if (
                    value is None
                    or str(value) == ""
                ):
                    missing_group += 1
                else:
                    groups.add(
                        str(value)
                    )

        if unlabeled:
            issues.append(
                f"{name} contains "
                f"{unlabeled} unlabeled "
                "records"
            )
        if missing_group:
            issues.append(
                f"{name} is missing "
                f"{effective_group_key!r} "
                f"on {missing_group} records"
            )

        group_sets[name] = groups
        split_report[name] = {
            "path": str(
                paths[name]
            ),
            "records": len(records),
            "semantic_states": len(
                semantics
            ),
            "groups": len(groups),
            "target_questions": (
                target_questions
            ),
            "unlabeled_records": (
                unlabeled
            ),
            "missing_group_records": (
                missing_group
            ),
        }

    semantic_overlaps = _overlaps(
        semantic_sets
    )
    if semantic_overlaps:
        issues.append(
            "semantic record leakage "
            "exists across dataset splits"
        )

    group_overlaps: list[
        dict[str, object]
    ] = []
    if effective_group_key is not None:
        group_overlaps = _overlaps(
            group_sets
        )
        if group_overlaps:
            issues.append(
                f"{effective_group_key} "
                "leakage exists across "
                "dataset splits"
            )

    report = {
        "schema_version": 1,
        "passed": not issues,
        "group_key": (
            effective_group_key
        ),
        "splits": split_report,
        "semantic_overlaps": (
            semantic_overlaps
        ),
        "group_overlaps": (
            group_overlaps
        ),
        "issues": issues,
    }

    if issues and fail_on_error:
        raise ValueError(
            "dataset audit failed: "
            + "; ".join(issues)
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit train/validation/"
            "calibration/test datasets for "
            "unlabeled rows and split leakage"
        )
    )
    parser.add_argument(
        "--train",
        required=True,
    )
    parser.add_argument(
        "--validation",
        required=True,
    )
    parser.add_argument(
        "--calibration",
        required=True,
    )
    parser.add_argument(
        "--test",
        required=True,
    )
    parser.add_argument(
        "--group-key",
        default="auto",
    )
    parser.add_argument(
        "--output",
    )
    args = parser.parse_args()

    report = audit_dataset_splits(
        {
            "train": args.train,
            "validation": (
                args.validation
            ),
            "calibration": (
                args.calibration
            ),
            "test": args.test,
        },
        group_key=args.group_key,
        fail_on_error=False,
    )
    rendered = json.dumps(
        report,
        indent=2,
        sort_keys=True,
    ) + "\n"

    if args.output:
        output = Path(
            args.output
        )
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output.write_text(
            rendered,
            encoding="utf-8",
        )
    print(
        rendered,
        end="",
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
