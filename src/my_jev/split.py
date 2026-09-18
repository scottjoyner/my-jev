from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .data import dump_jsonl, load_jsonl
from .manifest import dataset_manifest
from .schema import DecisionRecord


def _effective_group_key(
    records: list[DecisionRecord],
    requested: str | None,
) -> str | None:
    if requested in (None, "none"):
        return None
    if requested != "auto":
        missing = [
            index
            for index, record in enumerate(records)
            if requested not in record.metadata
        ]
        if missing:
            raise ValueError(
                f"group key {requested!r} missing from {len(missing)} records"
            )
        return requested

    if records and all(
        "family_id" in record.metadata
        for record in records
    ):
        return "family_id"
    return None


def split_records(
    records: list[DecisionRecord],
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.1,
    calibration_fraction: float = 0.1,
    seed: int = 17,
    group_key: str | None = "auto",
) -> tuple[dict[str, list[DecisionRecord]], str | None]:
    fractions = {
        "train": train_fraction,
        "validation": validation_fraction,
        "calibration": calibration_fraction,
    }
    if any(value < 0 for value in fractions.values()):
        raise ValueError("split fractions must be >= 0")
    if train_fraction <= 0:
        raise ValueError("train fraction must be > 0")
    if sum(fractions.values()) >= 1.0:
        raise ValueError(
            "train + validation + calibration fractions must be < 1"
        )

    effective_key = _effective_group_key(records, group_key)
    groups: dict[str, list[DecisionRecord]] = defaultdict(list)

    for index, record in enumerate(records):
        group_id = (
            str(record.metadata[effective_key])
            if effective_key is not None
            else f"record-{index:012d}"
        )
        groups[group_id].append(record)

    group_ids = list(groups)
    random.Random(seed).shuffle(group_ids)
    total_groups = len(group_ids)

    train_end = int(total_groups * train_fraction)
    validation_end = train_end + int(
        total_groups * validation_fraction
    )
    calibration_end = validation_end + int(
        total_groups * calibration_fraction
    )

    assignments = {
        "train": group_ids[:train_end],
        "validation": group_ids[train_end:validation_end],
        "calibration": group_ids[
            validation_end:calibration_end
        ],
        "test": group_ids[calibration_end:],
    }
    splits = {
        name: [
            record
            for group_id in ids
            for record in groups[group_id]
        ]
        for name, ids in assignments.items()
    }

    seen: set[str] = set()
    for name, ids in assignments.items():
        overlap = seen.intersection(ids)
        if overlap:
            raise RuntimeError(
                f"group leakage detected in {name}: {sorted(overlap)[:5]}"
            )
        seen.update(ids)

    return splits, effective_key


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create group-safe train/validation/calibration/test splits"
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.1,
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--group-key",
        default="auto",
        help=(
            "metadata key used to keep related records together; "
            "'auto' uses family_id when present, 'none' disables grouping"
        ),
    )
    args = parser.parse_args()

    records = load_jsonl(args.input)
    splits, effective_key = split_records(
        records,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
        group_key=args.group_key,
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    manifests: dict[str, object] = {}
    for name, split in splits.items():
        path = output / f"{name}.jsonl"
        dump_jsonl(iter(split), path)
        manifests[name] = dataset_manifest(path)

    summary = {
        "source": str(Path(args.input)),
        "seed": args.seed,
        "group_key": effective_key,
        "fractions": {
            "train": args.train_fraction,
            "validation": args.validation_fraction,
            "calibration": args.calibration_fraction,
            "test": (
                1.0
                - args.train_fraction
                - args.validation_fraction
                - args.calibration_fraction
            ),
        },
        "splits": manifests,
    }
    (output / "split_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
