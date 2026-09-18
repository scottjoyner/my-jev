from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .agentic_synth import (
    GENERATOR_VERSION,
    generate_agent_policy_records,
)
from .data import dump_jsonl
from .locking import atomic_write_json
from .manifest import dataset_manifest
from .split import split_records


_EXPECTED_FILES = (
    "source.jsonl",
    "train.jsonl",
    "validation.jsonl",
    "calibration.jsonl",
    "test.jsonl",
    "prepare_manifest.json",
)


def prepare_agent_policy_dataset(
    output_dir: str | Path,
    *,
    records: int = 10_000,
    seed: int = 23,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.10,
    calibration_fraction: float = 0.10,
    force: bool = False,
) -> dict[str, object]:
    destination = Path(
        output_dir
    )
    if records < 4:
        raise ValueError(
            "records must be >= 4"
        )

    existing = [
        destination / name
        for name in _EXPECTED_FILES
        if (
            destination / name
        ).exists()
    ]
    if existing and not force:
        raise FileExistsError(
            "dataset output already exists; "
            "use force=True to replace it: "
            + ", ".join(
                str(path)
                for path in existing
            )
        )

    if force and destination.exists():
        for name in _EXPECTED_FILES:
            path = (
                destination / name
            )
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(
                    path
                )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated = (
        generate_agent_policy_records(
            records,
            seed=seed,
        )
    )
    source_path = (
        destination
        / "source.jsonl"
    )
    dump_jsonl(
        iter(generated),
        source_path,
    )

    splits, group_key = split_records(
        generated,
        train_fraction=(
            train_fraction
        ),
        validation_fraction=(
            validation_fraction
        ),
        calibration_fraction=(
            calibration_fraction
        ),
        seed=seed,
        group_key="auto",
    )

    split_manifests: dict[
        str,
        dict[str, object],
    ] = {}
    for name, split in (
        splits.items()
    ):
        path = (
            destination
            / f"{name}.jsonl"
        )
        dump_jsonl(
            iter(split),
            path,
        )
        split_manifests[
            name
        ] = dataset_manifest(
            path
        )

    source_manifest = (
        dataset_manifest(
            source_path
        )
    )
    payload = {
        "schema_version": 1,
        "generator": (
            GENERATOR_VERSION
        ),
        "records_requested": (
            records
        ),
        "records_emitted": (
            source_manifest[
                "records"
            ]
        ),
        "seed": seed,
        "group_key": group_key,
        "fractions": {
            "train": (
                train_fraction
            ),
            "validation": (
                validation_fraction
            ),
            "calibration": (
                calibration_fraction
            ),
            "test": (
                1.0
                - train_fraction
                - validation_fraction
                - calibration_fraction
            ),
        },
        "source": source_manifest,
        "splits": (
            split_manifests
        ),
    }
    atomic_write_json(
        destination
        / "prepare_manifest.json",
        payload,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a deterministic "
            "Hermes/AssistX policy dataset "
            "with group-safe train/"
            "validation/calibration/test splits"
        )
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "data/assistx-policy-v1"
        ),
    )
    parser.add_argument(
        "--records",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    args = parser.parse_args()

    payload = (
        prepare_agent_policy_dataset(
            args.output_dir,
            records=args.records,
            seed=args.seed,
            train_fraction=(
                args.train_fraction
            ),
            validation_fraction=(
                args.validation_fraction
            ),
            calibration_fraction=(
                args.calibration_fraction
            ),
            force=args.force,
        )
    )
    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
