from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from .data import dump_jsonl, load_jsonl
from .manifest import dataset_manifest, write_manifest
from .schema import DecisionRecord

MIX_VERSION = "my-jev-mix-v1"

# These are evidence sources, not acceptable target provenance.  A record may
# still carry legacy/shadow predictions in metadata after adjudication; only
# claiming one of them as the *label source* is rejected.
UNTRUSTED_LABEL_SOURCES = frozenset(
    {
        "shadow_prediction",
        "shadow_model",
        "legacy_classifier",
        "legacy_router",
        "self_prediction",
        "model_prediction",
    }
)


@dataclass(frozen=True)
class MixSource:
    name: str
    path: Path
    max_records: int | None = None


def _canonical_bytes(
    record: DecisionRecord,
    *,
    include_targets: bool,
) -> bytes:
    payload: dict[str, object] = {
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
    if include_targets:
        payload["targets"] = {
            name: target.model_dump(
                mode="json"
            )
            for name, target in sorted(
                (record.targets or {}).items()
            )
        }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(
    record: DecisionRecord,
    *,
    include_targets: bool,
) -> str:
    return hashlib.sha256(
        _canonical_bytes(
            record,
            include_targets=include_targets,
        )
    ).hexdigest()


def _label_source(
    record: DecisionRecord,
) -> str | None:
    value = record.metadata.get(
        "label_source"
    )
    if value is None:
        return None
    return str(value).strip() or None


def validate_mix_record(
    record: DecisionRecord,
    *,
    source_name: str,
    allow_unlabeled: bool = False,
) -> None:
    if not record.targets:
        if allow_unlabeled:
            return
        raise ValueError(
            f"{source_name}: unlabeled record "
            "cannot enter a training mix"
        )

    status = str(
        record.metadata.get(
            "label_status",
            "",
        )
    ).strip().lower()
    if (
        status == "unlabeled"
        and not allow_unlabeled
    ):
        raise ValueError(
            f"{source_name}: record claims "
            "label_status=unlabeled"
        )

    source = _label_source(
        record
    )
    if (
        source is not None
        and source.lower()
        in UNTRUSTED_LABEL_SOURCES
    ):
        raise ValueError(
            f"{source_name}: refusing "
            f"untrusted label_source={source!r}"
        )


def _tag_record(
    record: DecisionRecord,
    *,
    source: MixSource,
    source_sha256: str,
) -> DecisionRecord:
    output = record.model_copy(
        deep=True
    )
    output.metadata.update(
        {
            "mix_version": MIX_VERSION,
            "mix_source": source.name,
            "mix_source_sha256": (
                source_sha256
            ),
        }
    )
    return output


def mix_records(
    sources: list[MixSource],
    *,
    seed: int = 17,
    allow_unlabeled: bool = False,
    allow_conflicting_duplicates: bool = False,
) -> tuple[
    list[DecisionRecord],
    dict[str, object],
]:
    if not sources:
        raise ValueError(
            "at least one mix source is required"
        )
    names = [
        source.name
        for source in sources
    ]
    if (
        len(set(names))
        != len(names)
    ):
        raise ValueError(
            "mix source names must be unique"
        )

    output: list[
        DecisionRecord
    ] = []
    seen_content: set[str] = set()
    identity_targets: dict[
        str,
        str,
    ] = {}
    composition: dict[
        str,
        object,
    ] = {}
    duplicate_count = 0

    for source in sources:
        if (
            source.max_records is not None
            and source.max_records < 1
        ):
            raise ValueError(
                f"{source.name}: max_records "
                "must be >= 1"
            )

        manifest = dataset_manifest(
            source.path
        )
        source_sha256 = str(
            manifest["sha256"]
        )
        records = load_jsonl(
            source.path
        )

        if (
            source.max_records
            is not None
            and len(records)
            > source.max_records
        ):
            rng = random.Random(
                f"{seed}:{source.name}:cap"
            )
            selected = list(
                range(len(records))
            )
            rng.shuffle(
                selected
            )
            selected = sorted(
                selected[
                    : source.max_records
                ]
            )
            records = [
                records[index]
                for index in selected
            ]

        accepted = 0
        source_duplicates = 0
        for record in records:
            validate_mix_record(
                record,
                source_name=source.name,
                allow_unlabeled=(
                    allow_unlabeled
                ),
            )
            identity = _digest(
                record,
                include_targets=False,
            )
            target_digest = _digest(
                record,
                include_targets=True,
            )

            previous = (
                identity_targets.get(
                    identity
                )
            )
            if (
                previous is not None
                and previous
                != target_digest
                and not (
                    allow_conflicting_duplicates
                )
            ):
                raise ValueError(
                    f"{source.name}: conflicting "
                    "targets for an identical "
                    "state/question schema"
                )

            if target_digest in seen_content:
                duplicate_count += 1
                source_duplicates += 1
                continue

            identity_targets.setdefault(
                identity,
                target_digest,
            )
            seen_content.add(
                target_digest
            )
            output.append(
                _tag_record(
                    record,
                    source=source,
                    source_sha256=(
                        source_sha256
                    ),
                )
            )
            accepted += 1

        composition[
            source.name
        ] = {
            "path": str(
                source.path
            ),
            "sha256": (
                source_sha256
            ),
            "available_records": (
                manifest["records"]
            ),
            "selected_records": (
                len(records)
            ),
            "accepted_records": (
                accepted
            ),
            "exact_duplicates_dropped": (
                source_duplicates
            ),
            "max_records": (
                source.max_records
            ),
        }

    if not output:
        raise ValueError(
            "mix produced no records"
        )

    random.Random(seed).shuffle(
        output
    )
    return (
        output,
        {
            "mix_version": MIX_VERSION,
            "seed": seed,
            "records": len(output),
            "exact_duplicates_dropped": (
                duplicate_count
            ),
            "allow_unlabeled": (
                allow_unlabeled
            ),
            "allow_conflicting_duplicates": (
                allow_conflicting_duplicates
            ),
            "sources": composition,
        },
    )


def _parse_key_value(
    values: list[str],
    *,
    field: str,
) -> dict[str, str]:
    parsed: dict[
        str,
        str,
    ] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(
                f"{field} expects NAME=VALUE, "
                f"got {raw!r}"
            )
        name, value = raw.split(
            "=",
            1,
        )
        name = name.strip()
        value = value.strip()
        if (
            not name
            or not value
        ):
            raise ValueError(
                f"{field} expects NAME=VALUE"
            )
        if name in parsed:
            raise ValueError(
                f"duplicate {field} name: "
                f"{name}"
            )
        parsed[name] = value
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically mix labeled "
            "my-jev datasets with provenance"
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "repeat for each input dataset"
        ),
    )
    parser.add_argument(
        "--cap",
        action="append",
        default=[],
        metavar="NAME=N",
        help=(
            "optional deterministic per-source "
            "record cap"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
    )
    parser.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help=(
            "unsafe for normal training; "
            "retain targetless records"
        ),
    )
    parser.add_argument(
        "--allow-conflicting-duplicates",
        action="store_true",
    )
    args = parser.parse_args()

    source_values = _parse_key_value(
        args.source,
        field="source",
    )
    cap_values = _parse_key_value(
        args.cap,
        field="cap",
    )
    unknown_caps = (
        set(cap_values)
        - set(source_values)
    )
    if unknown_caps:
        raise SystemExit(
            "caps reference unknown sources: "
            f"{sorted(unknown_caps)}"
        )

    sources: list[
        MixSource
    ] = []
    for name, raw_path in (
        source_values.items()
    ):
        max_records = None
        if name in cap_values:
            max_records = int(
                cap_values[name]
            )
        sources.append(
            MixSource(
                name=name,
                path=Path(
                    raw_path
                ),
                max_records=(
                    max_records
                ),
            )
        )

    records, composition = (
        mix_records(
            sources,
            seed=args.seed,
            allow_unlabeled=(
                args.allow_unlabeled
            ),
            allow_conflicting_duplicates=(
                args.allow_conflicting_duplicates
            ),
        )
    )

    output = Path(
        args.output
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    dump_jsonl(
        iter(records),
        output,
    )
    manifest_path = (
        output.with_suffix(
            output.suffix
            + ".manifest.json"
        )
    )
    manifest = write_manifest(
        output,
        manifest_path,
    )
    composition[
        "output"
    ] = {
        "path": str(output),
        "sha256": (
            manifest["sha256"]
        ),
        "records": (
            manifest["records"]
        ),
    }
    mix_manifest_path = (
        output.with_suffix(
            output.suffix
            + ".mix.json"
        )
    )
    mix_manifest_path.write_text(
        json.dumps(
            composition,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output": str(
                    output
                ),
                "manifest": str(
                    manifest_path
                ),
                "mix_manifest": str(
                    mix_manifest_path
                ),
                "records": (
                    manifest["records"]
                ),
                "sha256": (
                    manifest["sha256"]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
