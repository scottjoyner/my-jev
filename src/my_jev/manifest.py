from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from .data import load_jsonl


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_manifest(path: str | Path) -> dict[str, object]:
    source = Path(path)
    records = load_jsonl(source)

    domains: Counter[str] = Counter()
    question_types: Counter[str] = Counter()
    task_ids: Counter[str] = Counter()
    generator_versions: Counter[str] = Counter()
    hard_targets = 0
    soft_targets = 0
    option_counts: list[int] = []
    state_lengths: list[int] = []
    family_ids: set[str] = set()

    for record in records:
        domains[str(record.metadata.get("domain", "unknown"))] += 1
        version = record.metadata.get("generator_version")
        if version is not None:
            generator_versions[str(version)] += 1
        family_id = record.metadata.get("family_id")
        if family_id is not None:
            family_ids.add(str(family_id))
        state_lengths.append(len(record.state))

        for name, question in record.questions.items():
            question_types[question.type.value] += 1
            task_id = question.metadata.get("task_id", name)
            task_ids[str(task_id)] += 1
            option_counts.append(len(question.options or []))

            target = (record.targets or {}).get(name)
            if target is None:
                continue
            if target.distribution is not None:
                soft_targets += 1
            else:
                hard_targets += 1

    question_count = sum(question_types.values())
    return {
        "path": str(source),
        "sha256": file_sha256(source),
        "bytes": source.stat().st_size,
        "records": len(records),
        "questions": question_count,
        "families": len(family_ids),
        "domains": dict(sorted(domains.items())),
        "question_types": dict(sorted(question_types.items())),
        "task_ids": dict(sorted(task_ids.items())),
        "targets": {
            "hard": hard_targets,
            "soft": soft_targets,
        },
        "generator_versions": dict(sorted(generator_versions.items())),
        "state_chars": {
            "min": min(state_lengths, default=0),
            "max": max(state_lengths, default=0),
            "mean": mean(state_lengths) if state_lengths else 0.0,
        },
        "option_cardinality": {
            "min": min(option_counts, default=0),
            "max": max(option_counts, default=0),
            "mean": mean(option_counts) if option_counts else 0.0,
        },
    }


def write_manifest(
    path: str | Path,
    output: str | Path,
) -> dict[str, object]:
    manifest = dataset_manifest(path)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a reproducibility manifest for a decision dataset"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    output = (
        Path(args.output)
        if args.output
        else Path(args.input).with_suffix(
            Path(args.input).suffix + ".manifest.json"
        )
    )
    manifest = write_manifest(args.input, output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
