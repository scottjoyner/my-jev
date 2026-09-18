from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .data import dump_jsonl, load_jsonl
from .manifest import write_manifest
from .schema import DecisionRecord, QuestionSpec, QuestionType, TargetSpec

AnnotationValue = bool | int | str


class PolicyAnnotation(BaseModel):
    """One independent label source for a visited policy state.

    Labels may be partial. Multiple annotations for the same intent are
    aggregated into empirical probability distributions rather than collapsed
    to a majority-vote hard target.
    """

    source_intent_id: str = Field(min_length=1)
    labels: dict[str, AnnotationValue]
    source: str = Field(default="operator", min_length=1)
    annotator: str | None = None
    weight: float = Field(default=1.0, gt=0.0)
    evidence: dict[str, Any] = Field(default_factory=dict)
    note: str = ""

    @model_validator(mode="after")
    def validate_labels(self) -> PolicyAnnotation:
        if not self.labels:
            raise ValueError("annotation must label at least one question")
        return self


def load_annotations(path: str | Path) -> list[PolicyAnnotation]:
    annotations: list[PolicyAnnotation] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                annotations.append(
                    PolicyAnnotation.model_validate(payload)
                )
            except Exception as exc:
                raise ValueError(
                    f"invalid annotation at {path}:{line_number}: {exc}"
                ) from exc
    if not annotations:
        raise ValueError(f"no annotations in {path}")
    return annotations


def _record_key(record: DecisionRecord) -> str | None:
    for field in ("source_intent_id", "family_id"):
        value = record.metadata.get(field)
        if value is not None and str(value):
            return str(value)
    return None


def _label_index(
    question_name: str,
    question: QuestionSpec,
    value: AnnotationValue,
) -> int:
    options = list(question.options or [])

    if isinstance(value, bool):
        if question.type != QuestionType.NOUL:
            raise ValueError(
                f"{question_name}: boolean labels are only valid for Noul questions"
            )
        return int(value)

    if isinstance(value, int):
        if not 0 <= value < len(options):
            raise ValueError(
                f"{question_name}: option index {value} outside 0..{len(options) - 1}"
            )
        return value

    if value not in options:
        raise ValueError(
            f"{question_name}: {value!r} is not one of {options!r}"
        )
    return options.index(value)


def apply_annotations(
    record: DecisionRecord,
    annotations: list[PolicyAnnotation],
    *,
    replace_existing: bool = False,
) -> DecisionRecord:
    """Attach partial/consensus targets to one visited state."""

    if not annotations:
        return record.model_copy(deep=True)

    key = _record_key(record)
    if key is None:
        raise ValueError(
            "cannot adjudicate a record without source_intent_id or family_id"
        )

    target_mass: dict[str, list[float]] = {}
    vote_count: Counter[str] = Counter()
    source_count: Counter[str] = Counter()

    for annotation in annotations:
        if annotation.source_intent_id != key:
            raise ValueError(
                f"annotation {annotation.source_intent_id!r} does not match record {key!r}"
            )
        source_count[annotation.source] += 1

        for question_name, value in annotation.labels.items():
            question = record.questions.get(question_name)
            if question is None:
                raise ValueError(
                    f"{key}: annotation references unknown question {question_name!r}"
                )
            if (
                record.targets
                and question_name in record.targets
                and not replace_existing
            ):
                raise ValueError(
                    f"{key}: target for {question_name!r} already exists; "
                    "use replace_existing=True to replace it"
                )

            index = _label_index(
                question_name,
                question,
                value,
            )
            mass = target_mass.setdefault(
                question_name,
                [0.0] * len(question.options or []),
            )
            mass[index] += annotation.weight
            vote_count[question_name] += 1

    targets = dict(record.targets or {})
    for question_name, mass in target_mass.items():
        targets[question_name] = TargetSpec(
            distribution=mass
        )

    output = record.model_copy(deep=True)
    output.targets = targets or None

    covered = len(targets)
    total = len(record.questions)
    output.metadata.update(
        {
            "label_status": (
                "adjudicated"
                if covered == total
                else "partially_adjudicated"
            ),
            "label_source": "trajectory_adjudication",
            "label_sources": dict(sorted(source_count.items())),
            "annotation_count": len(annotations),
            "adjudicated_questions": sorted(target_mass),
            "label_votes": dict(sorted(vote_count.items())),
            "label_coverage": covered / total if total else 0.0,
        }
    )
    return DecisionRecord.model_validate(
        output.model_dump(mode="json")
    )


def adjudicate_records(
    records: list[DecisionRecord],
    annotations: list[PolicyAnnotation],
    *,
    keep_unlabeled: bool = False,
    allow_unmatched: bool = False,
    replace_existing: bool = False,
) -> list[DecisionRecord]:
    by_key: dict[str, list[PolicyAnnotation]] = defaultdict(list)
    for annotation in annotations:
        by_key[annotation.source_intent_id].append(annotation)

    record_keys = {
        key
        for record in records
        if (key := _record_key(record)) is not None
    }
    unmatched = set(by_key) - record_keys
    if unmatched and not allow_unmatched:
        raise ValueError(
            "annotations do not match any input record: "
            f"{sorted(unmatched)}"
        )

    output: list[DecisionRecord] = []
    for record in records:
        key = _record_key(record)
        matched = by_key.get(key or "", [])
        if matched:
            output.append(
                apply_annotations(
                    record,
                    matched,
                    replace_existing=replace_existing,
                )
            )
        elif keep_unlabeled:
            output.append(record.model_copy(deep=True))
    return output


def _summary(
    records: list[DecisionRecord],
    annotations: list[PolicyAnnotation],
) -> dict[str, object]:
    labeled_records = sum(
        1 for record in records if record.targets
    )
    labeled_questions = sum(
        len(record.targets or {})
        for record in records
    )
    sources = Counter(
        annotation.source
        for annotation in annotations
    )
    return {
        "records": len(records),
        "labeled_records": labeled_records,
        "labeled_questions": labeled_questions,
        "annotations": len(annotations),
        "annotation_sources": dict(sorted(sources.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Attach operator/outcome adjudications to imported "
            "Hermes/AssistX shadow states"
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--keep-unlabeled",
        action="store_true",
        help="retain states that have no matching annotation",
    )
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="ignore annotation IDs that are absent from the input dataset",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="allow adjudications to replace existing targets for the same question",
    )
    args = parser.parse_args()

    records = load_jsonl(args.input)
    annotations = load_annotations(args.annotations)
    adjudicated = adjudicate_records(
        records,
        annotations,
        keep_unlabeled=args.keep_unlabeled,
        allow_unmatched=args.allow_unmatched,
        replace_existing=args.replace_existing,
    )
    if not adjudicated:
        raise SystemExit(
            "no records were emitted; provide matching annotations "
            "or use --keep-unlabeled"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(iter(adjudicated), output)

    manifest_path = output.with_suffix(
        output.suffix + ".manifest.json"
    )
    manifest = write_manifest(
        output,
        manifest_path,
    )

    print(
        json.dumps(
            {
                "output": str(output),
                "manifest": str(manifest_path),
                "sha256": manifest["sha256"],
                **_summary(adjudicated, annotations),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
