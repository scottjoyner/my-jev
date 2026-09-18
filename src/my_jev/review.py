from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import dump_jsonl, load_jsonl
from .manifest import dataset_manifest, write_manifest
from .schema import DecisionRecord

REVIEW_QUEUE_VERSION = "assistx-active-review-v1"


@dataclass(frozen=True)
class ReviewWeights:
    correction: float = 4.0
    inconsistency: float = 3.0
    disagreement: float = 2.0
    uncertainty: float = 2.0


@dataclass(frozen=True)
class ReviewScore:
    priority: float
    uncertainty: float
    disagreement: bool
    inconsistency_count: int
    correction_evidence: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "priority": self.priority,
            "uncertainty": self.uncertainty,
            "disagreement": self.disagreement,
            "inconsistency_count": self.inconsistency_count,
            "correction_evidence": self.correction_evidence,
            "reasons": list(self.reasons),
        }


def _probability(value: object) -> float | None:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(probability):
        return None
    return min(1.0, max(0.0, probability))


def _normalized_entropy_distribution(
    values: object,
) -> float | None:
    if not isinstance(values, dict) or len(values) < 2:
        return None
    probabilities: list[float] = []
    for value in values.values():
        probability = _probability(value)
        if probability is None:
            return None
        probabilities.append(probability)
    total = sum(probabilities)
    if total <= 0.0:
        return None
    probabilities = [
        probability / total
        for probability in probabilities
    ]
    entropy = -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )
    maximum = math.log(len(probabilities))
    if maximum <= 0.0:
        return None
    return entropy / maximum


def _normalized_binary_entropy(
    value: object,
) -> float | None:
    probability = _probability(value)
    if probability is None:
        return None
    if probability in {0.0, 1.0}:
        return 0.0
    entropy = -(
        probability * math.log(probability)
        + (1.0 - probability)
        * math.log(1.0 - probability)
    )
    return entropy / math.log(2.0)


def model_uncertainty(
    record: DecisionRecord,
) -> float:
    """Return mean normalized entropy across available shadow policy fields."""

    scores = record.metadata.get(
        "shadow_scores"
    )
    entropies: list[float] = []
    if isinstance(scores, dict):
        for value in scores.values():
            if isinstance(value, dict):
                entropy = (
                    _normalized_entropy_distribution(
                        value
                    )
                )
            else:
                entropy = (
                    _normalized_binary_entropy(
                        value
                    )
                )
            if entropy is not None:
                entropies.append(entropy)

    if entropies:
        return sum(entropies) / len(entropies)

    confidence = _probability(
        record.metadata.get(
            "shadow_model_route_confidence"
        )
    )
    if confidence is not None:
        return 1.0 - confidence
    return 0.0


def _nonempty_text(
    metadata: dict[str, Any],
    key: str,
) -> str:
    value = metadata.get(key)
    if value is None:
        return ""
    return str(value).strip()


def legacy_disagreement(
    record: DecisionRecord,
) -> bool:
    metadata = record.metadata
    legacy_action = _nonempty_text(
        metadata,
        "legacy_policy_action",
    )
    shadow_action = _nonempty_text(
        metadata,
        "shadow_policy_action",
    )
    if (
        legacy_action
        and shadow_action
        and legacy_action != shadow_action
    ):
        return True

    legacy_classification = _nonempty_text(
        metadata,
        "legacy_classification",
    )
    shadow_classification = _nonempty_text(
        metadata,
        "shadow_classification",
    )
    return bool(
        legacy_classification
        and shadow_classification
        and legacy_classification
        != shadow_classification
    )


def consistency_violations(
    record: DecisionRecord,
) -> tuple[str, ...]:
    value = record.metadata.get(
        "shadow_consistency_violations"
    )
    if not isinstance(value, list):
        return ()
    return tuple(
        sorted(
            {
                str(item).strip()
                for item in value
                if str(item).strip()
            }
        )
    )


def has_correction_evidence(
    record: DecisionRecord,
) -> bool:
    fields = record.metadata.get(
        "correction_evidence_fields"
    )
    if isinstance(fields, list) and fields:
        return True
    return bool(
        record.metadata.get(
            "user_corrected",
            False,
        )
        or record.metadata.get(
            "operator_corrected",
            False,
        )
    )


def score_review_record(
    record: DecisionRecord,
    *,
    weights: ReviewWeights = ReviewWeights(),
) -> ReviewScore:
    uncertainty = model_uncertainty(
        record
    )
    disagreement = legacy_disagreement(
        record
    )
    violations = consistency_violations(
        record
    )
    correction = has_correction_evidence(
        record
    )

    reasons: list[str] = []
    if correction:
        reasons.append("correction_evidence")
    if violations:
        reasons.append(
            "policy_inconsistency:"
            + ",".join(violations)
        )
    if disagreement:
        reasons.append(
            "legacy_shadow_disagreement"
        )
    if uncertainty > 0.0:
        reasons.append(
            f"model_uncertainty={uncertainty:.4f}"
        )

    inconsistency_signal = min(
        1.0,
        len(violations) / 2.0,
    )
    priority = (
        weights.correction * float(correction)
        + weights.inconsistency
        * inconsistency_signal
        + weights.disagreement
        * float(disagreement)
        + weights.uncertainty
        * uncertainty
    )

    return ReviewScore(
        priority=priority,
        uncertainty=uncertainty,
        disagreement=disagreement,
        inconsistency_count=len(violations),
        correction_evidence=correction,
        reasons=tuple(reasons),
    )


def _intent_id(
    record: DecisionRecord,
) -> str:
    for key in (
        "source_intent_id",
        "family_id",
    ):
        value = record.metadata.get(key)
        if value is not None and str(value):
            return str(value)
    return ""


def build_review_queue(
    records: list[DecisionRecord],
    *,
    limit: int | None = 200,
    min_priority: float = 0.0,
    weights: ReviewWeights = ReviewWeights(),
) -> list[DecisionRecord]:
    ranked: list[
        tuple[ReviewScore, str, int, DecisionRecord]
    ] = []

    for index, record in enumerate(records):
        score = score_review_record(
            record,
            weights=weights,
        )
        if score.priority < min_priority:
            continue
        ranked.append(
            (
                score,
                _intent_id(record),
                index,
                record,
            )
        )

    ranked.sort(
        key=lambda item: (
            -item[0].priority,
            -float(
                item[0].correction_evidence
            ),
            -item[0].inconsistency_count,
            -float(item[0].disagreement),
            -item[0].uncertainty,
            item[1],
            item[2],
        )
    )

    if limit is not None:
        ranked = ranked[:limit]

    queue: list[DecisionRecord] = []
    for rank, (
        score,
        _,
        _,
        record,
    ) in enumerate(ranked, start=1):
        item = record.model_copy(
            deep=True
        )
        item.metadata["active_review"] = {
            "version": REVIEW_QUEUE_VERSION,
            "rank": rank,
            **score.as_dict(),
        }
        queue.append(item)
    return queue


def _queue_summary(
    records: list[DecisionRecord],
) -> dict[str, object]:
    priorities: list[float] = []
    uncertainties: list[float] = []
    disagreements = 0
    inconsistencies = 0
    corrections = 0

    for record in records:
        review = record.metadata.get(
            "active_review"
        )
        if not isinstance(review, dict):
            continue
        priorities.append(
            float(review.get("priority", 0.0))
        )
        uncertainties.append(
            float(review.get("uncertainty", 0.0))
        )
        disagreements += int(
            bool(review.get("disagreement"))
        )
        inconsistencies += int(
            int(
                review.get(
                    "inconsistency_count",
                    0,
                )
            )
            > 0
        )
        corrections += int(
            bool(
                review.get(
                    "correction_evidence"
                )
            )
        )

    return {
        "records": len(records),
        "priority": {
            "max": max(
                priorities,
                default=0.0,
            ),
            "min": min(
                priorities,
                default=0.0,
            ),
        },
        "uncertainty": {
            "max": max(
                uncertainties,
                default=0.0,
            ),
            "mean": (
                sum(uncertainties)
                / len(uncertainties)
                if uncertainties
                else 0.0
            ),
        },
        "signals": {
            "legacy_shadow_disagreement": (
                disagreements
            ),
            "policy_inconsistency": (
                inconsistencies
            ),
            "correction_evidence": corrections,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rank imported Hermes/AssistX shadow states "
            "for active-learning review"
        )
    )
    parser.add_argument(
        "--input",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help=(
            "maximum review rows; use 0 for no limit"
        ),
    )
    parser.add_argument(
        "--min-priority",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--correction-weight",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--inconsistency-weight",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--disagreement-weight",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--uncertainty-weight",
        type=float,
        default=2.0,
    )
    args = parser.parse_args()

    records = load_jsonl(args.input)
    weights = ReviewWeights(
        correction=args.correction_weight,
        inconsistency=(
            args.inconsistency_weight
        ),
        disagreement=(
            args.disagreement_weight
        ),
        uncertainty=args.uncertainty_weight,
    )
    queue = build_review_queue(
        records,
        limit=(
            None
            if args.limit == 0
            else args.limit
        ),
        min_priority=args.min_priority,
        weights=weights,
    )
    if not queue:
        raise SystemExit(
            "no records met the review threshold"
        )

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    dump_jsonl(
        iter(queue),
        output,
    )

    manifest_path = output.with_suffix(
        output.suffix
        + ".manifest.json"
    )
    output_manifest = write_manifest(
        output,
        manifest_path,
    )
    summary_path = output.with_suffix(
        output.suffix
        + ".review.json"
    )
    summary = {
        "version": REVIEW_QUEUE_VERSION,
        "input": dataset_manifest(
            args.input
        ),
        "output": output_manifest,
        "weights": {
            "correction": weights.correction,
            "inconsistency": (
                weights.inconsistency
            ),
            "disagreement": (
                weights.disagreement
            ),
            "uncertainty": weights.uncertainty,
        },
        "min_priority": args.min_priority,
        "limit": args.limit,
        "queue": _queue_summary(queue),
    }
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "manifest": str(
                    manifest_path
                ),
                "summary": str(
                    summary_path
                ),
                **summary["queue"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
