from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .calibration import load_temperature
from .checkpoint import load_checkpoint
from .data import load_jsonl
from .schema import DecisionRecord

REPORT_SCHEMA = "my-jev-assistx-policy-heldout-eval-v1"


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def evaluate_policy_predictions(
    records: list[DecisionRecord],
    predictions: list[dict[str, object]],
) -> dict[str, Any]:
    by_index: dict[int, dict[str, object]] = {}
    for prediction in predictions:
        if str(prediction.get("question") or "") != "execution_policy":
            continue
        index = int(prediction.get("record_index") or 0)
        if index in by_index:
            raise ValueError(
                f"duplicate execution_policy prediction for record {index}"
            )
        by_index[index] = prediction

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if record.metadata.get("domain") != "assistx_inference_policy":
            raise ValueError(
                f"record {index} is not an AssistX inference-policy record"
            )
        prediction = by_index.get(index)
        if prediction is None:
            raise ValueError(
                f"missing execution_policy prediction for record {index}"
            )

        question = record.questions.get("execution_policy")
        if question is None:
            raise ValueError(
                f"record {index} has no execution_policy question"
            )
        options = list(question.options or [])
        predicted_options = [
            str(value)
            for value in prediction.get("options", [])
        ]
        if predicted_options != options:
            raise ValueError(
                f"record {index} prediction option ordering drift"
            )

        choice = str(prediction.get("choice") or "")
        if choice not in options:
            raise ValueError(
                f"record {index} prediction chose unknown policy option"
            )
        evidence = record.metadata.get("candidate_evidence")
        if not isinstance(evidence, dict):
            raise ValueError(
                f"record {index} has no candidate_evidence"
            )
        selected = evidence.get(choice)
        if not isinstance(selected, dict):
            raise ValueError(
                f"record {index} choice has no frozen candidate evidence"
            )
        selected_wall = selected.get("wall_ms")
        best_wall = record.metadata.get("best_wall_ms")
        if not isinstance(selected_wall, (int, float)):
            raise ValueError(
                f"record {index} selected wall_ms is invalid"
            )
        if not isinstance(best_wall, (int, float)) or float(best_wall) <= 0:
            raise ValueError(
                f"record {index} best_wall_ms is invalid"
            )
        selected_wall = float(selected_wall)
        best_wall = float(best_wall)
        if selected_wall < best_wall:
            raise ValueError(
                f"record {index} selected wall_ms beats frozen oracle"
            )

        near_best = {
            str(value)
            for value in record.metadata.get(
                "near_best_signature_ids",
                [],
            )
        }
        probabilities = prediction.get("probabilities")
        if not isinstance(probabilities, list) or len(probabilities) != len(options):
            raise ValueError(
                f"record {index} prediction probabilities are invalid"
            )
        numeric_probabilities = [float(value) for value in probabilities]
        if any(value < 0 for value in numeric_probabilities):
            raise ValueError(
                f"record {index} prediction probability is negative"
            )
        total_probability = sum(numeric_probabilities)
        if abs(total_probability - 1.0) > 1e-4:
            raise ValueError(
                f"record {index} prediction probabilities do not sum to one"
            )

        expected_wall = 0.0
        for option, probability in zip(
            options,
            numeric_probabilities,
            strict=True,
        ):
            option_evidence = evidence.get(option)
            if not isinstance(option_evidence, dict):
                raise ValueError(
                    f"record {index} option has no candidate evidence: {option}"
                )
            wall = option_evidence.get("wall_ms")
            if not isinstance(wall, (int, float)):
                raise ValueError(
                    f"record {index} option wall_ms is invalid: {option}"
                )
            expected_wall += probability * float(wall)

        regret_ms = selected_wall - best_wall
        regret_ratio = selected_wall / best_wall
        expected_regret_ms = expected_wall - best_wall
        rows.append(
            {
                "record_index": index,
                "record_id": record.metadata.get("assistx_record_id"),
                "family_id": record.metadata.get("family_id"),
                "context_tokens": record.metadata.get("context_tokens"),
                "choice": choice,
                "confidence": float(
                    prediction.get("confidence") or max(numeric_probabilities)
                ),
                "near_best_hit": choice in near_best,
                "selected_wall_ms": selected_wall,
                "oracle_wall_ms": best_wall,
                "regret_ms": regret_ms,
                "regret_ratio": regret_ratio,
                "expected_wall_ms": expected_wall,
                "expected_regret_ms": expected_regret_ms,
            }
        )

    regrets = [float(row["regret_ms"]) for row in rows]
    ratios = [float(row["regret_ratio"]) for row in rows]
    expected_regrets = [
        float(row["expected_regret_ms"]) for row in rows
    ]
    selected_total = sum(float(row["selected_wall_ms"]) for row in rows)
    oracle_total = sum(float(row["oracle_wall_ms"]) for row in rows)
    return {
        "schema": REPORT_SCHEMA,
        "record_count": len(rows),
        "near_best_hit_rate": (
            sum(1 for row in rows if row["near_best_hit"]) / len(rows)
            if rows
            else 0.0
        ),
        "selected_latency_ms_total": selected_total,
        "oracle_latency_ms_total": oracle_total,
        "aggregate_regret_ms": selected_total - oracle_total,
        "aggregate_regret_ratio": (
            selected_total / oracle_total
            if oracle_total > 0
            else None
        ),
        "mean_regret_ms": mean(regrets) if regrets else None,
        "p50_regret_ms": _percentile(regrets, 0.50),
        "p95_regret_ms": _percentile(regrets, 0.95),
        "mean_regret_ratio": mean(ratios) if ratios else None,
        "mean_expected_regret_ms": (
            mean(expected_regrets) if expected_regrets else None
        ),
        "records": rows,
        "authority": {
            "evidence_only": True,
            "dispatch_allowed": False,
            "runtime_authority_changed": False,
        },
        "note": (
            "Held-out counterfactual evaluation only. The selected policy "
            "is scored against frozen evidence and is never dispatched."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate my-jev execution-policy recommendations against "
            "frozen held-out AssistX counterfactual latency evidence."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--device")
    parser.add_argument("--output")
    args = parser.parse_args()

    import torch

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_checkpoint(args.checkpoint, device=device)
    temperature = load_temperature(args.calibration)
    records = load_jsonl(args.data)
    predictions = model.predict(
        records,
        temperature=temperature,
    )
    report = evaluate_policy_predictions(records, predictions)
    report["checkpoint"] = args.checkpoint
    report["calibration"] = args.calibration
    report["temperature"] = temperature
    report["device"] = str(device)

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
