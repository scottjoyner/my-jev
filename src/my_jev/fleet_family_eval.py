from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .schema import DecisionRecord, TargetSpec


def _target_index(
    record: DecisionRecord,
    question: str,
) -> int:
    if not record.targets or question not in record.targets:
        raise ValueError(
            f"record is missing target for {question}"
        )
    target: TargetSpec = record.targets[question]
    if target.index is not None:
        return target.index
    if target.distribution is None:
        raise ValueError(
            f"target for {question} has no index or distribution"
        )
    return int(
        np.asarray(
            target.distribution,
            dtype=np.float64,
        ).argmax()
    )


def _target_signature(
    record: DecisionRecord,
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (
            question,
            _target_index(
                record,
                question,
            ),
        )
        for question in sorted(
            record.targets or {}
        )
    )


def _top1(
    prediction: dict[str, float],
) -> str:
    if not prediction:
        raise ValueError(
            "prediction distribution is empty"
        )
    return max(
        prediction,
        key=prediction.get,
    )


def _mean(values: list[float], *, empty: float = 1.0) -> float:
    return float(np.mean(values)) if values else empty


def evaluate_fleet_families(
    records: list[DecisionRecord],
    predictions: list[
        dict[str, dict[str, float]]
    ],
) -> dict[str, object]:
    """Evaluate paired fleet counterfactual families.

    Every complete synthetic family is expected to contain:
    - one node_permutation record with the same verified targets as a base record;
    - one semantic variant whose verified targets differ from that base.

    Predictions are option-name -> probability mappings in each record's native
    question option order. The evaluator never resolves or dispatches workloads.
    """

    if len(records) != len(predictions):
        raise ValueError(
            "records/predictions length mismatch"
        )

    by_family: dict[
        str,
        list[tuple[DecisionRecord, dict[str, dict[str, float]]]],
    ] = defaultdict(list)

    for record, prediction in zip(
        records,
        predictions,
        strict=True,
    ):
        family_id = record.metadata.get(
            "family_id"
        )
        if family_id is None:
            continue
        by_family[str(family_id)].append(
            (record, prediction)
        )

    invariant_top1: list[float] = []
    invariant_abs_delta: list[float] = []
    semantic_new_target_correct: list[float] = []
    semantic_direction_correct: list[float] = []
    semantic_unchanged_top1: list[float] = []
    family_success: list[float] = []
    scenario_totals: dict[str, int] = defaultdict(int)
    scenario_success: dict[str, int] = defaultdict(int)
    incomplete = 0
    complete = 0

    for items in by_family.values():
        permutation = next(
            (
                item
                for item in items
                if item[0].metadata.get("variant")
                == "node_permutation"
            ),
            None,
        )
        if permutation is None:
            incomplete += 1
            continue

        permutation_record, permutation_prediction = permutation
        permutation_signature = _target_signature(
            permutation_record
        )
        base = next(
            (
                item
                for item in items
                if item is not permutation
                and _target_signature(item[0])
                == permutation_signature
            ),
            None,
        )
        if base is None:
            incomplete += 1
            continue

        base_record, base_prediction = base
        semantic = [
            item
            for item in items
            if item is not permutation
            and item is not base
            and _target_signature(item[0])
            != permutation_signature
        ]
        if not semantic:
            incomplete += 1
            continue

        complete += 1
        scenario = str(
            base_record.metadata.get(
                "scenario",
                "unknown",
            )
        )
        scenario_totals[scenario] += 1

        invariant_ok = True
        for question in sorted(
            set(base_prediction)
            & set(permutation_prediction)
        ):
            left = base_prediction[question]
            right = permutation_prediction[question]
            options = sorted(
                set(left) & set(right)
            )
            if not options:
                continue

            top1_agree = float(
                _top1(left)
                == _top1(right)
            )
            invariant_top1.append(
                top1_agree
            )
            invariant_ok = (
                invariant_ok
                and bool(top1_agree)
            )
            invariant_abs_delta.extend(
                abs(
                    float(left[option])
                    - float(right[option])
                )
                for option in options
            )

        changed_ok = True
        family_has_changed_target = False

        for semantic_record, semantic_prediction in semantic:
            common_questions = sorted(
                set(base_record.targets or {})
                & set(semantic_record.targets or {})
                & set(base_prediction)
                & set(semantic_prediction)
            )
            for question in common_questions:
                options = (
                    semantic_record.questions[
                        question
                    ].options
                    or []
                )
                old_index = _target_index(
                    base_record,
                    question,
                )
                new_index = _target_index(
                    semantic_record,
                    question,
                )
                old_option = options[old_index]
                new_option = options[new_index]

                if old_index == new_index:
                    stable = float(
                        _top1(
                            base_prediction[
                                question
                            ]
                        )
                        == _top1(
                            semantic_prediction[
                                question
                            ]
                        )
                    )
                    semantic_unchanged_top1.append(
                        stable
                    )
                    continue

                family_has_changed_target = True

                semantic_top1 = _top1(
                    semantic_prediction[
                        question
                    ]
                )
                new_target_correct = float(
                    semantic_top1
                    == new_option
                )
                semantic_new_target_correct.append(
                    new_target_correct
                )

                before = base_prediction[
                    question
                ]
                after = semantic_prediction[
                    question
                ]
                direction_correct = float(
                    float(
                        after.get(
                            new_option,
                            0.0,
                        )
                    )
                    > float(
                        before.get(
                            new_option,
                            0.0,
                        )
                    )
                    and float(
                        after.get(
                            old_option,
                            0.0,
                        )
                    )
                    < float(
                        before.get(
                            old_option,
                            0.0,
                        )
                    )
                )
                semantic_direction_correct.append(
                    direction_correct
                )
                changed_ok = (
                    changed_ok
                    and bool(
                        new_target_correct
                    )
                    and bool(
                        direction_correct
                    )
                )

        succeeded = (
            invariant_ok
            and family_has_changed_target
            and changed_ok
        )
        family_success.append(
            float(succeeded)
        )
        if succeeded:
            scenario_success[
                scenario
            ] += 1

    scenario_rates = {
        name: {
            "families": total,
            "success_rate": (
                scenario_success[name]
                / total
                if total
                else 1.0
            ),
        }
        for name, total in sorted(
            scenario_totals.items()
        )
    }

    return {
        "schema_version": 1,
        "dispatch_allowed": False,
        "families": len(
            by_family
        ),
        "complete_families": complete,
        "incomplete_families": incomplete,
        "invariant_decisions": len(
            invariant_top1
        ),
        "invariant_top1_agreement": _mean(
            invariant_top1
        ),
        "invariant_mean_abs_probability_delta": _mean(
            invariant_abs_delta,
            empty=0.0,
        ),
        "invariant_max_abs_probability_delta": (
            max(
                invariant_abs_delta
            )
            if invariant_abs_delta
            else 0.0
        ),
        "semantic_changed_decisions": len(
            semantic_new_target_correct
        ),
        "semantic_new_target_accuracy": _mean(
            semantic_new_target_correct
        ),
        "semantic_probability_direction_rate": _mean(
            semantic_direction_correct
        ),
        "semantic_unchanged_decisions": len(
            semantic_unchanged_top1
        ),
        "semantic_unchanged_top1_agreement": _mean(
            semantic_unchanged_top1
        ),
        "family_success_rate": _mean(
            family_success
        ),
        "by_scenario": scenario_rates,
        "authority_boundary": (
            "advisory_only"
        ),
    }


def prediction_from_outputs(
    outputs: list[Any],
) -> dict[str, dict[str, float]]:
    result: dict[
        str,
        dict[str, float],
    ] = {}
    for output in outputs:
        probabilities = output[
            "probabilities"
        ]
        result[str(output["question"])] = {
            str(option): float(probability)
            for option, probability in zip(
                output["options"],
                probabilities,
                strict=True,
            )
        }
    return result
