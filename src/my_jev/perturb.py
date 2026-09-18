from __future__ import annotations

from copy import deepcopy

from .schema import DecisionRecord, QuestionType, TargetSpec


def reverse_choice_options(
    record: DecisionRecord,
) -> DecisionRecord:
    """Reverse Choice option order while preserving the semantic target.

    Noul has a fixed false/true order and Score order is semantically meaningful,
    so only Choice questions are perturbed.
    """
    payload = record.model_dump(mode="python")
    questions = payload["questions"]
    targets = payload.get("targets")

    for name, question in questions.items():
        if question["type"] != QuestionType.CHOICE.value:
            continue

        options = list(question["options"])
        reversed_options = list(reversed(options))
        question["options"] = reversed_options

        if not targets or name not in targets:
            continue

        target = targets[name]
        if target.get("index") is not None:
            old_index = int(target["index"])
            chosen = options[old_index]
            target["index"] = reversed_options.index(
                chosen
            )

        distribution = target.get(
            "distribution"
        )
        if distribution is not None:
            by_option = dict(
                zip(
                    options,
                    distribution,
                    strict=True,
                )
            )
            target["distribution"] = [
                by_option[option]
                for option in reversed_options
            ]

    payload["metadata"] = deepcopy(
        payload.get("metadata") or {}
    )
    payload["metadata"][
        "perturbation"
    ] = "reverse_choice_options"
    return DecisionRecord.model_validate(
        payload
    )


def align_probabilities_by_option(
    *,
    reference_options: list[str],
    candidate_options: list[str],
    candidate_probabilities: list[float],
) -> list[float]:
    if set(reference_options) != set(
        candidate_options
    ):
        raise ValueError(
            "option sets differ and cannot be aligned"
        )
    by_option = dict(
        zip(
            candidate_options,
            candidate_probabilities,
            strict=True,
        )
    )
    return [
        float(by_option[option])
        for option in reference_options
    ]
