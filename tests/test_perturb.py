import pytest

from my_jev.perturb import (
    align_probabilities_by_option,
    reverse_choice_options,
)
from my_jev.schema import DecisionRecord


def _record() -> DecisionRecord:
    return DecisionRecord.model_validate(
        {
            "state": "state",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "route?",
                    "options": [
                        "chat",
                        "act",
                        "clarify",
                    ],
                },
                "risk": {
                    "type": "score",
                    "instructions": "risk?",
                    "options": [
                        "low",
                        "moderate",
                        "high",
                    ],
                },
                "binary": {
                    "type": "noul",
                    "instructions": "binary?",
                },
            },
            "targets": {
                "route": {
                    "distribution": [
                        0.1,
                        0.8,
                        0.1,
                    ]
                },
                "risk": {"index": 1},
                "binary": {"index": 1},
            },
        }
    )


def test_reverse_choice_options_preserves_semantic_target():
    original = _record()
    perturbed = reverse_choice_options(
        original
    )

    assert (
        perturbed.questions[
            "route"
        ].options
        == [
            "clarify",
            "act",
            "chat",
        ]
    )
    assert perturbed.targets[
        "route"
    ].distribution == pytest.approx(
        [0.1, 0.8, 0.1]
    )

    assert (
        perturbed.questions[
            "risk"
        ].options
        == original.questions[
            "risk"
        ].options
    )
    assert (
        perturbed.questions[
            "binary"
        ].options
        == ["false", "true"]
    )


def test_align_probabilities_restores_reference_order():
    aligned = align_probabilities_by_option(
        reference_options=[
            "chat",
            "act",
            "clarify",
        ],
        candidate_options=[
            "clarify",
            "act",
            "chat",
        ],
        candidate_probabilities=[
            0.1,
            0.8,
            0.1,
        ],
    )
    assert aligned == pytest.approx(
        [0.1, 0.8, 0.1]
    )


def test_align_probabilities_rejects_different_sets():
    with pytest.raises(
        ValueError,
        match="option sets differ",
    ):
        align_probabilities_by_option(
            reference_options=[
                "a",
                "b",
            ],
            candidate_options=[
                "a",
                "c",
            ],
            candidate_probabilities=[
                0.5,
                0.5,
            ],
        )
