import pytest

from my_jev.schema import DecisionRecord, QuestionSpec, QuestionType


def test_noul_gets_implicit_options():
    question = QuestionSpec(
        type=QuestionType.NOUL,
        instructions="Is it urgent?",
    )
    assert question.options == ["false", "true"]


def test_choice_requires_options():
    with pytest.raises(ValueError):
        QuestionSpec(
            type=QuestionType.CHOICE,
            instructions="Where?",
        )


def test_record_validates_target_cardinality():
    with pytest.raises(ValueError):
        DecisionRecord.model_validate(
            {
                "state": "x",
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "route it",
                        "options": ["a", "b"],
                    }
                },
                "targets": {
                    "route": {"distribution": [1.0]}
                },
            }
        )
