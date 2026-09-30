from my_jev.causal_scalar import (
    causal_candidate_text,
)
from my_jev.schema import (
    QuestionSpec,
    QuestionType,
)


def test_causal_candidate_text_marks_typed_choice():
    question = QuestionSpec(
        type=QuestionType.CHOICE,
        instructions="What should Hermes do?",
        options=[
            "chat",
            "act",
        ],
    )
    text = causal_candidate_text(
        "user asked to inspect CI",
        question,
        "act",
        1,
    )

    assert "STATE:" in text
    assert "user asked to inspect CI" in text
    assert "QUESTION TYPE:" in text
    assert "choice" in text
    assert "choice option: act" in text


def test_causal_candidate_text_preserves_score_order():
    question = QuestionSpec(
        type=QuestionType.SCORE,
        instructions="How risky is this?",
        options=[
            "low",
            "moderate",
            "high",
        ],
    )
    text = causal_candidate_text(
        "restart production",
        question,
        "high",
        2,
    )

    assert "ordered level 3 of 3: high" in text
