from __future__ import annotations

from .schema import QuestionSpec, QuestionType


PROMPT_CONTRACT_VERSION = (
    "my-jev-typed-decision-prompt-v1"
)


def candidate_text(
    question: QuestionSpec,
    option: str,
    option_index: int,
) -> str:
    if (
        question.type
        == QuestionType.NOUL
    ):
        return (
            f"Question: "
            f"{question.instructions}\n"
            f"Answer: {option}"
        )
    if (
        question.type
        == QuestionType.SCORE
    ):
        total = len(
            question.options
            or []
        )
        return (
            f"Question: "
            f"{question.instructions}\n"
            f"Ordered level "
            f"{option_index + 1} "
            f"of {total}: "
            f"{option}"
        )
    return (
        f"Question: "
        f"{question.instructions}\n"
        f"Option: {option}"
    )
