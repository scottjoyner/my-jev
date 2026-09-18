from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class QuestionType(str, Enum):
    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"


class QuestionSpec(BaseModel):
    type: QuestionType
    instructions: str
    options: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_options(self) -> "QuestionSpec":
        if self.type == QuestionType.NOUL:
            if self.options not in (None, ["false", "true"]):
                raise ValueError("noul options are implicit: false/true")
            self.options = ["false", "true"]
            return self

        if not self.options or len(self.options) < 2:
            raise ValueError(f"{self.type.value} requires at least two options")
        if len(self.options) > 255:
            raise ValueError("v0 caps option cardinality at 255")
        if len(set(self.options)) != len(self.options):
            raise ValueError("options must be unique")
        return self


class TargetSpec(BaseModel):
    """Training target for one question.

    Supply either a hard target (index) or a probability distribution. A distribution is
    useful for consensus labels, repeated human labels, or teacher/reference probabilities.
    """

    index: int | None = None
    distribution: list[float] | None = None

    @model_validator(mode="after")
    def validate_target(self) -> "TargetSpec":
        if self.index is None and self.distribution is None:
            raise ValueError("target needs index or distribution")
        if self.distribution is not None:
            if not self.distribution:
                raise ValueError("distribution cannot be empty")
            if any(p < 0 for p in self.distribution):
                raise ValueError("distribution probabilities must be >= 0")
            total = sum(self.distribution)
            if total <= 0:
                raise ValueError("distribution must have positive mass")
            self.distribution = [p / total for p in self.distribution]
        return self


class DecisionRecord(BaseModel):
    state: str
    questions: dict[str, QuestionSpec]
    targets: dict[str, TargetSpec] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_targets(self) -> "DecisionRecord":
        if not self.questions:
            raise ValueError("record must contain at least one question")
        if self.targets is None:
            return self

        unknown = set(self.targets) - set(self.questions)
        if unknown:
            raise ValueError(f"targets reference unknown questions: {sorted(unknown)}")

        for name, target in self.targets.items():
            size = len(self.questions[name].options or [])
            if target.index is not None and not (0 <= target.index < size):
                raise ValueError(f"target index out of range for {name}")
            if target.distribution is not None and len(target.distribution) != size:
                raise ValueError(f"target distribution length mismatch for {name}")
        return self
