from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .schema import DecisionRecord, QuestionType

RECEIPT_SCHEMA = "system-one-decision-receipt-v1"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _ExactModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DecisionAuthority(_ExactModel):
    """Evidence-only authority contract.

    A decision receipt can describe a recommendation and resolver outcome, but it
    cannot encode permission to dispatch, approve, claim, route, or mutate.
    """

    dispatch_allowed: Literal[False] = False
    approval_granted: Literal[False] = False
    claim_acquired: Literal[False] = False
    mutation_allowed: Literal[False] = False
    routing_authority_changed: Literal[False] = False


class DecisionProvider(_ExactModel):
    provider_id: str = Field(min_length=1, max_length=128)
    provider_version: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=512)
    model_version: str | None = Field(default=None, max_length=256)
    model_artifact_sha256: str | None = Field(
        default=None,
        pattern=_SHA256_PATTERN,
    )
    runtime: dict[str, Any] = Field(default_factory=dict)


class DecisionQuestionResult(_ExactModel):
    name: str = Field(min_length=1, max_length=256)
    type: Literal["noul", "choice", "score"]
    options: list[str] = Field(min_length=2, max_length=255)
    probabilities: list[float] = Field(min_length=2, max_length=255)
    choice: str = Field(min_length=1, max_length=4096)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    noul: float | None = Field(default=None, ge=0.0, le=1.0)
    score: float | None = None

    @field_validator("options")
    @classmethod
    def _validate_options(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("decision receipt options must be unique")
        if any(not option for option in value):
            raise ValueError("decision receipt options must be non-empty")
        return value

    @field_validator("probabilities")
    @classmethod
    def _validate_probabilities(cls, value: list[float]) -> list[float]:
        if any(not math.isfinite(item) or item < 0.0 or item > 1.0 for item in value):
            raise ValueError("probabilities must be finite values in [0, 1]")
        if not math.isclose(sum(value), 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("probabilities must sum to 1")
        return value

    @model_validator(mode="after")
    def _validate_result(self) -> DecisionQuestionResult:
        if len(self.options) != len(self.probabilities):
            raise ValueError("option/probability cardinality mismatch")
        if self.choice not in self.options:
            raise ValueError("choice must be one of the supplied options")

        selected = self.probabilities[self.options.index(self.choice)]
        peak = max(self.probabilities)
        if not math.isclose(selected, peak, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("choice must correspond to a maximum-probability option")

        if self.confidence is not None and not math.isclose(
            self.confidence,
            selected,
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            raise ValueError("confidence must match the selected option probability")

        if self.type == QuestionType.NOUL.value:
            if self.options != ["false", "true"]:
                raise ValueError("noul receipt options must be false/true")
            if self.noul is None:
                raise ValueError("noul result requires the true probability")
            if not math.isclose(
                self.noul,
                self.probabilities[1],
                rel_tol=0.0,
                abs_tol=1e-5,
            ):
                raise ValueError("noul must match P(true)")
            if self.score is not None:
                raise ValueError("noul result cannot carry a score")
        elif self.type == QuestionType.SCORE.value:
            if self.score is None or not math.isfinite(self.score):
                raise ValueError("score result requires a finite score")
            if self.noul is not None:
                raise ValueError("score result cannot carry noul")
        else:
            if self.noul is not None or self.score is not None:
                raise ValueError("choice result cannot carry noul/score")
        return self


class DecisionReceipt(_ExactModel):
    schema: Literal["system-one-decision-receipt-v1"] = RECEIPT_SCHEMA
    input_sha256: str = Field(pattern=_SHA256_PATTERN)
    question_schema_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate_set_sha256: str = Field(pattern=_SHA256_PATTERN)
    response_sha256: str = Field(pattern=_SHA256_PATTERN)
    provider: DecisionProvider
    latency_ms: float = Field(ge=0.0)
    questions: list[DecisionQuestionResult] = Field(min_length=1)
    resolver_result: str | None = Field(default=None, max_length=256)
    evidence_only: Literal[True] = True
    authority: DecisionAuthority = Field(default_factory=DecisionAuthority)
    metadata: dict[str, Any] = Field(default_factory=dict)


def canonical_json_bytes(value: Any) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def decision_record_hashes(record: DecisionRecord) -> dict[str, str]:
    """Hash only model-visible decision inputs.

    Record metadata and training targets are intentionally excluded: inference sees
    the state plus the typed question schema. Candidate-set identity is recorded
    separately so option changes can be detected without diffing instructions.
    """

    question_schema = {
        name: {
            "type": question.type.value,
            "instructions": question.instructions,
            "options": list(question.options or []),
        }
        for name, question in record.questions.items()
    }
    candidate_set = {
        name: list(question.options or [])
        for name, question in record.questions.items()
    }
    return {
        "input_sha256": sha256_json({"state": record.state}),
        "question_schema_sha256": sha256_json(question_schema),
        "candidate_set_sha256": sha256_json(candidate_set),
    }


def _prediction_by_question(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for prediction in predictions:
        name = str(prediction.get("question", ""))
        if not name:
            raise ValueError("prediction is missing question name")
        if name in indexed:
            raise ValueError(f"duplicate prediction for question: {name}")
        indexed[name] = prediction
    return indexed


def _question_result(
    name: str,
    prediction: Mapping[str, Any],
) -> DecisionQuestionResult:
    question_type = str(prediction["type"])
    options = [str(item) for item in prediction["options"]]
    probabilities = [float(item) for item in prediction["probabilities"]]
    choice = str(prediction["choice"])

    confidence = prediction.get("confidence")
    noul = prediction.get("noul")
    score = prediction.get("score")

    if question_type == QuestionType.NOUL.value and noul is None:
        if options != ["false", "true"] or len(probabilities) != 2:
            raise ValueError("cannot derive noul from malformed prediction")
        noul = probabilities[1]

    if question_type == QuestionType.SCORE.value and score is None:
        if len(probabilities) < 2:
            raise ValueError("cannot derive score from fewer than two options")
        denominator = len(probabilities) - 1
        score = sum(
            (index / denominator) * probability
            for index, probability in enumerate(probabilities)
        )

    return DecisionQuestionResult(
        name=name,
        type=question_type,
        options=options,
        probabilities=probabilities,
        choice=choice,
        confidence=(None if confidence is None else float(confidence)),
        noul=(None if noul is None else float(noul)),
        score=(None if score is None else float(score)),
    )


def build_decision_receipt(
    *,
    record: DecisionRecord,
    predictions: Sequence[Mapping[str, Any]],
    provider: DecisionProvider | Mapping[str, Any],
    latency_ms: float,
    resolver_result: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DecisionReceipt:
    indexed = _prediction_by_question(predictions)
    expected = set(record.questions)
    observed = set(indexed)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"prediction question mismatch: missing={missing}, extra={extra}"
        )

    results: list[DecisionQuestionResult] = []
    for name, question in record.questions.items():
        result = _question_result(name, indexed[name])
        if result.type != question.type.value:
            raise ValueError(f"prediction type mismatch for question: {name}")
        if result.options != list(question.options or []):
            raise ValueError(f"prediction options mismatch for question: {name}")
        results.append(result)

    hashes = decision_record_hashes(record)
    response_sha256 = sha256_json(
        [result.model_dump(mode="json") for result in results]
    )
    provider_model = (
        provider
        if isinstance(provider, DecisionProvider)
        else DecisionProvider.model_validate(provider)
    )
    return DecisionReceipt(
        **hashes,
        response_sha256=response_sha256,
        provider=provider_model,
        latency_ms=float(latency_ms),
        questions=results,
        resolver_result=resolver_result,
        metadata=dict(metadata or {}),
    )


def deterministic_receipt_bytes(
    receipt: DecisionReceipt | Mapping[str, Any],
) -> bytes:
    value = (
        receipt
        if isinstance(receipt, DecisionReceipt)
        else DecisionReceipt.model_validate(receipt)
    )
    return canonical_json_bytes(value)
