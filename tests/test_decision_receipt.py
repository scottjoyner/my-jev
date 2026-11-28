from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from my_jev.agent_policy import (
    AgentPolicyState,
    agent_policy_questions,
)
from my_jev.decision_receipt import (
    DecisionProvider,
    DecisionReceipt,
    build_decision_receipt,
    deterministic_receipt_bytes,
)
from my_jev.schema import (
    DecisionRecord,
    QuestionSpec,
    QuestionType,
)
from my_jev.server import (
    AgentPolicyRequest,
    PolicyRuntime,
)


def sample_record() -> DecisionRecord:
    return DecisionRecord(
        state='{"ticket":"refund failed"}',
        questions={
            "route": QuestionSpec(
                type=QuestionType.CHOICE,
                instructions="Choose the route.",
                options=["answer", "review"],
            ),
            "urgent": QuestionSpec(
                type=QuestionType.NOUL,
                instructions="Is this urgent?",
            ),
            "severity": QuestionSpec(
                type=QuestionType.SCORE,
                instructions="Rate severity.",
                options=["low", "medium", "high"],
            ),
        },
    )


def sample_predictions() -> list[dict[str, object]]:
    return [
        {
            "record_index": 0,
            "question": "route",
            "type": "choice",
            "options": ["answer", "review"],
            "probabilities": [0.8, 0.2],
            "choice": "answer",
            "confidence": 0.8,
            "temperature": 1.0,
        },
        {
            "record_index": 0,
            "question": "urgent",
            "type": "noul",
            "options": ["false", "true"],
            "probabilities": [0.25, 0.75],
            "choice": "true",
            "confidence": 0.75,
            "noul": 0.75,
            "temperature": 1.0,
        },
        {
            "record_index": 0,
            "question": "severity",
            "type": "score",
            "options": ["low", "medium", "high"],
            "probabilities": [0.1, 0.3, 0.6],
            "choice": "high",
            "confidence": 0.6,
            "score": 0.75,
            "temperature": 1.0,
        },
    ]


def provider() -> DecisionProvider:
    return DecisionProvider(
        provider_id="my-jev",
        provider_version="test",
        model_id="fixture",
        model_version="fixture-v1",
        model_artifact_sha256="a" * 64,
        runtime={"device": "cpu"},
    )


def test_receipt_is_deterministic_and_binds_all_model_visible_inputs():
    receipt = build_decision_receipt(
        record=sample_record(),
        predictions=sample_predictions(),
        provider=provider(),
        latency_ms=12.5,
        resolver_result="answer",
    )
    raw = deterministic_receipt_bytes(receipt)
    round_trip = deterministic_receipt_bytes(
        json.loads(raw.decode("utf-8"))
    )

    assert raw == round_trip
    assert receipt.input_sha256 != receipt.question_schema_sha256
    assert receipt.question_schema_sha256 != receipt.candidate_set_sha256
    assert len(receipt.response_sha256) == 64
    assert [item.name for item in receipt.questions] == [
        "route",
        "urgent",
        "severity",
    ]
    assert receipt.authority.dispatch_allowed is False
    assert receipt.authority.mutation_allowed is False


def test_receipt_rejects_authority_widening_and_extra_fields():
    payload = build_decision_receipt(
        record=sample_record(),
        predictions=sample_predictions(),
        provider=provider(),
        latency_ms=1.0,
    ).model_dump(mode="json")

    payload["authority"]["dispatch_allowed"] = True
    with pytest.raises(ValidationError):
        DecisionReceipt.model_validate(payload)

    payload = build_decision_receipt(
        record=sample_record(),
        predictions=sample_predictions(),
        provider=provider(),
        latency_ms=1.0,
    ).model_dump(mode="json")
    payload["surprise"] = "field"
    with pytest.raises(ValidationError):
        DecisionReceipt.model_validate(payload)


def test_receipt_rejects_probability_and_question_mismatch():
    malformed = sample_predictions()
    malformed[0] = {
        **malformed[0],
        "probabilities": [0.7, 0.2],
        "confidence": 0.7,
    }
    with pytest.raises(ValidationError, match="sum to 1"):
        build_decision_receipt(
            record=sample_record(),
            predictions=malformed,
            provider=provider(),
            latency_ms=1.0,
        )

    missing = sample_predictions()[:-1]
    with pytest.raises(ValueError, match="question mismatch"):
        build_decision_receipt(
            record=sample_record(),
            predictions=missing,
            provider=provider(),
            latency_ms=1.0,
        )


class FakePolicyModel:
    def predict(
        self,
        records: list[DecisionRecord],
        *,
        temperature: float = 1.0,
    ) -> list[dict[str, object]]:
        record = records[0]
        preferred = {
            "route": "chat",
            "needs_tools": "false",
            "needs_task_graph": "false",
            "context_sufficient": "true",
            "external_effect": "false",
            "approval_likely": "false",
            "action_scope": "none",
            "risk": "low",
            "delegation": "self",
            "response_depth": "normal",
        }
        results: list[dict[str, object]] = []
        for name, question in record.questions.items():
            options = list(question.options or [])
            choice = preferred[name]
            selected = options.index(choice)
            remainder = 0.1 / (len(options) - 1)
            probabilities = [remainder for _ in options]
            probabilities[selected] = 0.9
            payload: dict[str, object] = {
                "record_index": 0,
                "question": name,
                "type": question.type.value,
                "options": options,
                "probabilities": probabilities,
                "choice": choice,
                "confidence": 0.9,
                "temperature": temperature,
            }
            if question.type == QuestionType.NOUL:
                payload["noul"] = probabilities[1]
            elif question.type == QuestionType.SCORE:
                denominator = len(options) - 1
                payload["score"] = sum(
                    (index / denominator) * probability
                    for index, probability in enumerate(probabilities)
                )
            results.append(payload)
        return results


def test_policy_service_adds_receipt_without_widening_authority():
    runtime = PolicyRuntime(
        model=FakePolicyModel(),
        temperature=1.0,
        checkpoint="fixture-checkpoint",
        device="cpu",
        checkpoint_sha256="b" * 64,
    )
    payload = runtime.agent_policy(
        AgentPolicyRequest(
            state=AgentPolicyState(
                utterance="Hello there.",
            )
        )
    )

    receipt = DecisionReceipt.model_validate(
        payload["decision_receipt"]
    )
    assert payload["contract"] == "assistx-agent-policy-v1"
    assert receipt.provider.provider_id == "my-jev"
    assert receipt.provider.model_artifact_sha256 == "b" * 64
    assert receipt.resolver_result == "chat"
    assert len(receipt.questions) == len(agent_policy_questions())
    assert receipt.authority.dispatch_allowed is False
    assert receipt.authority.approval_granted is False
    assert receipt.authority.claim_acquired is False
    assert receipt.authority.mutation_allowed is False
    assert receipt.authority.routing_authority_changed is False
