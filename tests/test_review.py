import pytest

from my_jev.agent_policy import (
    AgentPolicyState,
    build_agent_policy_record,
)
from my_jev.review import (
    build_review_queue,
    legacy_disagreement,
    load_review_records,
    model_uncertainty,
    score_review_record,
)


def _record(
    intent_id: str,
    *,
    scores=None,
    legacy_action="answer_inline",
    shadow_action="answer_inline",
    violations=None,
    correction_fields=None,
):
    record = build_agent_policy_record(
        AgentPolicyState(
            utterance=f"request {intent_id}",
            metadata={
                "assistx_intent_id": (
                    intent_id
                )
            },
        )
    )
    record.metadata.update(
        {
            "source_intent_id": intent_id,
            "family_id": intent_id,
            "legacy_policy_action": (
                legacy_action
            ),
            "shadow_policy_action": (
                shadow_action
            ),
            "shadow_scores": (
                scores or {}
            ),
            "shadow_consistency_violations": (
                violations or []
            ),
            "correction_evidence_fields": (
                correction_fields or []
            ),
            "label_status": "unlabeled",
        }
    )
    return record


def test_uncertainty_uses_full_shadow_distributions():
    record = _record(
        "uncertain",
        scores={
            "route": {
                "chat": 0.5,
                "act": 0.5,
            },
            "needs_tools": 0.5,
        },
    )

    assert model_uncertainty(
        record
    ) == pytest.approx(1.0)


def test_uncertainty_falls_back_to_route_confidence():
    record = _record("fallback")
    record.metadata[
        "shadow_model_route_confidence"
    ] = 0.8

    assert model_uncertainty(
        record
    ) == pytest.approx(0.2)


def test_legacy_disagreement_checks_policy_action():
    record = _record(
        "disagree",
        legacy_action="answer_inline",
        shadow_action="create_task_graph",
    )

    assert legacy_disagreement(record)


def test_review_score_prioritizes_corrections_and_inconsistency():
    record = _record(
        "high-value",
        scores={
            "route": {
                "chat": 0.5,
                "act": 0.5,
            }
        },
        legacy_action="answer_inline",
        shadow_action="direct_action",
        violations=[
            "act_without_tools",
            "act_with_none_scope",
        ],
        correction_fields=[
            "user_correction"
        ],
    )

    score = score_review_record(
        record
    )

    assert score.correction_evidence
    assert score.disagreement
    assert score.inconsistency_count == 2
    assert score.uncertainty == pytest.approx(
        1.0
    )
    assert score.priority == pytest.approx(
        11.0
    )
    assert (
        "correction_evidence"
        in score.reasons
    )


def test_review_queue_is_ranked_and_carries_audit_metadata():
    low = _record(
        "low",
        scores={
            "route": {
                "chat": 0.99,
                "act": 0.01,
            }
        },
    )
    disagreement = _record(
        "disagreement",
        scores={
            "route": {
                "chat": 0.99,
                "act": 0.01,
            }
        },
        shadow_action="create_task_graph",
    )
    correction = _record(
        "correction",
        scores={
            "route": {
                "chat": 0.99,
                "act": 0.01,
            }
        },
        correction_fields=[
            "user_correction"
        ],
    )

    queue = build_review_queue(
        [
            low,
            disagreement,
            correction,
        ],
        limit=2,
    )

    assert [
        item.metadata[
            "source_intent_id"
        ]
        for item in queue
    ] == [
        "correction",
        "disagreement",
    ]
    assert queue[0].metadata[
        "active_review"
    ]["rank"] == 1
    assert queue[1].metadata[
        "active_review"
    ]["rank"] == 2
    assert (
        queue[0].targets is None
    )


def test_min_priority_filters_clean_rows():
    clean = _record(
        "clean",
        scores={
            "route": {
                "chat": 1.0,
                "act": 0.0,
            },
            "needs_tools": 0.0,
        },
    )

    assert (
        build_review_queue(
            [clean],
            min_priority=0.1,
        )
        == []
    )


def test_review_loader_accepts_shadow_replay_rows(tmp_path):
    replay = {
        "schema_version": 1,
        "mode": "shadow_replay",
        "dispatch_allowed": False,
        "intent_id": "intent-replay",
        "source": "signal",
        "checkpoint": "runs/policy/best",
        "temperature": 1.0,
        "request": {
            "state": {
                "utterance": "Check CI.",
                "source": "signal",
                "metadata": {
                    "assistx_intent_id": (
                        "intent-replay"
                    )
                },
            },
            "constraints": {},
        },
        "legacy": {
            "classification": "query",
            "policy_action": "answer_inline",
        },
        "candidate": {
            "model_route": "act",
            "disposition": "propose_action",
            "classification": "task",
            "policy_action": "review_dispatch",
            "consistency_violations": [
                "act_with_none_scope"
            ],
        },
        "candidate_response": {
            "checkpoint": "runs/policy/best",
            "scores": {
                "route": {
                    "chat": 0.45,
                    "act": 0.55,
                },
                "needs_tools": 0.5,
            },
            "resolved": {
                "model_route": "act",
                "model_route_confidence": 0.55,
                "disposition": "propose_action",
                "consistency_violations": [
                    "act_with_none_scope"
                ],
            },
            "assistx": {
                "classification": "task",
                "policy_action": "review_dispatch",
            },
        },
        "correction_evidence_fields": [
            "user_correction"
        ],
    }
    path = tmp_path / "replay.jsonl"
    import json

    path.write_text(
        json.dumps(replay) + "\n",
        encoding="utf-8",
    )

    records = load_review_records(
        path
    )
    assert len(records) == 1
    record = records[0]
    assert (
        record.metadata[
            "evidence_mode"
        ]
        == "shadow_replay"
    )
    assert record.metadata[
        "shadow_scores"
    ]["needs_tools"] == 0.5
    score = score_review_record(
        record
    )
    assert score.correction_evidence
    assert score.disagreement
    assert score.inconsistency_count == 1
