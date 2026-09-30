import pytest

from my_jev.adjudicate import (
    PolicyAnnotation,
    adjudicate_records,
    apply_annotations,
)
from my_jev.agent_policy import (
    AgentPolicyState,
    build_agent_policy_record,
)


def _record(intent_id: str = "intent-7"):
    state = AgentPolicyState(
        utterance="Check CI and fix the failing workflow.",
        source="signal",
        metadata={"assistx_intent_id": intent_id},
    )
    record = build_agent_policy_record(state)
    record.metadata["source_intent_id"] = intent_id
    record.metadata["family_id"] = intent_id
    record.metadata["label_status"] = "unlabeled"
    return record


def test_repeated_annotations_become_soft_targets():
    record = _record()
    annotations = [
        PolicyAnnotation(
            source_intent_id="intent-7",
            source="operator",
            labels={
                "route": "act",
                "needs_tools": True,
            },
            weight=3.0,
        ),
        PolicyAnnotation(
            source_intent_id="intent-7",
            source="outcome_verifier",
            labels={
                "route": "chat",
                "needs_tools": True,
            },
            weight=1.0,
        ),
    ]

    result = apply_annotations(
        record,
        annotations,
    )

    assert result.targets is not None
    assert result.targets["route"].distribution == pytest.approx(
        [0.25, 0.0, 0.75, 0.0, 0.0, 0.0]
    )
    assert result.targets["needs_tools"].distribution == pytest.approx(
        [0.0, 1.0]
    )
    assert result.metadata["label_status"] == "partially_adjudicated"
    assert result.metadata["annotation_count"] == 2
    assert result.metadata["label_sources"] == {
        "operator": 1,
        "outcome_verifier": 1,
    }


def test_partial_annotations_leave_other_questions_unlabeled():
    record = _record()
    result = apply_annotations(
        record,
        [
            PolicyAnnotation(
                source_intent_id="intent-7",
                labels={"risk": "high"},
            )
        ],
    )

    assert result.targets is not None
    assert set(result.targets) == {"risk"}
    assert result.metadata["adjudicated_questions"] == ["risk"]
    assert result.metadata["label_coverage"] == pytest.approx(0.1)


def test_invalid_option_is_rejected():
    record = _record()
    with pytest.raises(
        ValueError,
        match="not one of",
    ):
        apply_annotations(
            record,
            [
                PolicyAnnotation(
                    source_intent_id="intent-7",
                    labels={"route": "teleport"},
                )
            ],
        )


def test_unmatched_annotation_ids_fail_closed():
    records = [_record()]
    annotations = [
        PolicyAnnotation(
            source_intent_id="missing-intent",
            labels={"route": "chat"},
        )
    ]

    with pytest.raises(
        ValueError,
        match="do not match any input record",
    ):
        adjudicate_records(
            records,
            annotations,
        )


def test_unlabeled_records_can_be_retained():
    records = [
        _record("intent-7"),
        _record("intent-8"),
    ]
    annotations = [
        PolicyAnnotation(
            source_intent_id="intent-7",
            labels={"route": "act"},
        )
    ]

    result = adjudicate_records(
        records,
        annotations,
        keep_unlabeled=True,
    )

    assert len(result) == 2
    assert result[0].targets is not None
    assert result[1].targets is None
