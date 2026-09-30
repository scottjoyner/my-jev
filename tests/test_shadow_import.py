import json

from my_jev.shadow_import import (
    load_shadow_export,
    shadow_row_to_record,
)


def _row():
    evidence = {
        "request": {
            "state": {
                "utterance": "Check the current CI run.",
                "conversation_summary": "",
                "source": "signal",
                "speaker_id": "speaker-primary",
                "speaker_verified": True,
                "foreground": True,
                "active_work": [],
                "pending_approvals": [],
                "available_capabilities": [
                    "chat",
                    "tools",
                ],
                "available_tools": [
                    "web_search"
                ],
                "actions_allowed": False,
                "external_actions_allowed": False,
                "privileged_actions_allowed": False,
                "metadata": {
                    "assistx_intent_id": "intent-7"
                },
            },
            "constraints": {
                "speaker_verified": True
            },
        },
        "response": {
            "contract": "assistx-agent-policy-v1",
            "checkpoint": "runs/policy/best",
            "temperature": 1.2,
            "scores": {
                "route": {
                    "chat": 0.30,
                    "act": 0.70,
                },
                "needs_tools": 0.85,
            },
            "resolved": {
                "model_route": "act",
                "model_route_confidence": 0.70,
                "disposition": "propose_action",
                "consistency_violations": [
                    "act_with_none_scope"
                ],
                "reasons": [
                    "runtime policy does not permit actions"
                ],
            },
            "assistx": {
                "classification": "task",
                "policy_action": "review_dispatch"
            },
        },
        "legacy": {
            "classification": "query",
            "policy_action": "answer_inline",
        },
    }
    return {
        "intent_id": "intent-7",
        "source": "signal",
        "legacy_classification": "query",
        "legacy_policy_action": "answer_inline",
        "policy_shadow_json": json.dumps(
            evidence
        ),
        "created_tasks": [],
        "redacted": False,
        "user_correction": True,
    }


def test_shadow_row_is_unlabeled_even_when_shadow_has_prediction():
    record = shadow_row_to_record(
        _row()
    )

    assert record.targets is None
    assert record.metadata[
        "label_status"
    ] == "unlabeled"
    assert record.metadata[
        "legacy_classification"
    ] == "query"
    assert record.metadata[
        "shadow_model_route"
    ] == "act"
    assert record.metadata[
        "shadow_model_route_confidence"
    ] == 0.70
    assert record.metadata[
        "shadow_disposition"
    ] == "propose_action"
    assert record.metadata[
        "shadow_classification"
    ] == "task"
    assert record.metadata[
        "shadow_scores"
    ]["needs_tools"] == 0.85
    assert record.metadata[
        "shadow_consistency_violations"
    ] == [
        "act_with_none_scope"
    ]
    assert record.metadata[
        "correction_evidence_fields"
    ] == [
        "user_correction"
    ]
    assert record.state


def test_shadow_export_loader_accepts_jsonl(tmp_path):
    path = tmp_path / "shadow.jsonl"
    path.write_text(
        json.dumps(_row()) + "\n",
        encoding="utf-8",
    )
    records = load_shadow_export(path)

    assert len(records) == 1
    assert (
        records[0].metadata[
            "source_intent_id"
        ]
        == "intent-7"
    )
