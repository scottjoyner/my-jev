import json

from my_jev.shadow_replay import replay_export, replay_row


class FakeRuntime:
    checkpoint = "runs/example/checkpoints/best"

    def agent_policy(self, request):
        assert request.state.utterance == "Check CI."
        assert request.constraints.actions_allowed is False
        return {
            "contract": "assistx-agent-policy-v1",
            "checkpoint": self.checkpoint,
            "temperature": 1.25,
            "resolved": {
                "model_route": "act",
                "disposition": "propose_action",
                "consistency_violations": [],
            },
            "assistx": {
                "classification": "task",
                "policy_action": "review_dispatch",
            },
            "predictions": [],
            "scores": {},
            "hermes": {},
        }


def _row():
    evidence = {
        "request": {
            "state": {
                "utterance": "Check CI.",
                "source": "signal",
                "speaker_verified": True,
                "actions_allowed": False,
                "external_actions_allowed": False,
                "privileged_actions_allowed": False,
                "metadata": {
                    "assistx_intent_id": "intent-9",
                },
            },
            "constraints": {
                "speaker_verified": True,
                "actions_allowed": False,
                "external_actions_allowed": False,
                "privileged_actions_allowed": False,
            },
        },
        "response": {
            "resolved": {
                "model_route": "act",
                "disposition": "propose_action",
            }
        },
        "legacy": {
            "classification": "query",
            "policy_action": "answer_inline",
        },
    }
    return {
        "intent_id": "intent-9",
        "source": "signal",
        "policy_shadow_json": json.dumps(evidence),
    }


def test_replay_row_is_explicitly_non_dispatching():
    result = replay_row(_row(), FakeRuntime())

    assert result["mode"] == "shadow_replay"
    assert result["dispatch_allowed"] is False
    assert result["candidate"]["disposition"] == "propose_action"
    assert result["candidate"]["policy_action"] == "review_dispatch"
    assert result["legacy"]["policy_action"] == "answer_inline"
    assert result["candidate_response"]["checkpoint"] == FakeRuntime.checkpoint


def test_replay_export_writes_hashed_summary(tmp_path):
    source = tmp_path / "shadow.jsonl"
    output = tmp_path / "replay.jsonl"
    source.write_text(
        json.dumps(_row()) + "\n" + json.dumps(_row()) + "\n",
        encoding="utf-8",
    )

    manifest = replay_export(
        input_path=source,
        output_path=output,
        runtime=FakeRuntime(),
    )

    assert manifest["dispatch_allowed"] is False
    assert manifest["records"] == 2
    assert len(manifest["input_sha256"]) == 64
    assert len(manifest["output_sha256"]) == 64
    assert manifest["summary"]["dispositions"] == {
        "propose_action": 2
    }
    assert (
        manifest["summary"][
            "legacy_policy_action_agreement_rate"
        ]
        == 0.0
    )
    assert (
        manifest["summary"][
            "previous_shadow_disposition_agreement_rate"
        ]
        == 1.0
    )
    assert (
        manifest["summary"][
            "consistency_violation_record_rate"
        ]
        == 0.0
    )

    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert all(row["dispatch_allowed"] is False for row in rows)
    assert output.with_suffix(".jsonl.manifest.json").is_file()
