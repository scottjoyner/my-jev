import json

from my_jev import shadow_evaluation


def test_shadow_evaluation_binds_candidate_and_export(tmp_path, monkeypatch):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "best"
    checkpoint.mkdir(parents=True)
    for path, payload in (
        (checkpoint / "model.pt", b"model"),
        (checkpoint / "my_jev_config.json", b"{}"),
        (run / "calibration.json", b"{}"),
        (run / "manifest.json", b"{}"),
        (run / "r9700-baseline-receipt.json", b"{}"),
    ):
        path.write_bytes(payload)

    export = tmp_path / "shadow.jsonl"
    export.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        shadow_evaluation,
        "validate_baseline_run",
        lambda root, expected_sha=None: {
            "git_sha": "a" * 40,
            "evidence_only": True,
        },
    )

    class Runtime:
        checkpoint = str(checkpoint)

    monkeypatch.setattr(
        shadow_evaluation.PolicyRuntime,
        "load",
        lambda *args, **kwargs: Runtime(),
    )

    replay = {
        "schema_version": 1,
        "mode": "shadow_replay",
        "dispatch_allowed": False,
        "intent_id": "intent-1",
        "source": "test",
        "checkpoint": str(checkpoint),
        "temperature": 1.0,
        "request": {
            "state": {
                "utterance": "Check CI.",
                "source": "test",
                "metadata": {"assistx_intent_id": "intent-1"},
            },
            "constraints": {},
        },
        "legacy": {"classification": "query", "policy_action": "answer_inline"},
        "candidate": {
            "model_route": "chat",
            "disposition": "chat",
            "classification": "query",
            "policy_action": "answer_inline",
            "consistency_violations": [],
        },
        "candidate_response": {
            "checkpoint": str(checkpoint),
            "scores": {"route": {"chat": 0.9, "act": 0.1}},
            "resolved": {
                "model_route": "chat",
                "model_route_confidence": 0.9,
                "disposition": "chat",
                "consistency_violations": [],
            },
            "assistx": {
                "classification": "query",
                "policy_action": "answer_inline",
            },
        },
        "correction_evidence_fields": [],
    }

    def fake_replay(*, input_path, output_path, runtime):
        output_path.write_text(json.dumps(replay) + "\n", encoding="utf-8")
        manifest = output_path.with_suffix(output_path.suffix + ".manifest.json")
        manifest.write_text("{}\n", encoding="utf-8")
        return {"summary": {"records": 1}}

    monkeypatch.setattr(shadow_evaluation, "replay_export", fake_replay)

    receipt = shadow_evaluation.run_shadow_evaluation(
        run_dir=run,
        shadow_export=export,
        expected_sha="a" * 40,
        min_priority=10.0,
    )

    assert receipt["candidate"]["git_sha"] == "a" * 40
    assert receipt["shadow_export"]["sha256"]
    assert receipt["dispatch_allowed"] is False
    assert receipt["runtime_authority_changed"] is False
    assert receipt["authority"]["may_dispatch"] is False
    assert receipt["review"]["records_queued"] == 0
    assert (run / "shadow-evaluation" / "shadow-evaluation-receipt.json").is_file()


def test_shadow_evaluation_refuses_nonempty_destination(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    export = tmp_path / "shadow.jsonl"
    export.write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "bundle"
    destination.mkdir()
    (destination / "old.txt").write_text("stale", encoding="utf-8")

    monkeypatch.setattr(
        shadow_evaluation,
        "validate_baseline_run",
        lambda root, expected_sha=None: {"git_sha": "b" * 40},
    )

    import pytest

    with pytest.raises(ValueError, match="must be empty"):
        shadow_evaluation.run_shadow_evaluation(
            run_dir=run,
            shadow_export=export,
            output_dir=destination,
        )
