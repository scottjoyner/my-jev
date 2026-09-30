import json
from datetime import UTC, datetime

from my_jev.heartbeat_compile_cli import main
from my_jev.heartbeat_snapshot import (
    FleetStateSummary,
    KnowledgeStateSummary,
    WorkStateSummary,
    build_heartbeat_snapshot,
    snapshot_sha256,
)


def test_compiler_cli_builds_bound_stored_response(tmp_path):
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    snapshot = build_heartbeat_snapshot(
        work=WorkStateSummary(
            work_id="work-cli",
            status="active",
            goal="Continue the compiler acceptance.",
        ),
        knowledge=KnowledgeStateSummary(
            knowledge_revision="knowledge-cli",
            neo4j_snapshot_id="neo4j-cli",
            note_refs=["20-Projects/local-studio/CURRENT_STATE.md"],
        ),
        fleet=FleetStateSummary(
            projection_generation="generation-cli",
            projection_checksum="fleet-cli",
            observation_snapshot_id="observation-cli",
            eligible_count=1,
            eligible_handles=["eligible:opaque:r9700-a"],
        ),
        observed_at=now,
        ttl_seconds=300,
    )
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")

    recommendation = {
        "schema": "hermes-system-one-recommendation-v1",
        "snapshot_sha256": snapshot_sha256(snapshot),
        "advice": {
            "mode": "act",
            "fleet_handle": "eligible:opaque:r9700-a",
            "context_focus": "20-Projects/local-studio/CURRENT_STATE.md",
        },
        "authority": {
            "dispatch_allowed": False,
            "approval_granted": False,
            "claim_acquired": False,
            "mutation_allowed": False,
            "routing_authority_changed": False,
        },
        "evidence_only": True,
        "runtime_authority_changed": False,
    }
    recommendation_path = tmp_path / "recommendation.json"
    recommendation_path.write_text(json.dumps(recommendation), encoding="utf-8")

    trace_path = tmp_path / "trace.json"
    trace_path.write_text('{"config_version":1}\n', encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    output = tmp_path / "latest.json"

    rc = main(
        [
            "--snapshot",
            str(snapshot_path),
            "--recommendation",
            str(recommendation_path),
            "--receipt-id",
            "receipt-cli",
            "--consumer-session-id",
            "pi-session-cli",
            "--project-cwd",
            str(project),
            "--compiled-at",
            "2026-09-24T12:00:30Z",
            "--ttl-seconds",
            "600",
            "--mode-confidence",
            "0.92",
            "--policy-disposition",
            "propose_action",
            "--system-one-config-version",
            "1",
            "--model-revision",
            "recorded/jev",
            "--trace",
            str(trace_path),
            "--response-id",
            "resp_cli",
            "--uhp-session-id",
            "hsess-cli",
            "--harness-id",
            "chrn_system_one",
            "--model",
            "recorded/jev",
            "--created-at",
            "2026-09-24T12:00:30Z",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    profile = payload["metadata"]["hermes_system_one"]
    assert profile["binding"]["work_id"] == "work-cli"
    assert profile["binding"]["consumer_session_id"] == "pi-session-cli"
    assert profile["binding"]["snapshot_sha256"] == snapshot_sha256(snapshot)
    assert profile["expires_at"] == "2026-09-24T12:05:00Z"
    assert profile["advice"]["mode"] == "act"
    assert profile["advice"]["policy_disposition"] == "propose_action"
    assert profile["authority"]["mutation_allowed"] is False
    assert len(profile["provenance"]["trace_sha256"]) == 64
