import json

from my_jev.heartbeat_snapshot_cli import main


def test_heartbeat_snapshot_cli_builds_bounded_record(tmp_path):
    output = tmp_path / "heartbeat.json"
    rc = main(
        [
            "--work",
            "examples/heartbeat/work.json",
            "--knowledge",
            "examples/heartbeat/knowledge.json",
            "--fleet",
            "examples/heartbeat/fleet.json",
            "--authority",
            "examples/heartbeat/authority.json",
            "--metadata",
            "examples/heartbeat/metadata.json",
            "--capability",
            "read_repo",
            "--capability",
            "run_tests",
            "--tool",
            "github.read",
            "--tool",
            "filesystem.read",
            "--observed-at",
            "2026-09-23T22:30:00Z",
            "--ttl-seconds",
            "300",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "hermes-heartbeat-snapshot-v1"
    assert payload["observed_at"] == "2026-09-23T22:30:00Z"
    assert payload["expires_at"] == "2026-09-23T22:35:00Z"
    assert payload["knowledge"]["knowledge_revision"] == "knowledge-example-sha"
    assert payload["fleet"]["projection_checksum"] == "fleet-projection-example-sha"
    assert payload["authority_context"]["external_actions_allowed"] is False
    assert payload["authority_context"]["privileged_actions_allowed"] is False
