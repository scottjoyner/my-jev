import json
from pathlib import Path

from my_jev.uhp_fixture import main


def test_fixture_cli_emits_local_studio_compatible_response(tmp_path):
    output = tmp_path / "latest.json"

    rc = main(
        [
            "--decision",
            "examples/uhp/decision.json",
            "--fleet-resolution",
            "examples/uhp/fleet-resolution.json",
            "--fleet-handle-map",
            "examples/uhp/fleet-handles.json",
            "--provenance",
            "examples/uhp/provenance.json",
            "--receipt-id",
            "fixture-receipt-1",
            "--response-id",
            "resp_fixture1",
            "--session-id",
            "hsess-fixture1",
            "--harness-id",
            "chrn_system_one",
            "--model",
            "recorded/jev",
            "--observed-at",
            "2026-09-23T22:30:00Z",
            "--created-at",
            "2026-09-23T22:30:00Z",
            "--ttl-seconds",
            "600",
            "--task-focus",
            "Continue only inside the existing authority boundary.",
            "--context-priority",
            "current-pr",
            "--context-priority",
            "latest-handoff",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["metadata"]["hermes_system_one"]["uhp_version"] == "2026-09-12"
    assert payload["metadata"]["hermes_system_one"]["authority"] == {
        "dispatch_allowed": False,
        "approval_granted": False,
        "claim_acquired": False,
        "mutation_allowed": False,
        "routing_authority_changed": False,
    }

    rendered = output.read_text(encoding="utf-8")
    assert "r9700-primary" not in rendered
    assert "x1-370" not in rendered
    assert "eligible:opaque:r9700-a" in rendered
    assert "eligible:opaque:cpu-b" in rendered
