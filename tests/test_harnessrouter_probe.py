import json
import subprocess
import sys
from datetime import UTC, datetime

import pytest

from my_jev.harnessrouter_probe import (
    PINNED_HARNESSROUTER_DRIVER_BLOB_SHA1,
    PINNED_HARNESSROUTER_HEAD,
    PINNED_SYSTEMONE_CONFIG_SHA256,
    PINNED_SYSTEMONE_PACKAGE_MANIFEST_SHA256,
    PINNED_SYSTEMONE_PROVIDER_BLOB_SHA1,
    _confidence,
    _isolated_python_command,
    _probe_choices,
    _recommend_step,
    _sanitized_python_env,
    _script_entry,
)
from my_jev.heartbeat_snapshot import (
    FleetStateSummary,
    KnowledgeStateSummary,
    WorkStateSummary,
    build_heartbeat_snapshot,
)


def snapshot(*, context="20-Projects/local-studio/CURRENT_STATE.md"):
    return build_heartbeat_snapshot(
        work=WorkStateSummary(
            work_id="work-probe",
            session_id="work-session-probe",
            status="active",
            goal="Continue the bounded implementation.",
        ),
        knowledge=KnowledgeStateSummary(
            knowledge_revision="knowledge-sha",
            neo4j_snapshot_id="neo4j-snapshot",
            note_refs=[context] if context else [],
        ),
        fleet=FleetStateSummary(
            projection_generation="generation-1",
            projection_checksum="fleet-sha",
            observation_snapshot_id="observation-1",
            eligible_count=1,
            eligible_handles=["eligible:opaque:r9700-a"],
        ),
        observed_at=datetime(2026, 9, 24, 12, 0, tzinfo=UTC),
        ttl_seconds=300,
    )


def test_probe_pins_reviewed_harnessrouter_head_and_finite_recommendation():
    snap = snapshot()
    assert PINNED_HARNESSROUTER_HEAD == "250de65d6e690abdef40e39d21591b4a807984a3"
    assert PINNED_HARNESSROUTER_DRIVER_BLOB_SHA1 == "7cb3516a14b4f947e396a20735db4eb419a3db12"
    assert PINNED_SYSTEMONE_PROVIDER_BLOB_SHA1 == "008ddd09fe8e2c85ee3b8316cf25062c28b59c1c"
    assert PINNED_SYSTEMONE_PACKAGE_MANIFEST_SHA256 == "3a69281583ccccefd3e4d5422939703c842b00b92e2bfa9fc374293a94e15a74"
    assert PINNED_SYSTEMONE_CONFIG_SHA256 == "459cc500b481878aa1445a6176bb8a6b61db51981696afcc6dd65f9fe3700f4e"
    assert _probe_choices(snap) == (
        "act",
        "eligible:opaque:r9700-a",
        "20-Projects/local-studio/CURRENT_STATE.md",
    )
    assert _script_entry(snap) == (
        "recommend("
        "mode=act,"
        "fleet_handle=eligible:opaque:r9700-a,"
        "context_focus=20-Projects/local-studio/CURRENT_STATE.md"
        ")"
    )


def test_probe_uses_none_when_snapshot_has_no_optional_candidates():
    snap = snapshot(context="")
    snap = snap.model_copy(
        update={
            "fleet": snap.fleet.model_copy(
                update={"eligible_count": 0, "eligible_handles": []}
            )
        }
    )
    assert _probe_choices(snap) == ("act", "none", "none")


def test_probe_rejects_scriptprovider_delimiter_in_snapshot_candidate():
    with pytest.raises(ValueError, match="ScriptProvider"):
        _script_entry(snapshot(context="notes/a,b.md"))


def test_recommend_step_requires_exactly_one_terminal_action():
    trace = {
        "steps": [
            {
                "action": "recommend",
                "verdict": "run",
                "action_confidence": 0.99,
            }
        ]
    }
    step = _recommend_step(trace)
    assert step["action"] == "recommend"
    assert _confidence(step) == 0.99

    with pytest.raises(RuntimeError, match="exactly one"):
        _recommend_step({"steps": []})

def test_probe_sanitizes_python_injection_and_provider_credentials(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/tmp/injected")
    monkeypatch.setenv("PYTHONHOME", "/tmp/fake-home")
    monkeypatch.setenv("PYTHONSTARTUP", "/tmp/startup.py")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-openrouter")
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret-typesafe")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/inject.so")
    monkeypatch.setenv("KEEP_ME", "safe")

    env, removed = _sanitized_python_env()

    assert "KEEP_ME" in env
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert "PYTHONSTARTUP" not in env
    assert "OPENROUTER_API_KEY" not in env
    assert "TYPESAFE_API_KEY" not in env
    assert "LD_PRELOAD" not in env
    assert {
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "OPENROUTER_API_KEY",
        "TYPESAFE_API_KEY",
        "LD_PRELOAD",
    }.issubset(set(removed))


def test_isolated_python_command_inserts_only_explicit_search_paths():
    runtime = {"search_paths": ["/probe/site-a", "/probe/site-b"]}
    command = _isolated_python_command(
        sys.executable,
        runtime,
        code="print(json.dumps({'path': sys.path[:2], 'argv': sys.argv}))",
        argv=["driver.py", "job-json"],
    )
    proc = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        env={},
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["path"] == runtime["search_paths"]
    assert payload["argv"] == ["driver.py", "job-json"]

