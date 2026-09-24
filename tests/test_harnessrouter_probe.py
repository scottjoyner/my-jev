from datetime import UTC, datetime

import pytest

from my_jev.harnessrouter_probe import (
    PINNED_HARNESSROUTER_HEAD,
    _confidence,
    _probe_choices,
    _recommend_step,
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
