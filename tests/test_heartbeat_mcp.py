from datetime import UTC, datetime, timedelta

import pytest

from my_jev.heartbeat_mcp import HeartbeatAdvisoryEnvironment
from my_jev.heartbeat_snapshot import (
    AuthorityContext,
    FleetStateSummary,
    KnowledgeFact,
    KnowledgeStateSummary,
    WorkStateSummary,
    build_heartbeat_snapshot,
)


def write_snapshot(tmp_path, *, observed_at=None, ttl_seconds=300):
    snapshot = build_heartbeat_snapshot(
        work=WorkStateSummary(
            work_id="work-1",
            status="active",
            goal="Continue the bounded acceptance slice.",
            active_claims=["claim:opaque:1"],
        ),
        knowledge=KnowledgeStateSummary(
            knowledge_revision="knowledge-sha",
            markdown_revision="markdown-sha",
            neo4j_snapshot_id="neo4j-snapshot",
            note_refs=[
                "20-Projects/local-studio/CURRENT_STATE.md",
                "20-Projects/my-jev/CURRENT_STATE.md",
            ],
            facts=[
                KnowledgeFact(
                    subject="work-1",
                    predicate="boundary",
                    object="System-One is advisory only.",
                    source="markdown",
                    source_ref="20-Projects/my-jev/CURRENT_STATE.md",
                )
            ],
        ),
        fleet=FleetStateSummary(
            projection_generation="generation-1",
            projection_checksum="fleet-sha",
            observation_snapshot_id="observation-1",
            eligible_count=2,
            eligible_handles=[
                "eligible:opaque:r9700-a",
                "eligible:opaque:x1-b",
            ],
            pressure="light",
        ),
        authority_context=AuthorityContext(
            speaker_verified=True,
            actions_allowed=True,
            local_writes_allowed=True,
            external_actions_allowed=False,
            privileged_actions_allowed=False,
            approval_gate_available=True,
        ),
        available_capabilities=["read_repo", "run_tests"],
        available_tools=["github.read", "filesystem.read"],
        metadata={"project": "local-studio"},
        observed_at=observed_at or datetime.now(UTC),
        ttl_seconds=ttl_seconds,
    )
    path = tmp_path / "heartbeat.json"
    path.write_text(
        snapshot.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return path


def test_environment_exposes_only_bounded_advisory_candidates(tmp_path):
    env = HeartbeatAdvisoryEnvironment(write_snapshot(tmp_path))

    observed = env.observe()
    assert observed["terminal"] is False
    assert observed["candidates"]["eligible_fleet_handles"] == [
        "none",
        "eligible:opaque:r9700-a",
        "eligible:opaque:x1-b",
    ]
    assert observed["candidates"]["context_focus"] == [
        "none",
        "20-Projects/local-studio/CURRENT_STATE.md",
        "20-Projects/my-jev/CURRENT_STATE.md",
    ]
    assert observed["fields"]["evidence_only"] is True
    assert observed["fields"]["runtime_authority_changed"] is False
    assert "password" not in observed["text"].lower()


def test_recommendation_terminates_without_granting_authority(tmp_path):
    env = HeartbeatAdvisoryEnvironment(write_snapshot(tmp_path))

    result = env.recommend(
        mode="act",
        fleet_handle="eligible:opaque:r9700-a",
        context_focus="20-Projects/local-studio/CURRENT_STATE.md",
    )

    assert result["terminal"] is True
    assert result["fields"]["advice"] == {
        "mode": "act",
        "fleet_handle": "eligible:opaque:r9700-a",
        "context_focus": "20-Projects/local-studio/CURRENT_STATE.md",
    }
    assert result["fields"]["authority"] == {
        "dispatch_allowed": False,
        "approval_granted": False,
        "claim_acquired": False,
        "mutation_allowed": False,
        "routing_authority_changed": False,
    }
    assert result["fields"]["evidence_only"] is True
    assert result["fields"]["runtime_authority_changed"] is False
    assert env.observe()["terminal"] is True


def test_recommendation_cannot_invent_a_fleet_destination(tmp_path):
    env = HeartbeatAdvisoryEnvironment(write_snapshot(tmp_path))

    with pytest.raises(ValueError, match="eligible set"):
        env.recommend(
            mode="act",
            fleet_handle="eligible:opaque:not-projected",
            context_focus="none",
        )


def test_recommendation_cannot_invent_context(tmp_path):
    env = HeartbeatAdvisoryEnvironment(write_snapshot(tmp_path))

    with pytest.raises(ValueError, match="bounded snapshot"):
        env.recommend(
            mode="chat",
            fleet_handle="none",
            context_focus="secrets/credentials.md",
        )


def test_expired_snapshot_fails_closed_before_an_episode_starts(tmp_path):
    path = write_snapshot(
        tmp_path,
        observed_at=datetime.now(UTC) - timedelta(minutes=10),
        ttl_seconds=300,
    )

    with pytest.raises(RuntimeError, match="expired"):
        HeartbeatAdvisoryEnvironment(path)



def test_recommendation_rechecks_snapshot_freshness(tmp_path):
    env = HeartbeatAdvisoryEnvironment(write_snapshot(tmp_path))
    env.snapshot = env.snapshot.model_copy(
        update={
            "observed_at": "2026-09-24T11:00:00Z",
            "expires_at": "2026-09-24T11:05:00Z",
        }
    )

    with pytest.raises(RuntimeError, match="expired"):
        env.recommend(
            mode="act",
            fleet_handle="eligible:opaque:r9700-a",
            context_focus="none",
        )
