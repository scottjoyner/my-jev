from datetime import UTC, datetime

import pytest

from my_jev.heartbeat_snapshot import (
    AuthorityContext,
    FleetStateSummary,
    KnowledgeFact,
    KnowledgeStateSummary,
    WorkStateSummary,
    agent_policy_state_from_snapshot,
    build_heartbeat_snapshot,
    canonical_snapshot_json,
    snapshot_sha256,
)


def make_snapshot(**overrides):
    kwargs = {
        "work": WorkStateSummary(
            work_id="work-1",
            session_id="session-1",
            status="active",
            goal="Continue the bounded implementation.",
            blockers=["awaiting one validation"],
            pending_approvals=["approval:opaque:1"],
            active_claims=["claim:opaque:1"],
        ),
        "knowledge": KnowledgeStateSummary(
            knowledge_revision="knowledge-sha",
            markdown_revision="markdown-sha",
            neo4j_snapshot_id="neo4j-snapshot-1",
            note_refs=["20-Projects/local-studio/CURRENT_STATE.md"],
            facts=[
                KnowledgeFact(
                    subject="local-studio-pr3",
                    predicate="state",
                    object="advisory consumer open and mergeable",
                    source="markdown",
                    source_ref="20-Projects/local-studio/CURRENT_STATE.md",
                ),
                KnowledgeFact(
                    subject="assistx-runtime-projection",
                    predicate="authority",
                    object="signed admission remains host-authoritative",
                    source="neo4j",
                    source_ref="graph:runtime-projection",
                ),
            ],
        ),
        "fleet": FleetStateSummary(
            projection_generation="generation-1",
            projection_checksum="fleet-sha",
            observation_snapshot_id="observation-1",
            eligible_count=3,
            drained_count=1,
            pressure="light",
        ),
        "authority_context": AuthorityContext(
            speaker_verified=True,
            actions_allowed=True,
            local_writes_allowed=True,
            external_actions_allowed=False,
            privileged_actions_allowed=False,
            approval_gate_available=True,
        ),
        "available_capabilities": ["read_repo", "run_tests"],
        "available_tools": ["github.read", "filesystem.read"],
        "metadata": {"project": "local-studio"},
        "observed_at": datetime(2026, 9, 23, 22, 30, tzinfo=UTC),
        "ttl_seconds": 300,
    }
    kwargs.update(overrides)
    return build_heartbeat_snapshot(**kwargs)


def test_snapshot_is_stable_bounded_and_revisioned():
    snapshot = make_snapshot()

    assert snapshot.observed_at == "2026-09-23T22:30:00Z"
    assert snapshot.expires_at == "2026-09-23T22:35:00Z"
    assert snapshot.knowledge.knowledge_revision == "knowledge-sha"
    assert snapshot.fleet.projection_checksum == "fleet-sha"
    assert len(snapshot_sha256(snapshot)) == 64


def test_snapshot_hash_is_deterministic():
    first = make_snapshot()
    second = make_snapshot()

    assert canonical_snapshot_json(first) == canonical_snapshot_json(second)
    assert snapshot_sha256(first) == snapshot_sha256(second)


def test_snapshot_rejects_credential_like_metadata_keys():
    with pytest.raises(ValueError, match="credential-like"):
        make_snapshot(metadata={"api_token": "must-not-cross-projector"})


def test_snapshot_rejects_long_ttl():
    with pytest.raises(ValueError, match="1..600"):
        make_snapshot(ttl_seconds=601)


def test_policy_projection_carries_snapshot_provenance_not_authority_grants():
    snapshot = make_snapshot()
    state = agent_policy_state_from_snapshot(
        snapshot,
        utterance="Keep working on the current slice.",
        speaker_id="operator",
    )

    assert state.source == "hermes_heartbeat"
    assert state.speaker_verified is True
    assert state.external_actions_allowed is False
    assert state.privileged_actions_allowed is False
    assert state.metadata["heartbeat_snapshot_sha256"] == snapshot_sha256(snapshot)
    assert state.metadata["knowledge_revision"] == "knowledge-sha"
    assert state.metadata["fleet_projection_checksum"] == "fleet-sha"
    assert "work-1" in state.active_work


def test_knowledge_fact_must_be_attested_non_secret():
    with pytest.raises(ValueError):
        KnowledgeFact(
            subject="x",
            predicate="contains",
            object="sensitive",
            source="neo4j",
            source_ref="graph:x",
            sensitivity="secret",
        )
