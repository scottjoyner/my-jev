import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from my_jev.heartbeat_compile import compile_heartbeat_recommendation
from my_jev.heartbeat_snapshot import (
    FleetStateSummary,
    KnowledgeStateSummary,
    WorkStateSummary,
    build_heartbeat_snapshot,
    snapshot_sha256,
)
from my_jev.uhp_advisory import CONTRACT_SHA256, project_fingerprint


def snapshot(now):
    return build_heartbeat_snapshot(
        work=WorkStateSummary(
            work_id="work-1",
            session_id="work-session-1",
            status="active",
            goal="Continue the bounded implementation.",
        ),
        knowledge=KnowledgeStateSummary(
            knowledge_revision="knowledge-sha",
            neo4j_snapshot_id="neo4j-snapshot",
            note_refs=["20-Projects/local-studio/CURRENT_STATE.md"],
        ),
        fleet=FleetStateSummary(
            projection_generation="generation-1",
            projection_checksum="fleet-sha",
            observation_snapshot_id="observation-1",
            eligible_count=1,
            eligible_handles=["eligible:opaque:r9700-a"],
        ),
        observed_at=now,
        ttl_seconds=300,
    )


def recommendation(snap):
    return {
        "schema": "hermes-system-one-recommendation-v1",
        "snapshot_sha256": snapshot_sha256(snap),
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


def test_contract_sha_matches_checked_in_schema():
    schema = Path("contracts/hermes-system-one-heartbeat-v1.schema.json").read_bytes()
    assert hashlib.sha256(schema).hexdigest() == CONTRACT_SHA256


def test_compile_binds_receipt_to_exact_session_project_and_snapshot(tmp_path):
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    snap = snapshot(now)
    project = tmp_path / "project"
    project.mkdir()

    profile = compile_heartbeat_recommendation(
        snap,
        recommendation(snap),
        receipt_id="receipt-1",
        consumer_session_id="pi-session-1",
        project_cwd=project,
        compiled_at=now + timedelta(seconds=30),
        ttl_seconds=600,
        mode_confidence=0.88,
        policy_disposition="propose_action",
        approval_recommended=True,
        provenance={
            "system_one_config_version": "1",
            "model_revision": "recorded/jev",
            "trace_sha256": "a" * 64,
        },
    )

    assert profile.contract_sha256 == CONTRACT_SHA256
    assert profile.binding.consumer == "local-studio"
    assert profile.binding.work_id == "work-1"
    assert profile.binding.consumer_session_id == "pi-session-1"
    assert profile.binding.project_fingerprint == project_fingerprint(project)
    assert profile.binding.snapshot_sha256 == snapshot_sha256(snap)
    assert profile.expires_at == "2026-09-24T12:05:00Z"
    assert profile.advice.mode == "act"
    assert profile.advice.policy_disposition == "propose_action"
    assert profile.advice.approval_recommended is True
    assert profile.advice.fleet_priority[0].handle == "eligible:opaque:r9700-a"
    assert profile.authority.mutation_allowed is False
    assert profile.provenance.knowledge_revision == "knowledge-sha"
    assert profile.provenance.fleet_projection_checksum == "fleet-sha"


def test_compile_rejects_snapshot_mismatch(tmp_path):
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    snap = snapshot(now)
    rec = recommendation(snap)
    rec["snapshot_sha256"] = "b" * 64

    with pytest.raises(ValueError, match="snapshot_sha256"):
        compile_heartbeat_recommendation(
            snap,
            rec,
            receipt_id="receipt-2",
            consumer_session_id="pi-session-1",
            project_cwd=tmp_path,
            compiled_at=now,
            mode_confidence=0.5,
        )


def test_compile_rejects_authority_bearing_recommendation(tmp_path):
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    snap = snapshot(now)
    rec = recommendation(snap)
    rec["authority"]["mutation_allowed"] = True

    with pytest.raises(ValueError, match="authority"):
        compile_heartbeat_recommendation(
            snap,
            rec,
            receipt_id="receipt-3",
            consumer_session_id="pi-session-1",
            project_cwd=tmp_path,
            compiled_at=now,
            mode_confidence=0.5,
        )


def test_compile_rejects_candidate_outside_snapshot(tmp_path):
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    snap = snapshot(now)
    rec = recommendation(snap)
    rec["advice"]["fleet_handle"] = "eligible:opaque:not-offered"

    with pytest.raises(ValueError, match="eligible set"):
        compile_heartbeat_recommendation(
            snap,
            rec,
            receipt_id="receipt-4",
            consumer_session_id="pi-session-1",
            project_cwd=tmp_path,
            compiled_at=now,
            mode_confidence=0.5,
        )


def test_compile_rejects_expired_snapshot(tmp_path):
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    snap = snapshot(now)

    with pytest.raises(ValueError, match="expired snapshot"):
        compile_heartbeat_recommendation(
            snap,
            recommendation(snap),
            receipt_id="receipt-5",
            consumer_session_id="pi-session-1",
            project_cwd=tmp_path,
            compiled_at=now + timedelta(minutes=6),
            mode_confidence=0.5,
        )



def test_project_fingerprint_matches_realpath_workspace_identity(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    canonical = hashlib.sha256(target.resolve().as_posix().encode("utf-8")).hexdigest()
    assert project_fingerprint(link) == canonical
    assert project_fingerprint(link) == project_fingerprint(target)



def test_compile_rejects_missing_authority_assertion(tmp_path):
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    snap = snapshot(now)
    rec = recommendation(snap)
    del rec["authority"]["mutation_allowed"]

    with pytest.raises(Exception):
        compile_heartbeat_recommendation(
            snap,
            rec,
            receipt_id="receipt-6",
            consumer_session_id="pi-session-1",
            project_cwd=tmp_path,
            compiled_at=now,
            mode_confidence=0.5,
        )


def test_compile_rejects_authority_extension_even_when_false(tmp_path):
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    snap = snapshot(now)
    rec = recommendation(snap)
    rec["authority"]["future_grant"] = False

    with pytest.raises(Exception):
        compile_heartbeat_recommendation(
            snap,
            rec,
            receipt_id="receipt-7",
            consumer_session_id="pi-session-1",
            project_cwd=tmp_path,
            compiled_at=now,
            mode_confidence=0.5,
        )
