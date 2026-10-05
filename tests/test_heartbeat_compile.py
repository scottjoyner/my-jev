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


def provenance():
    return {
        "system_one_config_version": "1",
        "model_revision": "recorded/jev",
        "trace_sha256": "a" * 64,
    }


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
        provenance=provenance(),
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
    assert profile.advice.fleet_priority[0].score == 0.88
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
            provenance=provenance(),
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
            provenance=provenance(),
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
            provenance=provenance(),
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
            provenance=provenance(),
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
            provenance=provenance(),
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
            provenance=provenance(),
        )



def test_compile_rejects_missing_required_provenance(tmp_path):
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    snap = snapshot(now)

    with pytest.raises(ValueError, match="trace_sha256"):
        compile_heartbeat_recommendation(
            snap,
            recommendation(snap),
            receipt_id="receipt-8",
            consumer_session_id="pi-session-1",
            project_cwd=tmp_path,
            compiled_at=now,
            mode_confidence=0.5,
            provenance={
                "system_one_config_version": "1",
                "model_revision": "recorded/jev",
            },
        )


def test_compile_rejects_future_dated_snapshot(tmp_path):
    compiled = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    snap = snapshot(compiled + timedelta(minutes=6))

    with pytest.raises(ValueError, match="future-dated"):
        compile_heartbeat_recommendation(
            snap,
            recommendation(snap),
            receipt_id="receipt-9",
            consumer_session_id="pi-session-1",
            project_cwd=tmp_path,
            compiled_at=compiled,
            mode_confidence=0.5,
            provenance=provenance(),
        )


def test_compile_rejects_missing_recommendation_schema(tmp_path):
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    snap = snapshot(now)
    rec = recommendation(snap)
    del rec["schema"]

    with pytest.raises(Exception):
        compile_heartbeat_recommendation(
            snap,
            rec,
            receipt_id="receipt-10",
            consumer_session_id="pi-session-1",
            project_cwd=tmp_path,
            compiled_at=now,
            mode_confidence=0.5,
            provenance=provenance(),
        )


def test_compile_rejects_missing_explicit_none_candidate(tmp_path):
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    snap = snapshot(now)
    rec = recommendation(snap)
    del rec["advice"]["context_focus"]

    with pytest.raises(Exception):
        compile_heartbeat_recommendation(
            snap,
            rec,
            receipt_id="receipt-11",
            consumer_session_id="pi-session-1",
            project_cwd=tmp_path,
            compiled_at=now,
            mode_confidence=0.5,
            provenance=provenance(),
        )


def test_compile_does_not_implicitly_promote_snapshot_goal_to_task_focus(tmp_path):
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    snap = snapshot(now)
    profile = compile_heartbeat_recommendation(
        snap,
        recommendation(snap),
        receipt_id="receipt-12",
        consumer_session_id="pi-session-1",
        project_cwd=tmp_path,
        compiled_at=now,
        mode_confidence=0.5,
        provenance=provenance(),
    )

    assert snap.work.goal
    assert profile.advice.task_focus is None


def _benchmark_advisory(preferred, *, role, shape="preferred_node", fallback=None):
    from my_jev.fleet_benchmark_qualification import (
        BenchmarkPreference,
        FleetBenchmarkAdvisory,
    )
    from my_jev.fleet_policy import PlacementShape

    return FleetBenchmarkAdvisory(
        campaign_id="campaign-1",
        observed_at="2026-09-24T11:00:00Z",
        expires_at="2026-09-25T11:00:00Z",
        execution_shape=PlacementShape(shape),
        role_assignment=role,
        decomposition_fallback=fallback,
        preferred=[
            BenchmarkPreference(role=r, handle=h, score=0.9, reason="measured")
            for h, r in preferred
        ],
        eligible_node_count=len(preferred),
    )


def test_terminal_emitter_carries_benchmark_qualification(tmp_path):
    """The receipt an operator reads is built here, not by the profile builder."""
    from my_jev.fleet_benchmark_qualification import BenchmarkRole

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
        benchmark_advisory=_benchmark_advisory(
            [("eligible:opaque:r9700-a", BenchmarkRole.CODE)], role=BenchmarkRole.CODE
        ),
        provenance=provenance(),
    )

    qualification = profile.advice.benchmark_qualification
    assert qualification is not None
    assert qualification.next_action == "implement"
    assert qualification.preferred_handles == ["eligible:opaque:r9700-a"]
    assert profile.authority.dispatch_allowed is False


def test_terminal_emitter_omits_qualification_when_none_supplied(tmp_path):
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    snap = snapshot(now)
    project = tmp_path / "project"
    project.mkdir()

    profile = compile_heartbeat_recommendation(
        snap, recommendation(snap), receipt_id="r", consumer_session_id="cs",
        project_cwd=project, compiled_at=now + timedelta(seconds=30),
        ttl_seconds=600, mode_confidence=0.88, provenance=provenance(),
    )
    assert profile.advice.benchmark_qualification is None


def test_benchmark_handle_outside_the_snapshot_fleet_is_rejected(tmp_path):
    """A qualification built against a different snapshot must not ride along."""
    from my_jev.fleet_benchmark_qualification import BenchmarkRole

    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    snap = snapshot(now)
    project = tmp_path / "project"
    project.mkdir()

    with pytest.raises(ValueError, match="outside the authoritative eligible set"):
        compile_heartbeat_recommendation(
            snap, recommendation(snap), receipt_id="r", consumer_session_id="cs",
            project_cwd=project, compiled_at=now + timedelta(seconds=30),
            ttl_seconds=600, mode_confidence=0.88,
            benchmark_advisory=_benchmark_advisory(
                [("eligible:opaque:from-another-snapshot", BenchmarkRole.CODE)],
                role=BenchmarkRole.CODE,
            ),
            provenance=provenance(),
        )


def test_selecting_a_non_preferred_handle_is_rejected(tmp_path):
    from my_jev.fleet_benchmark_qualification import BenchmarkRole

    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    snap = build_heartbeat_snapshot(
        work=WorkStateSummary(
            work_id="work-1", session_id="work-session-1", status="active",
            goal="Continue the bounded implementation.",
        ),
        knowledge=KnowledgeStateSummary(
            knowledge_revision="knowledge-sha", neo4j_snapshot_id="neo4j-snapshot",
            note_refs=["20-Projects/local-studio/CURRENT_STATE.md"],
        ),
        fleet=FleetStateSummary(
            projection_generation="generation-1", projection_checksum="fleet-sha",
            observation_snapshot_id="observation-1", eligible_count=2,
            eligible_handles=["eligible:opaque:r9700-a", "eligible:opaque:r9700-b"],
        ),
        observed_at=now, ttl_seconds=300,
    )
    project = tmp_path / "project"
    project.mkdir()

    rec = recommendation(snap)
    rec["advice"]["fleet_handle"] = "eligible:opaque:r9700-b"

    with pytest.raises(ValueError, match="not benchmark-preferred"):
        compile_heartbeat_recommendation(
            snap, rec, receipt_id="r", consumer_session_id="cs", project_cwd=project,
            compiled_at=now + timedelta(seconds=30), ttl_seconds=600,
            mode_confidence=0.88,
            benchmark_advisory=_benchmark_advisory(
                [("eligible:opaque:r9700-a", BenchmarkRole.CODE)], role=BenchmarkRole.CODE
            ),
            provenance=provenance(),
        )


def test_qualification_survives_the_signed_response_envelope(tmp_path):
    """Additive fields must reach the envelope an operator actually reads."""
    from my_jev.fleet_benchmark_qualification import BenchmarkRole
    from my_jev.uhp_advisory import build_uhp_response_fixture

    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    snap = snapshot(now)
    project = tmp_path / "project"
    project.mkdir()

    profile = compile_heartbeat_recommendation(
        snap, recommendation(snap), receipt_id="r", consumer_session_id="cs",
        project_cwd=project, compiled_at=now + timedelta(seconds=30),
        ttl_seconds=600, mode_confidence=0.88,
        benchmark_advisory=_benchmark_advisory(
            [("eligible:opaque:r9700-a", BenchmarkRole.CODE)], role=BenchmarkRole.CODE
        ),
        provenance=provenance(),
    )
    envelope = build_uhp_response_fixture(
        profile, response_id="resp_1", session_id="hsess_1", harness_id="chrn_1",
        model="test-model", created_at=now,
    )

    carried = envelope["metadata"]["hermes_system_one"]
    qualification = carried["advice"]["benchmark_qualification"]
    assert qualification["next_action"] == "implement"
    assert qualification["preferred_handles"] == ["eligible:opaque:r9700-a"]
    assert set(carried["authority"].values()) == {False}


def test_the_documented_digest_matches_the_checked_in_schema():
    """Documentation drift is not caught by a code-vs-schema check.

    `test_contract_sha_matches_checked_in_schema` verifies the constant against the
    schema bytes. It says nothing about prose, so the moment the schema changed the
    documented digest was stale by one whole revision and every test still passed.
    A consumer reading the doc would have pinned a digest the code no longer emits.
    """
    import re

    from my_jev.uhp_advisory import CONTRACT_SHA256

    docs = Path("docs/UHP_ADVISORY.md").read_text(encoding="utf-8")

    # The digest stated as *the* contract, i.e. outside the revision-history table.
    stated = re.search(
        r"Exact schema SHA-256:\s*\n\s*`([0-9a-f]{64})`",
        docs,
    )
    assert stated is not None, "the documented digest must remain discoverable"
    assert stated.group(1) == CONTRACT_SHA256

    # And the current digest must appear in the revision history too, so a consumer
    # arriving at the old one can find out what changed.
    assert f"`{CONTRACT_SHA256[:8]}...`" in docs


def test_a_superseded_digest_is_only_mentioned_as_history():
    """An old digest must not survive anywhere as if it were current.

    That is exactly how the stale value sat in this file: correct-looking,
    unlabelled, and wrong.
    """
    import re

    from my_jev.uhp_advisory import CONTRACT_SHA256

    docs = Path("docs/UHP_ADVISORY.md").read_text(encoding="utf-8")
    digests = set(re.findall(r"`([0-9a-f]{64})`", docs))

    superseded = digests - {CONTRACT_SHA256}
    assert not superseded, f"undocumented-as-superseded digests present: {superseded}"
    # Short forms in the history table are fine; full stale ones are not.
    for short in re.findall(r"`([0-9a-f]{8})\.\.\.`", docs):
        assert docs.count(CONTRACT_SHA256[:8]) >= 1, short
