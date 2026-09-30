from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from my_jev.model_evidence import (
    ModelCandidateEvidence,
    ModelEvidenceProvenance,
    ModelExecutionEnvelope,
    ModelNeedProfile,
    ModelRequestNeeds,
    build_model_evidence_snapshot,
    build_model_selection_outcome,
    build_model_selection_record,
    canonical_model_evidence_json,
    canonical_model_selection_outcome_json,
    model_evidence_sha256,
    model_selection_outcome_sha256,
    rank_candidate_handles,
)


def _provenance() -> ModelEvidenceProvenance:
    return ModelEvidenceProvenance(
        knowledge_revision="knowledge@test",
        scoring_contract_sha256="a" * 64,
        local_benchmark_snapshot_sha256="b" * 64,
        heartbeat_snapshot_sha256="c" * 64,
        artificial_analysis_snapshot_sha256="d" * 64,
        task_bundle_revision="tasks@test",
    )


def _candidate(
    handle: str,
    *,
    balanced: float,
    task_fit: float | None = None,
    max_context: int = 131072,
    confidence: float = 0.9,
) -> ModelCandidateEvidence:
    return ModelCandidateEvidence(
        handle=handle,
        scenario_scores={
            "fast": balanced,
            "balanced": balanced,
            "complex": balanced,
            "agentic": balanced,
            "long-context": balanced,
        },
        task_fit_scores={
            "fast": task_fit if task_fit is not None else balanced,
            "balanced": task_fit if task_fit is not None else balanced,
            "complex": task_fit if task_fit is not None else balanced,
            "agentic": task_fit if task_fit is not None else balanced,
            "long-context": task_fit if task_fit is not None else balanced,
        },
        identity_confidence=0.95,
        evidence_coverage=0.9,
        quantization_confidence=0.85,
        capability_score=50,
        local_weight_gib=7.0,
        local_reliability=0.95,
        quantization_class="PQ2_0",
        modalities=["text"],
        measured_task_families=["repo_work", "debugging"],
        execution=ModelExecutionEnvelope(
            benchmark_lane_count=6,
            distinct_node_count=4,
            currently_eligible_replica_count=2,
            generation_tps_best=78.0,
            generation_tps_median=45.6,
            prompt_tps_best=2043.0,
            prompt_tps_median=1114.0,
            min_verified_context_tokens=32768,
            max_verified_context_tokens=max_context,
            execution_evidence_confidence=confidence,
            backend_classes=["rocm", "cpu", "vulkan"],
            evidence_states=["current_verified", "historical_valid"],
        ),
    )


def test_cross_node_envelope_is_model_visible_without_node_identity() -> None:
    candidate = _candidate("model:bonsai2:v1", balanced=63.0)
    snapshot = build_model_evidence_snapshot(
        request=ModelRequestNeeds(
            profile=ModelNeedProfile.BALANCED,
            required_context_tokens=32768,
            required_modalities=["text"],
        ),
        candidates=[candidate],
        provenance=_provenance(),
        observed_at=datetime(2026, 9, 24, 14, 0, tzinfo=UTC),
    )

    state = snapshot.as_model_state()
    assert '"benchmark_lane_count":6' in state
    assert '"distinct_node_count":4' in state
    assert '"generation_tps_best":78.0' in state
    assert "x1-370" not in state
    assert "xwing" not in state
    assert "deathstar" not in state


def test_required_context_accepts_any_qualified_lane_in_envelope() -> None:
    candidate = _candidate(
        "model:ornith:v1",
        balanced=70.0,
        max_context=262144,
    )
    snapshot = build_model_evidence_snapshot(
        request=ModelRequestNeeds(
            profile=ModelNeedProfile.LONG_CONTEXT,
            required_context_tokens=131072,
            required_modalities=["text"],
        ),
        candidates=[candidate],
        provenance=_provenance(),
        observed_at=datetime(2026, 9, 24, 14, 0, tzinfo=UTC),
    )
    assert snapshot.candidates[0].execution.max_verified_context_tokens == 262144


def test_required_context_rejects_candidate_when_no_lane_meets_it() -> None:
    candidate = _candidate(
        "model:k2:v1",
        balanced=70.0,
        max_context=65536,
    )
    with pytest.raises(ValueError, match="does not meet required context"):
        build_model_evidence_snapshot(
            request=ModelRequestNeeds(
                profile=ModelNeedProfile.LONG_CONTEXT,
                required_context_tokens=131072,
                required_modalities=["text"],
            ),
            candidates=[candidate],
            provenance=_provenance(),
            observed_at=datetime(2026, 9, 24, 14, 0, tzinfo=UTC),
        )


def test_physical_node_fields_are_rejected_not_silently_forwarded() -> None:
    with pytest.raises(ValidationError):
        ModelExecutionEnvelope(
            benchmark_lane_count=1,
            distinct_node_count=1,
            node="x1-370",
        )


def test_rank_is_deterministic_and_advisory_only() -> None:
    lower = _candidate("model:small:v1", balanced=55.0)
    higher = _candidate("model:large:v1", balanced=70.0)
    snapshot = build_model_evidence_snapshot(
        request=ModelRequestNeeds(profile=ModelNeedProfile.BALANCED),
        candidates=[lower, higher],
        provenance=_provenance(),
        observed_at=datetime(2026, 9, 24, 14, 0, tzinfo=UTC),
    )
    assert rank_candidate_handles(snapshot) == [
        "model:large:v1",
        "model:small:v1",
    ]

    record = build_model_selection_record(snapshot)
    assert record.metadata["dispatch_allowed"] is False
    assert record.metadata["approval_granted"] is False
    assert record.metadata["claim_acquired"] is False
    assert record.metadata["mutation_allowed"] is False
    assert record.metadata["routing_authority_changed"] is False


def test_canonical_snapshot_hash_is_stable() -> None:
    candidate = _candidate("model:stable:v1", balanced=61.0)
    snapshot = build_model_evidence_snapshot(
        request=ModelRequestNeeds(profile=ModelNeedProfile.BALANCED),
        candidates=[candidate],
        provenance=_provenance(),
        observed_at=datetime(2026, 9, 24, 14, 0, tzinfo=UTC),
    )
    first = canonical_model_evidence_json(snapshot)
    second = canonical_model_evidence_json(snapshot)
    assert first == second
    assert model_evidence_sha256(snapshot) == model_evidence_sha256(snapshot)


def test_agentic_judge_evidence_is_preserved_without_node_identity() -> None:
    candidate = _candidate("model:k2-tiny:v1", balanced=64.0)
    candidate.execution.task_success_rate = 1.0
    candidate.execution.task_trial_count = 6
    candidate.execution.measurement_classes = [
        "agentic_judge_acceptance",
        "agentic_live_loop",
    ]
    candidate.execution.roles_observed = ["judge"]

    snapshot = build_model_evidence_snapshot(
        request=ModelRequestNeeds(profile=ModelNeedProfile.AGENTIC),
        candidates=[candidate],
        provenance=_provenance(),
        observed_at=datetime(2026, 9, 24, 15, 30, tzinfo=UTC),
    )

    state = snapshot.as_model_state()
    assert '"task_success_rate":1.0' in state
    assert '"task_trial_count":6' in state
    assert '"agentic_judge_acceptance"' in state
    assert '"judge"' in state
    assert "destroyer" not in state
    assert "1238" not in state


def test_narrow_perfect_sample_does_not_beat_broad_evidence_by_default() -> None:
    narrow = _candidate("model:tiny-judge:v1", balanced=100.0, confidence=0.65)
    narrow.evidence_coverage = 0.20
    narrow.identity_confidence = 0.60
    narrow.quantization_confidence = 0.50
    narrow.execution.task_success_rate = 1.0
    narrow.execution.task_trial_count = 6

    broad = _candidate("model:broad:v1", balanced=72.0, confidence=0.95)
    broad.evidence_coverage = 0.90
    broad.identity_confidence = 0.95
    broad.quantization_confidence = 0.85

    snapshot = build_model_evidence_snapshot(
        request=ModelRequestNeeds(profile=ModelNeedProfile.BALANCED),
        candidates=[narrow, broad],
        provenance=_provenance(),
        observed_at=datetime(2026, 9, 24, 15, 45, tzinfo=UTC),
    )

    assert rank_candidate_handles(snapshot) == [
        "model:broad:v1",
        "model:tiny-judge:v1",
    ]


def test_task_fit_precedes_bounded_execution_adjustment() -> None:
    faster_but_weaker = _candidate(
        "model:fast-but-weak:v1",
        balanced=78.0,
        task_fit=58.0,
    )
    slower_but_capable = _candidate(
        "model:capable:v1",
        balanced=70.0,
        task_fit=76.0,
    )

    snapshot = build_model_evidence_snapshot(
        request=ModelRequestNeeds(
            profile=ModelNeedProfile.BALANCED,
            task_families=["repo_work"],
        ),
        candidates=[faster_but_weaker, slower_but_capable],
        provenance=_provenance(),
        observed_at=datetime(2026, 9, 24, 16, 0, tzinfo=UTC),
    )

    assert rank_candidate_handles(snapshot) == [
        "model:capable:v1",
        "model:fast-but-weak:v1",
    ]
    state = snapshot.as_model_state()
    assert '"task_families":["repo_work"]' in state
    assert '"task_fit_score":76.0' in state


def test_outcome_feedback_is_bound_to_snapshot_and_task_family() -> None:
    candidate = _candidate(
        "model:capable:v1",
        balanced=72.0,
        task_fit=78.0,
    )
    snapshot = build_model_evidence_snapshot(
        request=ModelRequestNeeds(
            profile=ModelNeedProfile.BALANCED,
            task_families=["repo_work"],
        ),
        candidates=[candidate],
        provenance=_provenance(),
        observed_at=datetime(2026, 9, 24, 16, 0, tzinfo=UTC),
    )
    outcome = build_model_selection_outcome(
        snapshot,
        selected_handle="model:capable:v1",
        completed_at=datetime(2026, 9, 24, 16, 2, tzinfo=UTC),
        success=True,
        quality_score=0.92,
        latency_ms=4700,
        output_tokens=612,
        retry_count=0,
        tool_call_count=4,
    )
    assert outcome.evidence_snapshot_sha256 == model_evidence_sha256(snapshot)
    assert outcome.task_families == ["repo_work"]
    assert outcome.routing_authority_changed is False
    assert outcome.dispatch_authority_changed is False
    assert outcome.mutation_authority_changed is False
    assert model_selection_outcome_sha256(outcome) == model_selection_outcome_sha256(outcome)
    assert '"repo_work"' in canonical_model_selection_outcome_json(outcome)


def test_outcome_cannot_claim_model_not_in_snapshot() -> None:
    snapshot = build_model_evidence_snapshot(
        request=ModelRequestNeeds(
            profile=ModelNeedProfile.BALANCED,
            task_families=["repo_work"],
        ),
        candidates=[_candidate("model:allowed:v1", balanced=70.0)],
        provenance=_provenance(),
        observed_at=datetime(2026, 9, 24, 16, 0, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="not present"):
        build_model_selection_outcome(
            snapshot,
            selected_handle="model:not-present:v1",
            completed_at=datetime(2026, 9, 24, 16, 2, tzinfo=UTC),
            success=True,
        )


def test_failed_outcome_requires_failure_class() -> None:
    snapshot = build_model_evidence_snapshot(
        request=ModelRequestNeeds(
            profile=ModelNeedProfile.AGENTIC,
            task_families=["terminal_agent"],
        ),
        candidates=[_candidate("model:agent:v1", balanced=70.0)],
        provenance=_provenance(),
        observed_at=datetime(2026, 9, 24, 16, 0, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="failure_class"):
        build_model_selection_outcome(
            snapshot,
            selected_handle="model:agent:v1",
            completed_at=datetime(2026, 9, 24, 16, 5, tzinfo=UTC),
            success=False,
        )


def test_outcome_task_family_cannot_drift_from_request() -> None:
    snapshot = build_model_evidence_snapshot(
        request=ModelRequestNeeds(
            profile=ModelNeedProfile.BALANCED,
            task_families=["repo_work"],
        ),
        candidates=[_candidate("model:repo:v1", balanced=70.0)],
        provenance=_provenance(),
        observed_at=datetime(2026, 9, 24, 16, 0, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="scoped to the request"):
        build_model_selection_outcome(
            snapshot,
            selected_handle="model:repo:v1",
            completed_at=datetime(2026, 9, 24, 16, 2, tzinfo=UTC),
            task_families=["terminal_agent"],
            success=True,
        )
