from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .heartbeat_snapshot import HeartbeatSnapshot, snapshot_sha256
from .uhp_advisory import (
    CONTRACT_SHA256,
    DEFAULT_TTL_SECONDS,
    MAX_TTL_SECONDS,
    FleetPriorityItem,
    HermesSystemOneProfile,
    SystemOneAdvice,
    SystemOneAuthority,
    SystemOneBinding,
    SystemOneProvenance,
    project_fingerprint,
)

RECOMMENDATION_SCHEMA = "hermes-system-one-recommendation-v1"
MAX_SOURCE_CLOCK_SKEW = timedelta(minutes=5)


class TerminalRecommendationAdvice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["chat", "create_tasks", "act", "clarify", "cancel", "abstain"]
    fleet_handle: str | None
    context_focus: str | None


class TerminalRecommendationAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dispatch_allowed: Literal[False]
    approval_granted: Literal[False]
    claim_acquired: Literal[False]
    mutation_allowed: Literal[False]
    routing_authority_changed: Literal[False]


class TerminalRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema: Literal["hermes-system-one-recommendation-v1"]
    snapshot_sha256: str = Field(min_length=64, max_length=64)
    advice: TerminalRecommendationAdvice
    authority: TerminalRecommendationAuthority
    evidence_only: Literal[True]
    runtime_authority_changed: Literal[False]


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("compiled_at must be timezone-aware")
    return value.astimezone(UTC)


def _parse_stamp(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("snapshot timestamps must be timezone-aware")
    return parsed.astimezone(UTC)


def _stamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def compile_heartbeat_recommendation(
    snapshot: HeartbeatSnapshot,
    recommendation: TerminalRecommendation | Mapping[str, Any],
    *,
    receipt_id: str,
    consumer_session_id: str,
    project_cwd: str | Path,
    consumer: str = "local-studio",
    compiled_at: datetime | None = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    mode_confidence: float,
    policy_disposition: str | None = None,
    approval_recommended: bool | None = None,
    task_focus: str | None = None,
    provenance: SystemOneProvenance | Mapping[str, Any],
) -> HermesSystemOneProfile:
    """Compile one terminal recommendation into a bound, fail-closed advisory receipt."""

    rec = (
        recommendation
        if isinstance(recommendation, TerminalRecommendation)
        else TerminalRecommendation.model_validate(recommendation)
    )
    expected_snapshot = snapshot_sha256(snapshot)
    if rec.snapshot_sha256 != expected_snapshot:
        raise ValueError("recommendation snapshot_sha256 does not match the supplied snapshot")
    if not (0.0 <= mode_confidence <= 1.0):
        raise ValueError("mode_confidence must be within [0,1]")
    if ttl_seconds <= 0 or ttl_seconds > MAX_TTL_SECONDS:
        raise ValueError(f"ttl_seconds must be within 1..{MAX_TTL_SECONDS}")

    selected_handle = rec.advice.fleet_handle
    if selected_handle is not None and selected_handle not in snapshot.fleet.eligible_handles:
        raise ValueError("recommendation fleet handle is outside the authoritative eligible set")
    context_focus = rec.advice.context_focus
    if context_focus is not None and context_focus not in snapshot.knowledge.note_refs:
        raise ValueError("recommendation context focus is outside the bounded snapshot")

    compiled = _utc(compiled_at)
    snapshot_observed = _parse_stamp(snapshot.observed_at)
    snapshot_expires = _parse_stamp(snapshot.expires_at)
    if snapshot_observed > compiled + MAX_SOURCE_CLOCK_SKEW:
        raise ValueError("cannot compile a recommendation from a future-dated snapshot")
    if compiled >= snapshot_expires:
        raise ValueError("cannot compile a recommendation from an expired snapshot")
    expires = min(compiled + timedelta(seconds=ttl_seconds), snapshot_expires)
    if expires <= compiled:
        raise ValueError("compiled advisory has no remaining lifetime")

    base_provenance = (
        provenance.model_dump(mode="json")
        if isinstance(provenance, SystemOneProvenance)
        else dict(provenance)
    )
    base_provenance.setdefault("knowledge_revision", snapshot.knowledge.knowledge_revision)
    base_provenance.setdefault("neo4j_snapshot_id", snapshot.knowledge.neo4j_snapshot_id)
    base_provenance.setdefault(
        "fleet_projection_generation",
        snapshot.fleet.projection_generation,
    )
    base_provenance.setdefault(
        "fleet_projection_checksum",
        snapshot.fleet.projection_checksum,
    )
    prov = SystemOneProvenance.model_validate(base_provenance)
    if not prov.system_one_config_version:
        raise ValueError("system_one_config_version is required")
    if not prov.model_revision:
        raise ValueError("model_revision is required")
    if not prov.trace_sha256:
        raise ValueError("trace_sha256 is required")

    receipt = receipt_id.strip()
    session = consumer_session_id.strip()
    audience = consumer.strip()
    if not receipt:
        raise ValueError("receipt_id must be non-empty")
    if not session:
        raise ValueError("consumer_session_id must be non-empty")
    if not audience:
        raise ValueError("consumer must be non-empty")

    fleet_priority = (
        [
            FleetPriorityItem(
                handle=selected_handle,
                score=mode_confidence,
                reason="selected by the finite System-One advisory environment at recommendation confidence",
            )
        ]
        if selected_handle is not None
        else []
    )
    context_priority = [context_focus] if context_focus is not None else []

    return HermesSystemOneProfile(
        contract_sha256=CONTRACT_SHA256,
        receipt_id=receipt,
        observed_at=_stamp(compiled),
        expires_at=_stamp(expires),
        binding=SystemOneBinding(
            consumer=audience,
            work_id=snapshot.work.work_id,
            consumer_session_id=session,
            project_fingerprint=project_fingerprint(project_cwd),
            snapshot_sha256=expected_snapshot,
        ),
        advice=SystemOneAdvice(
            mode=rec.advice.mode,
            mode_confidence=mode_confidence,
            policy_disposition=policy_disposition,
            approval_recommended=approval_recommended,
            task_focus=task_focus if task_focus is not None else (snapshot.work.goal or None),
            context_priority=context_priority,
            fleet_priority=fleet_priority,
        ),
        authority=SystemOneAuthority(),
        provenance=prov,
    )
