from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .agent_policy import AgentPolicyState

HEARTBEAT_SNAPSHOT_VERSION = "hermes-heartbeat-snapshot-v1"
DEFAULT_SNAPSHOT_TTL_SECONDS = 300
MAX_SNAPSHOT_TTL_SECONDS = 600
MAX_FACTS = 64
MAX_NOTE_REFS = 24
MAX_BLOCKERS = 24
MAX_PENDING_APPROVALS = 24
MAX_ACTIVE_CLAIMS = 24
MAX_TEXT = 600
MAX_REF = 256
MAX_ELIGIBLE_HANDLES = 16
_OPAQUE_HANDLE = re.compile(r"^[A-Za-z0-9._:-]+$")

_FORBIDDEN_METADATA_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)


class KnowledgeFact(BaseModel):
    subject: str = Field(min_length=1, max_length=MAX_REF)
    predicate: str = Field(min_length=1, max_length=128)
    object: str = Field(min_length=1, max_length=MAX_TEXT)
    source: Literal["neo4j", "markdown", "assistx", "operator"]
    source_ref: str = Field(min_length=1, max_length=MAX_REF)
    sensitivity: Literal["non_secret"] = "non_secret"


class WorkStateSummary(BaseModel):
    work_id: str = Field(min_length=1, max_length=128)
    session_id: str | None = Field(default=None, max_length=128)
    status: Literal[
        "idle",
        "active",
        "blocked",
        "waiting",
        "complete",
        "cancelled",
    ]
    goal: str = Field(default="", max_length=MAX_TEXT)
    priority: int = Field(default=50, ge=0, le=100)
    blockers: list[str] = Field(default_factory=list, max_length=MAX_BLOCKERS)
    pending_approvals: list[str] = Field(
        default_factory=list,
        max_length=MAX_PENDING_APPROVALS,
    )
    active_claims: list[str] = Field(default_factory=list, max_length=MAX_ACTIVE_CLAIMS)
    last_action_class: str | None = Field(default=None, max_length=128)


class KnowledgeStateSummary(BaseModel):
    knowledge_revision: str = Field(min_length=1, max_length=128)
    markdown_revision: str | None = Field(default=None, max_length=128)
    neo4j_snapshot_id: str | None = Field(default=None, max_length=128)
    note_refs: list[str] = Field(default_factory=list, max_length=MAX_NOTE_REFS)
    facts: list[KnowledgeFact] = Field(default_factory=list, max_length=MAX_FACTS)


class FleetStateSummary(BaseModel):
    projection_generation: str = Field(min_length=1, max_length=128)
    projection_checksum: str = Field(min_length=1, max_length=128)
    observation_snapshot_id: str = Field(min_length=1, max_length=128)
    eligible_count: int = Field(ge=0)
    eligible_handles: list[str] = Field(default_factory=list, max_length=MAX_ELIGIBLE_HANDLES)
    drained_count: int = Field(default=0, ge=0)
    unhealthy_count: int = Field(default=0, ge=0)
    pressure: Literal["idle", "light", "moderate", "high", "saturated"] = "idle"

    @model_validator(mode="after")
    def _validate_handles(self) -> FleetStateSummary:
        if self.eligible_count != len(self.eligible_handles):
            raise ValueError("eligible_count must match eligible_handles length")
        if len(set(self.eligible_handles)) != len(self.eligible_handles):
            raise ValueError("eligible_handles must be unique")
        for handle in self.eligible_handles:
            if (
                handle == "none"
                or not handle
                or len(handle) > MAX_REF
                or not _OPAQUE_HANDLE.fullmatch(handle)
            ):
                raise ValueError("invalid opaque eligible handle")
        return self


class AuthorityContext(BaseModel):
    """Observed host policy facts; these are context, never grants from the model.

    Missing projector evidence is deliberately fail-closed. A caller must
    explicitly assert every permissive host fact it wants the decision model to
    observe.
    """

    speaker_verified: bool = False
    actions_allowed: bool = False
    local_writes_allowed: bool = False
    external_actions_allowed: bool = False
    privileged_actions_allowed: bool = False
    approval_gate_available: bool = False


class HeartbeatSnapshot(BaseModel):
    schema_version: Literal["hermes-heartbeat-snapshot-v1"] = HEARTBEAT_SNAPSHOT_VERSION
    observed_at: str
    expires_at: str
    work: WorkStateSummary
    knowledge: KnowledgeStateSummary
    fleet: FleetStateSummary
    authority_context: AuthorityContext = Field(default_factory=AuthorityContext)
    available_capabilities: list[str] = Field(default_factory=list, max_length=32)
    available_tools: list[str] = Field(default_factory=list, max_length=32)
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_bounded_text(self) -> HeartbeatSnapshot:
        for group in (
            self.work.blockers,
            self.work.pending_approvals,
            self.work.active_claims,
            self.knowledge.note_refs,
            self.available_capabilities,
            self.available_tools,
        ):
            for value in group:
                if not value or len(value) > MAX_REF:
                    raise ValueError("snapshot list values must be non-empty and bounded")

        for key, value in self.metadata.items():
            lowered = key.lower()
            if any(part in lowered for part in _FORBIDDEN_METADATA_PARTS):
                raise ValueError(f"credential-like metadata key is forbidden: {key}")
            if not key or len(key) > 128 or len(value) > MAX_REF:
                raise ValueError("snapshot metadata keys/values must be bounded")
        return self


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware")
    return value.astimezone(UTC)


def _stamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def build_heartbeat_snapshot(
    *,
    work: WorkStateSummary | Mapping[str, Any],
    knowledge: KnowledgeStateSummary | Mapping[str, Any],
    fleet: FleetStateSummary | Mapping[str, Any],
    authority_context: AuthorityContext | Mapping[str, Any] | None = None,
    available_capabilities: Sequence[str] = (),
    available_tools: Sequence[str] = (),
    metadata: Mapping[str, str] | None = None,
    observed_at: datetime | None = None,
    ttl_seconds: int = DEFAULT_SNAPSHOT_TTL_SECONDS,
) -> HeartbeatSnapshot:
    if ttl_seconds <= 0 or ttl_seconds > MAX_SNAPSHOT_TTL_SECONDS:
        raise ValueError(
            f"ttl_seconds must be within 1..{MAX_SNAPSHOT_TTL_SECONDS}"
        )
    observed = _utc(observed_at)
    expires = observed + timedelta(seconds=ttl_seconds)
    return HeartbeatSnapshot(
        observed_at=_stamp(observed),
        expires_at=_stamp(expires),
        work=work if isinstance(work, WorkStateSummary) else WorkStateSummary.model_validate(work),
        knowledge=(
            knowledge
            if isinstance(knowledge, KnowledgeStateSummary)
            else KnowledgeStateSummary.model_validate(knowledge)
        ),
        fleet=fleet if isinstance(fleet, FleetStateSummary) else FleetStateSummary.model_validate(fleet),
        authority_context=(
            authority_context
            if isinstance(authority_context, AuthorityContext)
            else AuthorityContext.model_validate(authority_context or {})
        ),
        available_capabilities=list(available_capabilities),
        available_tools=list(available_tools),
        metadata={str(k): str(v) for k, v in (metadata or {}).items()},
    )


def canonical_snapshot_json(snapshot: HeartbeatSnapshot) -> str:
    return json.dumps(
        snapshot.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def snapshot_sha256(snapshot: HeartbeatSnapshot) -> str:
    return hashlib.sha256(canonical_snapshot_json(snapshot).encode("utf-8")).hexdigest()


def agent_policy_state_from_snapshot(
    snapshot: HeartbeatSnapshot,
    *,
    utterance: str,
    conversation_summary: str = "",
    source: str = "hermes_heartbeat",
    speaker_id: str = "",
) -> AgentPolicyState:
    """Project the bounded heartbeat snapshot into the existing mode-decision contract."""

    active = snapshot.work.status in {"active", "blocked", "waiting"}
    return AgentPolicyState(
        utterance=utterance,
        conversation_summary=conversation_summary,
        source=source,
        speaker_id=speaker_id,
        speaker_verified=snapshot.authority_context.speaker_verified,
        foreground=True,
        active_work=(
            [
                snapshot.work.work_id,
                *snapshot.work.blockers,
            ]
            if active
            else []
        ),
        pending_approvals=list(snapshot.work.pending_approvals),
        available_capabilities=list(snapshot.available_capabilities),
        available_tools=list(snapshot.available_tools),
        actions_allowed=snapshot.authority_context.actions_allowed,
        external_actions_allowed=snapshot.authority_context.external_actions_allowed,
        privileged_actions_allowed=snapshot.authority_context.privileged_actions_allowed,
        metadata={
            "heartbeat_snapshot_version": snapshot.schema_version,
            "heartbeat_snapshot_sha256": snapshot_sha256(snapshot),
            "knowledge_revision": snapshot.knowledge.knowledge_revision,
            "markdown_revision": snapshot.knowledge.markdown_revision or "",
            "neo4j_snapshot_id": snapshot.knowledge.neo4j_snapshot_id or "",
            "fleet_projection_generation": snapshot.fleet.projection_generation,
            "fleet_projection_checksum": snapshot.fleet.projection_checksum,
            "fleet_observation_snapshot_id": snapshot.fleet.observation_snapshot_id,
            **snapshot.metadata,
        },
    )
