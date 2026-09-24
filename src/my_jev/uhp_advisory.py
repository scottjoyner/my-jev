from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .agent_policy import ResolvedAgentPolicy, ResolvedDisposition
from .fleet_resolver import FleetPlacementResolution

UHP_VERSION = "2026-09-12"
HERMES_SYSTEM_ONE_PROFILE = "hermes-system-one-heartbeat-v1"
CONTRACT_SHA256 = "5e88c73e7cbb2e46f3b5171951d2a84f0549633fbcb420458d56ae5ada0ffc8f"
DEFAULT_TTL_SECONDS = 600
MAX_TTL_SECONDS = 900
MAX_CONTEXT_PRIORITY = 16
MAX_FLEET_PRIORITY = 16
MAX_LABEL_LENGTH = 128
MAX_REASON_LENGTH = 256
MAX_TASK_FOCUS_LENGTH = 600
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OPAQUE_HANDLE = re.compile(r"^[A-Za-z0-9._:-]+$")


class FleetPriorityItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    handle: str = Field(min_length=1, max_length=MAX_LABEL_LENGTH)
    score: float = Field(ge=0.0, le=1.0)
    reason: str | None = Field(default=None, max_length=MAX_REASON_LENGTH)


class SystemOneAdvice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["chat", "create_tasks", "act", "clarify", "cancel", "abstain"]
    mode_confidence: float = Field(ge=0.0, le=1.0)
    policy_disposition: str | None = Field(default=None, max_length=64)
    approval_recommended: bool | None = None
    task_focus: str | None = Field(default=None, max_length=MAX_TASK_FOCUS_LENGTH)
    context_priority: list[str] = Field(default_factory=list, max_length=MAX_CONTEXT_PRIORITY)
    fleet_priority: list[FleetPriorityItem] = Field(
        default_factory=list,
        max_length=MAX_FLEET_PRIORITY,
    )


class SystemOneAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dispatch_allowed: Literal[False] = False
    approval_granted: Literal[False] = False
    claim_acquired: Literal[False] = False
    mutation_allowed: Literal[False] = False
    routing_authority_changed: Literal[False] = False


class SystemOneBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    consumer: str = Field(min_length=1, max_length=64)
    work_id: str = Field(min_length=1, max_length=128)
    consumer_session_id: str = Field(min_length=1, max_length=128)
    project_fingerprint: str
    snapshot_sha256: str

    def model_post_init(self, __context: Any) -> None:
        if not _SHA256.fullmatch(self.project_fingerprint):
            raise ValueError("project_fingerprint must be a lowercase SHA-256")
        if not _SHA256.fullmatch(self.snapshot_sha256):
            raise ValueError("snapshot_sha256 must be a lowercase SHA-256")


class SystemOneProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_one_config_version: str | None = None
    model_revision: str | None = None
    knowledge_revision: str | None = None
    neo4j_snapshot_id: str | None = None
    fleet_projection_generation: str | None = None
    fleet_projection_checksum: str | None = None
    trace_sha256: str | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.trace_sha256 is not None and not _SHA256.fullmatch(self.trace_sha256):
            raise ValueError("trace_sha256 must be a lowercase SHA-256")


class HermesSystemOneProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str = HERMES_SYSTEM_ONE_PROFILE
    uhp_version: str = UHP_VERSION
    contract_sha256: str = CONTRACT_SHA256
    receipt_id: str = Field(min_length=1, max_length=200)
    observed_at: str
    expires_at: str
    binding: SystemOneBinding
    advice: SystemOneAdvice
    authority: SystemOneAuthority = Field(default_factory=SystemOneAuthority)
    provenance: SystemOneProvenance = Field(default_factory=SystemOneProvenance)


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware")
    return value.astimezone(UTC)


def _rfc3339(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def project_fingerprint(cwd: str | Path) -> str:
    # Local Studio resolves the agent workspace with fs.realpath() before the
    # advisory extension sees it. Mirror that canonical filesystem identity so
    # a symlinked project and its target bind to the same workspace.
    normalized = Path(cwd).expanduser().resolve().as_posix().rstrip("/") or "/"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _clean_labels(values: Sequence[str]) -> list[str]:
    if len(values) > MAX_CONTEXT_PRIORITY:
        raise ValueError("too many context-priority labels")
    out: list[str] = []
    for value in values:
        label = str(value).strip()
        if not label or len(label) > MAX_LABEL_LENGTH:
            raise ValueError("invalid context-priority label")
        out.append(label)
    return out


def _clean_task_focus(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or len(cleaned) > MAX_TASK_FOCUS_LENGTH:
        raise ValueError("invalid task_focus")
    return cleaned


def advisory_mode(decision: ResolvedAgentPolicy) -> str:
    mapping = {
        ResolvedDisposition.CHAT: "chat",
        ResolvedDisposition.CREATE_TASKS: "create_tasks",
        ResolvedDisposition.ACT: "act",
        ResolvedDisposition.ACT_WITH_APPROVAL: "act",
        ResolvedDisposition.PROPOSE_ACTION: "act",
        ResolvedDisposition.CLARIFY: "clarify",
        ResolvedDisposition.CANCEL: "cancel",
        ResolvedDisposition.ABSTAIN: "abstain",
    }
    return mapping[decision.disposition]


def _fleet_priority(
    resolution: FleetPlacementResolution | None,
    handle_by_node_id: Mapping[str, str] | None,
) -> list[FleetPriorityItem]:
    if resolution is None:
        return []
    if not resolution.observer_only or resolution.dispatch_allowed:
        raise ValueError("fleet resolution must remain observer-only")
    ranked = list(resolution.ranked_node_ids)
    if len(ranked) > MAX_FLEET_PRIORITY:
        raise ValueError("too many ranked fleet candidates")
    if not ranked:
        return []
    handles = handle_by_node_id or {}
    missing = [node_id for node_id in ranked if node_id not in handles]
    if missing:
        raise ValueError("every ranked node requires an opaque handle")

    seen: set[str] = set()
    items: list[FleetPriorityItem] = []
    count = len(ranked)
    confidence = max(0.0, min(1.0, float(resolution.model_placement_confidence)))
    for index, node_id in enumerate(ranked):
        handle = str(handles[node_id]).strip()
        if (
            not handle
            or len(handle) > MAX_LABEL_LENGTH
            or not _OPAQUE_HANDLE.fullmatch(handle)
            or handle in seen
        ):
            raise ValueError("invalid or duplicate opaque fleet handle")
        rank_fraction = (count - index) / count
        score = round(confidence * rank_fraction, 6)
        reason = "observer-only rank inside the authoritative eligible fleet set"
        items.append(FleetPriorityItem(handle=handle, score=score, reason=reason))
        seen.add(handle)
    return items


def build_hermes_system_one_profile(
    decision: ResolvedAgentPolicy,
    *,
    receipt_id: str,
    binding: SystemOneBinding | Mapping[str, Any],
    observed_at: datetime | None = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    expires_at_cap: datetime | None = None,
    task_focus: str | None = None,
    context_priority: Sequence[str] = (),
    fleet_resolution: FleetPlacementResolution | None = None,
    fleet_handle_by_node_id: Mapping[str, str] | None = None,
    provenance: SystemOneProvenance | Mapping[str, Any] | None = None,
) -> HermesSystemOneProfile:
    """Build an advisory-only Hermes profile bound to one consumer/work/session.

    The result cannot grant dispatch, approval, claims, mutations, or routing
    authority. Fleet ranking is emitted only after the caller maps authoritative
    node identities to opaque handles.
    """

    receipt = receipt_id.strip()
    if not receipt or len(receipt) > 200:
        raise ValueError("invalid receipt_id")
    if ttl_seconds <= 0 or ttl_seconds > MAX_TTL_SECONDS:
        raise ValueError(f"ttl_seconds must be within 1..{MAX_TTL_SECONDS}")

    observed = _utc(observed_at)
    expires = observed + timedelta(seconds=ttl_seconds)
    if expires_at_cap is not None:
        cap = _utc(expires_at_cap)
        expires = min(expires, cap)
    if expires <= observed:
        raise ValueError("profile expiry must be after observation time")

    prov = (
        provenance
        if isinstance(provenance, SystemOneProvenance)
        else SystemOneProvenance.model_validate(provenance or {})
    )
    bound = (
        binding
        if isinstance(binding, SystemOneBinding)
        else SystemOneBinding.model_validate(binding)
    )

    return HermesSystemOneProfile(
        receipt_id=receipt,
        observed_at=_rfc3339(observed),
        expires_at=_rfc3339(expires),
        binding=bound,
        advice=SystemOneAdvice(
            mode=advisory_mode(decision),
            mode_confidence=max(
                0.0,
                min(1.0, float(decision.model_route_confidence)),
            ),
            policy_disposition=decision.disposition.value,
            approval_recommended=decision.approval_required,
            task_focus=_clean_task_focus(task_focus),
            context_priority=_clean_labels(context_priority),
            fleet_priority=_fleet_priority(
                fleet_resolution,
                fleet_handle_by_node_id,
            ),
        ),
        authority=SystemOneAuthority(),
        provenance=prov,
    )


def build_uhp_response_fixture(
    profile: HermesSystemOneProfile,
    *,
    response_id: str,
    session_id: str,
    harness_id: str,
    model: str,
    created_at: datetime | None = None,
    previous_response_id: str | None = None,
) -> dict[str, Any]:
    """Wrap a profile in the stored UHP response shape used for deterministic probes."""

    if not response_id.startswith("resp_"):
        raise ValueError("response_id must use the UHP resp_ prefix")
    if not session_id.startswith("hsess"):
        raise ValueError("session_id must use the UHP hsess prefix")
    if not harness_id.startswith("chrn_"):
        raise ValueError("harness_id must use the UHP chrn_ prefix")
    if previous_response_id is not None and not previous_response_id.startswith("resp_"):
        raise ValueError("previous_response_id must use the UHP resp_ prefix")
    served_model = model.strip()
    if not served_model:
        raise ValueError("model must be non-empty")

    created = _utc(created_at)
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(created.timestamp()),
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "previous_response_id": previous_response_id,
        "model": served_model,
        "output": [],
        "store": True,
        "usage": None,
        "metadata": {
            "session_id": session_id,
            "harness_id": harness_id,
            "hermes_system_one": profile.model_dump(mode="json"),
        },
    }


def canonical_json(value: Mapping[str, Any] | BaseModel) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Mapping[str, Any] | BaseModel) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
