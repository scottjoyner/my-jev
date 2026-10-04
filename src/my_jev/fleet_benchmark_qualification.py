from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from .fleet_policy import (
    FleetPlacementState,
    PlacementShape,
    eligible_nodes,
)
from .uhp_advisory import (
    MAX_FLEET_PRIORITY,
    MAX_LABEL_LENGTH,
    MAX_REASON_LENGTH,
    SystemOneAuthority,
)

ADVISORY_SCHEMA_VERSION = "fleet-benchmark-advisory-v1"
QUALIFICATION_CONTRACT = "fleet-benchmark-qualification-v1"
DEFAULT_MAX_EVIDENCE_AGE_SECONDS = 900
DEFAULT_MIN_QUALITY_CONFIDENCE = 0.6
DEFAULT_MAX_HEALTH_FRESHNESS_SECONDS = 180.0
MAX_LANES = 128
MAX_REASONS = 32
LATENCY_REFERENCE_SECONDS = 1.0
THROUGHPUT_REFERENCE_TPS = 32.0
_OPAQUE_HANDLE = re.compile(r"^[A-Za-z0-9._:-]+$")

REASON_NO_ELIGIBLE_NODES = (
    "no node survives authoritative health/capability/capacity/locality filters; "
    "benchmark evidence cannot widen eligibility"
)
REASON_NO_CODE_LANE = (
    "no code-qualified lane has fresh evidence above the quality floor; coding work "
    "is deferred instead of being reassigned to a weaker role"
)
REASON_SCOUT_FALLBACK = (
    "a scout-qualified lane is available; recommend decomposition/scout, which is "
    "never an implementation lane"
)
REASON_SCOUT_ONLY = (
    "scout decomposition is recommended for this lane; implementation is not "
    "recommended without code qualification"
)
REASON_SPLIT = (
    "distinct code and review lanes are available; split is advisory across "
    "eligible nodes only"
)
REASON_SPLIT_NO_CHECKPOINT = (
    "split placement is not advised without checkpoint/restart support; staying on "
    "one code lane"
)
REASON_SPLIT_SINGLE_NODE = (
    "split placement is not advised with fewer than two eligible nodes; staying on "
    "one code lane"
)
REASON_SINGLE_LANE = (
    "recommended the highest-ranked eligible lane for the requested role; physical "
    "node selection remains with the deterministic resolver"
)
REASON_STALE_IGNORED = (
    "benchmark evidence older than the configured TTL was ignored for role purposes"
)
REASON_LOW_CONFIDENCE_IGNORED = (
    "benchmark evidence below the quality-confidence floor was ignored for role "
    "purposes"
)
REASON_HEALTH_STALE_IGNORED = (
    "benchmark evidence whose node health observation is older than the configured "
    "freshness bound was ignored for role purposes"
)
REASON_INELIGIBLE_IGNORED = (
    "benchmark evidence for nodes outside the authoritative eligible set was "
    "ignored for preference"
)


class BenchmarkRole(StrEnum):
    """Advisory role vocabulary. A role is a recommendation, never a dispatch."""

    CODE = "code"
    REVIEW = "review"
    SCOUT = "scout"
    SUMMARY = "summary"


class BenchmarkWorkIntent(StrEnum):
    """The role the workload under advisory is asking for."""

    CODING = "coding"
    REVIEW = "review"
    SCOUT = "scout"
    SUMMARY = "summary"


_ROLE_FOR_INTENT: dict[BenchmarkWorkIntent, BenchmarkRole] = {
    BenchmarkWorkIntent.CODING: BenchmarkRole.CODE,
    BenchmarkWorkIntent.REVIEW: BenchmarkRole.REVIEW,
    BenchmarkWorkIntent.SCOUT: BenchmarkRole.SCOUT,
    BenchmarkWorkIntent.SUMMARY: BenchmarkRole.SUMMARY,
}


class QualificationThresholds(BaseModel):
    """Bounded, configurable trust bounds for benchmark evidence.

    Nothing here gates eligibility. Eligibility stays with ``eligible_nodes``.
    These bounds decide only whether a lane is trusted enough to inform an
    advisory preference at all.
    """

    model_config = ConfigDict(extra="forbid")

    max_evidence_age_seconds: int = Field(
        default=DEFAULT_MAX_EVIDENCE_AGE_SECONDS,
        ge=1,
        le=86_400,
    )
    min_quality_confidence: float = Field(
        default=DEFAULT_MIN_QUALITY_CONFIDENCE,
        ge=0.0,
        le=1.0,
    )
    max_health_freshness_seconds: float = Field(
        default=DEFAULT_MAX_HEALTH_FRESHNESS_SECONDS,
        ge=0.0,
    )


class BenchmarkLaneEvidence(BaseModel):
    """One benchmark lane's measured evidence for one physical node.

    Node identity lives here on the host side only. The advisory wire pairs the
    evidence with an opaque handle supplied by the caller, exactly like
    ``_fleet_priority`` in ``uhp_advisory``: handles are never derived here.
    """

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, max_length=128)
    code_qualified: bool = False
    review_qualified: bool = False
    scout_qualified: bool = False
    summary_only: bool = False
    measured_task_family: str = Field(min_length=1, max_length=128)
    quality_confidence: float = Field(ge=0.0, le=1.0)
    latency_seconds: float | None = Field(default=None, ge=0.0)
    throughput_tps: float | None = Field(default=None, ge=0.0)
    resource_pressure: float = Field(ge=0.0, le=1.0)
    health_freshness_seconds: float = Field(ge=0.0)
    observed_at: datetime

    @model_validator(mode="after")
    def _validate_lane(self) -> BenchmarkLaneEvidence:
        if not (
            self.code_qualified
            or self.review_qualified
            or self.scout_qualified
            or self.summary_only
        ):
            raise ValueError("a benchmark lane must claim at least one qualification")
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("benchmark observed_at must be timezone-aware")
        return self


class FleetBenchmarkMatrix(BaseModel):
    """A fleet campaign's benchmark matrix: one bounded lane per node observation."""

    model_config = ConfigDict(extra="forbid")

    campaign_id: str = Field(min_length=1, max_length=128)
    lanes: list[BenchmarkLaneEvidence] = Field(
        min_length=1,
        max_length=MAX_LANES,
    )

    @model_validator(mode="after")
    def _validate_matrix(self) -> FleetBenchmarkMatrix:
        keys = [
            (lane.node_id, lane.observed_at.isoformat())
            for lane in self.lanes
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("benchmark lanes must be unique per (node_id, observed_at)")
        return self


class BenchmarkPreference(BaseModel):
    """One advisory preference for an opaque handle, already proven eligible."""

    model_config = ConfigDict(extra="forbid")

    role: BenchmarkRole
    handle: str = Field(min_length=1, max_length=MAX_LABEL_LENGTH)
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=MAX_REASON_LENGTH)


#: Fields that are *meant* to carry an opaque surrogate rather than identity.
#: A caller-supplied handle is the deliberate indirection this whole design
#: rests on - ``eligible:opaque:<slug>`` routinely contains the node's own
#: name. Scanning it would flag every legitimate handle. Whether a handle is
#: genuinely opaque is the caller's contract, exactly as in ``_fleet_priority``.
_OPAQUE_SURROGATE_KEYS = frozenset({"handle"})


def _contains_identity(payload: object, needle: str) -> bool:
    """True when ``needle`` appears in free text inside a nested payload.

    Values under an opaque-surrogate key (a handle) are skipped: they are the
    intended replacement for identity, not a leak of it.
    """

    if isinstance(payload, str):
        return needle in payload
    if isinstance(payload, Mapping):
        return any(
            _contains_identity(value, needle)
            for key, value in payload.items()
            if key not in _OPAQUE_SURROGATE_KEYS
        )
    if isinstance(payload, (list, tuple, set)):
        return any(_contains_identity(item, needle) for item in payload)
    return False


class FleetBenchmarkAdvisory(BaseModel):
    """Advisory-only benchmark-qualification result.

    Three concerns are kept explicitly separate:

    * ``execution_shape`` is a recommended execution shape, not a placement.
    * ``role_assignment`` is a recommended role, not a dispatch instruction.
    * ``physical_node_selection`` stays with the deterministic resolver.

    ``selected_node_id`` is an assertion surface only: it can name a node solely
    when ``eligible_nodes`` already admitted it, so a maximal qualification
    score on an ineligible node cannot widen eligibility. The model-facing wire
    (``as_model_state``) omits it entirely, along with every other identity.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = ADVISORY_SCHEMA_VERSION
    contract: str = QUALIFICATION_CONTRACT
    campaign_id: str
    observed_at: str
    expires_at: str
    execution_shape: PlacementShape
    role_assignment: BenchmarkRole | None = None
    decomposition_fallback: BenchmarkRole | None = None
    physical_node_selection: Literal["deterministic_resolver"] = "deterministic_resolver"
    preferred: list[BenchmarkPreference] = Field(
        default_factory=list,
        max_length=2 * MAX_FLEET_PRIORITY,
    )
    eligible_node_count: int = Field(default=0, ge=0, le=MAX_LANES)
    selected_node_id: str | None = None
    ignored_stale_lane_count: int = Field(default=0, ge=0, le=MAX_LANES)
    ignored_low_confidence_lane_count: int = Field(default=0, ge=0, le=MAX_LANES)
    ignored_health_stale_lane_count: int = Field(default=0, ge=0, le=MAX_LANES)
    ignored_ineligible_lane_count: int = Field(default=0, ge=0, le=MAX_LANES)
    reasons: list[str] = Field(default_factory=list, max_length=MAX_REASONS)
    authority: SystemOneAuthority = Field(default_factory=SystemOneAuthority)
    observer_only: Literal[True] = True
    dispatch_allowed: Literal[False] = False

    #: Node identities this advisory was derived from, retained so the
    #: identity-free surfaces can be checked rather than trusted. ``reasons``
    #: is operator-facing free text; without this guard a hostname could reach
    #: the decision model inside a reason string.
    _node_ids: frozenset[str] = PrivateAttr(default_factory=frozenset)

    def _reject_identity(self, payload: object) -> object:
        """Fail closed if any emitted string carries a node identity.

        Mirrors ``_fleet_priority`` and ``heartbeat_compile``: an invalid
        identity on the wire is an error, never a silent pass.
        """

        for node_id in self._node_ids:
            if not node_id:
                continue
            if _contains_identity(payload, node_id):
                raise ValueError(
                    "advisory surface leaked a node identity; reasons must be "
                    "authored from opaque handles"
                )
        return payload

    def semantic_decision(self) -> dict[str, Any]:
        """Identity-free decision summary, safe to compare across renamings."""

        return self._reject_identity({
            "execution_shape": self.execution_shape.value,
            "role_assignment": (
                self.role_assignment.value if self.role_assignment is not None else None
            ),
            "decomposition_fallback": (
                self.decomposition_fallback.value
                if self.decomposition_fallback is not None
                else None
            ),
            "preferred_roles": [item.role.value for item in self.preferred],
            "preferred_scores": [item.score for item in self.preferred],
            "eligible_node_count": self.eligible_node_count,
            "ignored_lane_counts": {
                "stale": self.ignored_stale_lane_count,
                "low_confidence": self.ignored_low_confidence_lane_count,
                "health_stale": self.ignored_health_stale_lane_count,
                "ineligible": self.ignored_ineligible_lane_count,
            },
            "reasons": list(self.reasons),
        })

    def model_wire(self) -> dict[str, Any]:
        """The identity-free model-facing payload. No node IDs, no hostnames."""

        payload = self.model_dump(mode="json")
        payload.pop("selected_node_id", None)
        return self._reject_identity(payload)

    def as_model_state(self) -> str:
        # Mirrors FleetPlacementState.as_model_state: hostnames and node IDs are
        # intentionally excluded. The model sees role, shape, and handle-level
        # preference; the deterministic controller retains identity.
        return json.dumps(
            self.model_wire(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class _LaneAssessment:
    lane: BenchmarkLaneEvidence
    age_seconds: float
    fresh: bool
    trusted: bool
    health_fresh: bool
    in_eligible_set: bool

    @property
    def usable(self) -> bool:
        return (
            self.fresh
            and self.trusted
            and self.health_fresh
            and self.in_eligible_set
        )


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("advisory observed_at must be timezone-aware")
    return value.astimezone(UTC)


def _rfc3339(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _role_for(work_intent: BenchmarkWorkIntent | str) -> BenchmarkRole:
    intent = (
        work_intent
        if isinstance(work_intent, BenchmarkWorkIntent)
        else BenchmarkWorkIntent(str(work_intent))
    )
    return _ROLE_FOR_INTENT[intent]


def _assess(
    lane: BenchmarkLaneEvidence,
    observed: datetime,
    thresholds: QualificationThresholds,
    eligible_ids: set[str],
) -> _LaneAssessment:
    age = (observed - lane.observed_at.astimezone(UTC)).total_seconds()
    return _LaneAssessment(
        lane=lane,
        age_seconds=age,
        # Future-dated evidence is not fresh either: clock skew must not buy trust.
        fresh=0.0 <= age <= float(thresholds.max_evidence_age_seconds),
        trusted=lane.quality_confidence >= thresholds.min_quality_confidence,
        health_fresh=lane.health_freshness_seconds <= thresholds.max_health_freshness_seconds,
        in_eligible_set=lane.node_id in eligible_ids,
    )


def _qualifies(lane: BenchmarkLaneEvidence, role: BenchmarkRole) -> bool:
    # A summary-only lane certifies summarisation only. It can never certify
    # implementation, review, or scouting, even if another flag is also set.
    if lane.summary_only:
        return role is BenchmarkRole.SUMMARY
    if role is BenchmarkRole.CODE:
        return lane.code_qualified
    if role is BenchmarkRole.REVIEW:
        return lane.review_qualified
    if role is BenchmarkRole.SCOUT:
        return lane.scout_qualified
    return False


def _qualification_score(
    assessment: _LaneAssessment,
    thresholds: QualificationThresholds,
) -> float:
    """Multiply trust INTO an ordering. Never gate eligibility on the result."""

    lane = assessment.lane
    freshness = max(
        0.0,
        1.0 - (assessment.age_seconds / thresholds.max_evidence_age_seconds),
    )
    health = max(
        0.0,
        1.0
        - (
            lane.health_freshness_seconds
            / max(thresholds.max_health_freshness_seconds, 1e-9)
        ),
    )
    headroom = max(0.0, 1.0 - lane.resource_pressure)
    latency = lane.latency_seconds
    latency_factor = (
        0.0
        if latency is None
        else min(1.0, LATENCY_REFERENCE_SECONDS / max(latency, 1e-9))
    )
    throughput = lane.throughput_tps
    throughput_factor = (
        0.0
        if throughput is None
        else min(1.0, throughput / THROUGHPUT_REFERENCE_TPS)
    )
    score = (
        lane.quality_confidence
        * freshness
        * health
        * headroom
        * latency_factor
        * throughput_factor
    )
    return round(max(0.0, min(1.0, score)), 6)


def _opaque_handle(
    node_id: str,
    handle_by_node_id: Mapping[str, str] | None,
) -> str:
    handles = handle_by_node_id or {}
    handle = str(handles.get(node_id, "")).strip()
    if (
        not handle
        or len(handle) > MAX_LABEL_LENGTH
        or not _OPAQUE_HANDLE.fullmatch(handle)
    ):
        raise ValueError("every advisory lane requires a caller-supplied opaque handle")
    return handle


def _preferences(
    assessments: list[_LaneAssessment],
    role: BenchmarkRole,
    thresholds: QualificationThresholds,
    handle_by_node_id: Mapping[str, str] | None,
) -> tuple[list[BenchmarkPreference], str | None]:
    scored = [
        (
            _qualification_score(assessment, thresholds),
            assessment.lane.node_id,
            assessment.lane,
        )
        for assessment in assessments
        if assessment.usable and _qualifies(assessment.lane, role)
    ]
    if not scored:
        return [], None
    scored.sort(key=lambda item: (-item[0], item[1]))

    seen: set[str] = set()
    items: list[BenchmarkPreference] = []
    leader: str | None = None
    for score, node_id, lane in scored[:MAX_FLEET_PRIORITY]:
        handle = _opaque_handle(node_id, handle_by_node_id)
        if handle in seen:
            raise ValueError("invalid or duplicate opaque fleet handle")
        seen.add(handle)
        items.append(
            BenchmarkPreference(
                role=role,
                handle=handle,
                score=score,
                reason=(
                    f"qualification score {score} for {role.value} on measured family "
                    f"{lane.measured_task_family} inside the eligible fleet set"
                ),
            )
        )
        if leader is None:
            leader = node_id
    return items, leader


def build_fleet_benchmark_advisory(
    matrix: FleetBenchmarkMatrix,
    state: FleetPlacementState,
    *,
    work_intent: BenchmarkWorkIntent | str = BenchmarkWorkIntent.CODING,
    thresholds: QualificationThresholds | None = None,
    observed_at: datetime | None = None,
    handle_by_node_id: Mapping[str, str] | None = None,
) -> FleetBenchmarkAdvisory:
    """Turn a fleet benchmark matrix into a bounded shape/role recommendation.

    The deterministic resolver owns eligibility and physical node selection. This
    function is deliberately downstream of ``eligible_nodes``: it reorders and
    annotates the already-eligible set, recommends an execution shape and a role,
    and never adds a node, dispatches, claims, approves, or mutates anything.
    """

    bounds = thresholds or QualificationThresholds()
    role = _role_for(work_intent)
    observed = _utc(observed_at)

    # Eligibility is computed BEFORE any advisory branch, mirroring the
    # resolve_fleet_placement cascade ordering.
    eligible = eligible_nodes(state)
    eligible_ids = {node.node_id for node in eligible}

    assessments = [
        _assess(lane, observed, bounds, eligible_ids)
        for lane in matrix.lanes
    ]
    stale = [item for item in assessments if not item.fresh]
    low = [item for item in assessments if item.fresh and not item.trusted]
    unhealthy = [
        item for item in assessments if item.fresh and item.trusted and not item.health_fresh
    ]
    ineligible = [
        item for item in assessments
        if item.fresh and item.trusted and item.health_fresh and not item.in_eligible_set
    ]
    usable = [item for item in assessments if item.usable]

    reasons: list[str] = []

    def advisory(
        shape: PlacementShape,
        *,
        assigned: BenchmarkRole | None = None,
        fallback: BenchmarkRole | None = None,
        preferred: list[BenchmarkPreference] | None = None,
        selected: str | None = None,
    ) -> FleetBenchmarkAdvisory:
        collected = list(reasons)
        if stale:
            collected.append(REASON_STALE_IGNORED)
        if low:
            collected.append(REASON_LOW_CONFIDENCE_IGNORED)
        if unhealthy:
            collected.append(REASON_HEALTH_STALE_IGNORED)
        if ineligible:
            collected.append(REASON_INELIGIBLE_IGNORED)
        built = FleetBenchmarkAdvisory(
            campaign_id=matrix.campaign_id,
            observed_at=_rfc3339(observed),
            expires_at=_rfc3339(
                observed + timedelta(seconds=bounds.max_evidence_age_seconds)
            ),
            execution_shape=shape,
            role_assignment=assigned,
            decomposition_fallback=fallback,
            preferred=preferred or [],
            eligible_node_count=len(eligible_ids),
            selected_node_id=selected,
            ignored_stale_lane_count=len(stale),
            ignored_low_confidence_lane_count=len(low),
            ignored_health_stale_lane_count=len(unhealthy),
            ignored_ineligible_lane_count=len(ineligible),
            reasons=collected[:MAX_REASONS],
        )
        built._node_ids = frozenset(node.node_id for node in state.nodes)
        return built

    if not eligible:
        reasons.append(REASON_NO_ELIGIBLE_NODES)
        return advisory(PlacementShape.DEFER)

    primary, leader = _preferences(
        usable,
        role,
        bounds,
        handle_by_node_id,
    )

    if not primary:
        fallback: BenchmarkRole | None = None
        if role is BenchmarkRole.CODE:
            reasons.append(REASON_NO_CODE_LANE)
            scouts, _ = _preferences(
                usable,
                BenchmarkRole.SCOUT,
                bounds,
                handle_by_node_id,
            )
            if scouts:
                fallback = BenchmarkRole.SCOUT
                reasons.append(REASON_SCOUT_FALLBACK)
        else:
            reasons.append(
                f"no {role.value}-qualified lane has fresh evidence above the quality "
                "floor for the requested intent"
            )
        return advisory(PlacementShape.DEFER, fallback=fallback)

    if role is not BenchmarkRole.CODE:
        if role is BenchmarkRole.SCOUT:
            reasons.append(REASON_SCOUT_ONLY)
        reasons.append(REASON_SINGLE_LANE)
        return advisory(
            PlacementShape.PREFERRED_NODE,
            assigned=role,
            preferred=primary,
            selected=leader,
        )

    reviewers, reviewer_leader = _preferences(
        [item for item in usable if item.lane.node_id != leader],
        BenchmarkRole.REVIEW,
        bounds,
        handle_by_node_id,
    )
    if reviewers and reviewer_leader is not None:
        if len(eligible_ids) < 2:
            reasons.append(REASON_SPLIT_SINGLE_NODE)
        elif not state.checkpoint_supported:
            reasons.append(REASON_SPLIT_NO_CHECKPOINT)
        else:
            reasons.append(REASON_SPLIT)
            return advisory(
                PlacementShape.SPLIT,
                assigned=BenchmarkRole.CODE,
                preferred=primary + reviewers,
            )

    reasons.append(REASON_SINGLE_LANE)
    return advisory(
        PlacementShape.PREFERRED_NODE,
        assigned=BenchmarkRole.CODE,
        preferred=primary,
        selected=leader,
    )