from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schema import DecisionRecord, QuestionSpec, QuestionType, TargetSpec

MODEL_EVIDENCE_SNAPSHOT_VERSION = "model-evidence-snapshot-v1"
MODEL_SELECTION_POLICY_CONTRACT = "model-selection-advisory-v1"

DEFAULT_MODEL_EVIDENCE_TTL_SECONDS = 300
MAX_MODEL_EVIDENCE_TTL_SECONDS = 600
MAX_MODEL_CANDIDATES = 16
MAX_MODALITIES = 8
MAX_TASK_FAMILIES = 16
MAX_REQUEST_TASK_FAMILIES = 8
MAX_TEXT = 256

_OPAQUE_HANDLE = re.compile(r"^[A-Za-z0-9._:-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ModelNeedProfile(StrEnum):
    FAST = "fast"
    BALANCED = "balanced"
    COMPLEX = "complex"
    AGENTIC = "agentic"
    LONG_CONTEXT = "long-context"


PROFILE_OPTIONS = [profile.value for profile in ModelNeedProfile]


class ModelRequestNeeds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: ModelNeedProfile = ModelNeedProfile.BALANCED
    complexity: int = Field(default=2, ge=0, le=4)
    minimum_task_fit_score: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
    )
    required_context_tokens: int = Field(default=0, ge=0)
    latency_target_ms: int | None = Field(default=None, gt=0)
    required_modalities: list[str] = Field(default_factory=list, max_length=MAX_MODALITIES)
    task_families: list[str] = Field(
        default_factory=list,
        max_length=MAX_REQUEST_TASK_FAMILIES,
    )

    @model_validator(mode="after")
    def _validate_modalities(self) -> ModelRequestNeeds:
        if len(set(self.required_modalities)) != len(self.required_modalities):
            raise ValueError("required_modalities must be unique")
        for modality in self.required_modalities:
            if not modality or len(modality) > 64:
                raise ValueError("required modality names must be non-empty and bounded")
        if len(set(self.task_families)) != len(self.task_families):
            raise ValueError("task_families must be unique")
        for family in self.task_families:
            if not family or len(family) > 128:
                raise ValueError("task family names must be non-empty and bounded")
        return self


class ModelExecutionEnvelope(BaseModel):
    """Cross-node execution evidence with physical identities intentionally removed."""

    model_config = ConfigDict(extra="forbid")

    benchmark_lane_count: int = Field(default=0, ge=0, le=128)
    distinct_node_count: int = Field(default=0, ge=0, le=128)
    currently_eligible_replica_count: int = Field(default=0, ge=0, le=128)

    generation_tps_best: float | None = Field(default=None, ge=0.0)
    generation_tps_median: float | None = Field(default=None, ge=0.0)
    prompt_tps_best: float | None = Field(default=None, ge=0.0)
    prompt_tps_median: float | None = Field(default=None, ge=0.0)

    min_verified_context_tokens: int | None = Field(default=None, ge=0)
    max_verified_context_tokens: int | None = Field(default=None, ge=0)

    execution_evidence_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    task_success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    task_trial_count: int = Field(default=0, ge=0)
    backend_classes: list[str] = Field(default_factory=list, max_length=16)
    evidence_states: list[str] = Field(default_factory=list, max_length=16)
    measurement_classes: list[str] = Field(default_factory=list, max_length=16)
    roles_observed: list[str] = Field(default_factory=list, max_length=16)
    task_families_observed: list[str] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def _validate_envelope(self) -> ModelExecutionEnvelope:
        if (
            self.min_verified_context_tokens is not None
            and self.max_verified_context_tokens is not None
            and self.min_verified_context_tokens > self.max_verified_context_tokens
        ):
            raise ValueError("execution context envelope is inverted")
        for name, values in (
            ("backend_classes", self.backend_classes),
            ("evidence_states", self.evidence_states),
            ("measurement_classes", self.measurement_classes),
            ("roles_observed", self.roles_observed),
            ("task_families_observed", self.task_families_observed),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must be unique")
            for value in values:
                if not value or len(value) > 64:
                    raise ValueError(f"{name} values must be non-empty and bounded")
        return self


class ModelCandidateEvidence(BaseModel):
    """Bounded evidence for one already-eligible opaque artifact handle.

    The same exact artifact may have useful benchmark history on many nodes and
    backends. Those measurements are summarized into an execution envelope rather
    than collapsed to one host. Physical node/provider coordinates are deliberately
    absent from model-visible state. Eligibility is resolved before this object is
    created; the decision model can rank handles but cannot make a blocked artifact
    eligible.
    """

    model_config = ConfigDict(extra="forbid")

    handle: str = Field(min_length=1, max_length=MAX_TEXT)
    scenario_scores: dict[str, float]
    task_fit_scores: dict[str, float] = Field(default_factory=dict)

    identity_confidence: float = Field(ge=0.0, le=1.0)
    evidence_coverage: float = Field(ge=0.0, le=1.0)
    quantization_confidence: float = Field(ge=0.0, le=1.0)

    capability_score: float | None = Field(default=None, ge=0.0, le=100.0)
    coding_score: float | None = Field(default=None, ge=0.0, le=100.0)
    agentic_score: float | None = Field(default=None, ge=0.0, le=100.0)

    execution: ModelExecutionEnvelope = Field(default_factory=ModelExecutionEnvelope)
    local_weight_gib: float | None = Field(default=None, ge=0.0)
    local_reliability: float | None = Field(default=None, ge=0.0, le=1.0)

    quantization_class: str = Field(default="unknown", min_length=1, max_length=64)
    quant_quality_retention: float | None = Field(default=None, ge=0.0, le=2.0)

    modalities: list[str] = Field(default_factory=list, max_length=MAX_MODALITIES)
    measured_task_families: list[str] = Field(
        default_factory=list,
        max_length=MAX_TASK_FAMILIES,
    )

    @model_validator(mode="after")
    def _validate_candidate(self) -> ModelCandidateEvidence:
        if (
            self.handle == "none"
            or not _OPAQUE_HANDLE.fullmatch(self.handle)
        ):
            raise ValueError("model candidate handle must be opaque and token-safe")

        allowed = set(PROFILE_OPTIONS)
        unknown_profiles = set(self.scenario_scores) - allowed
        if unknown_profiles:
            raise ValueError(
                f"unknown scenario score profiles: {sorted(unknown_profiles)}"
            )
        unknown_task_fit_profiles = set(self.task_fit_scores) - allowed
        if unknown_task_fit_profiles:
            raise ValueError(
                "unknown task-fit score profiles: "
                f"{sorted(unknown_task_fit_profiles)}"
            )
        if not self.scenario_scores:
            raise ValueError("at least one scenario score is required")
        for score_name, values in (
            ("scenario", self.scenario_scores),
            ("task-fit", self.task_fit_scores),
        ):
            for name, value in values.items():
                if not 0.0 <= value <= 100.0:
                    raise ValueError(
                        f"{score_name} score out of range for {name}"
                    )

        for group_name, values in (
            ("modalities", self.modalities),
            ("measured_task_families", self.measured_task_families),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{group_name} must be unique")
            for value in values:
                if not value or len(value) > 64:
                    raise ValueError(f"{group_name} values must be non-empty and bounded")
        return self


class ModelEvidenceProvenance(BaseModel):
    """Revision/hash bindings for the evidence producer.

    These identifiers support replay and audit. They are not model authority.
    """

    knowledge_revision: str = Field(min_length=1, max_length=128)
    scoring_contract_sha256: str
    local_benchmark_snapshot_sha256: str
    heartbeat_snapshot_sha256: str
    artificial_analysis_snapshot_sha256: str | None = None
    task_bundle_revision: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def _validate_hashes(self) -> ModelEvidenceProvenance:
        for name in (
            "scoring_contract_sha256",
            "local_benchmark_snapshot_sha256",
            "heartbeat_snapshot_sha256",
            "artificial_analysis_snapshot_sha256",
        ):
            value = getattr(self, name)
            if value is not None and not _SHA256.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
        return self


class ModelEvidenceSnapshot(BaseModel):
    schema_version: str = MODEL_EVIDENCE_SNAPSHOT_VERSION
    observed_at: str
    expires_at: str
    request: ModelRequestNeeds
    candidates: list[ModelCandidateEvidence] = Field(
        default_factory=list,
        max_length=MAX_MODEL_CANDIDATES,
    )
    excluded_counts: dict[str, int] = Field(default_factory=dict)
    provenance: ModelEvidenceProvenance

    @model_validator(mode="after")
    def _validate_snapshot(self) -> ModelEvidenceSnapshot:
        if self.schema_version != MODEL_EVIDENCE_SNAPSHOT_VERSION:
            raise ValueError(
                f"schema_version must be {MODEL_EVIDENCE_SNAPSHOT_VERSION}"
            )

        handles = [candidate.handle for candidate in self.candidates]
        if len(set(handles)) != len(handles):
            raise ValueError("candidate handles must be unique")

        required_modalities = set(self.request.required_modalities)
        profile = self.request.profile.value
        for candidate in self.candidates:
            if profile not in candidate.scenario_scores:
                raise ValueError(
                    f"candidate {candidate.handle} is missing {profile!r} score"
                )
            if self.request.required_context_tokens:
                verified = candidate.execution.max_verified_context_tokens
                if verified is None:
                    raise ValueError(
                        f"candidate {candidate.handle} lacks verified context evidence"
                    )
                if verified < self.request.required_context_tokens:
                    raise ValueError(
                        f"candidate {candidate.handle} does not meet required context"
                    )
            if not required_modalities.issubset(set(candidate.modalities)):
                raise ValueError(
                    f"candidate {candidate.handle} does not meet required modalities"
                )

        for reason, count in self.excluded_counts.items():
            if not reason or len(reason) > 128:
                raise ValueError("excluded-count reasons must be non-empty and bounded")
            if count < 0:
                raise ValueError("excluded-count values must be non-negative")

        observed = _parse_timestamp(self.observed_at, "observed_at")
        expires = _parse_timestamp(self.expires_at, "expires_at")
        if expires <= observed:
            raise ValueError("expires_at must be after observed_at")
        if expires - observed > timedelta(seconds=MAX_MODEL_EVIDENCE_TTL_SECONDS):
            raise ValueError(
                f"model evidence TTL must not exceed "
                f"{MAX_MODEL_EVIDENCE_TTL_SECONDS} seconds"
            )
        return self

    def as_model_state(self) -> str:
        """Return bounded decision features without provider/runtime coordinates."""

        payload = {
            "request": self.request.model_dump(mode="json"),
            "candidates": [
                {
                    "handle": candidate.handle,
                    "scenario_score": candidate.scenario_scores[self.request.profile.value],
                    "task_fit_score": candidate.task_fit_scores.get(
                        self.request.profile.value
                    ),
                    "identity_confidence": candidate.identity_confidence,
                    "evidence_coverage": candidate.evidence_coverage,
                    "quantization_confidence": candidate.quantization_confidence,
                    "capability_score": candidate.capability_score,
                    "coding_score": candidate.coding_score,
                    "agentic_score": candidate.agentic_score,
                    "execution": candidate.execution.model_dump(mode="json"),
                    "local_weight_gib": candidate.local_weight_gib,
                    "local_reliability": candidate.local_reliability,
                    "quantization_class": candidate.quantization_class,
                    "quant_quality_retention": candidate.quant_quality_retention,
                    "modalities": sorted(candidate.modalities),
                    "measured_task_families": sorted(candidate.measured_task_families),
                }
                for candidate in self.candidates
            ],
            "excluded_counts": dict(sorted(self.excluded_counts.items())),
            "evidence": {
                "knowledge_revision": self.provenance.knowledge_revision,
                "task_bundle_revision": self.provenance.task_bundle_revision,
            },
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _parse_timestamp(value: str, field: str) -> datetime:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be valid ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(UTC)


def _stamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_model_evidence_snapshot(
    *,
    request: ModelRequestNeeds,
    candidates: list[ModelCandidateEvidence],
    provenance: ModelEvidenceProvenance,
    excluded_counts: dict[str, int] | None = None,
    observed_at: datetime | None = None,
    ttl_seconds: int = DEFAULT_MODEL_EVIDENCE_TTL_SECONDS,
) -> ModelEvidenceSnapshot:
    if ttl_seconds <= 0 or ttl_seconds > MAX_MODEL_EVIDENCE_TTL_SECONDS:
        raise ValueError(
            f"ttl_seconds must be within 1..{MAX_MODEL_EVIDENCE_TTL_SECONDS}"
        )
    observed = observed_at or datetime.now(UTC)
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware")
    observed = observed.astimezone(UTC)
    return ModelEvidenceSnapshot(
        observed_at=_stamp(observed),
        expires_at=_stamp(observed + timedelta(seconds=ttl_seconds)),
        request=request,
        candidates=candidates,
        excluded_counts=excluded_counts or {},
        provenance=provenance,
    )


def canonical_model_evidence_json(snapshot: ModelEvidenceSnapshot) -> str:
    return json.dumps(
        snapshot.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def model_evidence_sha256(snapshot: ModelEvidenceSnapshot) -> str:
    return hashlib.sha256(
        canonical_model_evidence_json(snapshot).encode("utf-8")
    ).hexdigest()


def evidence_adjusted_score(
    candidate: ModelCandidateEvidence,
    profile: ModelNeedProfile | str,
) -> float:
    """Conservative deterministic score used only for the baseline fallback.

    The raw scenario score remains model-visible. This adjustment prevents a
    perfect result on a tiny/narrow sample from outranking broadly supported
    evidence solely because its observed score is 100.
    """

    profile_name = profile.value if isinstance(profile, ModelNeedProfile) else str(profile)
    raw_task_fit = float(
        candidate.task_fit_scores.get(
            profile_name,
            candidate.scenario_scores[profile_name],
        )
    )
    coverage = max(0.0, min(1.0, candidate.evidence_coverage))
    execution_confidence = max(
        0.0,
        min(1.0, candidate.execution.execution_evidence_confidence),
    )
    return (
        raw_task_fit
        * (coverage ** 0.5)
        * candidate.identity_confidence
        * candidate.quantization_confidence
        * execution_confidence
    )


def rank_candidate_handles(snapshot: ModelEvidenceSnapshot) -> list[str]:
    """Deterministic evidence-adjusted ranking for already-eligible handles only."""

    profile = snapshot.request.profile
    ranked = sorted(
        snapshot.candidates,
        key=lambda candidate: (
            -evidence_adjusted_score(candidate, profile),
            -candidate.task_fit_scores.get(
                profile.value,
                candidate.scenario_scores[profile.value],
            ),
            -candidate.scenario_scores[profile.value],
            -candidate.evidence_coverage,
            -candidate.identity_confidence,
            candidate.handle,
        ),
    )
    return [candidate.handle for candidate in ranked]


def model_selection_question(snapshot: ModelEvidenceSnapshot) -> QuestionSpec:
    options = [candidate.handle for candidate in snapshot.candidates]
    options.append("abstain")
    if len(options) < 2:
        options.append("no_eligible_model")
    return QuestionSpec(
        type=QuestionType.CHOICE,
        instructions=(
            "Choose the best already-eligible opaque model handle for the bounded "
            "request evidence. Choose abstain when the evidence is insufficient. "
            "This recommendation cannot grant routing, dispatch, approval, claims, "
            "tool access, or mutation authority."
        ),
        options=options,
        metadata={
            "task_id": "model.select",
            "policy_contract": MODEL_SELECTION_POLICY_CONTRACT,
            "loss_weight": 2.0,
        },
    )


def build_model_selection_record(
    snapshot: ModelEvidenceSnapshot,
    *,
    target_handle: str | None = None,
) -> DecisionRecord:
    question = model_selection_question(snapshot)
    targets = None
    if target_handle is not None:
        if target_handle not in question.options:
            raise ValueError("target_handle must be an eligible handle or abstain")
        targets = {
            "model_selection": TargetSpec(index=question.options.index(target_handle))
        }

    return DecisionRecord(
        state=snapshot.as_model_state(),
        questions={"model_selection": question},
        targets=targets,
        metadata={
            "domain": "model_selection",
            "policy_contract": MODEL_SELECTION_POLICY_CONTRACT,
            "model_evidence_snapshot_sha256": model_evidence_sha256(snapshot),
            "heartbeat_snapshot_sha256": snapshot.provenance.heartbeat_snapshot_sha256,
            "dispatch_allowed": False,
            "approval_granted": False,
            "claim_acquired": False,
            "mutation_allowed": False,
            "routing_authority_changed": False,
        },
    )
