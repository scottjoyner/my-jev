"""Plan GPU time for a model version before the version exists.

Two things are missing from the run lifecycle and this module supplies both.

**Planned versions.** ``registry.ExperimentEntry`` records what happened: a run
that finished, with the SHAs of everything it produced. Nothing names the version
that is *about to* exist -- its identity, its parent, and the GPU it is asking
for -- while the decision to spend that GPU is still cheap to change. A plan is
that artifact. It is derived from a fine-tuning job (``ExperimentSpec``) and
carries a stable ``version_id``, so the same job always plans to the same version
and changing the job changes the version rather than silently amending it.

**GPU allocation.** ``scale_readiness.evaluate_scale_readiness`` already answers
"is a small baseline good enough to justify the full run", and documents itself
as a resource-allocation gate. This module does not duplicate its thresholds; it
*consults* it, and refuses to allocate training GPU when the answer is no. That
ordering matters: the cheapest question is asked first, and a version that cannot
be justified never reaches the planner's budget arithmetic.

Advisory only. A plan reserves nothing, dispatches nothing, and grants no
authority. Executing it is someone else's decision, which is why
``PlanAuthority`` exists and is all-false.

Fail-closed in one specific way, documented on :func:`build_model_version_plan`:
the stage chain is all-or-nothing from the front. Benchmarking and evaluating a
version that was never trained produces no evidence about the version, so the
planner would rather fund nothing than fund a chain it cannot finish verifying.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StageKind(StrEnum):
    """One link in the chain that makes a version verifiable.

    The order is the dependency order and is load-bearing: ``benchmark`` and
    ``evaluate`` are evidence *about* a trained version, so scheduling either
    without ``train`` yields nothing about the thing being planned.
    """

    train = "train"
    benchmark = "benchmark"
    evaluate = "evaluate"


#: Dependency order. A plan always schedules a prefix of this list.
STAGE_CHAIN: tuple[StageKind, ...] = (
    StageKind.train,
    StageKind.benchmark,
    StageKind.evaluate,
)


class PickerKind(StrEnum):
    """The named selections a plan makes, recorded so a plan is auditable.

    "Which parent do we branch from", "which training config", "which benchmark
    suite", "which gate thresholds". Each is resolved by a deterministic rule and
    recorded by name, so a reviewer can see not just what was chosen but which
    rule chose it -- and two plans can be compared rule-for-rule.
    """

    parent_version = "parent_version"
    train_config = "train_config"
    benchmark_suite = "benchmark_suite"
    eval_thresholds = "eval_thresholds"


class ReadinessVerdict(BaseModel):
    """Whether a small baseline justifies spending GPU on the full run.

    Mirrors ``scale_readiness.evaluate_scale_readiness``, whose verdict carries
    ``passed`` and ``completed`` rather than raising for the ordinary
    not-yet-ready case. Both are required here: an incomplete baseline is not a
    pass, and a completed baseline that failed is also not a pass.
    """

    model_config = ConfigDict(extra="forbid")

    completed: bool = False
    passed: bool = False
    #: Set when readiness could not be determined. Never used to grant a pass.
    reason: str | None = None
    #: Provenance of the verdict: where it came from and when.
    source: str | None = None
    evaluated_at: str | None = None


def readiness_from_run(run_dir: str, *, source: str | None = None) -> ReadinessVerdict:
    """Read a readiness verdict, mapping "incomplete" to not-ready rather than raising.

    ``evaluate_scale_readiness`` raises ``ValueError`` for an incomplete baseline.
    That is the right behaviour for a caller that needs the evidence, and the
    wrong behaviour for a planner: an unfinished baseline is an ordinary state,
    not a crash, and it means "do not spend the GPU yet".
    """
    from .scale_readiness import evaluate_scale_readiness

    moment = datetime.now(UTC).replace(microsecond=0).isoformat()
    try:
        raw = evaluate_scale_readiness(run_dir)
    except ValueError as exc:
        return ReadinessVerdict(
            completed=False,
            passed=False,
            reason=str(exc),
            source=source or f"scale_readiness:{run_dir}",
            evaluated_at=moment,
        )
    return ReadinessVerdict(
        completed=bool(raw.get("completed", False)),
        passed=bool(raw.get("passed", False)),
        source=source or f"scale_readiness:{run_dir}",
        evaluated_at=moment,
    )


class PlanAuthority(BaseModel):
    """All-false authority block.

    ``Literal[False]`` means a caller supplying ``True`` is a validation error
    rather than a silently-ignored field, which is the same choice
    ``SystemOneAuthority`` and ``BenchmarkAdvisoryAuthority`` make.
    """

    model_config = ConfigDict(extra="forbid")

    dispatch_allowed: Literal[False] = False
    gpu_acquired: Literal[False] = False
    lease_acquired: Literal[False] = False
    promotion_granted: Literal[False] = False
    runtime_authority_changed: Literal[False] = False


@dataclass(frozen=True, order=True)
class GpuCandidate:
    """One GPU the planner may allocate from.

    ``free_gpu_minutes`` is a *claim about availability*, supplied by the caller
    from a fleet inventory. The planner never probes hardware and never assumes a
    device is idle: it can only allocate from what it is told exists, so a stale
    inventory yields a plan that reserves too little rather than one that
    double-books a device somebody is using.
    """

    gpu_id: str
    free_gpu_minutes: float
    #: Whether this device is trusted for the planned stage at all. A device
    #: marked untrusted is never allocated, regardless of free time.
    trusted: bool = True

    def __post_init__(self) -> None:
        if not self.gpu_id.strip():
            raise ValueError("gpu_id must be non-empty")
        if self.free_gpu_minutes < 0.0:
            raise ValueError("free_gpu_minutes must not be negative")


class PlannedStage(BaseModel):
    """One stage of the chain, with the picker that resolved it."""

    model_config = ConfigDict(extra="forbid")

    kind: StageKind
    #: Which selection rule ran. Recorded so a plan can be audited and compared
    #: rule-for-rule rather than only by outcome.
    picker: PickerKind
    #: What the picker chose, in the form the rule emits.
    selection: str
    #: Why that choice, in one sentence. Composed from facts, not scraped from a
    #: source's prose, so an edit to unrelated wording cannot silently repoint it.
    rationale: str
    estimated_gpu_minutes: float = Field(ge=0.0)
    #: Set once the stage has actually run. Plans are estimates until they are not.
    measured_gpu_minutes: float | None = Field(default=None, ge=0.0)
    #: GPUs this stage would be placed on. Empty only when unallocated.
    gpu_ids: list[str] = Field(default_factory=list)

    @property
    def minutes_used_for_budget(self) -> float:
        """Measured wins over estimated.

        A plan built after some stages have run must not keep budgeting the
        estimate for them, or repeated planning inflates the total without any
        new work being justified.
        """
        if self.measured_gpu_minutes is not None:
            return self.measured_gpu_minutes
        return self.estimated_gpu_minutes

    @property
    def calibrated(self) -> bool:
        return self.measured_gpu_minutes is not None


class ModelVersionPlan(BaseModel):
    """GPU plan for one model version that does not exist yet."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "my-jev-gpu-plan-v1"
    authority: PlanAuthority = PlanAuthority()
    advisory_only: Literal[True] = True

    #: Stable identity derived from the fine-tuning job. Same job, same version.
    version_id: str
    #: The job this was planned from, identified by content hash.
    job_sha256: str
    #: Human-readable job name, carried for operator readability only.
    experiment: str
    #: Resolved parent, or None when the job names no parent to branch from.
    parent: str | None = None
    backbone: str
    head_kind: str
    #: Hash over the data paths, so two jobs differing only in dataset do not
    #: collide on a version id.
    data_lineage_sha256: str

    #: The full chain, always. ``gpu_ids`` is empty on a stage that was not
    #: placed, which is how an unfunded plan stays inspectable.
    stages: list[PlannedStage] = Field(default_factory=list)
    budget_gpu_minutes: float = Field(ge=0.0)
    total_estimated_gpu_minutes: float = Field(default=0.0, ge=0.0)

    #: True when the whole chain fits. A partial chain is never "within budget":
    #: see :func:`build_model_version_plan`.
    within_budget: bool = False
    #: Chain prefix dropped for budget, in the order it would have run.
    deferred_stages: list[StageKind] = Field(default_factory=list)
    #: Why the plan is not fundable, in one sentence, or None when it is.
    budget_reason: str | None = None

    #: Readiness verdict the plan consulted, if any.
    readiness: ReadinessVerdict | None = None
    #: True when training was withheld because readiness did not pass.
    train_withheld: bool = False

    planned_at: str
    notes: list[str] = Field(default_factory=list)

    @property
    def gpu_ids_used(self) -> list[str]:
        """Every distinct GPU the plan touches, sorted for stable comparison."""
        return sorted({gpu for stage in self.stages for gpu in stage.gpu_ids})


def _canonical_sha256(payload: Any) -> str:
    """Hash on sorted, compact JSON so the digest is independent of key order."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def version_identity(job: Mapping[str, Any]) -> dict[str, str]:
    """Derive the version identity of a fine-tuning job.

    Deliberately content-addressed. Two jobs that differ anywhere in model,
    data, or training config get different version ids, so a plan can never
    silently describe one version while the job trains another. The digest is
    taken over a normalised projection rather than the raw spec: bookkeeping
    fields like output paths do not change what gets trained, and including them
    would mint a new version for every run directory.
    """
    experiment = job.get("experiment") or {}
    model = job.get("model") or {}
    data = job.get("data") or {}
    train = job.get("train") or {}

    # Only fields that change the trained artefact. Anything omitted here is a
    # candidate for a version-id collision, so the list is deliberately explicit.
    identity_payload = {
        "backbone": model.get("backbone"),
        "backend": model.get("backend"),
        "head_kind": model.get("head_kind"),
        "head_rank": model.get("head_rank"),
        "lora_r": model.get("lora_r"),
        "lora_alpha": model.get("lora_alpha"),
        "lora_dropout": model.get("lora_dropout"),
        "target_modules": model.get("target_modules"),
        "train": {
            key: train.get(key)
            for key in (
                "epochs",
                "batch_size",
                "grad_accum",
                "learning_rate",
                "weight_decay",
                "max_state_length",
                "max_candidate_length",
                "max_sequence_length",
                "seed",
                "freeze_backbone",
                "gradient_checkpointing",
            )
        },
        "data": {key: data.get(key) for key in ("train", "validation", "calibration", "test")},
        "parent": experiment.get("parent"),
    }
    job_sha = _canonical_sha256(identity_payload)
    return {
        "version_id": f"plan-{job_sha[:12]}",
        "job_sha256": job_sha,
        "backbone": str(model.get("backbone") or "unknown"),
        "head_kind": str(model.get("head_kind") or "unknown"),
        "data_lineage_sha256": _canonical_sha256(
            {key: data.get(key) for key in ("train", "validation", "calibration", "test")}
        ),
    }


def estimate_gpu_minutes(
    stage: StageKind,
    job: Mapping[str, Any],
) -> float:
    """Estimate GPU minutes for one stage from the shape of the job.

    A heuristic, and labelled as one in the plan notes. It reads only the spec's
    shape -- epochs, effective batch, sequence lengths, LoRA width -- because
    nothing else is available before the run exists. The intent is a *ranking*
    and an order of magnitude for budgeting, not a prediction. Any stage with a
    ``measured_gpu_minutes`` from a completed run overrides this, which is how a
    plan stops being a guess after the first version.
    """
    train = job.get("train") or {}
    epochs = float(train.get("epochs") or 1)
    batch = float(train.get("batch_size") or 1)
    accum = float(train.get("grad_accum") or 1)
    state_len = float(train.get("max_state_length") or 512)
    lora_r = float((job.get("model") or {}).get("lora_r") or 8)
    frozen = bool(train.get("freeze_backbone"))

    # Rough constant calibrated to the shape of the existing runs rather than to
    # any hardware measurement; see the plan note emitted alongside it.
    per_step = (state_len / 1024.0) * (1.0 + lora_r / 32.0) * (0.4 if frozen else 1.0)
    train_minutes = max(1.0, epochs * batch * accum * per_step * 4.0)

    if stage is StageKind.train:
        return round(train_minutes, 2)
    if stage is StageKind.benchmark:
        return round(train_minutes * 0.35, 2)
    return round(train_minutes * 0.2, 2)


def _select_gpus(
    stage: StageKind,
    needed: float,
    pool: Sequence[GpuCandidate],
) -> list[str]:
    """Choose devices for one stage, most-trusted-and-largest first.

    Ordering is by descending free time with the device id as an explicit
    tie-break, so the same inventory always yields the same placement and two
    plans can be diffed meaningfully. Untrusted devices are skipped entirely
    rather than merely deprioritised.
    """
    usable = sorted(
        (candidate for candidate in pool if candidate.trusted),
        key=lambda candidate: (-candidate.free_gpu_minutes, candidate.gpu_id),
    )
    chosen: list[str] = []
    remaining = needed
    for candidate in usable:
        if remaining <= 0.0:
            break
        if candidate.free_gpu_minutes <= 0.0:
            continue
        take = min(candidate.free_gpu_minutes, remaining)
        remaining -= take
        chosen.append(candidate.gpu_id)
    return chosen


def build_model_version_plan(
    job: Mapping[str, Any],
    *,
    pool: Sequence[GpuCandidate] = (),
    budget_gpu_minutes: float = 0.0,
    readiness: ReadinessVerdict | None = None,
    measured: Mapping[StageKind, float] | None = None,
    planned_at: datetime | None = None,
) -> ModelVersionPlan:
    """Plan GPU time for the version a fine-tuning job would produce.

    The chain is **all-or-nothing from the front**. Benchmarking and evaluating
    a version that was never trained yields no evidence about the version, so a
    budget that funds two of three stages funds nothing: the plan reports the
    shortfall and schedules no GPU. This is the module's one deliberate
    fail-closed choice, and it is why ``within_budget`` cannot be true for a
    partial chain.

    Readiness is consulted before any budget arithmetic. When it does not pass,
    the train stage is withheld and no GPU is planned, because the cheapest
    available question -- "is the baseline good enough to justify this at all" --
    has not been answered yes.
    """
    moment = planned_at or datetime.now(UTC)
    notes: list[str] = []
    identity = version_identity(job)
    experiment = job.get("experiment") or {}
    gates = job.get("gates") or {}
    measured_map = dict(measured or {})

    readiness_passed = readiness is None or (readiness.completed and readiness.passed)
    if readiness is None:
        notes.append(
            "no readiness verdict supplied; planning proceeded on the job's own "
            "budget. Pass one to gate training on a baseline first."
        )
    elif not readiness_passed:
        notes.append(
            "scale readiness did not pass; no training GPU planned. "
            f"reason: {readiness.reason or 'incomplete or failed baseline'}"
        )

    stages: list[PlannedStage] = []
    planned_total = 0.0
    for kind in STAGE_CHAIN:
        estimate = estimate_gpu_minutes(kind, job)
        measured_minutes = measured_map.get(kind)
        stage = PlannedStage(
            kind=kind,
            picker=_picker_for(kind),
            selection=_selection_for(kind, job, experiment, gates, stages),
            rationale=_rationale_for(kind, job, gates),
            estimated_gpu_minutes=estimate,
            measured_gpu_minutes=measured_minutes,
        )
        if not stage.calibrated:
            notes.append(
                f"{kind.value} estimate {estimate} min is derived from the job "
                "shape, not measured; supply measured minutes once it has run"
            )
        planned_total += stage.minutes_used_for_budget
        stages.append(stage)

    full_chain_minutes = round(sum(stage.minutes_used_for_budget for stage in stages), 2)
    available = sum(candidate.free_gpu_minutes for candidate in pool if candidate.trusted)

    within_budget = full_chain_minutes <= budget_gpu_minutes
    budget_reason: str | None = None
    if not readiness_passed:
        within_budget = False
        budget_reason = (
            "scale readiness did not pass, so no version is worth its GPU yet"
        )
    elif not within_budget:
        budget_reason = (
            f"the full {StageKind.train}/{StageKind.benchmark}/{StageKind.evaluate} "
            f"chain needs {full_chain_minutes} GPU-minutes but the budget is "
            f"{budget_gpu_minutes}; a partial chain produces no evidence about "
            "the version, so nothing is scheduled"
        )

    # The full chain is always present, placed or not. An unfunded plan that hid
    # its stages would make ``stages[0]`` an IndexError rather than an unscheduled
    # stage, and would hide the cost breakdown that explains the shortfall.
    scheduled: list[PlannedStage] = []
    for stage in stages:
        scheduled.append(
            stage.model_copy(
                update={
                    "gpu_ids": (
                        _select_gpus(stage.kind, stage.minutes_used_for_budget, pool)
                        if within_budget
                        else []
                    )
                }
            )
        )
    if within_budget:
        notes.append(f"inventory reports {available} free GPU-minutes in total")

    deferred = [] if within_budget else [stage.kind for stage in stages]

    if available < full_chain_minutes and within_budget:
        notes.append(
            f"budget allows the chain but the inventory only reports {available} "
            f"free GPU-minutes against {full_chain_minutes} needed; stages are "
            "placed on whatever is available and will contend"
        )

    return ModelVersionPlan(
        version_id=identity["version_id"],
        job_sha256=identity["job_sha256"],
        experiment=str(experiment.get("name") or "unnamed"),
        parent=experiment.get("parent"),
        backbone=identity["backbone"],
        head_kind=identity["head_kind"],
        data_lineage_sha256=identity["data_lineage_sha256"],
        stages=scheduled,
        budget_gpu_minutes=budget_gpu_minutes,
        total_estimated_gpu_minutes=full_chain_minutes,
        within_budget=within_budget,
        deferred_stages=deferred,
        budget_reason=budget_reason,
        readiness=readiness,
        train_withheld=not readiness_passed,
        planned_at=moment.replace(microsecond=0).isoformat(),
        notes=notes,
    )


def _picker_for(kind: StageKind) -> PickerKind:
    return {
        StageKind.train: PickerKind.train_config,
        StageKind.benchmark: PickerKind.benchmark_suite,
        StageKind.evaluate: PickerKind.eval_thresholds,
    }[kind]


def _selection_for(
    kind: StageKind,
    job: Mapping[str, Any],
    experiment: Mapping[str, Any],
    gates: Mapping[str, Any],
    scheduled: Sequence[PlannedStage],
) -> str:
    """Emit the picker's choice, or an explicit refusal to choose.

    ``parent_version`` is resolved from the job when the job names one, and
    refused as ``<unresolved>`` when it does not. Guessing a parent from a name
    pattern would fork lineage silently, so the planner states the absence
    instead and leaves resolving it to whoever owns the lineage.
    """
    if kind is StageKind.train:
        train = job.get("train") or {}
        return (
            f"lr={train.get('learning_rate')} epochs={train.get('epochs')} "
            f"effective_batch={int(float(train.get('batch_size') or 1) * float(train.get('grad_accum') or 1))} "
            f"seed={train.get('seed')}"
        )
    if kind is StageKind.benchmark:
        benchmark = job.get("benchmark") or {}
        return (
            f"batch={benchmark.get('batch_size')} "
            f"fleet_states={'set' if benchmark.get('fleet_states') else 'unset'}"
        )
    gate_names = ",".join(sorted(str(name) for name in gates)) or "none"
    return f"gates={gate_names}"


def _rationale_for(
    kind: StageKind,
    job: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> str:
    train = job.get("train") or {}
    if kind is StageKind.train:
        return (
            f"training cost is driven by {train.get('epochs')} epochs at "
            f"effective batch "
            f"{int(float(train.get('batch_size') or 1) * float(train.get('grad_accum') or 1))}"
        )
    if kind is StageKind.benchmark:
        return (
            "benchmarking reuses the trained checkpoint, so it is costed as a "
            "fraction of training rather than estimated independently"
        )
    return (
        f"evaluation is gated on {len(gates)} named threshold(s); without it a "
        "version cannot be compared to its parent"
    )


def plan_from_experiment_spec(
    spec: BaseModel,
    **kwargs: Any,
) -> ModelVersionPlan:
    """Convenience wrapper for a validated ``ExperimentSpec``.

    Accepts the spec model rather than re-validating a payload, so the planner
    cannot be handed something that ``ExperimentSpec`` would have rejected.
    """
    return build_model_version_plan(spec.model_dump(mode="json"), **kwargs)
