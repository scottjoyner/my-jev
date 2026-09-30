from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from .schema import DecisionRecord, QuestionSpec, QuestionType, TargetSpec


class PlacementShape(StrEnum):
    LOCAL = "local"
    PREFERRED_NODE = "preferred_node"
    ANY_ELIGIBLE = "any_eligible"
    SPLIT = "split"
    DEFER = "defer"
    ABSTAIN = "abstain"


class ResourceShape(StrEnum):
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    IO = "io"
    NETWORK = "network"
    MIXED = "mixed"


class LatencyClass(StrEnum):
    INTERACTIVE = "interactive"
    NEARLINE = "nearline"
    BATCH = "batch"
    BACKGROUND = "background"


class StateLocality(StrEnum):
    NONE = "none"
    WEAK = "weak"
    STRONG = "strong"
    PINNED = "pinned"


class MigrationTolerance(StrEnum):
    FREE = "free"
    CHECKPOINTABLE = "checkpointable"
    STICKY = "sticky"
    IMMOVABLE = "immovable"


class RedundancyShape(StrEnum):
    SINGLE = "single"
    RETRY_ELSEWHERE = "retry_elsewhere"
    REPLICATED = "replicated"


PLACEMENT_OPTIONS = [x.value for x in PlacementShape]
RESOURCE_OPTIONS = [x.value for x in ResourceShape]
LATENCY_OPTIONS = [x.value for x in LatencyClass]
LOCALITY_OPTIONS = [x.value for x in StateLocality]
MIGRATION_OPTIONS = [x.value for x in MigrationTolerance]
REDUNDANCY_OPTIONS = [x.value for x in RedundancyShape]
PRESSURE_OPTIONS = ["idle", "light", "moderate", "high", "saturated"]
CONFIDENCE_OPTIONS = ["very_low", "low", "medium", "high", "very_high"]


class FleetNodeSnapshot(BaseModel):
    node_id: str
    capabilities: list[str] = Field(default_factory=list)
    healthy: bool = True
    health_fresh: bool = True
    drained: bool = False
    reachable: bool = True
    cpu_free_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    ram_free_gib: float = Field(default=0.0, ge=0.0)
    vram_free_gib: float = Field(default=0.0, ge=0.0)
    active_claims: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FleetPlacementState(BaseModel):
    workload_id: str
    workload_type: str
    description: str = ""
    required_capabilities: list[str] = Field(default_factory=list)
    preferred_capabilities: list[str] = Field(default_factory=list)
    estimated_ram_gib: float = Field(default=0.0, ge=0.0)
    estimated_vram_gib: float = Field(default=0.0, ge=0.0)
    estimated_duration_seconds: float = Field(default=0.0, ge=0.0)
    deadline_seconds: float | None = Field(default=None, ge=0.0)
    checkpoint_supported: bool = False
    state_locality_hint: StateLocality = StateLocality.NONE
    pinned_node_id: str | None = None
    nodes: list[FleetNodeSnapshot] = Field(default_factory=list)
    source: str = "fleet_shadow"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def as_model_state(self) -> str:
        # Hostnames/node IDs are intentionally excluded. The model sees capability
        # and pressure facts; the deterministic controller retains identity.
        nodes = [
            {
                "capabilities": sorted(node.capabilities),
                "healthy": node.healthy,
                "health_fresh": node.health_fresh,
                "drained": node.drained,
                "reachable": node.reachable,
                "cpu_free_fraction": node.cpu_free_fraction,
                "ram_free_gib": node.ram_free_gib,
                "vram_free_gib": node.vram_free_gib,
                "active_claims": node.active_claims,
            }
            for node in self.nodes
        ]
        nodes.sort(
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":")
            )
        )
        payload = {
            "workload": {
                "type": self.workload_type,
                "description": self.description,
                "required_capabilities": sorted(self.required_capabilities),
                "preferred_capabilities": sorted(self.preferred_capabilities),
                "estimated_ram_gib": self.estimated_ram_gib,
                "estimated_vram_gib": self.estimated_vram_gib,
                "estimated_duration_seconds": self.estimated_duration_seconds,
                "deadline_seconds": self.deadline_seconds,
                "checkpoint_supported": self.checkpoint_supported,
                "state_locality_hint": self.state_locality_hint.value,
                "pinned": self.pinned_node_id is not None,
            },
            "fleet": nodes,
            "source": self.source,
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


class FleetPlacementLabel(BaseModel):
    placement: PlacementShape
    resource_shape: ResourceShape
    latency_class: LatencyClass
    state_locality: StateLocality
    migration_tolerance: MigrationTolerance
    redundancy: RedundancyShape
    fleet_pressure: int = Field(ge=0, le=4)
    placement_confidence: int = Field(ge=0, le=4)


def _question(kind, instructions, options, task_id, weight=1.0):
    return QuestionSpec(
        type=kind,
        instructions=instructions,
        options=options,
        metadata={
            "task_id": task_id,
            "policy_contract": "fleet-placement-v1",
            "loss_weight": weight,
        },
    )


def fleet_placement_questions() -> dict[str, QuestionSpec]:
    return {
        "placement": _question(
            QuestionType.CHOICE,
            "What execution placement shape best fits this workload and fleet state? "
            "Predict a shape, never a hostname or permission.",
            PLACEMENT_OPTIONS,
            "fleet.placement",
            3.0,
        ),
        "resource_shape": _question(
            QuestionType.CHOICE,
            "Which resource is most important to successful placement?",
            RESOURCE_OPTIONS,
            "fleet.resource_shape",
            2.0,
        ),
        "latency_class": _question(
            QuestionType.CHOICE,
            "What latency/service class does this workload require?",
            LATENCY_OPTIONS,
            "fleet.latency_class",
        ),
        "state_locality": _question(
            QuestionType.CHOICE,
            "How strongly does execution depend on existing local state or data?",
            LOCALITY_OPTIONS,
            "fleet.state_locality",
            2.0,
        ),
        "migration_tolerance": _question(
            QuestionType.CHOICE,
            "How safely can this workload move between eligible nodes?",
            MIGRATION_OPTIONS,
            "fleet.migration_tolerance",
            2.0,
        ),
        "redundancy": _question(
            QuestionType.CHOICE,
            "What retry/replication shape best matches the workload?",
            REDUNDANCY_OPTIONS,
            "fleet.redundancy",
        ),
        "fleet_pressure": _question(
            QuestionType.SCORE,
            "Rate current fleet resource pressure from idle through saturated.",
            PRESSURE_OPTIONS,
            "fleet.pressure",
            1.5,
        ),
        "placement_confidence": _question(
            QuestionType.SCORE,
            "Rate confidence that the observed fleet state supports a useful placement "
            "recommendation. This is advisory and never grants dispatch authority.",
            CONFIDENCE_OPTIONS,
            "fleet.placement_confidence",
            1.5,
        ),
    }


def _index(options: list[str], value: StrEnum | str) -> int:
    raw = value.value if isinstance(value, StrEnum) else str(value)
    return options.index(raw)


def label_to_targets(label: FleetPlacementLabel) -> dict[str, TargetSpec]:
    return {
        "placement": TargetSpec(index=_index(PLACEMENT_OPTIONS, label.placement)),
        "resource_shape": TargetSpec(index=_index(RESOURCE_OPTIONS, label.resource_shape)),
        "latency_class": TargetSpec(index=_index(LATENCY_OPTIONS, label.latency_class)),
        "state_locality": TargetSpec(index=_index(LOCALITY_OPTIONS, label.state_locality)),
        "migration_tolerance": TargetSpec(index=_index(MIGRATION_OPTIONS, label.migration_tolerance)),
        "redundancy": TargetSpec(index=_index(REDUNDANCY_OPTIONS, label.redundancy)),
        "fleet_pressure": TargetSpec(index=label.fleet_pressure),
        "placement_confidence": TargetSpec(index=label.placement_confidence),
    }


def build_fleet_placement_record(
    state: FleetPlacementState,
    *,
    label: FleetPlacementLabel | None = None,
) -> DecisionRecord:
    return DecisionRecord(
        state=state.as_model_state(),
        questions=fleet_placement_questions(),
        targets=label_to_targets(label) if label is not None else None,
        metadata={
            "domain": "fleet_placement",
            "policy_contract": "fleet-placement-v1",
            "source": state.source,
            "workload_id": state.workload_id,
            "dispatch_allowed": False,
            **state.metadata,
        },
    )


def eligible_nodes(state: FleetPlacementState) -> list[FleetNodeSnapshot]:
    required = set(state.required_capabilities)
    result = []
    for node in state.nodes:
        if not node.healthy or not node.health_fresh or node.drained or not node.reachable:
            continue
        if not required.issubset(set(node.capabilities)):
            continue
        if node.ram_free_gib < state.estimated_ram_gib:
            continue
        if node.vram_free_gib < state.estimated_vram_gib:
            continue
        if state.pinned_node_id is not None and node.node_id != state.pinned_node_id:
            continue
        result.append(node)
    return result
