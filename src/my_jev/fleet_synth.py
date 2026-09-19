from __future__ import annotations

import argparse
import random
from pathlib import Path

from .data import dump_jsonl
from .fleet_policy import (
    FleetNodeSnapshot,
    FleetPlacementLabel,
    FleetPlacementState,
    LatencyClass,
    MigrationTolerance,
    PlacementShape,
    RedundancyShape,
    ResourceShape,
    StateLocality,
    build_fleet_placement_record,
)
from .manifest import write_manifest
from .schema import DecisionRecord

GENERATOR_VERSION = "fleet-placement-synth-v2"


def _node(
    node_id: str,
    *,
    capabilities: list[str],
    ram: float,
    vram: float = 0.0,
    cpu: float = 0.8,
    claims: int = 0,
    drained: bool = False,
    fresh: bool = True,
    reachable: bool = True,
) -> FleetNodeSnapshot:
    return FleetNodeSnapshot(
        node_id=node_id,
        capabilities=capabilities,
        ram_free_gib=ram,
        vram_free_gib=vram,
        cpu_free_fraction=cpu,
        active_claims=claims,
        drained=drained,
        health_fresh=fresh,
        reachable=reachable,
    )


def _label(
    *,
    placement: PlacementShape,
    resource: ResourceShape,
    latency: LatencyClass,
    locality: StateLocality,
    migration: MigrationTolerance,
    redundancy: RedundancyShape,
    pressure: int,
    confidence: int,
) -> FleetPlacementLabel:
    return FleetPlacementLabel(
        placement=placement,
        resource_shape=resource,
        latency_class=latency,
        state_locality=locality,
        migration_tolerance=migration,
        redundancy=redundancy,
        fleet_pressure=pressure,
        placement_confidence=confidence,
    )


def _record(
    state: FleetPlacementState,
    label: FleetPlacementLabel,
    *,
    family_id: str,
    scenario: str,
    variant: str,
) -> DecisionRecord:
    state.metadata.update(
        {
            "family_id": family_id,
            "scenario": scenario,
            "variant": variant,
            "generator_version": (
                GENERATOR_VERSION
            ),
            "label_status": "verified",
            "label_source": (
                "programmatic_verifier"
            ),
        }
    )
    return build_fleet_placement_record(
        state,
        label=label,
    )


def _permuted(
    state: FleetPlacementState,
    *,
    workload_id: str,
) -> FleetPlacementState:
    result = state.model_copy(
        deep=True
    )
    result.workload_id = workload_id
    result.nodes = list(
        reversed(result.nodes)
    )
    result.metadata = {
        **result.metadata,
        "counterfactual": (
            "node_permutation"
        ),
    }
    return result


def _gpu_preference_family(
    rng: random.Random,
    family_id: str,
) -> list[DecisionRecord]:
    preferred = _node(
        f"{family_id}-preferred",
        capabilities=[
            "gpu",
            "rocm",
        ],
        ram=48.0,
        vram=rng.choice(
            (24.0, 28.0, 32.0)
        ),
        cpu=rng.uniform(0.5, 0.9),
    )
    fallback = _node(
        f"{family_id}-fallback",
        capabilities=["gpu"],
        ram=64.0,
        vram=24.0,
        cpu=rng.uniform(0.5, 0.9),
    )
    base = FleetPlacementState(
        workload_id=f"{family_id}-base",
        workload_type="model_training",
        description=(
            "Long accelerator training job "
            "with ROCm preference."
        ),
        required_capabilities=["gpu"],
        preferred_capabilities=["rocm"],
        estimated_ram_gib=16.0,
        estimated_vram_gib=16.0,
        estimated_duration_seconds=7200,
        checkpoint_supported=True,
        state_locality_hint=(
            StateLocality.WEAK
        ),
        nodes=[
            preferred,
            fallback,
        ],
        source="synthetic_fleet",
        metadata={
            "counterfactual_axis": (
                "preferred_node_drain"
            )
        },
    )
    normal_label = _label(
        placement=(
            PlacementShape.PREFERRED_NODE
        ),
        resource=ResourceShape.GPU,
        latency=LatencyClass.BATCH,
        locality=StateLocality.WEAK,
        migration=(
            MigrationTolerance
            .CHECKPOINTABLE
        ),
        redundancy=(
            RedundancyShape
            .RETRY_ELSEWHERE
        ),
        pressure=1,
        confidence=4,
    )
    drained = base.model_copy(
        deep=True
    )
    drained.workload_id = (
        f"{family_id}-drained"
    )
    drained.nodes[0].drained = True
    drained.metadata = {
        **drained.metadata,
        "counterfactual": (
            "preferred_drained"
        ),
    }

    return [
        _record(
            base,
            normal_label,
            family_id=family_id,
            scenario="gpu_preference",
            variant="base",
        ),
        _record(
            _permuted(
                base,
                workload_id=(
                    f"{family_id}-permuted"
                ),
            ),
            normal_label,
            family_id=family_id,
            scenario="gpu_preference",
            variant="node_permutation",
        ),
        _record(
            drained,
            _label(
                placement=(
                    PlacementShape
                    .ANY_ELIGIBLE
                ),
                resource=(
                    ResourceShape.GPU
                ),
                latency=(
                    LatencyClass.BATCH
                ),
                locality=(
                    StateLocality.WEAK
                ),
                migration=(
                    MigrationTolerance
                    .CHECKPOINTABLE
                ),
                redundancy=(
                    RedundancyShape
                    .RETRY_ELSEWHERE
                ),
                pressure=2,
                confidence=3,
            ),
            family_id=family_id,
            scenario="gpu_preference",
            variant="preferred_drained",
        ),
    ]


def _pinned_recovery_family(
    rng: random.Random,
    family_id: str,
) -> list[DecisionRecord]:
    owner = _node(
        f"{family_id}-owner",
        capabilities=[
            "storage",
            "database",
        ],
        ram=32.0,
        cpu=rng.uniform(0.35, 0.7),
    )
    other = _node(
        f"{family_id}-other",
        capabilities=[
            "storage",
            "database",
        ],
        ram=96.0,
        cpu=0.9,
    )
    base = FleetPlacementState(
        workload_id=f"{family_id}-base",
        workload_type="database_recovery",
        description=(
            "Recovery job pinned to local "
            "database state."
        ),
        required_capabilities=[
            "storage",
            "database",
        ],
        estimated_ram_gib=8.0,
        estimated_duration_seconds=21600,
        checkpoint_supported=False,
        state_locality_hint=(
            StateLocality.PINNED
        ),
        pinned_node_id=owner.node_id,
        nodes=[
            owner,
            other,
        ],
        source="synthetic_fleet",
        metadata={
            "counterfactual_axis": (
                "pinned_health_freshness"
            )
        },
    )
    normal_label = _label(
        placement=PlacementShape.LOCAL,
        resource=ResourceShape.IO,
        latency=LatencyClass.BACKGROUND,
        locality=StateLocality.PINNED,
        migration=(
            MigrationTolerance.IMMOVABLE
        ),
        redundancy=(
            RedundancyShape.SINGLE
        ),
        pressure=2,
        confidence=4,
    )
    stale = base.model_copy(
        deep=True
    )
    stale.workload_id = (
        f"{family_id}-stale"
    )
    stale.nodes[0].health_fresh = False
    stale.metadata = {
        **stale.metadata,
        "counterfactual": (
            "pinned_health_stale"
        ),
    }

    return [
        _record(
            base,
            normal_label,
            family_id=family_id,
            scenario="pinned_recovery",
            variant="base",
        ),
        _record(
            _permuted(
                base,
                workload_id=(
                    f"{family_id}-permuted"
                ),
            ),
            normal_label,
            family_id=family_id,
            scenario="pinned_recovery",
            variant="node_permutation",
        ),
        _record(
            stale,
            _label(
                placement=(
                    PlacementShape.DEFER
                ),
                resource=ResourceShape.IO,
                latency=(
                    LatencyClass.BACKGROUND
                ),
                locality=(
                    StateLocality.PINNED
                ),
                migration=(
                    MigrationTolerance
                    .IMMOVABLE
                ),
                redundancy=(
                    RedundancyShape.SINGLE
                ),
                pressure=4,
                confidence=0,
            ),
            family_id=family_id,
            scenario="pinned_recovery",
            variant="pinned_health_stale",
        ),
    ]


def _capacity_boundary_family(
    rng: random.Random,
    family_id: str,
) -> list[DecisionRecord]:
    capacity = rng.choice(
        (16.0, 24.0, 32.0)
    )
    node = _node(
        f"{family_id}-gpu",
        capabilities=[
            "gpu",
            "rocm",
        ],
        ram=64.0,
        vram=capacity,
        cpu=rng.uniform(0.45, 0.85),
    )
    base = FleetPlacementState(
        workload_id=f"{family_id}-base",
        workload_type="gpu_batch",
        required_capabilities=[
            "gpu",
            "rocm",
        ],
        estimated_ram_gib=12.0,
        estimated_vram_gib=(
            capacity - 1.0
        ),
        estimated_duration_seconds=1800,
        checkpoint_supported=True,
        nodes=[node],
        source="synthetic_fleet",
        metadata={
            "counterfactual_axis": (
                "vram_capacity_boundary"
            )
        },
    )
    normal_label = _label(
        placement=(
            PlacementShape.ANY_ELIGIBLE
        ),
        resource=ResourceShape.GPU,
        latency=LatencyClass.BATCH,
        locality=StateLocality.NONE,
        migration=(
            MigrationTolerance
            .CHECKPOINTABLE
        ),
        redundancy=(
            RedundancyShape
            .RETRY_ELSEWHERE
        ),
        pressure=2,
        confidence=4,
    )
    exceeds = base.model_copy(
        deep=True
    )
    exceeds.workload_id = (
        f"{family_id}-exceeds"
    )
    exceeds.estimated_vram_gib = (
        capacity + 1.0
    )
    exceeds.metadata = {
        **exceeds.metadata,
        "counterfactual": (
            "vram_exceeds_capacity"
        ),
    }

    return [
        _record(
            base,
            normal_label,
            family_id=family_id,
            scenario="capacity_boundary",
            variant="fits",
        ),
        _record(
            _permuted(
                base,
                workload_id=(
                    f"{family_id}-permuted"
                ),
            ),
            normal_label,
            family_id=family_id,
            scenario="capacity_boundary",
            variant="node_permutation",
        ),
        _record(
            exceeds,
            _label(
                placement=(
                    PlacementShape.DEFER
                ),
                resource=(
                    ResourceShape.GPU
                ),
                latency=(
                    LatencyClass.BATCH
                ),
                locality=(
                    StateLocality.NONE
                ),
                migration=(
                    MigrationTolerance
                    .CHECKPOINTABLE
                ),
                redundancy=(
                    RedundancyShape
                    .RETRY_ELSEWHERE
                ),
                pressure=4,
                confidence=1,
            ),
            family_id=family_id,
            scenario="capacity_boundary",
            variant="exceeds_capacity",
        ),
    ]


def _split_batch_family(
    rng: random.Random,
    family_id: str,
) -> list[DecisionRecord]:
    nodes = [
        _node(
            f"{family_id}-{index}",
            capabilities=[
                "cpu",
                "batch",
            ],
            ram=rng.choice(
                (16.0, 24.0, 32.0)
            ),
            cpu=rng.uniform(
                0.45,
                0.9,
            ),
            claims=rng.choice(
                (0, 0, 1)
            ),
        )
        for index in range(3)
    ]
    base = FleetPlacementState(
        workload_id=f"{family_id}-base",
        workload_type="parallel_batch",
        required_capabilities=[
            "cpu",
            "batch",
        ],
        estimated_ram_gib=4.0,
        estimated_duration_seconds=14400,
        checkpoint_supported=True,
        nodes=nodes,
        source="synthetic_fleet",
        metadata={
            "counterfactual_axis": (
                "checkpoint_support"
            )
        },
    )
    normal_label = _label(
        placement=PlacementShape.SPLIT,
        resource=ResourceShape.CPU,
        latency=LatencyClass.BACKGROUND,
        locality=StateLocality.NONE,
        migration=(
            MigrationTolerance
            .CHECKPOINTABLE
        ),
        redundancy=(
            RedundancyShape.REPLICATED
        ),
        pressure=2,
        confidence=4,
    )
    sticky = base.model_copy(
        deep=True
    )
    sticky.workload_id = (
        f"{family_id}-sticky"
    )
    sticky.checkpoint_supported = False
    sticky.metadata = {
        **sticky.metadata,
        "counterfactual": (
            "checkpoint_disabled"
        ),
    }

    return [
        _record(
            base,
            normal_label,
            family_id=family_id,
            scenario="split_batch",
            variant="checkpointable",
        ),
        _record(
            _permuted(
                base,
                workload_id=(
                    f"{family_id}-permuted"
                ),
            ),
            normal_label,
            family_id=family_id,
            scenario="split_batch",
            variant="node_permutation",
        ),
        _record(
            sticky,
            _label(
                placement=(
                    PlacementShape
                    .ANY_ELIGIBLE
                ),
                resource=(
                    ResourceShape.CPU
                ),
                latency=(
                    LatencyClass.BACKGROUND
                ),
                locality=(
                    StateLocality.NONE
                ),
                migration=(
                    MigrationTolerance.STICKY
                ),
                redundancy=(
                    RedundancyShape.SINGLE
                ),
                pressure=2,
                confidence=3,
            ),
            family_id=family_id,
            scenario="split_batch",
            variant="checkpoint_disabled",
        ),
    ]


_FAMILY_BUILDERS = (
    _gpu_preference_family,
    _pinned_recovery_family,
    _capacity_boundary_family,
    _split_batch_family,
)


def generate_fleet_records(
    count: int,
    seed: int = 31,
) -> list[DecisionRecord]:
    if count < 1:
        raise ValueError(
            "count must be >= 1"
        )

    rng = random.Random(seed)
    records: list[DecisionRecord] = []
    family_index = 0

    while len(records) < count:
        builder = rng.choice(
            _FAMILY_BUILDERS
        )
        family_id = (
            f"fleet-{seed}-"
            f"{family_index:08d}"
        )
        records.extend(
            builder(
                rng,
                family_id,
            )
        )
        family_index += 1

    return records[:count]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate verifier-labeled "
            "fleet-placement counterfactuals"
        )
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--records",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=31,
    )
    args = parser.parse_args()

    if args.records < 1:
        raise SystemExit(
            "--records must be >= 1"
        )
    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    dump_jsonl(
        iter(
            generate_fleet_records(
                args.records,
                args.seed,
            )
        ),
        output,
    )
    manifest_path = output.with_suffix(
        output.suffix
        + ".manifest.json"
    )
    manifest = write_manifest(
        output,
        manifest_path,
    )
    print(
        f"wrote {manifest['records']} "
        "fleet-placement states, "
        f"sha256={manifest['sha256']}"
    )
    print(
        f"manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
