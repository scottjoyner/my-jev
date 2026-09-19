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


def generate_fleet_records(count: int, seed: int = 31):
    rng = random.Random(seed)
    records = []
    for index in range(count):
        gpu = index % 3 == 0
        interactive = index % 4 == 0
        pressure = index % 5
        checkpoint = index % 2 == 0
        required = ["gpu"] if gpu else ["cpu"]
        nodes = [
            FleetNodeSnapshot(
                node_id=f"node-{index}-a",
                capabilities=["cpu", "gpu"] if gpu else ["cpu"],
                ram_free_gib=max(8.0, 64.0 - pressure * 8),
                vram_free_gib=32.0 - pressure * 4 if gpu else 0.0,
                cpu_free_fraction=max(0.1, 0.9 - pressure * 0.18),
                active_claims=pressure,
            ),
            FleetNodeSnapshot(
                node_id=f"node-{index}-b",
                capabilities=["cpu"],
                ram_free_gib=32.0,
                cpu_free_fraction=0.5,
                drained=(index % 11 == 0),
            ),
        ]
        state = FleetPlacementState(
            workload_id=f"synthetic-{index}",
            workload_type="interactive-inference" if interactive else "batch-job",
            description="synthetic fleet placement counterfactual",
            required_capabilities=required,
            estimated_ram_gib=8.0,
            estimated_vram_gib=16.0 if gpu else 0.0,
            estimated_duration_seconds=30.0 if interactive else 1800.0,
            deadline_seconds=2.0 if interactive else None,
            checkpoint_supported=checkpoint,
            nodes=nodes,
            source="synthetic_fleet",
            metadata={
                "family_id": f"fleet-{index // 2}",
                "synthetic": True,
            },
        )
        label = FleetPlacementLabel(
            placement=(
                PlacementShape.PREFERRED_NODE
                if interactive
                else PlacementShape.ANY_ELIGIBLE
            ),
            resource_shape=ResourceShape.GPU if gpu else ResourceShape.CPU,
            latency_class=(
                LatencyClass.INTERACTIVE if interactive else LatencyClass.BATCH
            ),
            state_locality=StateLocality.NONE,
            migration_tolerance=(
                MigrationTolerance.CHECKPOINTABLE
                if checkpoint
                else MigrationTolerance.STICKY
            ),
            redundancy=(
                RedundancyShape.RETRY_ELSEWHERE
                if checkpoint
                else RedundancyShape.SINGLE
            ),
            fleet_pressure=pressure,
            placement_confidence=3 if pressure < 4 else 2,
        )
        records.append(build_fleet_placement_record(state, label=label))

        # Counterfactual: node identities and ordering change while semantic fleet
        # facts remain identical. Targets must remain exactly unchanged.
        if len(records) < count and index % 2 == 0:
            swapped = state.model_copy(
                update={
                    "nodes": list(reversed(state.nodes)),
                    "workload_id": f"synthetic-{index}-permuted",
                    "metadata": {
                        **state.metadata,
                        "counterfactual": "node_permutation",
                    },
                }
            )
            records.append(build_fleet_placement_record(swapped, label=label))

    rng.shuffle(records)
    return records[:count]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate authority-safe synthetic fleet placement records"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--records", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    if args.records < 1:
        raise SystemExit("--records must be >= 1")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(iter(generate_fleet_records(args.records, args.seed)), output)


if __name__ == "__main__":
    main()
