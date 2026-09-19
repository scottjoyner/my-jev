from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from .fleet_policy import FleetNodeSnapshot, FleetPlacementState, StateLocality

CORPUS_VERSION = "fleet-benchmark-states-v1"


def generate_benchmark_states(count: int = 64, seed: int = 47) -> list[FleetPlacementState]:
    if count < 1:
        raise ValueError("count must be >= 1")
    rng = random.Random(seed)
    states = []
    for index in range(count):
        gpu = index % 2 == 0
        required = ["gpu", "rocm"] if gpu else ["cpu", "batch"]
        ram = float(rng.choice((8, 12, 16, 24)))
        vram = float(rng.choice((8, 12, 16, 20))) if gpu else 0.0
        nodes = [
            FleetNodeSnapshot(
                node_id=f"opaque-{index}-a",
                capabilities=required,
                ram_free_gib=ram + rng.choice((8, 16, 32)),
                vram_free_gib=vram + rng.choice((4, 8, 12)) if gpu else 0.0,
                cpu_free_fraction=rng.uniform(0.45, 0.95),
                active_claims=rng.choice((0, 0, 1, 2)),
            ),
            FleetNodeSnapshot(
                node_id=f"opaque-{index}-b",
                capabilities=required,
                ram_free_gib=ram + rng.choice((8, 16, 32)),
                vram_free_gib=vram + rng.choice((4, 8, 12)) if gpu else 0.0,
                cpu_free_fraction=rng.uniform(0.45, 0.95),
                active_claims=rng.choice((0, 1, 2)),
            ),
        ]
        states.append(
            FleetPlacementState(
                workload_id=f"fleet-benchmark-{seed}-{index:05d}",
                workload_type="gpu_batch" if gpu else "cpu_batch",
                description="Held-out fleet safety benchmark workload.",
                required_capabilities=required,
                preferred_capabilities=["rocm"] if gpu else [],
                estimated_ram_gib=ram,
                estimated_vram_gib=vram,
                estimated_duration_seconds=float(rng.choice((900, 1800, 7200))),
                checkpoint_supported=True,
                state_locality_hint=StateLocality.NONE,
                nodes=nodes,
                source="fleet_benchmark_v1",
                metadata={
                    "corpus_version": CORPUS_VERSION,
                    "seed": seed,
                    "index": index,
                },
            )
        )
    return states


def write_benchmark_states(path: str | Path, count: int = 64, seed: int = 47) -> dict[str, object]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    states = generate_benchmark_states(count, seed)
    payload = "".join(state.model_dump_json() + "\n" for state in states)
    output.write_text(payload, encoding="utf-8")
    return {
        "version": CORPUS_VERSION,
        "path": str(output),
        "records": len(states),
        "seed": seed,
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic fleet benchmark states")
    parser.add_argument("--output", required=True)
    parser.add_argument("--records", type=int, default=64)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    manifest = write_benchmark_states(args.output, args.records, args.seed)
    manifest_path = Path(args.output).with_suffix(Path(args.output).suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
