from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .calibration import load_temperature
from .checkpoint import load_checkpoint
from .fleet_eval import FleetGateThresholds, evaluate_fleet_promotion, evaluate_fleet_safety
from .fleet_policy import FleetPlacementState, PlacementShape, build_fleet_placement_record


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_states(path: str | Path) -> list[FleetPlacementState]:
    source = Path(path)
    states = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            states.append(FleetPlacementState.model_validate_json(line))
    if not states:
        raise ValueError("fleet benchmark requires at least one state")
    return states


def fleet_benchmark(
    checkpoint: str,
    states_path: str,
    *,
    calibration: str | None = None,
    thresholds: FleetGateThresholds = FleetGateThresholds(),
) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(checkpoint, device=device)
    temperature = load_temperature(calibration)

    def predict(state: FleetPlacementState) -> PlacementShape:
        record = build_fleet_placement_record(state)
        outputs = model.forward_records([record])
        placement = next(output for output in outputs if output.name == "placement")
        probabilities = placement.probabilities_at_temperature(temperature)
        return PlacementShape(placement.options[int(probabilities.argmax().item())])

    states = load_states(states_path)
    with torch.inference_mode():
        metrics = evaluate_fleet_safety(states, predict)
    promotion = evaluate_fleet_promotion(metrics, thresholds)
    source = Path(states_path)
    return {
        "schema_version": 1,
        "checkpoint": str(Path(checkpoint)),
        "calibration": calibration,
        "temperature": temperature,
        "device": str(device),
        "states": {
            "path": str(source),
            "sha256": _sha256(source),
            "count": len(states),
        },
        "metrics": metrics,
        "promotion": promotion,
        "authority_boundary": "advisory_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a checkpoint against fleet safety counterfactuals"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--states", required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-hard-failure-defer-rate", type=float, default=1.0)
    parser.add_argument("--min-capacity-boundary-accuracy", type=float, default=1.0)
    parser.add_argument("--min-node-permutation-agreement", type=float, default=1.0)
    parser.add_argument("--min-pressure-sensitivity-rate", type=float, default=0.0)
    args = parser.parse_args()

    thresholds = FleetGateThresholds(
        min_hard_failure_defer_rate=args.min_hard_failure_defer_rate,
        min_capacity_boundary_accuracy=args.min_capacity_boundary_accuracy,
        min_node_permutation_agreement=args.min_node_permutation_agreement,
        min_pressure_sensitivity_rate=args.min_pressure_sensitivity_rate,
    )
    result = fleet_benchmark(
        args.checkpoint,
        args.states,
        calibration=args.calibration,
        thresholds=thresholds,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, sort_keys=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
