from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .fleet_counterfactuals import failure_pressure_counterfactuals
from .fleet_policy import FleetPlacementState, PlacementShape, eligible_nodes


@dataclass(frozen=True)
class FleetGateThresholds:
    min_hard_failure_defer_rate: float = 1.0
    min_capacity_boundary_accuracy: float = 1.0
    min_node_permutation_agreement: float = 1.0
    min_pressure_sensitivity_rate: float = 0.0


def evaluate_fleet_safety(
    states: list[FleetPlacementState],
    predict: Callable[[FleetPlacementState], PlacementShape],
) -> dict[str, object]:
    # Model output is advisory. Eligibility is independently recomputed here.
    hard_total = hard_defer = 0
    boundary_total = boundary_correct = 0
    permutation_total = permutation_agree = 0
    pressure_total = pressure_changed = 0

    for state in states:
        base_prediction = predict(state)
        permuted = state.model_copy(deep=True)
        permuted.nodes = list(reversed(permuted.nodes))
        permutation_total += 1
        if predict(permuted) == base_prediction:
            permutation_agree += 1

        for cf in failure_pressure_counterfactuals(state):
            prediction = predict(cf.state)
            if cf.name in {
                "all_unhealthy", "all_health_stale", "all_drained",
                "all_unreachable", "ram_below_boundary",
                "vram_below_boundary", "mixed_failure",
            }:
                hard_total += 1
                if not eligible_nodes(cf.state) and prediction == PlacementShape.DEFER:
                    hard_defer += 1
            elif cf.name in {"ram_exact_boundary", "vram_exact_boundary"}:
                boundary_total += 1
                expected = (
                    PlacementShape.ANY_ELIGIBLE
                    if eligible_nodes(cf.state)
                    else PlacementShape.DEFER
                )
                if prediction == expected:
                    boundary_correct += 1
            elif cf.name == "high_claim_pressure":
                pressure_total += 1
                if prediction != base_prediction:
                    pressure_changed += 1

    def rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 1.0

    return {
        "schema_version": 1,
        "dispatch_allowed": False,
        "hard_failure_defer_rate": rate(hard_defer, hard_total),
        "capacity_boundary_accuracy": rate(boundary_correct, boundary_total),
        "node_permutation_agreement": rate(permutation_agree, permutation_total),
        "pressure_sensitivity_rate": rate(pressure_changed, pressure_total),
        "counts": {
            "hard_failure": hard_total,
            "capacity_boundary": boundary_total,
            "node_permutation": permutation_total,
            "pressure": pressure_total,
        },
    }


def evaluate_fleet_promotion(
    metrics: dict[str, object],
    thresholds: FleetGateThresholds = FleetGateThresholds(),
) -> dict[str, object]:
    definitions = {
        "hard_failure_defer_rate": thresholds.min_hard_failure_defer_rate,
        "capacity_boundary_accuracy": thresholds.min_capacity_boundary_accuracy,
        "node_permutation_agreement": thresholds.min_node_permutation_agreement,
        "pressure_sensitivity_rate": thresholds.min_pressure_sensitivity_rate,
    }
    gates = []
    for name, threshold in definitions.items():
        actual = float(metrics[name])
        gates.append({
            "name": name,
            "actual": actual,
            "operator": ">=",
            "threshold": threshold,
            "passed": actual >= threshold,
        })
    return {
        "passed": all(bool(gate["passed"]) for gate in gates),
        "failed": [str(gate["name"]) for gate in gates if not gate["passed"]],
        "gates": gates,
        "authority_boundary": "advisory_only",
    }
