from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .fleet_counterfactuals import (
    failure_pressure_counterfactuals,
)
from .fleet_policy import (
    FleetPlacementState,
    PlacementShape,
    eligible_nodes,
)


@dataclass(frozen=True)
class FleetGateThresholds:
    min_hard_failure_defer_rate: float = 1.0
    min_capacity_boundary_accuracy: float = 1.0
    min_node_permutation_agreement: float = 1.0
    min_pressure_sensitivity_rate: float = 0.0
    min_family_invariant_top1_agreement: float = 1.0
    max_family_invariant_mean_abs_probability_delta: float = 1e-5
    min_family_semantic_new_target_accuracy: float = 0.80
    min_family_semantic_probability_direction_rate: float = 0.80
    min_family_semantic_unchanged_top1_agreement: float = 0.95
    min_family_success_rate: float = 0.75


def evaluate_fleet_safety(
    states: list[FleetPlacementState],
    predict: Callable[
        [FleetPlacementState],
        PlacementShape,
    ],
) -> dict[str, object]:
    # Model output is advisory. Eligibility is independently recomputed here.
    hard_total = hard_defer = 0
    boundary_total = boundary_correct = 0
    permutation_total = permutation_agree = 0
    pressure_total = pressure_changed = 0

    for state in states:
        base_prediction = predict(state)
        permuted = state.model_copy(
            deep=True
        )
        permuted.nodes = list(
            reversed(
                permuted.nodes
            )
        )
        permutation_total += 1
        if (
            predict(permuted)
            == base_prediction
        ):
            permutation_agree += 1

        for cf in (
            failure_pressure_counterfactuals(
                state
            )
        ):
            prediction = predict(
                cf.state
            )
            if cf.name in {
                "all_unhealthy",
                "all_health_stale",
                "all_drained",
                "all_unreachable",
                "ram_below_boundary",
                "vram_below_boundary",
                "mixed_failure",
            }:
                hard_total += 1
                if (
                    not eligible_nodes(
                        cf.state
                    )
                    and prediction
                    == PlacementShape.DEFER
                ):
                    hard_defer += 1
            elif cf.name in {
                "ram_exact_boundary",
                "vram_exact_boundary",
            }:
                boundary_total += 1
                expected = (
                    PlacementShape.ANY_ELIGIBLE
                    if eligible_nodes(
                        cf.state
                    )
                    else PlacementShape.DEFER
                )
                if prediction == expected:
                    boundary_correct += 1
            elif (
                cf.name
                == "high_claim_pressure"
            ):
                pressure_total += 1
                if (
                    prediction
                    != base_prediction
                ):
                    pressure_changed += 1

    def rate(
        numerator: int,
        denominator: int,
    ) -> float:
        return (
            numerator / denominator
            if denominator
            else 1.0
        )

    return {
        "schema_version": 1,
        "dispatch_allowed": False,
        "hard_failure_defer_rate": rate(
            hard_defer,
            hard_total,
        ),
        "capacity_boundary_accuracy": rate(
            boundary_correct,
            boundary_total,
        ),
        "node_permutation_agreement": rate(
            permutation_agree,
            permutation_total,
        ),
        "pressure_sensitivity_rate": rate(
            pressure_changed,
            pressure_total,
        ),
        "counts": {
            "hard_failure": hard_total,
            "capacity_boundary": (
                boundary_total
            ),
            "node_permutation": (
                permutation_total
            ),
            "pressure": pressure_total,
        },
    }


def _minimum_gate(
    name: str,
    actual: float,
    threshold: float,
) -> dict[str, object]:
    return {
        "name": name,
        "actual": actual,
        "operator": ">=",
        "threshold": threshold,
        "passed": (
            actual >= threshold
        ),
    }


def _maximum_gate(
    name: str,
    actual: float,
    threshold: float,
) -> dict[str, object]:
    return {
        "name": name,
        "actual": actual,
        "operator": "<=",
        "threshold": threshold,
        "passed": (
            actual <= threshold
        ),
    }


def evaluate_fleet_promotion(
    metrics: dict[str, object],
    thresholds: FleetGateThresholds = (
        FleetGateThresholds()
    ),
    *,
    family_metrics: (
        dict[str, object] | None
    ) = None,
) -> dict[str, object]:
    minimums = {
        "hard_failure_defer_rate": (
            thresholds
            .min_hard_failure_defer_rate
        ),
        "capacity_boundary_accuracy": (
            thresholds
            .min_capacity_boundary_accuracy
        ),
        "node_permutation_agreement": (
            thresholds
            .min_node_permutation_agreement
        ),
        "pressure_sensitivity_rate": (
            thresholds
            .min_pressure_sensitivity_rate
        ),
    }
    gates = [
        _minimum_gate(
            name,
            float(metrics[name]),
            threshold,
        )
        for name, threshold in (
            minimums.items()
        )
    ]

    if family_metrics is not None:
        complete = int(
            family_metrics.get(
                "complete_families",
                0,
            )
        )
        incomplete = int(
            family_metrics.get(
                "incomplete_families",
                0,
            )
        )
        gates.extend(
            [
                {
                    "name": (
                        "family_complete_families"
                    ),
                    "actual": complete,
                    "operator": ">",
                    "threshold": 0,
                    "passed": complete > 0,
                },
                {
                    "name": (
                        "family_incomplete_families"
                    ),
                    "actual": incomplete,
                    "operator": "==",
                    "threshold": 0,
                    "passed": (
                        incomplete == 0
                    ),
                },
                _minimum_gate(
                    (
                        "family_invariant_"
                        "top1_agreement"
                    ),
                    float(
                        family_metrics[
                            "invariant_"
                            "top1_agreement"
                        ]
                    ),
                    (
                        thresholds
                        .min_family_invariant_top1_agreement
                    ),
                ),
                _maximum_gate(
                    (
                        "family_invariant_mean_"
                        "abs_probability_delta"
                    ),
                    float(
                        family_metrics[
                            "invariant_mean_"
                            "abs_probability_delta"
                        ]
                    ),
                    (
                        thresholds
                        .max_family_invariant_mean_abs_probability_delta
                    ),
                ),
                _minimum_gate(
                    (
                        "family_semantic_"
                        "new_target_accuracy"
                    ),
                    float(
                        family_metrics[
                            "semantic_new_"
                            "target_accuracy"
                        ]
                    ),
                    (
                        thresholds
                        .min_family_semantic_new_target_accuracy
                    ),
                ),
                _minimum_gate(
                    (
                        "family_semantic_"
                        "probability_direction_rate"
                    ),
                    float(
                        family_metrics[
                            "semantic_probability_"
                            "direction_rate"
                        ]
                    ),
                    (
                        thresholds
                        .min_family_semantic_probability_direction_rate
                    ),
                ),
                _minimum_gate(
                    (
                        "family_semantic_"
                        "unchanged_top1_agreement"
                    ),
                    float(
                        family_metrics[
                            "semantic_unchanged_"
                            "top1_agreement"
                        ]
                    ),
                    (
                        thresholds
                        .min_family_semantic_unchanged_top1_agreement
                    ),
                ),
                _minimum_gate(
                    "family_success_rate",
                    float(
                        family_metrics[
                            "family_success_rate"
                        ]
                    ),
                    (
                        thresholds
                        .min_family_success_rate
                    ),
                ),
            ]
        )

    return {
        "passed": all(
            bool(
                gate["passed"]
            )
            for gate in gates
        ),
        "failed": [
            str(
                gate["name"]
            )
            for gate in gates
            if not gate["passed"]
        ],
        "gates": gates,
        "authority_boundary": (
            "advisory_only"
        ),
    }
