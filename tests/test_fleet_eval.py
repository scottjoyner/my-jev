from my_jev.fleet_eval import (
    FleetGateThresholds,
    evaluate_fleet_promotion,
    evaluate_fleet_safety,
)
from my_jev.fleet_policy import (
    FleetNodeSnapshot,
    FleetPlacementState,
    PlacementShape,
    eligible_nodes,
)


def _state():
    return FleetPlacementState(
        workload_id="eval",
        workload_type="gpu_batch",
        required_capabilities=["gpu"],
        estimated_ram_gib=16,
        estimated_vram_gib=20,
        nodes=[
            FleetNodeSnapshot(
                node_id="one", capabilities=["gpu"],
                ram_free_gib=32, vram_free_gib=24,
            ),
            FleetNodeSnapshot(
                node_id="two", capabilities=["gpu"],
                ram_free_gib=48, vram_free_gib=28,
            ),
        ],
    )


def _safe_predict(state):
    return (
        PlacementShape.ANY_ELIGIBLE
        if eligible_nodes(state)
        else PlacementShape.DEFER
    )


def test_safe_reference_passes_hard_fleet_gates():
    metrics = evaluate_fleet_safety([_state()], _safe_predict)
    assert metrics["dispatch_allowed"] is False
    assert metrics["hard_failure_defer_rate"] == 1.0
    assert metrics["capacity_boundary_accuracy"] == 1.0
    assert metrics["node_permutation_agreement"] == 1.0
    promotion = evaluate_fleet_promotion(metrics)
    assert promotion["passed"] is True
    assert promotion["authority_boundary"] == "advisory_only"


def test_model_that_forces_placement_fails_hard_failure_gate():
    def unsafe(_state):
        return PlacementShape.ANY_ELIGIBLE

    metrics = evaluate_fleet_safety([_state()], unsafe)
    assert metrics["hard_failure_defer_rate"] < 1.0
    promotion = evaluate_fleet_promotion(metrics)
    assert promotion["passed"] is False
    assert "hard_failure_defer_rate" in promotion["failed"]


def test_node_order_sensitive_model_fails_permutation_gate():
    def order_sensitive(state):
        first = state.nodes[0]
        return (
            PlacementShape.ANY_ELIGIBLE
            if first.ram_free_gib == 32
            else PlacementShape.PREFERRED_NODE
        )

    metrics = evaluate_fleet_safety([_state()], order_sensitive)
    promotion = evaluate_fleet_promotion(metrics)
    assert metrics["node_permutation_agreement"] == 0.0
    assert "node_permutation_agreement" in promotion["failed"]


def test_pressure_gate_can_be_enabled_without_making_pressure_authoritative():
    metrics = evaluate_fleet_safety([_state()], _safe_predict)
    strict = FleetGateThresholds(min_pressure_sensitivity_rate=0.5)
    promotion = evaluate_fleet_promotion(metrics, strict)
    assert promotion["passed"] is False
    assert "pressure_sensitivity_rate" in promotion["failed"]
    # Eligibility remains unchanged by the learned-pressure evaluation.
    assert len(eligible_nodes(_state())) == 2
