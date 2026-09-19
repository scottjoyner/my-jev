from my_jev.fleet_counterfactuals import failure_pressure_counterfactuals
from my_jev.fleet_policy import (
    FleetNodeSnapshot,
    FleetPlacementState,
    PlacementShape,
    eligible_nodes,
)


def _base() -> FleetPlacementState:
    return FleetPlacementState(
        workload_id="counterfactual",
        workload_type="gpu_batch",
        required_capabilities=["gpu"],
        estimated_ram_gib=16,
        estimated_vram_gib=20,
        nodes=[
            FleetNodeSnapshot(
                node_id="a",
                capabilities=["gpu"],
                ram_free_gib=32,
                vram_free_gib=24,
            ),
            FleetNodeSnapshot(
                node_id="b",
                capabilities=["gpu"],
                ram_free_gib=48,
                vram_free_gib=28,
            ),
        ],
    )


def test_failure_counterfactuals_force_defer():
    by_name = {
        item.name: item
        for item in failure_pressure_counterfactuals(_base())
    }
    for name in (
        "all_unhealthy",
        "all_health_stale",
        "all_drained",
        "all_unreachable",
        "ram_below_boundary",
        "vram_below_boundary",
        "mixed_failure",
    ):
        item = by_name[name]
        assert item.expected_placement == PlacementShape.DEFER
        assert item.expected_eligible_count == 0
        assert eligible_nodes(item.state) == []


def test_exact_capacity_boundary_remains_eligible():
    by_name = {
        item.name: item
        for item in failure_pressure_counterfactuals(_base())
    }
    for name in ("ram_exact_boundary", "vram_exact_boundary"):
        item = by_name[name]
        assert item.expected_placement == PlacementShape.ANY_ELIGIBLE
        assert item.expected_eligible_count == 2


def test_pressure_is_not_silently_promoted_to_hard_ineligibility():
    by_name = {
        item.name: item
        for item in failure_pressure_counterfactuals(_base())
    }
    item = by_name["high_claim_pressure"]
    assert item.expected_placement == PlacementShape.ANY_ELIGIBLE
    assert item.expected_eligible_count == 2
    assert all(node.active_claims >= 8 for node in item.state.nodes)


def test_counterfactuals_remain_non_dispatching_policy_inputs():
    for item in failure_pressure_counterfactuals(_base()):
        assert item.state.metadata["counterfactual_suite"] == "fleet-failure-pressure-v1"
        encoded = item.state.as_model_state()
        assert '"node_id"' not in encoded
        assert '"a"' not in encoded
        assert '"b"' not in encoded
