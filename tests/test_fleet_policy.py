import json

from my_jev.fleet_policy import (
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
    eligible_nodes,
)


def _node(node_id, **kwargs):
    return FleetNodeSnapshot(node_id=node_id, **kwargs)


def test_model_state_omits_node_identity_and_is_permutation_invariant():
    first = _node(
        "xwing",
        capabilities=["gpu", "rocm"],
        ram_free_gib=48,
        vram_free_gib=30,
    )
    second = _node(
        "x1-370",
        capabilities=["cpu"],
        ram_free_gib=70,
    )
    a = FleetPlacementState(
        workload_id="w1",
        workload_type="inference",
        required_capabilities=["gpu"],
        nodes=[first, second],
    )
    b = a.model_copy(update={"nodes": [second, first]})

    assert a.as_model_state() == b.as_model_state()
    encoded = a.as_model_state()
    assert "xwing" not in encoded
    assert "x1-370" not in encoded


def test_record_is_explicitly_non_dispatching():
    state = FleetPlacementState(
        workload_id="w2",
        workload_type="benchmark",
    )
    record = build_fleet_placement_record(state)

    assert record.metadata["domain"] == "fleet_placement"
    assert record.metadata["dispatch_allowed"] is False
    assert record.targets is None
    assert set(record.questions) == {
        "placement",
        "resource_shape",
        "latency_class",
        "state_locality",
        "migration_tolerance",
        "redundancy",
        "fleet_pressure",
        "placement_confidence",
    }


def test_label_maps_to_typed_targets():
    state = FleetPlacementState(
        workload_id="w3",
        workload_type="training",
    )
    label = FleetPlacementLabel(
        placement=PlacementShape.ANY_ELIGIBLE,
        resource_shape=ResourceShape.GPU,
        latency_class=LatencyClass.BATCH,
        state_locality=StateLocality.WEAK,
        migration_tolerance=MigrationTolerance.CHECKPOINTABLE,
        redundancy=RedundancyShape.RETRY_ELSEWHERE,
        fleet_pressure=2,
        placement_confidence=3,
    )
    record = build_fleet_placement_record(state, label=label)

    assert record.targets is not None
    assert record.targets["placement"].index == 2
    assert record.targets["resource_shape"].index == 1
    assert record.targets["fleet_pressure"].index == 2


def test_hard_eligibility_filters_health_drain_capacity_and_capability():
    state = FleetPlacementState(
        workload_id="w4",
        workload_type="gpu-job",
        required_capabilities=["gpu"],
        estimated_ram_gib=16,
        estimated_vram_gib=20,
        nodes=[
            _node(
                "good",
                capabilities=["gpu"],
                ram_free_gib=32,
                vram_free_gib=24,
            ),
            _node(
                "drained",
                capabilities=["gpu"],
                ram_free_gib=32,
                vram_free_gib=24,
                drained=True,
            ),
            _node(
                "stale",
                capabilities=["gpu"],
                ram_free_gib=32,
                vram_free_gib=24,
                health_fresh=False,
            ),
            _node(
                "small",
                capabilities=["gpu"],
                ram_free_gib=32,
                vram_free_gib=12,
            ),
            _node(
                "cpu-only",
                capabilities=["cpu"],
                ram_free_gib=64,
                vram_free_gib=32,
            ),
        ],
    )

    assert [node.node_id for node in eligible_nodes(state)] == ["good"]


def test_pinned_locality_is_authoritative_filter_not_model_feature():
    state = FleetPlacementState(
        workload_id="w5",
        workload_type="recovery",
        required_capabilities=["storage"],
        pinned_node_id="nas-owner",
        nodes=[
            _node(
                "nas-owner",
                capabilities=["storage"],
                ram_free_gib=16,
            ),
            _node(
                "other",
                capabilities=["storage"],
                ram_free_gib=64,
            ),
        ],
    )

    assert [node.node_id for node in eligible_nodes(state)] == ["nas-owner"]
    payload = json.loads(state.as_model_state())
    assert payload["workload"]["pinned"] is True
    assert "nas-owner" not in state.as_model_state()
