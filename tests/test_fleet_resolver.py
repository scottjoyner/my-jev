import pytest

from my_jev.fleet_policy import (
    FleetNodeSnapshot,
    FleetPlacementState,
    PlacementShape,
)
from my_jev.fleet_resolver import (
    FleetPlacementScores,
    rank_eligible_nodes,
    resolve_fleet_placement,
)


def _distribution(
    options,
    winner,
    confidence=0.9,
):
    remaining = (
        (1.0 - confidence)
        / (len(options) - 1)
    )
    return {
        option: (
            confidence
            if option == winner
            else remaining
        )
        for option in options
    }


def _scores(
    placement="any_eligible",
    confidence="high",
):
    return FleetPlacementScores(
        placement=_distribution(
            [
                "local",
                "preferred_node",
                "any_eligible",
                "split",
                "defer",
                "abstain",
            ],
            placement,
        ),
        resource_shape=_distribution(
            [
                "cpu",
                "gpu",
                "memory",
                "io",
                "network",
                "mixed",
            ],
            "gpu",
        ),
        latency_class=_distribution(
            [
                "interactive",
                "nearline",
                "batch",
                "background",
            ],
            "batch",
        ),
        state_locality=_distribution(
            [
                "none",
                "weak",
                "strong",
                "pinned",
            ],
            "none",
        ),
        migration_tolerance=_distribution(
            [
                "free",
                "checkpointable",
                "sticky",
                "immovable",
            ],
            "checkpointable",
        ),
        redundancy=_distribution(
            [
                "single",
                "retry_elsewhere",
                "replicated",
            ],
            "retry_elsewhere",
        ),
        fleet_pressure=_distribution(
            [
                "idle",
                "light",
                "moderate",
                "high",
                "saturated",
            ],
            "light",
        ),
        placement_confidence=_distribution(
            [
                "very_low",
                "low",
                "medium",
                "high",
                "very_high",
            ],
            confidence,
        ),
    )


def _node(
    node_id,
    **kwargs,
):
    return FleetNodeSnapshot(
        node_id=node_id,
        **kwargs,
    )


def test_resolver_never_returns_ineligible_node():
    state = FleetPlacementState(
        workload_id="w1",
        workload_type="training",
        required_capabilities=[
            "gpu",
            "rocm",
        ],
        estimated_ram_gib=16,
        estimated_vram_gib=20,
        nodes=[
            _node(
                "eligible",
                capabilities=[
                    "gpu",
                    "rocm",
                ],
                ram_free_gib=64,
                vram_free_gib=28,
            ),
            _node(
                "drained",
                capabilities=[
                    "gpu",
                    "rocm",
                ],
                ram_free_gib=64,
                vram_free_gib=32,
                drained=True,
            ),
            _node(
                "too-small",
                capabilities=[
                    "gpu",
                    "rocm",
                ],
                ram_free_gib=64,
                vram_free_gib=12,
            ),
        ],
    )

    result = resolve_fleet_placement(
        _scores(),
        state,
    )

    assert result.dispatch_allowed is False
    assert result.observer_only is True
    assert result.eligible_node_ids == [
        "eligible"
    ]
    assert result.selected_node_id == (
        "eligible"
    )


def test_no_eligible_nodes_forces_defer():
    state = FleetPlacementState(
        workload_id="w2",
        workload_type="training",
        required_capabilities=["gpu"],
        estimated_vram_gib=24,
        nodes=[
            _node(
                "cpu-only",
                capabilities=["cpu"],
                ram_free_gib=64,
            )
        ],
    )

    result = resolve_fleet_placement(
        _scores(
            placement="preferred_node"
        ),
        state,
    )

    assert (
        result.resolved_placement
        == PlacementShape.DEFER
    )
    assert result.selected_node_id is None


def test_pinned_locality_overrides_learned_shape():
    state = FleetPlacementState(
        workload_id="w3",
        workload_type="recovery",
        required_capabilities=[
            "storage"
        ],
        pinned_node_id="owner",
        nodes=[
            _node(
                "owner",
                capabilities=["storage"],
                ram_free_gib=32,
            ),
            _node(
                "other",
                capabilities=["storage"],
                ram_free_gib=128,
            ),
        ],
    )

    result = resolve_fleet_placement(
        _scores(
            placement="any_eligible"
        ),
        state,
    )

    assert (
        result.resolved_placement
        == PlacementShape.LOCAL
    )
    assert result.selected_node_id == (
        "owner"
    )
    assert result.eligible_node_ids == [
        "owner"
    ]


def test_preferred_capability_is_only_ranked_inside_eligibility():
    state = FleetPlacementState(
        workload_id="w4",
        workload_type="inference",
        required_capabilities=["gpu"],
        preferred_capabilities=["rocm"],
        estimated_vram_gib=8,
        nodes=[
            _node(
                "generic-gpu",
                capabilities=["gpu"],
                ram_free_gib=64,
                vram_free_gib=30,
            ),
            _node(
                "rocm-gpu",
                capabilities=[
                    "gpu",
                    "rocm",
                ],
                ram_free_gib=32,
                vram_free_gib=16,
            ),
        ],
    )

    result = resolve_fleet_placement(
        _scores(
            placement="preferred_node"
        ),
        state,
    )

    assert result.ranked_node_ids[0] == (
        "rocm-gpu"
    )
    assert result.selected_node_id == (
        "rocm-gpu"
    )


def test_low_model_confidence_abstains_even_with_capacity():
    state = FleetPlacementState(
        workload_id="w5",
        workload_type="inference",
        nodes=[
            _node(
                "node-a",
                capabilities=["cpu"],
                ram_free_gib=64,
            )
        ],
    )

    result = resolve_fleet_placement(
        _scores(confidence="low"),
        state,
    )

    assert (
        result.resolved_placement
        == PlacementShape.ABSTAIN
    )
    assert result.selected_node_id is None


def test_split_requires_multiple_eligible_and_checkpoint_support():
    state = FleetPlacementState(
        workload_id="w6",
        workload_type="batch",
        checkpoint_supported=False,
        nodes=[
            _node(
                "a",
                capabilities=["cpu"],
                ram_free_gib=64,
            ),
            _node(
                "b",
                capabilities=["cpu"],
                ram_free_gib=64,
            ),
        ],
    )

    rejected = resolve_fleet_placement(
        _scores(placement="split"),
        state,
    )
    assert (
        rejected.resolved_placement
        == PlacementShape.DEFER
    )

    accepted = resolve_fleet_placement(
        _scores(placement="split"),
        state.model_copy(
            update={
                "checkpoint_supported": True
            }
        ),
    )
    assert (
        accepted.resolved_placement
        == PlacementShape.SPLIT
    )
    assert accepted.selected_node_id is None
    assert len(
        accepted.ranked_node_ids
    ) == 2


def test_rank_is_deterministic_under_input_permutation():
    state = FleetPlacementState(
        workload_id="w7",
        workload_type="batch",
        preferred_capabilities=["gpu"],
    )
    first = _node(
        "b",
        capabilities=["gpu"],
        ram_free_gib=32,
        vram_free_gib=16,
        cpu_free_fraction=0.8,
    )
    second = _node(
        "a",
        capabilities=["gpu"],
        ram_free_gib=32,
        vram_free_gib=16,
        cpu_free_fraction=0.8,
    )

    forward = [
        node.node_id
        for node in rank_eligible_nodes(
            state,
            [first, second],
        )
    ]
    reverse = [
        node.node_id
        for node in rank_eligible_nodes(
            state,
            [second, first],
        )
    ]

    assert forward == reverse == [
        "a",
        "b",
    ]


def test_local_without_authoritative_pin_fails_safe():
    state = FleetPlacementState(
        workload_id="w8",
        workload_type="locality-sensitive",
        nodes=[
            _node(
                "node-a",
                capabilities=["cpu"],
                ram_free_gib=64,
            )
        ],
    )

    result = resolve_fleet_placement(
        _scores(placement="local"),
        state,
    )

    assert (
        result.resolved_placement
        == PlacementShape.DEFER
    )
    assert result.selected_node_id is None
