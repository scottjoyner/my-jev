from __future__ import annotations

from dataclasses import dataclass

from .fleet_policy import (
    FleetPlacementState,
    PlacementShape,
    eligible_nodes,
)


@dataclass(frozen=True)
class FleetCounterfactual:
    name: str
    state: FleetPlacementState
    expected_placement: PlacementShape
    expected_eligible_count: int


def _copy(state: FleetPlacementState, name: str) -> FleetPlacementState:
    result = state.model_copy(deep=True)
    result.workload_id = f"{state.workload_id}-{name}"
    result.metadata = {
        **result.metadata,
        "counterfactual": name,
        "counterfactual_suite": "fleet-failure-pressure-v1",
    }
    return result


def failure_pressure_counterfactuals(
    state: FleetPlacementState,
) -> list[FleetCounterfactual]:
    """Generate hard fleet-state perturbations around one otherwise valid workload.

    These are verifier-oriented evaluation examples. Expected placement is derived
    from authoritative eligibility, not from a learned scorer.
    """
    if not state.nodes:
        return []

    variants: list[FleetCounterfactual] = []

    def add(name: str, changed: FleetPlacementState) -> None:
        count = len(eligible_nodes(changed))
        variants.append(
            FleetCounterfactual(
                name=name,
                state=changed,
                expected_placement=(
                    PlacementShape.ANY_ELIGIBLE
                    if count
                    else PlacementShape.DEFER
                ),
                expected_eligible_count=count,
            )
        )

    unhealthy = _copy(state, "all_unhealthy")
    for node in unhealthy.nodes:
        node.healthy = False
    add("all_unhealthy", unhealthy)

    stale = _copy(state, "all_health_stale")
    for node in stale.nodes:
        node.health_fresh = False
    add("all_health_stale", stale)

    drained = _copy(state, "all_drained")
    for node in drained.nodes:
        node.drained = True
    add("all_drained", drained)

    unreachable = _copy(state, "all_unreachable")
    for node in unreachable.nodes:
        node.reachable = False
    add("all_unreachable", unreachable)

    ram_below = _copy(state, "ram_below_boundary")
    if state.estimated_ram_gib > 0:
        for node in ram_below.nodes:
            node.ram_free_gib = max(0.0, state.estimated_ram_gib - 0.01)
        add("ram_below_boundary", ram_below)

    ram_exact = _copy(state, "ram_exact_boundary")
    if state.estimated_ram_gib > 0:
        for node in ram_exact.nodes:
            node.ram_free_gib = state.estimated_ram_gib
        add("ram_exact_boundary", ram_exact)

    if state.estimated_vram_gib > 0:
        vram_below = _copy(state, "vram_below_boundary")
        for node in vram_below.nodes:
            node.vram_free_gib = max(0.0, state.estimated_vram_gib - 0.01)
        add("vram_below_boundary", vram_below)

        vram_exact = _copy(state, "vram_exact_boundary")
        for node in vram_exact.nodes:
            node.vram_free_gib = state.estimated_vram_gib
        add("vram_exact_boundary", vram_exact)

    claims = _copy(state, "high_claim_pressure")
    for node in claims.nodes:
        node.active_claims = max(node.active_claims, 8)
        node.cpu_free_fraction = min(node.cpu_free_fraction, 0.1)
    add("high_claim_pressure", claims)

    mixed = _copy(state, "mixed_failure")
    for index, node in enumerate(mixed.nodes):
        if index % 4 == 0:
            node.healthy = False
        elif index % 4 == 1:
            node.drained = True
        elif index % 4 == 2:
            node.reachable = False
        else:
            node.health_fresh = False
    add("mixed_failure", mixed)

    return variants
