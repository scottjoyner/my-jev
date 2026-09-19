from __future__ import annotations

from pydantic import BaseModel, Field

from .fleet_policy import (
    CONFIDENCE_OPTIONS,
    LATENCY_OPTIONS,
    LOCALITY_OPTIONS,
    MIGRATION_OPTIONS,
    PLACEMENT_OPTIONS,
    PRESSURE_OPTIONS,
    REDUNDANCY_OPTIONS,
    RESOURCE_OPTIONS,
    FleetNodeSnapshot,
    FleetPlacementState,
    PlacementShape,
    eligible_nodes,
    fleet_placement_questions,
)


class FleetPlacementScores(BaseModel):
    placement: dict[str, float]
    resource_shape: dict[str, float]
    latency_class: dict[str, float]
    state_locality: dict[str, float]
    migration_tolerance: dict[str, float]
    redundancy: dict[str, float]
    fleet_pressure: dict[str, float]
    placement_confidence: dict[str, float]

    def choice(
        self,
        field: str,
    ) -> tuple[str, float]:
        distribution = getattr(
            self,
            field,
        )
        if (
            not isinstance(
                distribution,
                dict,
            )
            or not distribution
        ):
            raise ValueError(
                f"{field} is not a categorical distribution"
            )
        name = max(
            distribution,
            key=distribution.get,
        )
        return (
            name,
            float(distribution[name]),
        )


class FleetPlacementResolution(BaseModel):
    observer_only: bool = True
    dispatch_allowed: bool = False
    model_placement: PlacementShape
    model_placement_confidence: float
    resolved_placement: PlacementShape
    eligible_node_ids: list[str] = Field(
        default_factory=list
    )
    ranked_node_ids: list[str] = Field(
        default_factory=list
    )
    selected_node_id: str | None = None
    reasons: list[str] = Field(
        default_factory=list
    )


def _distribution(
    prediction: dict[str, object],
) -> dict[str, float]:
    options = prediction.get("options")
    probabilities = prediction.get(
        "probabilities"
    )
    if (
        not isinstance(options, list)
        or not isinstance(
            probabilities,
            list,
        )
    ):
        raise ValueError(
            "prediction is missing "
            "options/probabilities"
        )
    if len(options) != len(probabilities):
        raise ValueError(
            "prediction option/probability "
            "length mismatch"
        )
    return {
        str(option): float(probability)
        for option, probability in zip(
            options,
            probabilities,
            strict=True,
        )
    }


def scores_from_predictions(
    predictions: list[
        dict[str, object]
    ],
) -> FleetPlacementScores:
    by_name = {
        str(
            item.get("question")
        ): item
        for item in predictions
    }
    missing = (
        set(
            fleet_placement_questions()
        )
        - set(by_name)
    )
    if missing:
        raise ValueError(
            "missing fleet-placement "
            f"predictions: {sorted(missing)}"
        )

    return FleetPlacementScores(
        placement=_distribution(
            by_name["placement"]
        ),
        resource_shape=_distribution(
            by_name["resource_shape"]
        ),
        latency_class=_distribution(
            by_name["latency_class"]
        ),
        state_locality=_distribution(
            by_name["state_locality"]
        ),
        migration_tolerance=_distribution(
            by_name[
                "migration_tolerance"
            ]
        ),
        redundancy=_distribution(
            by_name["redundancy"]
        ),
        fleet_pressure=_distribution(
            by_name["fleet_pressure"]
        ),
        placement_confidence=_distribution(
            by_name[
                "placement_confidence"
            ]
        ),
    )


def _headroom(
    node: FleetNodeSnapshot,
    state: FleetPlacementState,
) -> tuple[float, float, float, int]:
    ram = (
        node.ram_free_gib
        - state.estimated_ram_gib
    )
    vram = (
        node.vram_free_gib
        - state.estimated_vram_gib
    )
    return (
        vram,
        ram,
        node.cpu_free_fraction,
        -node.active_claims,
    )


def _preferred_matches(
    node: FleetNodeSnapshot,
    state: FleetPlacementState,
) -> int:
    preferred = set(
        state.preferred_capabilities
    )
    if not preferred:
        return 0
    return len(
        preferred.intersection(
            node.capabilities
        )
    )


def rank_eligible_nodes(
    state: FleetPlacementState,
    nodes: list[FleetNodeSnapshot],
) -> list[FleetNodeSnapshot]:
    """Deterministically rank only nodes that already passed hard eligibility."""

    return sorted(
        nodes,
        key=lambda node: (
            -_preferred_matches(
                node,
                state,
            ),
            -_headroom(
                node,
                state,
            )[0],
            -_headroom(
                node,
                state,
            )[1],
            -_headroom(
                node,
                state,
            )[2],
            -_headroom(
                node,
                state,
            )[3],
            node.node_id,
        ),
    )


def _confidence_rank(
    name: str,
) -> int:
    try:
        return CONFIDENCE_OPTIONS.index(
            name
        )
    except ValueError:
        return 0


def resolve_fleet_placement(
    scores: FleetPlacementScores,
    state: FleetPlacementState,
    *,
    minimum_confidence: str = "medium",
) -> FleetPlacementResolution:
    """Resolve a learned shape only inside authoritative fleet eligibility.

    This function is observer-only. It can name a recommended eligible node,
    but it never grants dispatch authority or mutates scheduler state.
    """

    validate_score_options(scores)
    placement_name, placement_probability = (
        scores.choice("placement")
    )
    confidence_name, _ = scores.choice(
        "placement_confidence"
    )
    model_placement = PlacementShape(
        placement_name
    )

    eligible = eligible_nodes(state)
    ranked = rank_eligible_nodes(
        state,
        eligible,
    )
    eligible_ids = [
        node.node_id
        for node in eligible
    ]
    ranked_ids = [
        node.node_id
        for node in ranked
    ]
    reasons: list[str] = []

    def result(
        resolved: PlacementShape,
        *,
        selected: str | None = None,
    ) -> FleetPlacementResolution:
        return FleetPlacementResolution(
            observer_only=True,
            dispatch_allowed=False,
            model_placement=(
                model_placement
            ),
            model_placement_confidence=(
                placement_probability
            ),
            resolved_placement=resolved,
            eligible_node_ids=eligible_ids,
            ranked_node_ids=ranked_ids,
            selected_node_id=selected,
            reasons=reasons,
        )

    if not eligible:
        reasons.append(
            "no node survives authoritative "
            "health/capability/capacity/"
            "locality filters"
        )
        return result(
            PlacementShape.DEFER
        )

    if (
        _confidence_rank(
            confidence_name
        )
        < _confidence_rank(
            minimum_confidence
        )
    ):
        reasons.append(
            "model placement confidence "
            "is below observer threshold"
        )
        return result(
            PlacementShape.ABSTAIN
        )

    if model_placement in {
        PlacementShape.DEFER,
        PlacementShape.ABSTAIN,
    }:
        reasons.append(
            "model requested no placement"
        )
        return result(
            model_placement
        )

    if state.pinned_node_id is not None:
        # eligible_nodes already proved that the pinned node is legal.
        pinned = state.pinned_node_id
        reasons.append(
            "pinned locality is authoritative"
        )
        return result(
            PlacementShape.LOCAL,
            selected=pinned,
        )

    if (
        model_placement
        == PlacementShape.LOCAL
    ):
        reasons.append(
            "local placement has no "
            "authoritative pinned node"
        )
        return result(
            PlacementShape.DEFER
        )

    if (
        model_placement
        == PlacementShape.SPLIT
    ):
        if len(ranked) < 2:
            reasons.append(
                "split placement requires "
                "at least two eligible nodes"
            )
            return result(
                PlacementShape.DEFER
            )
        if not state.checkpoint_supported:
            reasons.append(
                "split placement requires "
                "checkpoint/restart support"
            )
            return result(
                PlacementShape.DEFER
            )
        reasons.append(
            "split is advisory across "
            "eligible nodes only"
        )
        return result(
            PlacementShape.SPLIT
        )

    selected = ranked[0].node_id
    if (
        model_placement
        == PlacementShape.PREFERRED_NODE
    ):
        if not state.preferred_capabilities:
            reasons.append(
                "no preferred capability "
                "was declared; using "
                "deterministic eligible rank"
            )
        elif _preferred_matches(
            ranked[0],
            state,
        ) == 0:
            reasons.append(
                "no eligible node satisfies "
                "a preferred capability; "
                "using deterministic fallback"
            )
        else:
            reasons.append(
                "selected highest-ranked "
                "eligible preferred-capability "
                "candidate"
            )
        return result(
            PlacementShape.PREFERRED_NODE,
            selected=selected,
        )

    reasons.append(
        "selected highest-ranked "
        "authoritatively eligible node"
    )
    return result(
        PlacementShape.ANY_ELIGIBLE,
        selected=selected,
    )


def validate_score_options(
    scores: FleetPlacementScores,
) -> None:
    expected = {
        "placement": PLACEMENT_OPTIONS,
        "resource_shape": RESOURCE_OPTIONS,
        "latency_class": LATENCY_OPTIONS,
        "state_locality": LOCALITY_OPTIONS,
        "migration_tolerance": MIGRATION_OPTIONS,
        "redundancy": REDUNDANCY_OPTIONS,
        "fleet_pressure": PRESSURE_OPTIONS,
        "placement_confidence": (
            CONFIDENCE_OPTIONS
        ),
    }
    for field, options in expected.items():
        actual = set(
            getattr(scores, field)
        )
        if actual != set(options):
            raise ValueError(
                f"{field} options mismatch: "
                f"expected {options}, "
                f"got {sorted(actual)}"
            )
