from datetime import UTC, datetime

import pytest

from my_jev.agent_policy import (
    ActionScope,
    AgentRoute,
    Delegation,
    ResolvedAgentPolicy,
    ResolvedDisposition,
    RiskLevel,
)
from my_jev.fleet_policy import PlacementShape
from my_jev.fleet_resolver import FleetPlacementResolution
from my_jev.uhp_advisory import (
    HERMES_SYSTEM_ONE_PROFILE,
    UHP_VERSION,
    SystemOneProvenance,
    build_hermes_system_one_profile,
    build_uhp_response_fixture,
    canonical_json,
    canonical_sha256,
)


def decision(
    disposition=ResolvedDisposition.ACT,
    *,
    confidence=0.91,
    approval_required=False,
):
    return ResolvedAgentPolicy(
        disposition=disposition,
        model_route=AgentRoute.ACT,
        model_route_confidence=confidence,
        action_scope=ActionScope.READ_ONLY,
        risk=RiskLevel.LOW,
        delegation=Delegation.SELF,
        needs_tools=True,
        needs_task_graph=False,
        approval_required=approval_required,
        reasons=[],
    )


def fleet():
    return FleetPlacementResolution(
        observer_only=True,
        dispatch_allowed=False,
        model_placement=PlacementShape.PREFERRED_NODE,
        model_placement_confidence=0.8,
        resolved_placement=PlacementShape.PREFERRED_NODE,
        eligible_node_ids=["x1-370", "xwing"],
        ranked_node_ids=["x1-370", "xwing"],
        selected_node_id="x1-370",
        reasons=["eligible rank only"],
    )


def test_profile_is_advisory_only_even_for_act_with_approval():
    profile = build_hermes_system_one_profile(
        decision(
            ResolvedDisposition.ACT_WITH_APPROVAL,
            approval_required=True,
        ),
        receipt_id="r-1",
        observed_at=datetime(2026, 9, 23, 22, 30, tzinfo=UTC),
    )

    assert profile.profile == HERMES_SYSTEM_ONE_PROFILE
    assert profile.uhp_version == UHP_VERSION
    assert profile.advice.mode == "act"
    assert profile.authority.model_dump() == {
        "dispatch_allowed": False,
        "approval_granted": False,
        "claim_acquired": False,
        "mutation_allowed": False,
        "routing_authority_changed": False,
    }


def test_fleet_wire_contains_only_opaque_handles_not_node_ids():
    profile = build_hermes_system_one_profile(
        decision(),
        receipt_id="r-2",
        observed_at=datetime(2026, 9, 23, 22, 30, tzinfo=UTC),
        fleet_resolution=fleet(),
        fleet_handle_by_node_id={
            "x1-370": "eligible:opaque:alpha",
            "xwing": "eligible:opaque:beta",
        },
    )

    wire = canonical_json(profile)
    assert "x1-370" not in wire
    assert "xwing" not in wire
    assert "eligible:opaque:alpha" in wire
    assert "eligible:opaque:beta" in wire
    assert [item.score for item in profile.advice.fleet_priority] == [0.8, 0.4]


def test_fleet_resolution_cannot_widen_dispatch_authority():
    bad = fleet().model_copy(update={"dispatch_allowed": True})

    with pytest.raises(ValueError, match="observer-only"):
        build_hermes_system_one_profile(
            decision(),
            receipt_id="r-3",
            fleet_resolution=bad,
            fleet_handle_by_node_id={
                "x1-370": "eligible:opaque:alpha",
                "xwing": "eligible:opaque:beta",
            },
        )


def test_every_ranked_node_requires_an_opaque_handle():
    with pytest.raises(ValueError, match="every ranked node"):
        build_hermes_system_one_profile(
            decision(),
            receipt_id="r-4",
            fleet_resolution=fleet(),
            fleet_handle_by_node_id={
                "x1-370": "eligible:opaque:alpha",
            },
        )


def test_ttl_is_bounded_to_local_studio_default_acceptance_window():
    with pytest.raises(ValueError, match="1..900"):
        build_hermes_system_one_profile(
            decision(),
            receipt_id="r-5",
            ttl_seconds=901,
        )


def test_uhp_fixture_carries_profile_and_no_fallback_fields():
    observed = datetime(2026, 9, 23, 22, 30, tzinfo=UTC)
    profile = build_hermes_system_one_profile(
        decision(ResolvedDisposition.CHAT),
        receipt_id="r-6",
        observed_at=observed,
        task_focus="Answer from the bounded project state.",
        context_priority=["current-pr", "latest-handoff"],
        provenance=SystemOneProvenance(
            model_revision="model-sha",
            knowledge_revision="knowledge-sha",
            fleet_projection_checksum="fleet-sha",
        ),
    )
    response = build_uhp_response_fixture(
        profile,
        response_id="resp_probe1",
        session_id="hsess-probe1",
        harness_id="chrn_system_one",
        model="recorded/jev",
        created_at=observed,
    )

    assert response["status"] == "completed"
    assert response["model"] == "recorded/jev"
    assert response["metadata"]["session_id"] == "hsess-probe1"
    assert response["metadata"]["harness_id"] == "chrn_system_one"
    assert response["metadata"]["hermes_system_one"]["receipt_id"] == "r-6"
    assert "requested_model" not in response["metadata"]
    assert "model_fallback" not in response["metadata"]


def test_canonical_hash_is_stable_across_mapping_order():
    first = {"b": 2, "a": 1}
    second = {"a": 1, "b": 2}

    assert canonical_json(first) == canonical_json(second)
    assert canonical_sha256(first) == canonical_sha256(second)
