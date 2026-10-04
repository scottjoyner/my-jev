import json
from datetime import UTC, datetime, timedelta

import pytest

from my_jev.fleet_benchmark_qualification import (
    ADVISORY_SCHEMA_VERSION,
    BenchmarkLaneEvidence,
    BenchmarkRole,
    BenchmarkWorkIntent,
    FleetBenchmarkMatrix,
    FleetBenchmarkAdvisory,
    QualificationThresholds,
    build_fleet_benchmark_advisory,
)
from my_jev.fleet_benchmark_qualification_cli import main
from my_jev.fleet_policy import (
    FleetNodeSnapshot,
    FleetPlacementState,
    PlacementShape,
    eligible_nodes,
)
from my_jev.uhp_advisory import SystemOneAuthority

NOW = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)
HANDLES = {
    "gpu-01.internal.lan": "lane.alpha",
    "gpu-02.internal.lan": "lane.beta",
    "gpu-03.internal.lan": "lane.gamma",
    "cpu-01.internal.lan": "lane.delta",
}


def _node(node_id: str, **kwargs) -> FleetNodeSnapshot:
    kwargs.setdefault("capabilities", ["gpu"])
    kwargs.setdefault("ram_free_gib", 64)
    kwargs.setdefault("vram_free_gib", 48)
    return FleetNodeSnapshot(node_id=node_id, **kwargs)


def _state(*nodes: FleetNodeSnapshot, **kwargs) -> FleetPlacementState:
    kwargs.setdefault("workload_id", "campaign-workload")
    kwargs.setdefault("workload_type", "gpu_coding")
    kwargs.setdefault("required_capabilities", ["gpu"])
    kwargs.setdefault("estimated_ram_gib", 16)
    kwargs.setdefault("estimated_vram_gib", 24)
    kwargs.setdefault("checkpoint_supported", True)
    if "nodes" not in kwargs:
        kwargs["nodes"] = list(nodes)
    return FleetPlacementState(**kwargs)


def _lane(node_id: str, **kwargs) -> BenchmarkLaneEvidence:
    kwargs.setdefault("code_qualified", True)
    kwargs.setdefault("measured_task_family", "code_patch_suite")
    kwargs.setdefault("quality_confidence", 0.9)
    kwargs.setdefault("latency_seconds", 0.25)
    kwargs.setdefault("throughput_tps", 48.0)
    kwargs.setdefault("resource_pressure", 0.1)
    kwargs.setdefault("health_freshness_seconds", 5.0)
    kwargs.setdefault("observed_at", NOW)
    return BenchmarkLaneEvidence(node_id=node_id, **kwargs)


def _matrix(*lanes: BenchmarkLaneEvidence) -> FleetBenchmarkMatrix:
    return FleetBenchmarkMatrix(campaign_id="campaign-2026-10-02", lanes=list(lanes))


def _fleet() -> FleetPlacementState:
    return _state(
        _node("gpu-01.internal.lan"),
        _node("gpu-02.internal.lan"),
        _node("cpu-01.internal.lan", capabilities=["cpu"], vram_free_gib=0),
    )


def _advisory(matrix: FleetBenchmarkMatrix, state: FleetPlacementState | None = None, **kwargs):
    handles = kwargs.pop("handle_by_node_id", HANDLES)
    return build_fleet_benchmark_advisory(
        matrix,
        state if state is not None else _fleet(),
        observed_at=NOW,
        handle_by_node_id=handles,
        **kwargs,
    )


def _system_one_authority_false() -> dict[str, bool]:
    return SystemOneAuthority().model_dump()


def test_no_code_qualified_lane_defers_coding_work():
    matrix = _matrix(
        _lane(
            "gpu-01.internal.lan",
            code_qualified=False,
            review_qualified=True,
            measured_task_family="review_suite",
        ),
        _lane(
            "gpu-02.internal.lan",
            code_qualified=False,
            summary_only=True,
            measured_task_family="summary_suite",
        ),
    )
    advisory = _advisory(matrix)

    assert advisory.execution_shape == PlacementShape.DEFER
    assert advisory.role_assignment is None
    assert advisory.preferred == []
    assert advisory.selected_node_id is None
    assert any("no code-qualified lane" in reason for reason in advisory.reasons)
    # The CPU node has benchmark-quality evidence but fails the hard capability filter.
    assert advisory.ignored_ineligible_lane_count == 0


def test_scout_only_lane_defers_coding_but_recommends_decomposition():
    matrix = _matrix(
        _lane(
            "gpu-01.internal.lan",
            code_qualified=False,
            scout_qualified=True,
            measured_task_family="scout_suite",
        ),
    )
    advisory = _advisory(matrix)

    assert advisory.execution_shape == PlacementShape.DEFER
    assert advisory.role_assignment is None
    assert advisory.decomposition_fallback == BenchmarkRole.SCOUT
    assert advisory.selected_node_id is None
    assert any("decomposition/scout" in reason for reason in advisory.reasons)
    assert any("never an implementation lane" in reason for reason in advisory.reasons)


def test_scout_intent_recommends_decomposition_not_implementation():
    matrix = _matrix(
        _lane(
            "gpu-01.internal.lan",
            code_qualified=False,
            scout_qualified=True,
            measured_task_family="scout_suite",
        ),
    )
    advisory = _advisory(matrix, work_intent=BenchmarkWorkIntent.SCOUT)

    assert advisory.execution_shape == PlacementShape.PREFERRED_NODE
    assert advisory.role_assignment == BenchmarkRole.SCOUT
    assert advisory.preferred[0].handle == "lane.alpha"
    assert any("implementation is not recommended" in reason for reason in advisory.reasons)


def test_code_and_review_lanes_advise_split():
    matrix = _matrix(
        _lane("gpu-01.internal.lan"),
        _lane(
            "gpu-02.internal.lan",
            code_qualified=False,
            review_qualified=True,
            measured_task_family="review_suite",
            quality_confidence=0.8,
        ),
    )
    advisory = _advisory(matrix)

    assert advisory.execution_shape == PlacementShape.SPLIT
    assert advisory.role_assignment == BenchmarkRole.CODE
    assert [item.role for item in advisory.preferred] == [
        BenchmarkRole.CODE,
        BenchmarkRole.REVIEW,
    ]
    assert [item.handle for item in advisory.preferred] == ["lane.alpha", "lane.beta"]
    # Split never names a node: physical selection stays with the resolver.
    assert advisory.selected_node_id is None


def test_split_is_not_advised_without_checkpoint_support():
    matrix = _matrix(
        _lane("gpu-01.internal.lan"),
        _lane(
            "gpu-02.internal.lan",
            code_qualified=False,
            review_qualified=True,
            measured_task_family="review_suite",
        ),
    )
    advisory = _advisory(matrix, _fleet().model_copy(update={"checkpoint_supported": False}))

    assert advisory.execution_shape == PlacementShape.PREFERRED_NODE
    assert advisory.role_assignment == BenchmarkRole.CODE
    assert [item.role for item in advisory.preferred] == [BenchmarkRole.CODE]
    assert any("checkpoint/restart" in reason for reason in advisory.reasons)


def test_same_fleet_coding_quality_removed_changes_coding_decision():
    qualified = _advisory(_matrix(_lane("gpu-01.internal.lan")))
    assert qualified.execution_shape == PlacementShape.PREFERRED_NODE
    assert qualified.role_assignment == BenchmarkRole.CODE
    assert qualified.selected_node_id == "gpu-01.internal.lan"

    degraded = _advisory(
        _matrix(
            _lane(
                "gpu-01.internal.lan",
                code_qualified=False,
                scout_qualified=True,
                measured_task_family="scout_suite",
            ),
        ),
    )
    assert degraded.execution_shape == PlacementShape.DEFER
    assert degraded.role_assignment is None
    assert degraded.selected_node_id is None
    assert qualified.semantic_decision() != degraded.semantic_decision()


def test_same_fleet_node_names_permuted_keeps_decision_stable():
    matrix = _matrix(
        _lane("gpu-01.internal.lan"),
        _lane(
            "gpu-02.internal.lan",
            code_qualified=False,
            review_qualified=True,
            measured_task_family="review_suite",
            quality_confidence=0.8,
        ),
    )
    forward = _advisory(matrix)
    reversed_order = _advisory(
        _matrix(*reversed(matrix.lanes)),
        _fleet().model_copy(update={"nodes": list(reversed(_fleet().nodes))}),
    )

    assert forward.model_dump(mode="json") == reversed_order.model_dump(mode="json")
    assert forward.as_model_state() == reversed_order.as_model_state()

    renamed_matrix = _matrix(
        _lane("alpha-17.internal.lan"),
        _lane(
            "beta-21.internal.lan",
            code_qualified=False,
            review_qualified=True,
            measured_task_family="review_suite",
            quality_confidence=0.8,
        ),
    )
    renamed_state = _state(
        _node("alpha-17.internal.lan"),
        _node("beta-21.internal.lan"),
        _node("cpu-01.internal.lan", capabilities=["cpu"], vram_free_gib=0),
    )
    renamed = _advisory(
        renamed_matrix,
        renamed_state,
        handle_by_node_id={
            "alpha-17.internal.lan": "lane.alpha",
            "beta-21.internal.lan": "lane.beta",
            "cpu-01.internal.lan": "lane.delta",
        },
    )

    assert renamed.semantic_decision() == forward.semantic_decision()
    for node_id in HANDLES:
        assert node_id not in renamed.as_model_state()


def test_high_throughput_with_failed_quality_cannot_become_code_lane():
    matrix = _matrix(
        _lane("gpu-01.internal.lan"),
        _lane(
            "gpu-02.internal.lan",
            review_qualified=True,
            measured_task_family="code_patch_suite",
            quality_confidence=0.4,
            latency_seconds=0.01,
            throughput_tps=4000.0,
            resource_pressure=0.0,
            health_freshness_seconds=0.0,
        ),
    )
    advisory = _advisory(matrix)

    assert advisory.ignored_low_confidence_lane_count == 1
    assert [item.handle for item in advisory.preferred] == ["lane.alpha"]
    assert advisory.selected_node_id == "gpu-01.internal.lan"
    assert all(item.score <= 1.0 for item in advisory.preferred)


def test_unqualified_high_throughput_lane_is_never_preferred_for_code():
    matrix = _matrix(
        _lane("gpu-01.internal.lan"),
        _lane(
            "gpu-02.internal.lan",
            code_qualified=False,
            scout_qualified=True,
            measured_task_family="code_patch_suite",
            quality_confidence=1.0,
            latency_seconds=0.01,
            throughput_tps=4000.0,
            resource_pressure=0.0,
            health_freshness_seconds=0.0,
        ),
    )
    advisory = _advisory(matrix)

    assert advisory.ignored_low_confidence_lane_count == 0
    assert [item.handle for item in advisory.preferred] == ["lane.alpha"]
    assert advisory.selected_node_id == "gpu-01.internal.lan"
    # The unqualified lane still measures near-maximally for its own role...
    scout_advisory = _advisory(matrix, work_intent=BenchmarkWorkIntent.SCOUT)
    assert scout_advisory.role_assignment == BenchmarkRole.SCOUT
    assert scout_advisory.preferred[0].score > 0.9
    # ...but throughput never buys the code role.
    assert all(item.role is BenchmarkRole.CODE for item in advisory.preferred)


def test_stale_benchmark_evidence_cannot_remain_preferred():
    fresh = _matrix(
        _lane("gpu-01.internal.lan"),
        _lane(
            "gpu-02.internal.lan",
            code_qualified=False,
            review_qualified=True,
            measured_task_family="review_suite",
            quality_confidence=0.8,
        ),
    )
    assert _advisory(fresh).execution_shape == PlacementShape.SPLIT

    stale = _matrix(
        _lane(
            "gpu-01.internal.lan",
            observed_at=NOW - timedelta(seconds=4000),
        ),
        _lane(
            "gpu-02.internal.lan",
            code_qualified=False,
            review_qualified=True,
            measured_task_family="review_suite",
            quality_confidence=0.8,
        ),
    )
    advisory = _advisory(stale)

    assert advisory.ignored_stale_lane_count == 1
    assert advisory.execution_shape == PlacementShape.DEFER
    assert advisory.role_assignment is None
    assert advisory.preferred == []
    assert advisory.selected_node_id is None
    assert any("TTL" in reason for reason in advisory.reasons)


def test_health_stale_lane_is_ignored_for_role():
    matrix = _matrix(
        _lane("gpu-01.internal.lan"),
        _lane(
            "gpu-02.internal.lan",
            code_qualified=False,
            review_qualified=True,
            measured_task_family="review_suite",
            health_freshness_seconds=5000.0,
        ),
    )
    advisory = _advisory(matrix)

    assert advisory.ignored_health_stale_lane_count == 1
    assert advisory.execution_shape == PlacementShape.PREFERRED_NODE
    assert [item.handle for item in advisory.preferred] == ["lane.alpha"]


def test_review_qualification_added_emerges_split_and_reviewer_preference():
    code_only = _matrix(_lane("gpu-01.internal.lan"))
    assert _advisory(code_only).execution_shape == PlacementShape.PREFERRED_NODE
    assert _advisory(code_only).selected_node_id == "gpu-01.internal.lan"

    with_reviewer = _matrix(
        _lane("gpu-01.internal.lan"),
        _lane(
            "gpu-02.internal.lan",
            code_qualified=False,
            review_qualified=True,
            measured_task_family="review_suite",
            quality_confidence=0.75,
        ),
    )
    advisory = _advisory(with_reviewer)

    assert advisory.execution_shape == PlacementShape.SPLIT
    assert advisory.selected_node_id is None
    reviewer = [item for item in advisory.preferred if item.role is BenchmarkRole.REVIEW]
    assert len(reviewer) == 1
    assert reviewer[0].handle == "lane.beta"
    assert reviewer[0].score > 0.0


def test_drained_node_deterministic_exclusion_wins_over_benchmark_evidence():
    state = _fleet()
    drained = _state(
        _node("gpu-01.internal.lan", drained=True),
        _node("gpu-02.internal.lan"),
        _node("cpu-01.internal.lan", capabilities=["cpu"], vram_free_gib=0),
    )
    matrix = _matrix(
        _lane("gpu-01.internal.lan"),
        _lane(
            "gpu-02.internal.lan",
            code_qualified=False,
            review_qualified=True,
            measured_task_family="review_suite",
        ),
    )
    advisory = _advisory(matrix, drained)

    assert [node.node_id for node in eligible_nodes(state)] == [
        "gpu-01.internal.lan",
        "gpu-02.internal.lan",
    ]
    assert "gpu-01.internal.lan" not in [
        node.node_id for node in eligible_nodes(drained)
    ]
    assert advisory.ignored_ineligible_lane_count == 1
    # The drained node's code lane is discarded outright; the review lane cannot
    # silently substitute for coding work.
    assert advisory.execution_shape == PlacementShape.DEFER
    assert advisory.role_assignment is None
    assert advisory.preferred == []
    assert advisory.selected_node_id is None
    assert any("outside the authoritative eligible set" in reason for reason in advisory.reasons)

    review_advisory = _advisory(matrix, drained, work_intent=BenchmarkWorkIntent.REVIEW)
    assert review_advisory.execution_shape == PlacementShape.PREFERRED_NODE
    assert [item.handle for item in review_advisory.preferred] == ["lane.beta"]
    assert review_advisory.selected_node_id == "gpu-02.internal.lan"


def test_unhealthy_node_deterministic_exclusion_wins():
    unhealthy = _state(
        _node("gpu-01.internal.lan", healthy=False),
        _node("gpu-02.internal.lan"),
        _node("cpu-01.internal.lan", capabilities=["cpu"], vram_free_gib=0),
    )
    advisory = _advisory(_matrix(_lane("gpu-01.internal.lan")), unhealthy)

    assert advisory.ignored_ineligible_lane_count == 1
    assert advisory.execution_shape == PlacementShape.DEFER
    assert advisory.selected_node_id is None


def test_stale_health_freshness_flag_deterministic_exclusion_wins():
    stale_health = _state(
        _node("gpu-01.internal.lan", health_fresh=False),
        _node("gpu-02.internal.lan"),
        _node("cpu-01.internal.lan", capabilities=["cpu"], vram_free_gib=0),
    )
    advisory = _advisory(_matrix(_lane("gpu-01.internal.lan")), stale_health)

    assert advisory.execution_shape == PlacementShape.DEFER
    assert advisory.selected_node_id is None


def test_maximal_score_on_ineligible_node_never_widens_eligibility():
    drained = _state(
        _node("gpu-01.internal.lan", drained=True),
        _node("gpu-02.internal.lan"),
        _node("cpu-01.internal.lan", capabilities=["cpu"], vram_free_gib=0),
    )
    maximal = _lane(
        "gpu-01.internal.lan",
        code_qualified=True,
        review_qualified=True,
        scout_qualified=True,
        measured_task_family="code_patch_suite",
        quality_confidence=1.0,
        latency_seconds=0.0,
        throughput_tps=10_000.0,
        resource_pressure=0.0,
        health_freshness_seconds=0.0,
        observed_at=NOW,
    )
    only_ineligible = _matrix(maximal)
    advisory = _advisory(only_ineligible, drained)

    assert advisory.execution_shape == PlacementShape.DEFER
    assert advisory.role_assignment is None
    assert advisory.preferred == []
    assert advisory.selected_node_id is None
    assert advisory.eligible_node_count == 1
    assert "gpu-01.internal.lan" not in advisory.as_model_state()
    assert any("outside the authoritative eligible set" in reason for reason in advisory.reasons)

    # The same evidence is a maximal 1.0 score when it lands on an eligible node,
    # which is exactly why it must not be able to leak onto the drained one.
    eligible_maximal = _advisory(
        _matrix(
            maximal.model_copy(update={"node_id": "gpu-02.internal.lan"}),
            _lane("gpu-01.internal.lan"),
        ),
        _fleet(),
    )
    assert eligible_maximal.preferred[0].handle == "lane.beta"
    assert eligible_maximal.preferred[0].score == 1.0


def test_no_eligible_node_is_defer_even_with_maximal_evidence():
    drained = _state(
        _node("gpu-01.internal.lan", drained=True),
        _node("gpu-02.internal.lan", reachable=False),
    )
    advisory = _advisory(_matrix(_lane("gpu-01.internal.lan")), drained)

    assert eligible_nodes(drained) == []
    assert advisory.eligible_node_count == 0
    assert advisory.execution_shape == PlacementShape.DEFER
    assert advisory.preferred == []
    assert advisory.selected_node_id is None
    assert any("cannot widen eligibility" in reason for reason in advisory.reasons)


def test_maximal_score_does_not_outrank_an_eligible_lower_score_lane():
    saturated = _state(
        _node("gpu-01.internal.lan", drained=True),
        _node("gpu-02.internal.lan"),
        _node("cpu-01.internal.lan", capabilities=["cpu"], vram_free_gib=0),
    )
    matrix = _matrix(
        _lane("gpu-01.internal.lan", quality_confidence=1.0, latency_seconds=0.0),
        _lane("gpu-02.internal.lan"),
    )
    advisory = _advisory(matrix, saturated)

    assert advisory.preferred[0].handle == "lane.beta"
    assert advisory.preferred[0].score < 1.0
    assert advisory.selected_node_id == "gpu-02.internal.lan"


def test_pressure_reorders_but_never_excludes_or_hard_inelibilizes():
    pressured = _matrix(
        _lane("gpu-01.internal.lan", resource_pressure=0.99),
        _lane("gpu-02.internal.lan", resource_pressure=0.05),
    )
    advisory = _advisory(pressured)

    assert advisory.execution_shape == PlacementShape.PREFERRED_NODE
    assert {item.handle for item in advisory.preferred} == {"lane.alpha", "lane.beta"}
    assert advisory.preferred[0].handle == "lane.beta"
    assert advisory.preferred[0].score > advisory.preferred[1].score
    assert advisory.selected_node_id == "gpu-02.internal.lan"


def test_unknown_telemetry_is_not_preferred_over_measured_evidence():
    matrix = _matrix(
        _lane("gpu-01.internal.lan", latency_seconds=None, throughput_tps=None),
        _lane(
            "gpu-02.internal.lan",
            quality_confidence=0.7,
            resource_pressure=0.8,
        ),
    )
    advisory = _advisory(matrix)

    assert advisory.preferred[0].handle == "lane.beta"
    assert advisory.preferred[1].score == 0.0


def test_authority_block_is_all_false_and_unchanged_by_any_score():
    advisory = _advisory(_matrix(_lane("gpu-01.internal.lan")))

    assert advisory.authority.model_dump() == _system_one_authority_false()
    assert advisory.authority.model_dump() == {
        "dispatch_allowed": False,
        "approval_granted": False,
        "claim_acquired": False,
        "mutation_allowed": False,
        "routing_authority_changed": False,
    }
    assert advisory.observer_only is True
    assert advisory.dispatch_allowed is False
    assert advisory.physical_node_selection == "deterministic_resolver"
    with pytest.raises(ValueError):
        SystemOneAuthority(dispatch_allowed=True)
    with pytest.raises(ValueError):
        FleetBenchmarkAdvisory(
            **{
                **advisory.model_dump(mode="json"),
                "dispatch_allowed": True,
            }
        )
    with pytest.raises(ValueError):
        FleetBenchmarkAdvisory(
            **{
                **advisory.model_dump(mode="json"),
                "observer_only": False,
            }
        )


def test_wire_contains_no_node_ids_or_hostnames():
    matrix = _matrix(
        _lane("gpu-01.internal.lan"),
        _lane(
            "gpu-02.internal.lan",
            code_qualified=False,
            review_qualified=True,
            measured_task_family="review_suite",
        ),
    )
    advisory = _advisory(matrix)
    wire = advisory.as_model_state()

    for node_id in HANDLES:
        assert node_id not in wire
        assert "gpu-0" not in wire
        assert "cpu-01" not in wire
        assert "internal.lan" not in wire
    assert "lane.alpha" in wire
    # No identity-bearing field survives onto the wire.
    wire_dict = advisory.model_wire()
    assert {
        "selected_node_id",
        "eligible_node_ids",
        "node_id",
        "node_ids",
        "handle_by_node_id",
    } & set(wire_dict) == set()


def test_preferences_require_caller_supplied_opaque_handles():
    matrix = _matrix(_lane("gpu-01.internal.lan"))

    with pytest.raises(ValueError, match="caller-supplied opaque handle"):
        _advisory(matrix, handle_by_node_id=None)
    with pytest.raises(ValueError, match="caller-supplied opaque handle"):
        _advisory(matrix, handle_by_node_id={"gpu-02.internal.lan": "lane.beta"})
    with pytest.raises(ValueError, match="caller-supplied opaque handle"):
        _advisory(matrix, handle_by_node_id={"gpu-01.internal.lan": "lane alpha/1"})


def test_preferences_reject_duplicate_opaque_handles():
    matrix = _matrix(
        _lane("gpu-01.internal.lan"),
        _lane("gpu-02.internal.lan", quality_confidence=0.8),
    )
    with pytest.raises(ValueError, match="duplicate opaque fleet handle"):
        _advisory(
            matrix,
            handle_by_node_id={
                "gpu-01.internal.lan": "lane.same",
                "gpu-02.internal.lan": "lane.same",
            },
        )


def test_lane_contract_rejects_naive_timestamp_and_unclaimed_lane():
    with pytest.raises(ValueError, match="timezone-aware"):
        _lane("gpu-01.internal.lan", observed_at=datetime(2026, 10, 2, 12, 0))
    with pytest.raises(ValueError, match="at least one qualification"):
        _lane("gpu-01.internal.lan", code_qualified=False)


def test_matrix_contract_rejects_duplicate_lane_observations():
    with pytest.raises(ValueError, match="unique per"):
        _matrix(_lane("gpu-01.internal.lan"), _lane("gpu-01.internal.lan"))


def test_naive_advisory_observation_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        build_fleet_benchmark_advisory(
            _matrix(_lane("gpu-01.internal.lan")),
            _fleet(),
            observed_at=datetime(2026, 10, 2, 12, 0),
            handle_by_node_id=HANDLES,
        )


def test_ttl_threshold_must_be_positive():
    with pytest.raises(ValueError):
        QualificationThresholds(max_evidence_age_seconds=0)


def test_summary_only_lane_never_certifies_implementation_even_with_code_flag():
    matrix = _matrix(
        _lane("gpu-01.internal.lan", code_qualified=True, summary_only=True),
    )
    advisory = _advisory(matrix)

    assert advisory.execution_shape == PlacementShape.DEFER
    assert advisory.role_assignment is None

    summary_advisory = _advisory(matrix, work_intent=BenchmarkWorkIntent.SUMMARY)
    assert summary_advisory.role_assignment == BenchmarkRole.SUMMARY
    assert summary_advisory.execution_shape == PlacementShape.PREFERRED_NODE


def test_advisory_is_deterministic_across_repeated_builds():
    matrix = _matrix(
        _lane("gpu-01.internal.lan"),
        _lane(
            "gpu-02.internal.lan",
            code_qualified=False,
            review_qualified=True,
            measured_task_family="review_suite",
            quality_confidence=0.8,
        ),
    )
    first = _advisory(matrix)
    second = _advisory(matrix)

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
    assert first.schema_version == ADVISORY_SCHEMA_VERSION


def test_advisory_serializes_with_schema_version_and_reason_list():
    advisory = _advisory(_matrix(_lane("gpu-01.internal.lan")))
    payload = advisory.model_dump(mode="json")

    assert payload["schema_version"] == ADVISORY_SCHEMA_VERSION
    assert payload["contract"] == "fleet-benchmark-qualification-v1"
    assert isinstance(payload["reasons"], list)
    assert payload["reasons"]
    assert advisory.observed_at == "2026-10-02T12:00:00Z"
    assert advisory.expires_at == "2026-10-02T12:15:00Z"
    assert isinstance(advisory, FleetBenchmarkAdvisory)


def _write_matrix(path, lanes) -> None:
    payload = {
        "campaign_id": "campaign-2026-10-02",
        "lanes": [json.loads(lane.model_dump_json()) for lane in lanes],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cli_emits_deterministic_identity_free_advisory(tmp_path, capsys):
    matrix_path = tmp_path / "matrix.json"
    state_path = tmp_path / "state.json"
    handle_path = tmp_path / "handles.json"
    _write_matrix(
        matrix_path,
        [
            _lane("gpu-01.internal.lan"),
            _lane(
                "gpu-02.internal.lan",
                code_qualified=False,
                review_qualified=True,
                measured_task_family="review_suite",
            ),
        ],
    )
    state_path.write_text(_fleet().model_dump_json(), encoding="utf-8")
    handle_path.write_text(json.dumps(HANDLES), encoding="utf-8")
    argv = [
        "--matrix",
        str(matrix_path),
        "--state",
        str(state_path),
        "--handle-map",
        str(handle_path),
        "--observed-at",
        "2026-10-02T12:00:00Z",
    ]

    assert main(argv) == 0
    first = capsys.readouterr()
    assert main(argv) == 0
    second = capsys.readouterr()

    assert first.out == second.out
    assert first.err == second.err
    payload = json.loads(first.out)
    assert payload["execution_shape"] == "split"
    assert payload["role_assignment"] == "code"
    assert payload["physical_node_selection"] == "deterministic_resolver"
    assert payload["authority"] == _system_one_authority_false()
    assert "selected_node_id" not in payload
    for node_id in HANDLES:
        assert node_id not in first.out
    summary = json.loads(first.err)
    assert summary["evidence_only"] is True
    assert summary["runtime_authority_changed"] is False
    assert summary["assistx_accessed"] is False
    assert summary["execution_shape"] == "split"
    assert len(summary["model_state_sha256"]) == 64


def test_cli_writes_output_file_without_node_identities(tmp_path, capsys):
    matrix_path = tmp_path / "matrix.json"
    state_path = tmp_path / "state.json"
    handle_path = tmp_path / "handles.json"
    output_path = tmp_path / "out" / "advisory.json"
    _write_matrix(matrix_path, [_lane("gpu-01.internal.lan")])
    state_path.write_text(_fleet().model_dump_json(), encoding="utf-8")
    handle_path.write_text(json.dumps(HANDLES), encoding="utf-8")

    assert main(
        [
            "--matrix",
            str(matrix_path),
            "--state",
            str(state_path),
            "--handle-map",
            str(handle_path),
            "--observed-at",
            "2026-10-02T12:00:00Z",
            "--output",
            str(output_path),
        ]
    ) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    rendered = output_path.read_text(encoding="utf-8")
    assert "gpu-01.internal.lan" not in rendered
    assert "internal.lan" not in rendered
    assert json.loads(rendered)["execution_shape"] == "preferred_node"
    assert json.loads(captured.err)["output"] == str(output_path)


def test_cli_returns_two_on_expected_failure(tmp_path, capsys):
    matrix_path = tmp_path / "matrix.json"
    state_path = tmp_path / "state.json"
    handle_path = tmp_path / "handles.json"
    _write_matrix(matrix_path, [_lane("gpu-01.internal.lan")])
    state_path.write_text(_fleet().model_dump_json(), encoding="utf-8")
    handle_path.write_text(json.dumps({"gpu-01.internal.lan": "not a handle"}), encoding="utf-8")

    assert main(
        [
            "--matrix",
            str(matrix_path),
            "--state",
            str(state_path),
            "--handle-map",
            str(handle_path),
            "--observed-at",
            "2026-10-02T12:00:00Z",
        ]
    ) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "caller-supplied opaque handle" in captured.err


def test_cli_returns_two_on_missing_matrix(tmp_path, capsys):
    assert main(
        [
            "--matrix",
            str(tmp_path / "absent.json"),
            "--state",
            str(tmp_path / "absent-state.json"),
            "--handle-map",
            str(tmp_path / "absent-handles.json"),
        ]
    ) == 2
    assert "my-jev-fleet-benchmark-advisory:" in capsys.readouterr().err

def test_reasons_cannot_carry_a_hostname_into_the_model_wire():
    """``reasons`` is operator-facing free text; identity must not ride along.

    Mutation-tested: appending the selected node id to the reason list used to
    reach the decision model unnoticed.
    """

    advisory = _advisory(_matrix(_lane("gpu-01.internal.lan")))
    advisory.reasons.append("preferred node is gpu-01.internal.lan")

    with pytest.raises(ValueError, match="leaked a node identity"):
        advisory.model_wire()
    with pytest.raises(ValueError, match="leaked a node identity"):
        advisory.semantic_decision()


def test_identity_free_reasons_still_pass_the_guard():
    advisory = _advisory(_matrix(_lane("gpu-01.internal.lan")))

    assert advisory.model_wire()["reasons"] == advisory.reasons
    assert advisory.semantic_decision()["reasons"] == advisory.reasons


def test_opaque_handles_may_contain_the_node_slug_without_tripping_the_guard():
    """``eligible:opaque:<slug>`` is the intended surrogate, not a leak.

    A guard that substring-matched handles would reject every legitimate
    handle the fleet convention produces.
    """

    advisory = _advisory(
        _matrix(_lane("gpu-01.internal.lan")),
        handle_by_node_id={"gpu-01.internal.lan": "eligible:opaque:gpu-01"},
    )

    # Does not raise: the handle is the deliberate indirection.
    wire = advisory.model_wire()
    assert wire["preferred"][0]["handle"] == "eligible:opaque:gpu-01"
    # ...but free text still cannot carry the identity.
    advisory.reasons.append("gpu-01.internal.lan is preferred")
    with pytest.raises(ValueError, match="leaked a node identity"):
        advisory.model_wire()


def _matrix_with_ages(ages, *, confidence=0.9, health_age=10.0):
    """One lane per age in hours, all otherwise qualified."""
    from my_jev.fleet_benchmark_bridge import BenchmarkLaneEvidence

    observed = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)
    lanes = [
        BenchmarkLaneEvidence(
            node_id=f"n{index}",
            code_qualified=True,
            review_qualified=False,
            scout_qualified=False,
            summary_only=False,
            measured_task_family="coding",
            quality_confidence=confidence,
            resource_pressure=0.2,
            health_freshness_seconds=health_age,
            observed_at=observed - timedelta(hours=hours),
        )
        for index, hours in enumerate(ages)
    ]
    return FleetBenchmarkMatrix(campaign_id="c1", lanes=lanes)


def _eligible_state(node_ids):
    from my_jev.fleet_policy import FleetNodeSnapshot

    return _state(
        *[
            FleetNodeSnapshot(node_id=node_id, capabilities=["gpu"], healthy=True)
            for node_id in node_ids
        ]
    )


def test_future_dated_evidence_is_reported_separately_from_stale():
    """Clock skew and staleness are different faults with different fixes.

    An operator told "stale" waits for a retest, which cannot help when the
    reporter's clock is wrong.
    """
    now = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)
    # 26h old is past the 24h TTL; -3h is dated after the evaluation moment.
    matrix = _matrix_with_ages([26.0, -3.0])
    advisory = build_fleet_benchmark_advisory(
        matrix,
        _eligible_state(["n0", "n1"]),
        work_intent=BenchmarkWorkIntent.CODING,
        observed_at=now,
        handle_by_node_id={"n0": "eligible:opaque:a", "n1": "eligible:opaque:b"},
    )

    assert advisory.ignored_stale_lane_count == 1
    assert advisory.ignored_future_dated_lane_count == 1
    counts = advisory.semantic_decision()["ignored_lane_counts"]
    assert counts["stale"] == 1
    assert counts["future_dated"] == 1
    assert any("clock skew" in reason for reason in advisory.reasons)
    assert not any(
        "future-dated evidence is not fresh" in reason for reason in advisory.reasons
    )


def test_future_dated_lane_never_becomes_preferred():
    now = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)
    matrix = _matrix_with_ages([-3.0])
    advisory = build_fleet_benchmark_advisory(
        matrix,
        _eligible_state(["n0"]),
        work_intent=BenchmarkWorkIntent.CODING,
        observed_at=now,
        handle_by_node_id={"n0": "eligible:opaque:a"},
    )

    assert advisory.ignored_future_dated_lane_count == 1
    assert advisory.ignored_stale_lane_count == 0
    assert advisory.preferred == []
    assert advisory.execution_shape is PlacementShape.DEFER


def test_future_dated_count_reaches_the_operator_surface():
    from my_jev.fleet_benchmark_projection import benchmark_qualification_for

    now = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)
    advisory = build_fleet_benchmark_advisory(
        _matrix_with_ages([26.0, -3.0]),
        _eligible_state(["n0", "n1"]),
        work_intent=BenchmarkWorkIntent.CODING,
        observed_at=now,
        handle_by_node_id={"n0": "eligible:opaque:a", "n1": "eligible:opaque:b"},
    )

    qualification = benchmark_qualification_for(advisory, None)
    assert qualification.rejected_evidence["stale"] == 1
    assert qualification.rejected_evidence["future_dated"] == 1


def test_a_handle_that_is_the_node_id_is_refused():
    """The pattern check alone cannot catch identity passed through.

    `_OPAQUE_HANDLE` accepts `gpu-01.internal.lan`, and `_reject_identity`
    deliberately skips handle values because a real surrogate often embeds the
    node's slug. So a caller mapping a node to itself produced an advisory whose
    `preferred_handles` named the host.
    """
    from datetime import UTC, datetime, timedelta

    from my_jev.fleet_benchmark_bridge import BenchmarkLaneEvidence
    from my_jev.fleet_policy import FleetNodeSnapshot

    now = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)
    lane = BenchmarkLaneEvidence(
        node_id="gpu-01.internal.lan",
        code_qualified=True,
        review_qualified=False,
        scout_qualified=False,
        summary_only=False,
        measured_task_family="coding",
        quality_confidence=0.9,
        resource_pressure=0.2,
        health_freshness_seconds=10.0,
        observed_at=now - timedelta(minutes=5),
    )
    state = _state(
        FleetNodeSnapshot(
            node_id="gpu-01.internal.lan",
            capabilities=["gpu"],
            vram_free_gib=80.0,
            ram_free_gib=64.0,
        )
    )

    with pytest.raises(ValueError, match="opaque surrogate"):
        build_fleet_benchmark_advisory(
            FleetBenchmarkMatrix(campaign_id="c1", lanes=[lane]),
            state,
            work_intent=BenchmarkWorkIntent.CODING,
            observed_at=now,
            handle_by_node_id={"gpu-01.internal.lan": "gpu-01.internal.lan"},
        )


def test_a_surrogate_containing_the_node_slug_is_still_accepted():
    """Equality, not substring: this is the case the pattern must not break."""
    from datetime import UTC, datetime, timedelta

    from my_jev.fleet_benchmark_bridge import BenchmarkLaneEvidence
    from my_jev.fleet_policy import FleetNodeSnapshot

    now = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)
    lane = BenchmarkLaneEvidence(
        node_id="gpu-01.internal.lan",
        code_qualified=True,
        review_qualified=False,
        scout_qualified=False,
        summary_only=False,
        measured_task_family="coding",
        quality_confidence=0.9,
        resource_pressure=0.2,
        health_freshness_seconds=10.0,
        observed_at=now - timedelta(minutes=5),
    )
    state = _state(
        FleetNodeSnapshot(
            node_id="gpu-01.internal.lan",
            capabilities=["gpu"],
            vram_free_gib=80.0,
            ram_free_gib=64.0,
        )
    )

    advisory = build_fleet_benchmark_advisory(
        FleetBenchmarkMatrix(campaign_id="c1", lanes=[lane]),
        state,
        work_intent=BenchmarkWorkIntent.CODING,
        observed_at=now,
        handle_by_node_id={"gpu-01.internal.lan": "eligible:opaque:gpu-01"},
    )

    assert [item.handle for item in advisory.preferred] == ["eligible:opaque:gpu-01"]
