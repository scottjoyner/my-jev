"""End-to-end tests for the auto-router -> my-jev campaign join.

Session 3's stated purpose is to consume "a benchmark matrix from the current
fleet campaign". These tests pin that a real auto-router qualification report
actually satisfies that, and that the two facts auto-router must never know -
resource pressure and health freshness - are joined in rather than invented.
"""

from __future__ import annotations

import pytest

from my_jev.fleet_benchmark_bridge import (
    BenchmarkBridgeError,
    fleet_health_document,
    matrix_from_campaign,
)
from my_jev.fleet_benchmark_qualification import (
    FleetBenchmarkMatrix,
    build_fleet_benchmark_advisory,
)
from my_jev.fleet_policy import FleetPlacementState


def _entry(node_id: str, role: str, family: str = "coding", **stats) -> dict:
    base = {
        "node_id": node_id,
        "model_id": f"m-{node_id}",
        "task_family": family,
        "role": role,
        "qualified_for_coding": role == "CODE_QUALIFIED",
        "qualified_for_review": role in {"CODE_QUALIFIED", "REVIEW_QUALIFIED"},
        "qualified_for_scouting": role != "SUMMARY_ONLY",
        "capabilities": {"axes_are_distinct": True},
        "role_stats": {
            "samples": 4,
            "samples_passed": 4,
            "pass_rate": 1.0,
            "confidence": 0.8,
            "hard_failure_modes": [],
            "advisory_throughput": {
                "median_time_to_first_token_ms": 200.0,
                "median_tokens_per_second": 50.0,
            },
        },
    }
    base["role_stats"].update(stats)
    return base


def _report(entries, generated_at: str = "2026-10-02T20:00:00+00:00", **extra):
    document = {
        "generated_at": generated_at,
        "entries": list(entries),
        "advisory_only": True,
        "auto_load_allowed": False,
        "executable": False,
        "mutates_model_config": False,
        "creates_provider_eligibility": False,
        "signed_admission_required": True,
        "mode": "simulation",
    }
    document.update(extra)
    return document


def _health(node_id: str, pressure: float = 0.2, freshness: float = 30.0):
    return {
        node_id: {"resource_pressure": pressure, "health_freshness_seconds": freshness}
    }


# --- the join works at all ------------------------------------------------


def test_real_campaign_report_satisfies_the_matrix():
    report = _report(
        [
            _entry("optiplex", "CODE_QUALIFIED"),
            _entry("lenovo", "SCOUT_QUALIFIED", family="extraction"),
        ]
    )

    matrix, join = matrix_from_campaign(report, {**_health("optiplex"), **_health("lenovo")})

    assert isinstance(matrix, FleetBenchmarkMatrix)
    assert matrix.campaign_id
    assert {lane.node_id for lane in matrix.lanes} == {"optiplex", "lenovo"}
    assert join["lanes_built"] == 2
    assert join["lanes_dropped"] == 0
    assert join["advisory_only"] is True
    assert join["creates_provider_eligibility"] is False


def test_role_fields_are_renamed_not_reinterpreted():
    report = _report([_entry("optiplex", "CODE_QUALIFIED")])

    matrix, _ = matrix_from_campaign(report, _health("optiplex"))
    lane = matrix.lanes[0]

    assert lane.code_qualified is True
    assert lane.review_qualified is True
    assert lane.scout_qualified is True
    assert lane.summary_only is False
    assert lane.measured_task_family == "coding"


def test_summary_only_lane_grants_no_implementation_capability():
    report = _report([_entry("lenovo", "SUMMARY_ONLY", family="summarization")])

    matrix, _ = matrix_from_campaign(report, _health("lenovo"))
    lane = matrix.lanes[0]

    assert lane.summary_only is True
    assert lane.code_qualified is False
    assert lane.review_qualified is False


def test_latency_is_converted_from_milliseconds_to_seconds():
    report = _report([_entry("optiplex", "CODE_QUALIFIED")])

    matrix, _ = matrix_from_campaign(report, _health("optiplex"))

    assert matrix.lanes[0].latency_seconds == pytest.approx(0.2)
    assert matrix.lanes[0].throughput_tps == pytest.approx(50.0)


def test_quality_confidence_is_carried_from_the_role_stats():
    report = _report([_entry("optiplex", "REVIEW_QUALIFIED", confidence=0.62)])

    matrix, _ = matrix_from_campaign(report, _health("optiplex"))

    assert matrix.lanes[0].quality_confidence == pytest.approx(0.62)


# --- several models per node collapse to one lane -------------------------


def test_multiple_models_on_one_node_reduce_to_the_strongest_lane():
    report = _report(
        [
            _entry("optiplex", "SCOUT_QUALIFIED", family="extraction"),
            _entry("optiplex", "CODE_QUALIFIED", family="coding"),
            _entry("optiplex", "REVIEW_QUALIFIED", family="reasoning"),
        ]
    )

    matrix, join = matrix_from_campaign(report, _health("optiplex"))

    assert len(matrix.lanes) == 1
    assert matrix.lanes[0].code_qualified is True
    assert matrix.lanes[0].measured_task_family == "coding"
    assert join["lanes_built"] == 1


def test_ties_break_deterministically_regardless_of_entry_order():
    entries = [
        _entry("optiplex", "REVIEW_QUALIFIED", family="reasoning"),
        _entry("optiplex", "REVIEW_QUALIFIED", family="coding"),
    ]

    first, _ = matrix_from_campaign(_report(entries), _health("optiplex"))
    second, _ = matrix_from_campaign(_report(list(reversed(entries))), _health("optiplex"))

    assert first.model_dump() == second.model_dump()
    assert first.lanes[0].measured_task_family == "coding"


# --- health evidence is joined in, never invented -------------------------


def test_node_without_fleet_health_is_dropped_not_assumed_healthy():
    report = _report([_entry("optiplex", "CODE_QUALIFIED"), _entry("ghost", "CODE_QUALIFIED")])

    matrix, join = matrix_from_campaign(report, _health("optiplex"))

    assert {lane.node_id for lane in matrix.lanes} == {"optiplex"}
    assert join["lanes_dropped"] == 1
    assert join["dropped"] == [{"node_id": "ghost", "reason": "no_fleet_health_for_node"}]


def test_unusable_health_values_drop_the_lane():
    report = _report([_entry("optiplex", "CODE_QUALIFIED")])
    health = {"optiplex": {"resource_pressure": "high", "health_freshness_seconds": 10}}

    with pytest.raises(BenchmarkBridgeError, match="optiplex:fleet_health_values_unusable"):
        matrix_from_campaign(report, health)


def test_pressure_and_freshness_are_taken_from_health_not_the_report():
    report = _report([_entry("optiplex", "CODE_QUALIFIED")])
    health = _health("optiplex", pressure=0.9, freshness=120.0)

    matrix, _ = matrix_from_campaign(report, health)

    assert matrix.lanes[0].resource_pressure == pytest.approx(0.9)
    assert matrix.lanes[0].health_freshness_seconds == pytest.approx(120.0)


# --- fail closed ----------------------------------------------------------


def test_report_claiming_provider_eligibility_is_rejected():
    report = _report([_entry("optiplex", "CODE_QUALIFIED")], creates_provider_eligibility=True)

    with pytest.raises(BenchmarkBridgeError, match="cannot inform an advisory layer"):
        matrix_from_campaign(report, _health("optiplex"))


def test_report_claiming_executability_is_rejected():
    report = _report([_entry("optiplex", "CODE_QUALIFIED")], executable=True)

    with pytest.raises(BenchmarkBridgeError, match="cannot inform an advisory layer"):
        matrix_from_campaign(report, _health("optiplex"))


def test_missing_campaign_timestamp_is_rejected():
    report = _report([_entry("optiplex", "CODE_QUALIFIED")], generated_at="")

    with pytest.raises(BenchmarkBridgeError, match="timestamp_missing"):
        matrix_from_campaign(report, _health("optiplex"))


def test_naive_timestamp_is_rejected():
    report = _report([_entry("optiplex", "CODE_QUALIFIED")], generated_at="2026-10-02 20:00:00")

    with pytest.raises(BenchmarkBridgeError, match="timestamp_missing"):
        matrix_from_campaign(report, _health("optiplex"))


def test_report_with_no_usable_lane_is_rejected():
    report = _report([_entry("ghost", "CODE_QUALIFIED")])

    with pytest.raises(BenchmarkBridgeError, match="no benchmark lane survived"):
        matrix_from_campaign(report, _health("optiplex"))


def test_entries_without_identity_are_dropped():
    report = _report([{"task_family": "coding", "role": "CODE_QUALIFIED"}])

    with pytest.raises(BenchmarkBridgeError, match="entry_missing_node_or_family"):
        matrix_from_campaign(report, _health("optiplex"))


def test_unrecognised_role_is_dropped():
    report = _report([_entry("optiplex", "SOMETHING_ELSE")])

    with pytest.raises(BenchmarkBridgeError, match="entry_role_not_recognised"):
        matrix_from_campaign(report, _health("optiplex"))


def test_a_diagnosable_partial_drop_still_returns_the_join_report():
    """One bad node among good ones reports per-node reasons, not a bare raise."""

    report = _report(
        [
            _entry("optiplex", "CODE_QUALIFIED"),
            _entry("destroyer", "CODE_QUALIFIED"),
        ]
    )
    health = {
        **_health("optiplex"),
        "destroyer": {"resource_pressure": "unknown", "health_freshness_seconds": 5},
    }

    matrix, join = matrix_from_campaign(report, health)

    assert [lane.node_id for lane in matrix.lanes] == ["optiplex"]
    assert join["lanes_dropped"] == 1
    assert join["dropped"] == [
        {"node_id": "destroyer", "reason": "fleet_health_values_unusable"}
    ]


def test_malformed_report_is_rejected():
    with pytest.raises(BenchmarkBridgeError, match="unreadable"):
        matrix_from_campaign(["not", "a", "mapping"], _health("optiplex"))


def test_report_without_entries_is_rejected():
    with pytest.raises(BenchmarkBridgeError, match="no_entries"):
        matrix_from_campaign({"advisory_only": True}, _health("optiplex"))


# --- the joined matrix really does drive an advisory ---------------------


def test_joined_matrix_reaches_a_full_advisory_with_all_false_authority():
    from my_jev.uhp_advisory import SystemOneAuthority

    report = _report(
        [
            _entry("optiplex", "CODE_QUALIFIED"),
            _entry("lenovo", "REVIEW_QUALIFIED", family="reasoning"),
        ]
    )
    matrix, _ = matrix_from_campaign(report, {**_health("optiplex"), **_health("lenovo")})

    state = FleetPlacementState(
        workload_id="w1",
        workload_type="coding",
        nodes=[
            _node("alpha.internal.lan", code=True),
            _node("beta.internal.lan", code=True),
        ],
    )
    advisory = build_fleet_benchmark_advisory(
        state=state,
        matrix=matrix,
        handle_by_node_id={
            "alpha.internal.lan": "lane.alpha",
            "beta.internal.lan": "lane.beta",
        },
    )

    assert advisory.authority.model_dump() == SystemOneAuthority().model_dump()
    assert advisory.authority.dispatch_allowed is False
    assert advisory.physical_node_selection == "deterministic_resolver"


def _node(node_id: str, code: bool = True):
    from my_jev.fleet_policy import FleetNodeSnapshot

    return FleetNodeSnapshot(
        node_id=node_id,
        capabilities=["gpu", "code"] if code else ["gpu"],
        healthy=True,
        health_fresh=True,
        drained=False,
        reachable=True,
        ram_free_gib=64.0,
        vram_free_gib=32.0,
    )


def test_fleet_health_document_envelope_is_self_describing():
    document = fleet_health_document(_health("optiplex"))

    assert document["schema"] == "benchmark-bridge-fleet-health-v1"
    assert document["advisory_only"] is True
    assert "optiplex" in document["nodes"]
