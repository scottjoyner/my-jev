"""Join auto-router benchmark qualification evidence into a fleet advisory matrix.

Session 3 consumes "a benchmark matrix from the current fleet campaign". That
campaign report is produced by auto-router's ``benchmark_qualification``. The
two schemas do not line up on their own, and the mismatch is not accidental:

* auto-router groups its report per ``(node, model, task_family)`` and names
  fields for the qualification ladder (``role``, ``qualified_for_coding``,
  ``role_stats.confidence``, ``advisory_throughput``). This module expects
  ``BenchmarkLaneEvidence`` fields.
* **auto-router must never read live node state**, so a benchmark report
  legitimately carries no ``resource_pressure`` and no
  ``health_freshness_seconds``. Those are fleet-health facts, not quality
  evidence. This module therefore takes them from a separate, caller-supplied
  health map and refuses to invent them.
* auto-router reports several entries per node (one per model, per family).
  ``FleetBenchmarkMatrix`` allows one lane per ``(node_id, observed_at)``, so
  entries are reduced to the best-supported lane per node.
* auto-router entries carry no per-entry timestamp; ``generated_at`` on the
  report envelope is the observation time for the whole campaign.

Consequences of that last point are handled fail-closed. A node with no fleet
health, an unparseable timestamp, or a report that claims authority it must not
have is dropped or rejected with a reason - never defaulted into looking
healthy. Benchmark evidence may inform an advisory preference; it can never
assert that a node is available.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from collections.abc import Mapping

from .fleet_benchmark_qualification import (
    FleetBenchmarkMatrix,
    BenchmarkLaneEvidence,
)

#: Strongest qualification first. Used only to reduce several benchmark entries
#: for one node down to the single lane the matrix can carry.
_ROLE_RANK: dict[str, int] = {
    "CODE_QUALIFIED": 0,
    "REVIEW_QUALIFIED": 1,
    "SCOUT_QUALIFIED": 2,
    "SUMMARY_ONLY": 3,
}

REASON_UNREADABLE_REPORT = "benchmark_report_unreadable"
REASON_NO_ENTRIES = "benchmark_report_has_no_entries"
REASON_MISSING_IDENTITY = "entry_missing_node_or_family"
REASON_MISSING_OBSERVED_AT = "campaign_timestamp_missing_or_unparseable"
REASON_NO_FLEET_HEALTH = "no_fleet_health_for_node"
REASON_BAD_HEALTH = "fleet_health_values_unusable"
REASON_UNKNOWN_ROLE = "entry_role_not_recognised"

CAMPAIGN_REPORT_SCHEMA = "benchmark-qualification-report-v1"
FLEET_HEALTH_SCHEMA = "benchmark-bridge-fleet-health-v1"


class BenchmarkBridgeError(ValueError):
    """Raised when a campaign report is not usable as advisory evidence."""


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _require_advisory_report(report: Mapping[str, Any]) -> None:
    """Refuse a report that asserts authority it is not allowed to hold.

    This refusal **raises** rather than recording a reason code, unlike every other
    rejection in this module, and a dead `REASON_AUTHORITY_CLAIM` constant used to
    imply otherwise -- advertising a reason string for a path that produces none.
    The split is deliberate: a report claiming authority is not a bad lane to be
    dropped with a note, it is an input this layer must not derive from at all.

    auto-router's report always carries these guards. A report claiming it can
    create provider eligibility or mutate model config is not something this
    advisory layer should be deriving from, so it is rejected outright rather
    than downgraded.
    """

    for field, expected in (
        ("advisory_only", True),
        ("auto_load_allowed", False),
        ("executable", False),
        ("mutates_model_config", False),
        ("creates_provider_eligibility", False),
    ):
        if field in report and report[field] is not expected:
            raise BenchmarkBridgeError(
                f"benchmark report must assert {field}={expected!r}; "
                "an authoritative report cannot inform an advisory layer"
            )


def _entry_rank(entry: Mapping[str, Any]) -> tuple[int, float, str]:
    role = str(entry.get("role") or "")
    rank = _ROLE_RANK.get(role, len(_ROLE_RANK))
    stats = entry.get("role_stats")
    confidence = 0.0
    if isinstance(stats, Mapping):
        try:
            confidence = float(stats.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
    return (rank, -confidence, str(entry.get("task_family") or ""))


def _lane_from_entry(
    entry: Mapping[str, Any],
    observed_at: datetime,
    health: Mapping[str, Any],
) -> BenchmarkLaneEvidence:
    stats = entry.get("role_stats")
    throughput = stats.get("advisory_throughput") if isinstance(stats, Mapping) else None
    throughput = throughput if isinstance(throughput, Mapping) else {}

    def _float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    ttft_ms = _float(throughput.get("median_time_to_first_token_ms"))
    code = bool(entry.get("qualified_for_coding"))
    review = bool(entry.get("qualified_for_review"))
    scout = bool(entry.get("qualified_for_scouting"))

    try:
        pressure = float(health.get("resource_pressure"))
        freshness = float(health.get("health_freshness_seconds"))
    except (TypeError, ValueError) as exc:
        raise BenchmarkBridgeError(REASON_BAD_HEALTH) from exc

    return BenchmarkLaneEvidence(
        node_id=str(entry["node_id"]),
        code_qualified=code,
        review_qualified=review,
        scout_qualified=scout,
        # True when the lane earns no code, review, or scout capability. This
        # is the "useful, but only for summarising" rung of the ladder.
        summary_only=not (code or review or scout),
        measured_task_family=str(entry["task_family"]),
        quality_confidence=max(0.0, min(1.0, float(
            stats.get("confidence") if isinstance(stats, Mapping) else 0.0
        ))),
        latency_seconds=None if ttft_ms is None else round(ttft_ms / 1000.0, 6),
        throughput_tps=_float(throughput.get("median_tokens_per_second")),
        resource_pressure=max(0.0, min(1.0, pressure)),
        health_freshness_seconds=max(0.0, freshness),
        observed_at=observed_at,
    )


def matrix_from_campaign(
    benchmark_report: Mapping[str, Any],
    fleet_health: Mapping[str, Mapping[str, Any]],
    *,
    campaign_id: str | None = None,
) -> tuple[FleetBenchmarkMatrix, dict[str, Any]]:
    """Join a campaign qualification report and fleet health into a matrix.

    Returns the matrix plus a join report listing every lane built and every
    lane dropped with its reason. Nothing is defaulted: a lane missing either
    its quality evidence or its health evidence is dropped, not guessed.
    """

    if not isinstance(benchmark_report, Mapping):
        raise BenchmarkBridgeError(REASON_UNREADABLE_REPORT)
    _require_advisory_report(benchmark_report)

    entries = benchmark_report.get("entries")
    if not isinstance(entries, list):
        raise BenchmarkBridgeError(REASON_NO_ENTRIES)

    observed_at = _timestamp(benchmark_report.get("generated_at"))
    built: list[BenchmarkLaneEvidence] = []
    dropped: list[dict[str, str]] = []
    reasons: list[str] = []

    def drop(node_id: str, reason: str) -> None:
        dropped.append({"node_id": node_id, "reason": reason})
        if reason not in reasons:
            reasons.append(reason)

    if observed_at is None:
        raise BenchmarkBridgeError(REASON_MISSING_OBSERVED_AT)

    # Reduce several benchmark entries per node to the strongest qualifying
    # lane, so the matrix's (node_id, observed_at) uniqueness holds.
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            drop("<malformed>", REASON_UNREADABLE_REPORT)
            continue
        node_id = str(entry.get("node_id") or "").strip()
        family = str(entry.get("task_family") or "").strip()
        if not node_id or not family:
            drop(node_id or "<unknown>", REASON_MISSING_IDENTITY)
            continue
        if str(entry.get("role") or "") not in _ROLE_RANK:
            drop(node_id, REASON_UNKNOWN_ROLE)
            continue
        grouped.setdefault(node_id, []).append(entry)

    for node_id in sorted(grouped):
        health = fleet_health.get(node_id)
        if not isinstance(health, Mapping):
            # No health evidence means we cannot claim the node is even
            # currently usable. Drop rather than assume healthy.
            drop(node_id, REASON_NO_FLEET_HEALTH)
            continue
        best = sorted(grouped[node_id], key=_entry_rank)[0]
        try:
            built.append(_lane_from_entry(best, observed_at, health))
        except (BenchmarkBridgeError, KeyError, TypeError, ValueError) as exc:
            drop(node_id, str(exc) or REASON_BAD_HEALTH)

    if not built:
        # Fail closed, but carry the per-lane diagnosis: an operator debugging a
        # campaign needs to know a node lacked health evidence rather than just
        # that "something" was unusable.
        detail = "; ".join(
            f"{item['node_id']}:{item['reason']}" for item in dropped
        ) or "no entries"
        raise BenchmarkBridgeError(f"no benchmark lane survived the join: {detail}")

    matrix = FleetBenchmarkMatrix(
        campaign_id=str(
            campaign_id or benchmark_report.get("campaign_id") or "fleet-campaign"
        ),
        lanes=built,
    )
    join_report = {
        "schema": CAMPAIGN_REPORT_SCHEMA,
        "observed_at": observed_at.isoformat(),
        "campaign_entries": len(entries),
        "lanes_built": len(built),
        "lanes_dropped": len(dropped),
        "dropped": dropped,
        "drop_reasons": reasons,
        "advisory_only": True,
        "creates_provider_eligibility": False,
        "evidence_only": True,
        "runtime_authority_changed": False,
    }
    return matrix, join_report


def fleet_health_document(
    nodes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Wrap a per-node health map in a checkable document envelope."""

    return {
        "schema": FLEET_HEALTH_SCHEMA,
        "nodes": {str(key): dict(value) for key, value in nodes.items()},
        "advisory_only": True,
    }


__all__ = [
    "CAMPAIGN_REPORT_SCHEMA",
    "FLEET_HEALTH_SCHEMA",
    "BenchmarkBridgeError",
    "fleet_health_document",
    "matrix_from_campaign",
]