"""Project benchmark qualification onto the System-One operator surface.

This lives in its own module because ``fleet_benchmark_qualification`` imports
``SystemOneAuthority`` from ``uhp_advisory`` — deliberately, so there is one
all-false authority block rather than a fourth copy. The projection needs the
advisory type *and* the advice models, so putting it here keeps the dependency
one-directional: this module imports both, ``uhp_advisory`` imports this one
lazily at the single call site.

The advisory built by ``build_fleet_benchmark_advisory`` was otherwise reachable
only through its own CLI. An operator reading a compiled System-One receipt had
no way to see that the only qualified lane was merely scout-capable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from .uhp_advisory import (
    MAX_FLEET_PRIORITY,
    MAX_LABEL_LENGTH,
    MAX_REASON_LENGTH,
    BenchmarkQualificationAdvice,
    NextAction,
    _OPAQUE_HANDLE,
)

if TYPE_CHECKING:
    from .fleet_benchmark_qualification import FleetBenchmarkAdvisory

from .fleet_benchmark_qualification import BenchmarkRole

#: Roles that can carry an implementation. A scout lane is explicitly excluded:
#: telling an operator that a scout-only fleet can implement is the exact failure
#: this projection exists to make visible where it is read.
_IMPLEMENTING_ROLES = frozenset(
    {BenchmarkRole.CODE, BenchmarkRole.REVIEW, BenchmarkRole.SUMMARY}
)


#: One operator-facing line per ``NextAction``, composed from structured facts.
#:
#: An earlier version scraped ``advisory.reasons`` for a sentence to show. The
#: advisory orders its list decision-reasons-first and rejections-last, so the
#: last element was usually "evidence older than the TTL was ignored" -- which
#: says nothing about why the decision went the way it did. Matching on the
#: prose would have been the second version of the same mistake, so the line is
#: composed here instead. The full reason list stays on the advisory for anyone
#: who wants it; the per-cause counts are already in ``rejected_evidence``.
_ACTION_LINES = {
    NextAction.implement: (
        "a {role}-qualified lane cleared benchmark qualification on measured "
        "evidence; {shape} is advisable"
    ),
    NextAction.implement_with_reviewer: (
        "a {role}-qualified lane and a distinct reviewer both cleared "
        "qualification; a split with a dedicated reviewer is advisable"
    ),
    NextAction.decompose_and_scout: (
        "no lane qualified to implement the requested work, but "
        "{fallback}-qualified lanes are available; recommend decomposition and "
        "scouting rather than implementation"
    ),
    NextAction.await_qualification: (
        "no lane cleared benchmark qualification for the requested role; retest "
        "or wait rather than assuming a reachable node is useful"
    ),
    NextAction.defer: (
        "no node survives authoritative deterministic eligibility; benchmark "
        "evidence cannot change this"
    ),
}


def decision_reason(advisory: FleetBenchmarkAdvisory, action: NextAction) -> str:
    """One line explaining the decision, composed rather than scraped."""
    template = _ACTION_LINES[action]
    return template.format(
        role=advisory.role_assignment.value if advisory.role_assignment else "requested",
        fallback=(
            advisory.decomposition_fallback.value
            if advisory.decomposition_fallback
            else "scout"
        ),
        shape=advisory.execution_shape.value,
    )[:MAX_REASON_LENGTH]


def classify_next_action(
    advisory: FleetBenchmarkAdvisory,
    preferred_count: int,
) -> NextAction:
    """Name the advisory's outcome, derived from the branch it actually took.

    Derived rather than inferred, so a consumer branches on one field instead of
    re-deriving the decision from shape, role and prose together. Every value
    corresponds to a real branch of ``build_fleet_benchmark_advisory``.
    """
    if preferred_count == 0 or advisory.role_assignment is None:
        if advisory.decomposition_fallback is not None:
            return NextAction.decompose_and_scout
        # Nothing was eligible at all is a different operational problem from
        # "eligible but unqualified": no benchmark result can change it, so it
        # must not be reported as if waiting for evidence would help.
        if advisory.eligible_node_count == 0:
            return NextAction.defer
        return NextAction.await_qualification
    if advisory.role_assignment not in _IMPLEMENTING_ROLES:
        return NextAction.await_qualification
    if len(advisory.preferred) >= 2:
        return NextAction.implement_with_reviewer
    return NextAction.implement


def benchmark_qualification_for(
    advisory: FleetBenchmarkAdvisory | None,
    handle_by_node_id: Mapping[str, str] | None,
) -> BenchmarkQualificationAdvice | None:
    """Project a benchmark advisory onto the operator surface, fail-closed.

    Every invariant here narrows. Qualification may withhold ``implement`` and
    order preference, but it cannot introduce a handle the advisory did not
    prefer, accept one outside the opaque pattern, or carry a node id across.
    """
    if advisory is None:
        return None
    if not advisory.observer_only or advisory.dispatch_allowed:
        raise ValueError("benchmark advisory must remain observer-only")
    if advisory.physical_node_selection != "deterministic_resolver":
        raise ValueError("benchmark advisory must leave selection to the resolver")

    # Node ids are the map's KEYS; handles are its values. A handle equal to a
    # key is identity passed through as a surrogate, which is the one thing the
    # indirection exists to prevent. Reading the keys is safe -- they are used
    # only to reject, never to emit.
    node_ids = set(handle_by_node_id or {})
    handles = set((handle_by_node_id or {}).values())
    seen: set[str] = set()
    preferred: list[str] = []
    roles: dict[str, str] = {}
    for item in advisory.preferred:
        handle = item.handle.strip()
        if (
            not handle
            or len(handle) > MAX_LABEL_LENGTH
            or not _OPAQUE_HANDLE.fullmatch(handle)
            or handle in seen
        ):
            raise ValueError("invalid or duplicate benchmark-preferred handle")
        seen.add(handle)
        preferred.append(handle)
        roles[handle] = item.role.value
    if len(preferred) > MAX_FLEET_PRIORITY:
        raise ValueError("too many benchmark-preferred handles")
    # Every preferred handle must be one the caller actually bound to a node.
    # Mirrors _fleet_priority: an unbound handle is a mapping the operator never
    # authorised, so it fails closed rather than being passed through.
    if handles and not set(preferred) <= handles:
        raise ValueError("every benchmark-preferred handle requires an opaque mapping")
    # Defence in depth: the advisory already asserts identity-freedom on its own
    # wire, but this is the last place a node id could escape into a receipt, so
    # re-check here rather than trusting the producer.
    leaked = sorted(h for h in preferred if h in node_ids)
    if leaked:
        raise ValueError("benchmark handle is a node id, not an opaque surrogate")

    rejected = {
        "stale": advisory.ignored_stale_lane_count,
        "future_dated": advisory.ignored_future_dated_lane_count,
        "low_confidence": advisory.ignored_low_confidence_lane_count,
        "health_stale": advisory.ignored_health_stale_lane_count,
        "ineligible": advisory.ignored_ineligible_lane_count,
    }
    rejected = {reason: count for reason, count in rejected.items() if count}

    action = classify_next_action(advisory, len(preferred))
    return BenchmarkQualificationAdvice(
        next_action=action,
        next_action_reason=decision_reason(advisory, action),
        execution_shape=str(advisory.execution_shape.value),
        implementation_advisable=bool(preferred)
        and advisory.role_assignment in _IMPLEMENTING_ROLES,
        preferred_handles=preferred,
        qualified_roles=roles,
        rejected_evidence=rejected,
        evaluated_at=advisory.observed_at,
    )
