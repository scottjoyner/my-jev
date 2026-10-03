# Fleet benchmark qualification advisory

This slice turns a fleet campaign's **benchmark matrix** into a bounded advisory
recommendation: an execution *shape*, a *role*, and a *preference ordering* over
handles.

It is deliberately **not** a placement authority.

## Boundary

The deterministic resolver `resolve_fleet_placement`
(`src/my_jev/fleet_resolver.py`) and the hard predicate `eligible_nodes`
(`src/my_jev/fleet_policy.py`) remain the final eligibility boundary. Neither is
modified by this slice, and `build_fleet_benchmark_advisory` calls
`eligible_nodes` *before* any advisory branch, mirroring the resolver cascade's
ordering property.

A qualification score can only **reorder and annotate** the already-eligible set,
alongside the existing `rank_eligible_nodes`. It can never:

- add a node to the eligible set,
- make an ineligible node eligible (drained, unhealthy, stale health,
  unreachable, missing capability, under capacity, or locality-pinned out),
- name a node on the model-facing wire,
- dispatch, claim, approve, mutate, or change routing authority.

A maximal score (1.0) on an ineligible node still yields `selected_node_id: null`.
The emitted authority block is always the existing `SystemOneAuthority` from
`src/my_jev/uhp_advisory.py` — there is no second copy:

```json
{
  "dispatch_allowed": false,
  "approval_granted": false,
  "claim_acquired": false,
  "mutation_allowed": false,
  "routing_authority_changed": false
}
```

## Three separate concerns

| Concern | Owner | Field |
| --- | --- | --- |
| Execution shape | this layer (recommendation) | `execution_shape` |
| Role assignment | this layer (recommendation) | `role_assignment`, `decomposition_fallback` |
| Physical node selection | deterministic resolver | `physical_node_selection: "deterministic_resolver"` |

`selected_node_id` exists on the in-process result only as an assertion surface:
it can name a node solely when `eligible_nodes` already admitted it. Split advice
never names a node, matching the resolver.

## Evidence contract

`BenchmarkLaneEvidence` — one lane per `(node_id, observed_at)`:

| Field | Meaning |
| --- | --- |
| `node_id` | authoritative identity, host-side only |
| `code_qualified` | lane passed the code/implementation task family |
| `review_qualified` | lane passed the review task family |
| `scout_qualified` | lane passed the scout/decomposition task family |
| `summary_only` | summary-only evidence; can never certify implementation |
| `measured_task_family` | which measured family the flags came from |
| `quality_confidence` | `0..1` measured quality confidence |
| `latency_seconds` | measured latency (unknown telemetry scores `0`) |
| `throughput_tps` | measured throughput (unknown telemetry scores `0`) |
| `resource_pressure` | `0..1` pressure; reorders, never excludes |
| `health_freshness_seconds` | age of the lane's own health observation |
| `observed_at` | timezone-aware; drives TTL staleness |

`FleetBenchmarkMatrix` bundles `campaign_id` plus a bounded, duplicate-free lane
list. `QualificationThresholds` carries the configurable bounds:
`max_evidence_age_seconds` (TTL), `min_quality_confidence` (floor),
`max_health_freshness_seconds`.

A lane is ignored for **role** purposes — never for eligibility — when it is
stale, below the confidence floor, health-stale, or outside the eligible set.
Each ignore category is counted separately on the result.

## Advisory behaviours

- **No code-qualified lane** → `DEFER` for coding work. Work is deferred, never
  silently reassigned to a weaker role.
- **Code + review lanes on distinct nodes** → `SPLIT` may be advised, with a code
  preference and a reviewer preference. It is withheld (falling back to
  `PREFERRED_NODE` on one code lane) when the state lacks checkpoint/restart
  support or has fewer than two eligible nodes.
- **Only scout-qualified lane** → coding defers, and `decomposition_fallback`
  names `scout`: recommend decomposition/scout, never implementation. With
  `--work-intent scout`, the advisory recommends the scout lane and says so
  explicitly.
- **Stale benchmark evidence** (older than the TTL, or future-dated) → ignored;
  the affected lane cannot remain preferred.
- **Quality confidence below the floor** → ignored for role purposes. High TPS
  does not buy a code lane.
- **Drained / unhealthy / stale-health node** → the deterministic exclusion wins;
  its evidence is counted as ignored-ineligible.
- **Pressure** only reorders. A saturated lane stays eligible and stays listed;
  pressure is never promoted to hard ineligibility.

Scoring mirrors `evidence_adjusted_score` in `model_evidence.py`: trust is
multiplied **into an ordering**, never used as an eligibility gate.

## Hostnames never reach learned state

`FleetPlacementState.as_model_state()` strips `node_id` deliberately. This
slice mirrors that: `FleetBenchmarkAdvisory.as_model_state()` /
`model_wire()` emit no node IDs and no hostnames.

Opaque handles (`^[A-Za-z0-9._:-]+$`) come only from the caller-supplied
`handle_by_node_id` mapping, exactly like `_fleet_priority` in
`src/my_jev/uhp_advisory.py`. Handles are never derived here; a preference
without a valid handle raises rather than guessing.

Dropping `selected_node_id` is not sufficient on its own. `reasons` is
operator-facing free text and also flows to the model, so the advisory keeps
the node IDs it was derived from in a private, dump-excluded attribute and
both identity-free surfaces **refuse to emit a payload containing any of
them**, raising rather than passing quietly. Reason strings must therefore be
authored from opaque handles, never from node IDs.

## CLI

```bash
my-jev-fleet-benchmark-advisory \
  --matrix matrix.json \
  --state fleet-state.json \
  --handle-map handles.json \
  --work-intent coding \
  --observed-at 2026-10-02T12:00:00Z
```

No live AssistX access is performed: the CLI reads local JSON only. stdout
carries the identity-free advisory wire; stderr carries the summary
(`evidence_only`, `runtime_authority_changed: false`, `assistx_accessed: false`,
`model_state_sha256`). Expected failures print to stderr and return `2`.