# UHP advisory bridge

This slice turns an already-resolved my-jev policy decision into the private
`hermes-system-one-heartbeat-v1` profile carried inside a stored Unified
Harness Protocol response.

It is deliberately **not** an execution or routing adapter.

## Boundary

The emitted authority block is always:

```json
{
  "dispatch_allowed": false,
  "approval_granted": false,
  "claim_acquired": false,
  "mutation_allowed": false,
  "routing_authority_changed": false
}
```

A learned decision can recommend `act`, but it cannot grant permission to act.

Existing Hermes/AssistX claims, approvals, fencing, tool guards, signed runtime
admission, and mutation controls remain authoritative.

## Fleet privacy / authority

`FleetPlacementResolution` remains observer-only and names internal node IDs.
Those IDs never cross the UHP advisory wire.

The caller must provide an explicit authoritative node-id -> opaque-handle map.
Every ranked node must have a handle. The profile contains only those handles.

A fleet resolution with either:

- `observer_only=false`, or
- `dispatch_allowed=true`

is rejected.

## Protocol

Pinned profile:

- UHP: `2026-09-12`
- Hermes profile: `hermes-system-one-heartbeat-v1`
- TTL: default 600 seconds, maximum 900 seconds

The maximum matches the default Local Studio consumer acceptance window.

## Recorded fixture

The checked-in example is intentionally deterministic and model-free:

```bash
my-jev-uhp-fixture \
  --decision examples/uhp/decision.json \
  --fleet-resolution examples/uhp/fleet-resolution.json \
  --fleet-handle-map examples/uhp/fleet-handles.json \
  --provenance examples/uhp/provenance.json \
  --receipt-id fixture-receipt-1 \
  --response-id resp_fixture1 \
  --session-id hsess-fixture1 \
  --harness-id chrn_system_one \
  --model recorded/jev \
  --work-id acceptance-work-1 \
  --consumer-session-id '<pi-session-id>' \
  --project-cwd /absolute/path/to/project \
  --snapshot-sha256 '<64-char-snapshot-sha256>' \
  --observed-at 2026-09-23T22:30:00Z \
  --created-at 2026-09-23T22:30:00Z \
  --ttl-seconds 600 \
  --task-focus "Continue only inside the existing authority boundary." \
  --context-priority current-pr \
  --context-priority latest-handoff \
  --output /tmp/system-one/latest.json
```

The CLI writes the stored UHP response and prints an evidence summary to stderr:

```json
{
  "evidence_only": true,
  "runtime_authority_changed": false,
  "response_sha256": "...",
  "profile_sha256": "...",
  "output": "/tmp/system-one/latest.json"
}
```

For Local Studio, place that output at one of its accepted locations:

```text
LOCAL_STUDIO_SYSTEM_ONE_ADVISORY_PATH
<LOCAL_STUDIO_DATA_DIR>/system-one/sessions/<pi-session-id>.json
<LOCAL_STUDIO_DATA_DIR>/system-one/latest.json
```

No network call is made by the Local Studio turn hook.

## HarnessRouter producer path

The recorded fixture is the first acceptance stage.

The next producer is the already-shipping HarnessRouter System-One backend at:

`HarnessRouter/harnessrouter@250de65d6e690abdef40e39d21591b4a807984a3`

Use its `metadata.systemone.script` ScriptProvider path first. That proves:

- configured-harness selection
- session continuity
- deterministic System-One actions
- `trace.json`
- stored UHP response
- refusal/escalation handoff
- consumer acceptance

without asking a learned provider anything.

Only after that path is deterministic should the provider be swapped to my-jev,
and later the Bonsai decision-attention path.

## State projection

The decision provider should not receive the generic Neo4j MCP server as an
action environment.

Instead a host-side projector should produce a bounded immutable snapshot from:

- current Hermes/AssistX work state
- relevant Neo4j knowledge/memory facts
- current Git-versioned Markdown/Obsidian state
- live fleet observations
- canonical signed runtime projection
- authoritative eligibility / reservation / claim facts

The System-One environment then exposes only finite advisory actions over that
snapshot.

## Disposition mapping

Resolved my-jev dispositions map to the stable heartbeat vocabulary:

| Resolved disposition | Advisory mode |
| --- | --- |
| chat | chat |
| create_tasks | create_tasks |
| act | act |
| act_with_approval | act |
| propose_action | act |
| clarify | clarify |
| cancel | cancel |
| abstain | abstain |

`act_with_approval` and `propose_action` still carry no approval or mutation
authority. The consumer sees mode advice; the authority layer independently
decides whether any effect may occur.

## Acceptance order

1. recorded fixture -> Local Studio
2. expired fixture -> rejected
3. authority-bearing fixture -> rejected
4. model-fallback fixture -> rejected
5. HarnessRouter scripted System-One response -> Local Studio
6. scripted refusal/handoff -> recorded, not injected as completed advice
7. my-jev provider behind the same UHP profile
8. Bonsai provider behind the same UHP profile
9. only then evaluate any broader Hermes control-plane adoption


## Local Studio acceptance suite

Render all consumer-boundary cases with one command:

```bash
my-jev-uhp-fixture-suite \
  --output-dir /tmp/system-one-acceptance \
  --consumer-session-id '<pi-session-id>' \
  --project-cwd /absolute/path/to/project \
  --snapshot-sha256 '<64-char-snapshot-sha256>'
```

The suite writes:

- `valid.json` -> expected `consumed`
- `expired.json` -> expected `expired`
- `authority-bearing.json` -> expected `authority_mutation_allowed`
- `model-fallback.json` -> expected `model_fallback`
- `handoff.json` -> expected `system_one_handoff`
- `manifest.json` -> exact SHA-256 for every case plus `evidence_only=true` and `runtime_authority_changed=false`

For a live Local Studio acceptance pass, point
`LOCAL_STUDIO_SYSTEM_ONE_ADVISORY_PATH` at one case at a time and run one
ordinary coding-agent turn. Only `valid.json` should enter the system prompt.
Every other existing case should append an ignored evidence row and inject no
advisory context.


## Finite System-One heartbeat environment

The bounded snapshot can now be served directly as a SystemOneHarness MCP
environment without exposing the generic Neo4j or AssistX tool surfaces.

Install the optional MCP dependency:

```bash
pip install -e '.[systemone]'
```

Build a fresh snapshot:

```bash
my-jev-heartbeat-snapshot \
  --work examples/heartbeat/work.json \
  --knowledge examples/heartbeat/knowledge.json \
  --fleet examples/heartbeat/fleet.json \
  --authority examples/heartbeat/authority.json \
  --metadata examples/heartbeat/metadata.json \
  --capability read_repo \
  --capability run_tests \
  --tool github.read \
  --tool filesystem.read \
  --output /tmp/hermes-heartbeat.json
```

Serve it over stdio:

```bash
my-jev-heartbeat-mcp --snapshot /tmp/hermes-heartbeat.json
```

The MCP environment exposes exactly three tools:

- `observe` — returns the bounded snapshot, exact source revisions/checksums,
  and finite candidate lists.
- `reset` — reloads only the local snapshot and clears the in-memory advisory
  episode.
- `recommend` — chooses one mode, one opaque eligible fleet handle or
  `none`, and one bounded knowledge-note focus or `none`, then terminates.

`recommend` is a no-external-effect action. Its result always carries:

```json
{
  "dispatch_allowed": false,
  "approval_granted": false,
  "claim_acquired": false,
  "mutation_allowed": false,
  "routing_authority_changed": false
}
```

The environment refuses:

- an expired snapshot,
- an observation materially in the future,
- a fleet handle not present in the host-projected eligible set,
- a context reference not present in the bounded snapshot,
- a mode outside the fixed heartbeat vocabulary.

The SystemOneHarness model therefore cannot invent a destination, endpoint,
credential, graph query, tool, or additional context source.

### HarnessRouter configuration

The checked-in configuration is:

`configs/systemone/hermes-heartbeat-advisory.yaml`

When packaging the environment for HarnessRouter, use the heartbeat MCP command
as the configured System-One harness's first MCP server and keep this
`config.yaml` beside that server in the package root.

The first real integration run should still use:

```json
{"metadata":{"systemone":{"script":["recommend"]}}}
```

or the equivalent scripted-provider probe expected by the concrete environment
fixture, so session creation, snapshot observation, finite action compilation,
trace generation, handoff, and stored UHP response can be proven before my-jev
or Bonsai supplies a learned decision.

The host remains responsible for converting the terminal System-One result into
the `metadata.hermes_system_one` UHP profile. That recorder/adapter is the next
networked seam; the environment itself has no routing or mutation capability.


## Bound contract and replay protection

The canonical checked-in profile schema is:

`contracts/hermes-system-one-heartbeat-v1.schema.json`

Exact schema SHA-256:

`5e88c73e7cbb2e46f3b5171951d2a84f0549633fbcb420458d56ae5ada0ffc8f`

Every emitted profile carries that value as `contract_sha256`. Consumers fail
closed on any contract mismatch instead of guessing how to interpret a newer or
older receipt.

Every profile is also bound to one intended consumer context:

```json
{
  "consumer": "local-studio",
  "work_id": "work-...",
  "consumer_session_id": "pi-session-...",
  "project_fingerprint": "<sha256(canonical realpath workspace)>",
  "snapshot_sha256": "<exact bounded heartbeat snapshot sha256>"
}
```

A fresh receipt is not portable to another Pi session or workspace. The consumer
session + project fingerprint are the Local Studio-enforced replay boundary.
`work_id` and `snapshot_sha256` preserve producer/source lineage; Local Studio
records them but cannot independently recompute the source work graph or heartbeat
snapshot. The compiler verifies the snapshot hash on the producer side, and
cryptographic producer signatures remain the next authenticity layer.

The advisory payload preserves the richer policy semantics in addition to the
small mode vocabulary:

- `policy_disposition`
- `approval_recommended`

These remain recommendations. `approval_granted` is always false.

## Recommendation compiler

The finite MCP environment and the UHP profile are joined by a fail-closed
compiler:

`src/my_jev/heartbeat_compile.py`

It accepts only a terminal `hermes-system-one-recommendation-v1` whose:

- `snapshot_sha256` exactly matches the supplied bounded snapshot
- all five authority fields are explicitly present and exactly false, with no unknown authority fields
- selected fleet handle is inside the snapshot's authoritative eligible set
- selected context focus is inside the snapshot's bounded note refs
- snapshot is still live and not materially future-dated
- exact System-One config version, model revision, and trace SHA-256 provenance are present

The compiled receipt expiry is capped by the source snapshot expiry. A
recommendation can never extend the lifetime of the state it was based on.

The operator bridge is:

```bash
my-jev-heartbeat-compile \
  --snapshot /tmp/hermes-heartbeat.json \
  --recommendation /tmp/hermes-system-one-recommendation.json \
  --receipt-id receipt-123 \
  --consumer-session-id '<pi-session-id>' \
  --project-cwd /absolute/path/to/project \
  --response-id resp_acceptance_123 \
  --uhp-session-id hsess-acceptance \
  --harness-id chrn_system_one \
  --model recorded/jev \
  --mode-confidence 1.0 \
  --system-one-config-version 1 \
  --model-revision recorded/jev \
  --trace /path/to/trace.json \
  --output /tmp/system-one/latest.json
```

The compiler requires the exact config version, model revision, explicit mode
confidence, and trace file. It records the exact trace hash and prints the source
snapshot hash, profile hash, response hash, binding, and expiry as evidence. The
stored UHP file is published with a write-then-rename so the consumer never sees
a partially written receipt.

## Single-use consumer expectation

A bound UHP response is intended to be consumed once per target coding-agent
session. The Local Studio consumer records an atomic consumption marker before
injecting the advisory. Replay identity is the receipt id within the target Pi
session, not the raw JSON byte hash. Re-presenting the same receipt produces
`replay_already_consumed`; reusing the same receipt id with changed content
produces `receipt_id_conflict`. A successful influence also requires the
detailed consumption ledger write to succeed.


## Snapshot trust defaults

Heartbeat authority context is fail-closed. When the projector omits an
authority fact, the snapshot now observes:

- `speaker_verified=false`
- `actions_allowed=false`
- `local_writes_allowed=false`
- `external_actions_allowed=false`
- `privileged_actions_allowed=false`
- `approval_gate_available=false`

A projector must explicitly assert permissive facts. Missing data never becomes
implicit permission even at the advisory/recommendation layer.

All snapshot text — goals, blockers, facts, note references, metadata, claims,
and approval labels — is treated as untrusted evidence by the finite System-One
environment. Commands embedded in those fields are not instructions, and a
selected context label is not permission to read the named resource.

## Remaining authenticity boundary

The current response/profile hashes establish stable evidence identity, not
authorship. Before live state is allowed to influence production coding-agent
turns, the producer should sign a standards-based canonical representation
(e.g. RFC 8785/JCS) with a host-owned key and consumers should hold
verification-only material. Do not sign the existing ad-hoc Python/JavaScript
JSON canonicalizations: their number formatting can differ for values such as
`1.0` versus `1`.
