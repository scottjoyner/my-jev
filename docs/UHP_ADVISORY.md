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
  --output-dir /tmp/system-one-acceptance
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
