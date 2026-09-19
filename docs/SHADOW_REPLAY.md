# Hermes / AssistX shadow replay

This slice evaluates a trained `my-jev` checkpoint against captured AssistX
policy states without giving the model any execution path.

The replay command is deliberately offline with respect to Hermes/AssistX:
it reads a JSONL export, runs local inference plus the existing deterministic
resolver, and writes JSONL evidence. It does not call AssistX, Hermes, tools,
task creation, approval APIs, messaging APIs, or mutation endpoints.

## Input

Use the same raw JSONL shape accepted by `my-jev-shadow-import`. Each row must
contain `policy_shadow_json` with the captured `request.state` and may contain
the old shadow response and legacy classification/action for comparison.

Preserve the raw export unchanged. The replay manifest hashes it so later
analysis can prove which evidence was measured.

## Run after the R9700 baseline

After `scripts/run-r9700-modernbert-baseline.sh` completes, take the emitted
run directory and replay a frozen AssistX export:

```bash
RUN_DIR="runs/experiments/assistx-policy-modernbert-r9700-baseline-..."
EXPORT="runs/shadow/assistx-export.jsonl"

my-jev-shadow-replay \
  --input "$EXPORT" \
  --checkpoint "$RUN_DIR/checkpoints/best" \
  --calibration "$RUN_DIR/calibration.json" \
  --device cuda \
  --output "$RUN_DIR/shadow-replay.jsonl"
```

This creates:

```text
shadow-replay.jsonl
shadow-replay.jsonl.manifest.json
```

Every output row carries `mode="shadow_replay"` and
`dispatch_allowed=false`. It retains the candidate response for later
disagreement inspection plus compact fields for:

- candidate route/disposition;
- candidate AssistX classification/policy action;
- policy-consistency violations;
- previous shadow route/disposition when present;
- legacy classification/policy action.

The manifest records input/output SHA-256s, record count, disposition counts,
legacy policy-action agreement, previous-shadow disposition agreement, and the
rate of records with policy-consistency violations.

## Acceptance before any integration work

The first replay is evidence gathering, not a promotion into runtime authority.
Review disagreements by source/domain and especially inspect any candidate
`act`, `act_with_approval`, or consistency-violation rows.

Do not wire candidate output into dispatch based on this replay. Existing
Hermes/AssistX identity, approvals, permissions, mutation controls, tool
guardrails, task claims/fencing, and recovery remain authoritative.

A later integration slice should consume replay evidence first and, if useful,
add a sidecar/shadow observer that cannot dispatch. Direct routing is a separate
milestone with separate acceptance criteria.
