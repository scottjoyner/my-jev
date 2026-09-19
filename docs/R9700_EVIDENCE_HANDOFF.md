# R9700 baseline evidence handoff

Use this after the physical R9700 run. It is intentionally a verification and
handoff step only; it does not deploy the model or modify Hermes/AssistX.

## 1. Run the exact-head baseline

Use the SHA you actually checked out:

```bash
SHA="$(git rev-parse HEAD)"
bash scripts/run-r9700-modernbert-baseline.sh "$SHA"
```

The launcher prints the resulting experiment run directory.

## 2. Independently validate the artifact bundle

```bash
my-jev-baseline-evidence \
  --run-dir "$RUN_DIR" \
  --expected-sha "$SHA" \
  --output "$RUN_DIR/r9700-evidence-validation.json"
```

The validator fails closed when the bundle is incomplete, the manifest records
a dirty tree, the manifest and receipt disagree on the Git SHA, the effective
checkpoint configuration differs from the pinned one-epoch baseline, HIP/R9700
evidence is absent, or a receipt hash disagrees with the artifact on disk.

A rejected promotion is still accepted as valid experimental evidence. Do not
rewrite or delete a rejected run merely because it misses quality gates.

## 3. Replay frozen AssistX evidence

Only after validation succeeds:

```bash
my-jev-shadow-replay \
  --input "$ASSISTX_SHADOW_EXPORT" \
  --output "$RUN_DIR/shadow-replay.jsonl" \
  --checkpoint "$RUN_DIR/checkpoints/best" \
  --calibration "$RUN_DIR/calibration.json" \
  --device cuda

my-jev-review \
  --input "$RUN_DIR/shadow-replay.jsonl" \
  --output "$RUN_DIR/shadow-review-queue.jsonl" \
  --limit 200 \
  --min-priority 0.25
```

Replay is non-dispatching. Review output remains unlabeled. Neither artifact is
permission to execute, create tasks, send messages, mutate state, or bypass an
approval gate.

## Evidence to return to the repo/session

Preserve these together:

- `r9700-baseline-receipt.json`
- `r9700-evidence-validation.json`
- `benchmark.json`
- `promotion.json`
- `calibration.json`
- `r9700-doctor.json`
- `shadow-replay.jsonl.manifest.json`
- `shadow-review-queue.jsonl.manifest.json`
- `shadow-review-queue.jsonl.review.json`

From those files we can decide the next experimental slice without treating the
candidate as runtime authority.

## Authority boundary

The candidate model remains evidence/advice only throughout this handoff.
Hermes/AssistX speaker verification, action permissions, approval gates,
mutation authority, tool guardrails, task claims/fencing, and recovery controls
remain authoritative.
