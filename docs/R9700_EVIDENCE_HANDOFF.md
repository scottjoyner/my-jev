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

## 3. Build the frozen AssistX shadow-evaluation bundle

Only after validation succeeds, use the bundled path so candidate identity,
shadow-export identity, replay, review artifacts, and authority assertions are
bound into one receipt:

```bash
my-jev-shadow-evaluate \
  --run-dir "$RUN_DIR" \
  --expected-sha "$SHA" \
  --shadow-export "$ASSISTX_SHADOW_EXPORT" \
  --device cuda \
  --limit 200 \
  --min-priority 0.25
```

The command re-validates the R9700 baseline before loading the checkpoint,
refuses to write into a non-empty evaluation directory, hashes the frozen
shadow export, performs non-dispatching replay, builds the disagreement/
uncertainty review queue, and writes
`shadow-evaluation/shadow-evaluation-receipt.json`.

The lower-level `my-jev-shadow-replay` and `my-jev-review` commands remain
available for diagnostics, but the bundled command is the preferred evidence
path because it prevents provenance from being assembled manually.

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
