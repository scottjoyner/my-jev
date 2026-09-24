# Model Evidence Selection Contract

Status: advisory-only v1  
Producer: knowledge `fleet_model_evidence_compiler.py`  
Consumer: my-jev / System-One

## Purpose

The model-selection head should answer a bounded question:

> Among opaque model handles that the trusted host has already declared eligible,
> which handle best fits the request's complexity, latency, context, modality,
> capability, quantization, and execution evidence?

It does not choose a physical node, provider endpoint, runtime coordinate,
approval state, claim, tool set, or mutation authority.

## Evidence layers

The producer preserves three separate evidence layers.

### Base-model capability prior

Artificial Analysis or another external benchmark source may supply intelligence,
coding, or agentic capability priors. Provider throughput is reference-only and
is never substituted for local hardware throughput.

### Exact local execution evidence

Local evidence is keyed by execution lane:

```text
physical node
+ runtime/backend
+ artifact identity
+ quantization
+ harness/task family
+ context/flags
+ timestamp/source revision
```

A benchmark remains historical evidence when its node is offline, reserved, or
not currently admitted.

### Current eligibility

A trusted projector supplies `eligible-model-handle-projection-v1`. This is the
only input that can introduce a candidate into a live model-selection snapshot.
The compiler may use internal node/artifact selectors to summarize the matching
execution evidence, but those selectors are removed before the snapshot reaches
the decision model.

## Model-visible envelope

`ModelExecutionEnvelope` exposes only bounded aggregate features:

- benchmark lane count
- distinct-node count
- currently eligible replica count
- raw task-fit scores by request profile
- comparable generation/prompt throughput envelope
- verified context envelope
- task success rate and trial count
- execution-evidence confidence
- backend classes
- evidence states
- measurement classes
- roles observed

Physical node names, ports, provider IDs, and runtime coordinates are not
model-visible.

## Measurement-class rule

Throughput from unlike measurements must not be pooled.

Examples:

- controlled `llama-bench` pp/tg -> generic throughput feature
- agent-loop judge generation -> agentic measurement class, not generic speed
- TerminalBench success -> task evidence
- short end-to-end canary -> separate wall-clock/server evidence
- runtime memory release -> scheduling/capacity event

The Destroyer 2026-09-24 example demonstrates this boundary:

```text
Ornith generator -> K2-1B judge -> accept/retry/delegate
K2 judge: 6/6 correct
```

The 6/6 result is useful agentic evidence. The judge's displayed 13.2 t/s is not
treated as generic generation throughput.

## Quality-first deterministic fallback

`rank_candidate_handles()` exists as a conservative baseline and acceptance
oracle. It ranks only candidates already present in the bounded snapshot.

The producer supplies both:

- `task_fit_scores`: capability/task evidence before local speed adjustment;
- `scenario_scores`: task fit plus the producer's bounded execution adjustment.

The evidence-adjusted primary key is:

```text
raw task-fit score
* sqrt(evidence coverage)
* identity confidence
* quantization confidence
* execution-evidence confidence
```

The bounded scenario score is only a later tie-break. This prevents a fast but
less capable model from outranking a better task match merely because it emits
tokens quickly, while still allowing latency and throughput to matter when
quality is close.

This also keeps narrow perfect samples from dominating broadly supported
evidence. Raw task-fit and scenario scores remain visible and are not rewritten.

The learned System-One head may later learn a better calibration from historical
request/evidence/outcome records. Promotion of that learned behavior is a
separate evidence gate.

## Authority invariants

Every model-selection `DecisionRecord` sets:

```text
dispatch_allowed = false
approval_granted = false
claim_acquired = false
mutation_allowed = false
routing_authority_changed = false
```

The selected opaque model handle is advisory. Existing trusted projection and
Auto-Router exact-artifact authority decide whether/how it can be executed.

## Acceptance sequence

The smallest deterministic acceptance is:

1. Freeze one heartbeat snapshot and its SHA-256.
2. Freeze one benchmark/catalog revision.
3. Supply an eligible-handle projection with two or more opaque model handles.
4. Bind a concrete request task family when one is known and compile
   `model-evidence-snapshot-v1`.
5. Verify no node/provider/port identity appears in canonical model state.
6. Verify changing request profile can change advisory ranking without changing
   candidate eligibility.
7. Verify context/modality/identity-conflicted candidates are excluded before
   the model sees them.
8. Verify a narrow 6/6 judge result retains its task evidence but does not gain
   generic throughput credit.
9. Verify the compiled decision record retains the all-false authority block.
10. Record the evidence snapshot SHA, selected handle, raw/evidence-adjusted
    scores, and eventual execution outcome for shadow learning.


## Request task family

`ModelRequestNeeds.task_families` carries bounded semantic task labels such as
`repo_work`, `debugging`, or a frozen benchmark task-family ID. The producer
may use an exact task family to select local historical success evidence; the
learned System-One head sees the same request label and the model's measured
task families.

Task-family labels do not grant any tool or routing authority. They are decision
features only.

## Why speed is not the objective

Generation speed is a deployment characteristic, not a capability proof. A
model that fails a coding-agent or reasoning task in 0.5 seconds is not more
useful for that task than a slower model that succeeds.

For this reason, the local scoring producer now fails to score a speed-only lane
when capability/task-fit evidence is absent. The learned policy should preserve
that semantic separation: learn *what model can do the job*, then learn the
cheapest/fastest acceptable execution choice.
