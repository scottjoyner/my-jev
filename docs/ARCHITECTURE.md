# Architecture: System-One v0

## Scope

TypeSafe publicly describes Jev as a System One model that consumes application
state plus typed questions and returns bounded probabilistic Noul, Choice, and
Score outputs. TypeSafe has also described a parallel sampler and RLCD, but has
not published enough implementation detail to reproduce the proprietary model.

This repository is therefore an independent implementation optimized for the
same general decision shape and, specifically, for Hermes/AssistX policy
routing.

## Data flow

```text
                 +------------------------------+
state ---------->| shared bidirectional encoder |---- state token memory
                 +------------------------------+             |
                                                              |
question+option -+ +----------------------------+             |
question+option -+>| shared bidirectional encoder|-- options  |
question+option -+ +----------------------------+             |
                                      |                       |
                                      v                       |
                              option query vectors            |
                                      |                       |
                                      +---- attend over -------+
                                      |
                                      v
                              scalar option logits
                                      |
                         group-wise softmax only
                                      |
                                      v
                    Noul / Choice / ordered Score probabilities
```

The model does not decode vocabulary tokens at inference time.

## Option-query decision head

The default head is `option_query`.

For a batch of states:

- state token memory has shape `[B, L, H]`;
- all runtime question/option candidates for a state are padded to
  `[B, C, H]`;
- candidates project to queries;
- state tokens project to keys and values;
- all candidate queries attend over the same state token memory in parallel;
- one shared query/context dot product produces one score per candidate.

This avoids materializing the state once per candidate, which matters for
Hermes policy turns containing many independent typed questions.

The pattern is informed by the independent MIT-licensed
`vinnylarouge/jevlike` project, which demonstrates variable-option queries over
shared context. The implementation here is independently structured around
multiple questions per state, soft targets, ordered scores, calibration, and
Hermes/AssistX policy contracts.

The original candidate-by-candidate cross-attention implementation remains
available as `head_kind=legacy` so older checkpoints can still load.

## Why dynamic option scoring

A fixed classifier head assumes every task has one permanent label vocabulary.
System-One workflows need runtime-defined schemas.

For Hermes this means the same model can simultaneously score questions such
as:

```text
route:
  chat | create_tasks | act | clarify | cancel | abstain

action_scope:
  none | read_only | local_write | external_side_effect | privileged

risk:
  low | moderate | high | critical
```

without introducing a new neural output head for each field.

Each option is text, not a hard-coded class neuron.

## Primitive semantics

### Noul

Implicit options: `false`, `true`. Output includes `P(true)`.

### Choice

Two to 255 caller-defined alternatives. Output is a categorical probability
distribution.

### Score

Two to 255 ordered levels. Training adds an ordinal earth-mover penalty so
near misses cost less than distant misses.

## Hermes / AssistX authority split

The learned model predicts **what the turn appears to require**. It does not
decide what the runtime is allowed to execute.

```text
                         learned
                     policy probabilities
                             |
                             v
hard runtime facts ---> deterministic resolver
                             |
                             v
                    resolved disposition
                             |
             +---------------+----------------+
             |               |                |
           Hermes          AssistX       approval/review
           tools           task graph        gates
```

Hard facts include speaker verification, permitted action scopes, approval
availability, active work, and the existing tool/mutation policy.

A high `P(act)` cannot widen those permissions.

See `docs/ASSISTX_POLICY.md`.

## Candidate lifecycle: exact-SHA baseline to shadow evidence

The first R9700 checkpoint is not a deployment milestone. It enters a frozen,
evidence-only candidate lifecycle before any larger training or integration
decision is made.

```text
exact clean PR SHA
        |
        v
R9700 ModernBERT baseline
        |
        +--> checkpoint + calibration + benchmark + fleet benchmark
        |
        v
independent artifact validation
        |
        v
frozen Hermes/AssistX shadow export
        |
        v
non-dispatching shadow replay
        |
        +--> calibrated policy vector
        +--> legacy/resolver evidence
        +--> policy-consistency checks
        |
        v
disagreement / uncertainty review queue
        |
        v
operator + outcome-verifier + user-correction adjudication
        |
        v
group-safe next-iteration corpus
        |
        +--> larger supervised experiment
        +--> calibration/regression comparison
        +--> remain rejected / evidence-only
```

### Frozen-candidate contract

A shadow evaluation must bind all evidence to one immutable candidate identity:

- exact 40-character Git SHA;
- checkpoint and calibration artifact hashes;
- experiment-spec hash and dataset split hashes;
- hardware/runtime receipt from the R9700 run;
- immutable shadow-export identity/hash;
- replay and review manifests derived from those inputs.

Changing the checkpoint, calibration, policy schema, resolver contract, or shadow
input creates a new evaluation identity rather than silently updating an
existing result.

### Shadow adapter boundary

The shadow adapter may read the same normalized policy state that is available
to the existing Hermes/AssistX routing path and may emit candidate probability
vectors plus evaluation metadata. It is observer-only.

It MUST NOT:

- dispatch tools or agents;
- create, cancel, claim, or mutate tasks;
- send messages or perform external side effects;
- grant permissions or satisfy approvals;
- alter speaker-verification results;
- acquire scheduler/worker execution authority;
- replace the deterministic resolver's disposition.

The existing resolver and runtime continue normally. Candidate outputs are
written only to an evaluation sink keyed by candidate identity and source turn.

### Replay and disagreement analysis

Offline replay consumes a frozen shadow export and produces predictions without
calling the live dispatcher. Each replay row should retain enough provenance to
reconstruct:

```text
source turn/state
candidate identity
calibrated probability vector
candidate top disposition
legacy/resolver evidence
policy-consistency findings
uncertainty/disagreement signals
```

The review queue is a prioritization artifact, not a label source. Priority may
combine explicit correction evidence, policy-consistency violations,
disagreement with existing routing evidence, and calibrated uncertainty.

### Adjudication boundary

Training truth enters only through explicit adjudication or deterministic
outcome verification. Existing router decisions and model predictions are
evidence, not ground truth.

Partial labels are valid when evidence supports only part of the policy vector.
Independent labels should remain independently attributable and may be
aggregated into soft targets rather than forcing false consensus.

Family/provenance identifiers must survive adjudication so related
counterfactuals and replay-derived examples cannot leak across train,
validation, calibration, and test partitions.

### Promotion boundary

There is intentionally no path from a successful shadow replay directly to
runtime authority. The next decision after shadow evaluation is experimental:
whether the evidence justifies a larger training iteration.

Any future runtime integration requires a separate design/release slice with
explicit resolver tests proving that learned output cannot widen permissions,
bypass approvals, override claims/fencing, or directly dispatch work.

## Training stages

### Stage A — supervised probabilistic training

Train against hard labels and/or target distributions using:

- categorical negative log likelihood;
- Brier score;
- ordinal EMD for Score;
- a small confidence/correctness calibration regularizer.

Soft targets may represent repeated human labels, empirical outcome
frequencies, adjudicated consensus, or teacher distributions.

### Stage B — held-out calibration

Fit temperature scaling only on a calibration split. Never fit calibration
parameters on the final test set.

Track NLL, Brier, ECE, reliability, and selective risk/coverage.

### Stage C — trajectory aggregation

For the Hermes policy model, collect states the current policy actually visits:

```text
state -> policy -> resolver -> tools/tasks/chat -> outcome/correction
  ^                                                  |
  +---------------- next training round -------------+
```

This is closer to imitation/DAgger-style policy improvement than static prompt
classification.

Observable action/outcome traces are useful supervision. Hidden chain-of-thought
is not required.

### Identifiers that name a wire contract are defined once

Any string that appears in a document another system reads — a schema name, a
filename, a digest — is defined once as a constant and referenced everywhere else.
`tests/test_no_copied_contract_strings.py` enforces it, and names the copies in
its failure message because a guard that only says "duplicate" gives an operator
nothing to act on.

Three of these had drifted, all in the same direction: the constant was dead
because the code was written against a literal instead.

* `hermes-system-one-recommendation-v1` was a bare literal in three modules while
  `RECOMMENDATION_SCHEMA` sat unused. The `Literal` annotation in
  `heartbeat_compile` cannot reference a constant, so it keeps its literal — and
  a test asserts the annotation and the constant agree.
* `stages/<name>.timing.json` was a literal in four places: the pattern constant,
  the writer in `experiment`, `gpu_plan`'s reader glob, and the suffix stripped to
  recover a stage name. The glob and suffix are now derived from the pattern, so
  the reader and writer cannot disagree about the file's name.
* `AFTER_HOURS_LINGER_SECONDS` and `REASON_AUTHORITY_CLAIM` were deleted: the
  first duplicated `AFTER_HOURS_CLOSE`, and the second advertised a reason code for
  an authority refusal that raises rather than records.

That last one is why the sweep is a test rather than a review step. The refusal it
named exists and works; only the constant was wrong, advertising a channel that
produces nothing.

Two items the sweep deliberately did **not** delete, because being unused is not
the same as being wrong:

* `sampled_policy_gradient_loss` — documented in this file as the Stage D
  policy-gradient fallback, so it is forward-looking API rather than dead code.
* `prediction_from_outputs` in `fleet_family_eval` — **deleted.** It looked like
  the adapter turning model outputs into the predictions `evaluate_fleet_families`
  consumes, and produced the right inner shape. But it subscripts its inputs
  (`output["probabilities"]`, `output["question"]`) and `QuestionOutput` is a
  dataclass with no `__getitem__`, whose field is `name` rather than `question`.
  It could only ever have run against hand-built dicts nothing produces, and it
  would have skipped the temperature transform. The transformation that *is* used
  lives inline in `fleet_benchmark._predict_family_records`.

### Correction evidence is a contract, and 9 of its 10 fields were unpinned

Which fields of a record show that a human corrected it is a producer/consumer
contract spanning three modules: `shadow_replay` and `shadow_import` write it,
`review.has_correction_evidence` reads it and turns it into a share of the record's
review priority. It was a bare string literal in all three, with no constant and no
test naming it. Rename it on one side and nothing fails — the consumer falls through
to the older `user_corrected` / `operator_corrected` booleans, which the replay
importer never writes, and a human-corrected record quietly stops counting as
corrected.

Worse, the *field list* was unpinned: of the ten names `_CORRECTION_FIELDS`
recognises, only `user_correction` appeared anywhere in the suite. A field dropped
from that tuple would have been silent too.

Both are now pinned. The key is `CORRECTION_EVIDENCE_FIELDS` in `schema.py`, and the
ten names are written down independently in
`tests/test_correction_evidence_contract.py` — deliberately *not* imported from the
list, because the obvious version of that test is
`@pytest.mark.parametrize("field", _CORRECTION_FIELDS)`, which iterates whatever the
list contains. Deleting a field deleted its own test case and the suite stayed
green. A contract list has to be written down somewhere independent or it is not
pinned at all.

### A falsy correction flag counted as a human correction

The check was `value is None or value is False` — an identity comparison. A producer
emitting `0` or `0.0` for an unset flag, which JSON producers do routinely, had that
record counted as human-corrected, raising the review priority of a record nobody
touched. The failure direction is the loud-looking one: the correction signal feeds
priority, so false positives inflate it.

### Stage D — verifier-reward fine-tuning

For decisions whose consequences can be scored programmatically, optimize
expected verifier reward while retaining supervised/calibration anchors.

If every legal option can be scored, use exact expected reward under the model
distribution. If only the selected action is observable, use the sampled
policy-gradient fallback in `my_jev.reward`.

This is intentionally described as RLCD-inspired, not TypeSafe RLCD.

## Shortcut controls

Accuracy alone is not enough for an agent policy.

The benchmark includes a **shuffled-state control** inspired by Jevlike: the
candidate questions/options remain associated with each record while the state
token memory is rotated across the batch.

A useful policy should degrade materially when given the wrong state.

Additional promotion controls should include:

1. authority counterfactuals;
2. option-order permutation;
3. conversation-context ablation;
4. speaker/source holdouts;
5. request-template and task-family holdouts;
6. user-correction replay.

## Efficiency path

The option-query head already removes the largest v0 duplication: candidate
queries share one state K/V tensor.

Next optimizations should be measured rather than assumed:

1. candidate length bucketing and packing;
2. encoder output/state cache for repeated policy questions;
3. compile / SDPA kernel experiments;
4. distillation into smaller encoder backbones;
5. INT8/FP8 inference where calibration remains stable;
6. batch coalescing for concurrent AssistX turns.

## Calibration and promotion gates

A checkpoint should not be promoted on top-1 accuracy alone.

Useful release gates include:

- ECE within an explicit workload-specific budget;
- NLL and Brier no-regression;
- positive accuracy delta over shuffled-state control;
- positive KL sensitivity to relevant state;
- no regression on authority counterfactuals;
- monotonic selective accuracy as confidence rises;
- acceptable p50/p95 latency and decisions/sec;
- no observed widening of execution authority in resolver tests.

Thresholds should be derived from real Hermes/AssistX shadow traffic rather
than copied from generic classification benchmarks.
