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
