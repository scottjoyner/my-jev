# Open-Jev / pngwn findings for my-jev

Reference implementation:

- Space: https://huggingface.co/spaces/pngwn/open-jev
- scorer: https://huggingface.co/pngwn/system-one-qwen3.5-4b-scorer
- causal-vs-masked report:
  https://huggingface.co/datasets/pngwn/typed-decisions-causal-experiment/blob/main/REPORT.md

This note records ideas worth testing in `my-jev`. It is not a code import.

## What the Space demonstrates

The public demo uses a Qwen3.5-4B base model adapted as a scalar scorer.

For each typed decision it scores `(state, question, option)` branches, then
normalizes the scalar logits within each question. The optimized inference path
prefills the state once and branches question/option work from the cached model
state instead of re-encoding the full state for every option.

The published model card reports:

- Qwen3.5-4B-Base;
- LoRA rank 16 over linear projections plus a scalar score head;
- grouped softmax training;
- 12,913 training questions over nine task families;
- 2,200 optimizer steps;
- batch 8 questions;
- max training sequence length 384;
- training option cap 16;
- bf16 + gradient checkpointing;
- a single held-out temperature of 1.75;
- raw test ECE 0.135 -> 0.044 after calibration;
- 112.3 ms per four-option question in the reported environment;
- 559.7 ms in a 77-option smoke run.

Those numbers are reference measurements from that project, not performance
claims for `my-jev`.

## Findings that matter for Hermes / AssistX

### 1. Calibration must be a first-class stage

The scorer's top-1 accuracy did not change after temperature scaling, while its
reported ECE improved sharply.

Implication: keep calibration as a separate held-out artifact. Do not choose a
checkpoint only by classification accuracy.

### 2. Choice order invariance is a real property to test

The report compares an independent scalar cross-encoder with a cached causal
letter scorer.

The scalar scorer is effectively invariant to candidate permutation because
each option receives an independent scalar before group softmax.

The cached letter scorer is not: the report observes substantial rank changes
when candidate order is reversed.

Implication: `my-jev-benchmark` now includes a Choice-order perturbation. The
same Choice options reversed must recover the same probability distribution
after aligning by option text.

This invariant is intentionally **not** applied to Score questions, because
Score order carries semantic meaning. Noul has its fixed false/true ordering.

### 3. Candidate-set size is different from candidate order

Even an order-invariant scorer changes normalized probability mass when new
candidates are added. That is expected from a softmax over a changing set.

The report shows meaningful candidate-cardinality sensitivity even for the
scalar scorer.

Implication: add a future cardinality/distractor suite that reports:

- top-1 agreement;
- gold logit margin;
- probability dilution;
- entropy;
- duplicate/synonym behavior;
- abstention behavior.

Do not call probability movement under an expanded option set "miscalibration"
without defining the target distribution for the expanded set.

### 4. Independently accurate fields can still disagree

The report's dependent-decision experiment shows that independent decisions can
violate known cross-field constraints. Iterative masked refinement removed the
measured violations, but at roughly four times the sequential forward work in
that experiment and without improving joint accuracy.

Hermes already has explicit semantics and authority rules. We do not need a
neural iterative reconciliation loop for rules the control plane already knows.

Implication: keep one-pass policy inference and run a deterministic
cross-field consistency audit before execution. Examples:

- `act` should not pair with `needs_tools=false`;
- `act` should not pair with `action_scope=none`;
- `create_tasks` should not pair with `needs_task_graph=false`;
- external/privileged scopes should imply an external effect;
- mutation scopes should require tool use.

Material contradictions prevent direct action and fall back to proposal or
clarification.

### 5. Shared-state reuse is worth a second experimental lane

Our current ModernBERT option-query model encodes state once and lets option
queries attend over shared state-token memory. That remains a good small,
bidirectional baseline.

The Open-Jev work suggests a second lane worth implementing after the baseline
is trained:

```text
Qwen3.5-4B-Base
  + LoRA
  + scalar sequence-classification head
  + state prefill/cache branching
  + grouped softmax
  + held-out calibration
```

The purpose is not to replace the encoder baseline automatically. It is to test
whether a pretrained causal backbone gives Hermes better language/world
knowledge while retaining bounded, non-generative decisions.

Measure both on the same AssistX policy corpus and the same held-out sessions.

## License boundary

The published `pngwn/system-one-qwen3.5-4b-scorer` artifact is marked
CC-BY-NC-4.0 because its training mixture includes non-commercial ticket data.

Do **not** make that checkpoint or restricted dataset the foundation of our
deployable model.

A Qwen experimental lane in `my-jev` should start from the Apache-2.0 Qwen
base and our own commercially clean / appropriately licensed policy corpus.

## Promotion matrix

Before any policy checkpoint gains routing authority, record at least:

| Gate | Required evidence |
| --- | --- |
| held-out behavior | accuracy / NLL / Brier / ECE |
| state dependence | shuffled-state degradation |
| Choice invariance | option-permutation agreement |
| candidate-set robustness | cardinality / distractor sweep |
| policy coherence | cross-field consistency violations |
| authority safety | authority-counterfactual replay |
| selective routing | error rate above confidence thresholds |
| speed | p50/p95 states/sec and decisions/sec |
| real-world fit | shadow Hermes/AssistX trajectory replay |

The result we want is not merely a classifier with a System-One-looking API.
It is a calibrated, state-sensitive policy model whose uncertainty and failure
modes are measurable enough to sit in front of Hermes.
