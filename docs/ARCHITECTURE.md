# Architecture: System-One v0

## What we know publicly

TypeSafe describes Jev as a System One model that consumes application state plus typed questions and returns bounded, probabilistic `Noul`, `Choice`, and `Score` outputs. TypeSafe also publicly describes a parallel sampler and a training method called Reinforcement Learning for Calibrated Decisions (RLCD), but does not disclose enough architectural or optimization detail to independently reproduce its implementation.

This repository therefore implements an original architecture optimized for the same problem shape.

## Data flow

```text
                 ┌──────────────────────────────┐
state ──────────►│ shared bidirectional encoder │──── state token memory
                 └──────────────────────────────┘              │
                                                               │
question+option ┐ ┌──────────────────────────────┐              │
question+option ├►│ shared bidirectional encoder │── candidates │
question+option ┘ └──────────────────────────────┘              │
                                      │                         │
                                      └──── parallel cross-attn ┘
                                                    │
                                                    ▼
                                           scalar option logits
                                                    │
                                       group-wise softmax only
                                                    │
                                                    ▼
                               Noul / Choice / ordered Score probabilities
```

The model does not decode vocabulary tokens at inference time.

## Why dynamic option scoring

A fixed classifier head assumes every task has the same labels. System-One workflows need the caller to define the schema at runtime. Each option is therefore represented as text, scored against the state, and normalized only against sibling options for that question.

This makes one model reusable across support, observability, routing, extraction-as-choice, risk checks, and other workflows without creating a new output layer for every task.

## Primitive semantics

### Noul

Implicit options: `false`, `true`. Output includes `P(true)`.

### Choice

Two to 255 caller-provided options. Output is a categorical probability distribution.

### Score

Two to 255 ordered levels. Output is a distribution plus a normalized expected score in `[0, 1]`. Training adds an ordinal earth-mover penalty so near misses cost less than distant misses.

## Training stages

### Stage A — supervised probabilistic training

Train against hard labels and/or target distributions using:

- categorical negative log likelihood
- Brier score
- ordinal EMD for Score
- a small confidence/correctness calibration regularizer

Soft targets are first-class. They can come from repeated labels, adjudicated annotators, consensus teachers, or empirical outcome frequencies.

### Stage B — held-out calibration

Fit temperature scaling only on a calibration split. Never fit calibration parameters on the final test set.

Track NLL, Brier, ECE, reliability curves, and accuracy-vs-coverage at confidence thresholds.

### Stage C — verifier-reward fine-tuning

For decisions whose consequences can be checked programmatically, optimize expected verifier reward while retaining the calibration objectives.

If the verifier can score every option, use the exact expected reward under the model distribution. If the verifier only evaluates the sampled choice, use the sampled policy-gradient fallback in `my_jev.reward`.

This is intentionally described as RLCD-inspired, not TypeSafe RLCD.

## Efficiency path

v0 prioritizes correctness and trainability. Candidate cross-attention temporarily expands state token memory per candidate. Once the learning setup is validated, optimize in this order:

1. grouped/fused attention that reuses state K/V without materializing copies
2. packed candidate batches and length bucketing
3. compile/SDPA kernels
4. distillation into smaller encoder backbones
5. INT8/FP8 inference where calibration remains stable
6. optional state embedding cache for repeated workflows

## Calibration gates

A model is not considered release-ready based on accuracy alone. A candidate checkpoint should meet explicit holdout gates, for example:

- ECE <= 0.05
- no confidence bucket with >10 percentage-point calibration gap when bucket sample size >= 100
- monotonic selective accuracy as confidence threshold rises
- tracked NLL/Brier regression budget
- latency measured separately from tokenizer and I/O

Thresholds should be tuned to the target workload rather than copied blindly.
