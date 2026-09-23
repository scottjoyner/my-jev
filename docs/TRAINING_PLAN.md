# Training Plan

This is the first reproducible path from an empty repository to a useful System-One-style decision model. The objective is not to imitate hidden Jev internals; it is to build a fast bounded decision model whose probabilities are empirically meaningful.

## Hardware lane

Primary training target:

- AMD Radeon AI PRO R9700 / 32 GB VRAM class ROCm GPU
- BF16 where supported
- ModernBERT-base first, not a large decoder model
- gradient checkpointing available when state length or candidate count pushes memory

Support lane:

- x1-370 / Strix Halo class host with 96 GB unified memory for dataset preparation, local teacher inference, evaluation sweeps, and overflow experiments
- run teacher generation independently from model training so either lane can restart without losing labels

The first baseline should optimize for reproducibility rather than maximum context length.

## Dataset ladder

### Smoke — 1k to 5k states

Purpose: prove data parsing, loss movement, checkpointing, calibration, and inference.

Aim for 3-8 questions per state and a mix of all three primitives.

### Baseline — 25k to 100k states

Purpose: determine whether the architecture learns reusable runtime-defined decisions.

A useful target is 200k-750k individual decisions, with enough repeated schemas to learn task semantics and enough schema variation to prove that the dynamic option head generalizes.

### Scale — 250k+ states

Only scale after the baseline beats simple fixed-task classifiers and structured-output teacher baselines on held-out data.

The expensive part should be adding diverse, verifiable states and decisions, not blindly increasing epochs.

## Label hierarchy

Prefer labels in this order:

1. programmatically verifiable outcomes
2. observed real-world outcomes
3. adjudicated human labels
4. repeated human labels converted to empirical distributions
5. consensus from multiple independent teacher models
6. repeated samples from one teacher model
7. single synthetic hard labels

Store probability distributions whenever the evidence is genuinely probabilistic. Do not turn teacher confidence into "ground truth calibration."

## Teacher distillation

The repository includes an OpenAI-compatible teacher client so local LM Studio/Ollama-compatible gateways or hosted providers can produce bounded target distributions.

Example:

```bash
my-jev-label \
  --input data/unlabeled.jsonl \
  --output data/labeled.jsonl \
  --endpoint http://localhost:1234/v1/chat/completions \
  --model YOUR_TEACHER_MODEL \
  --samples 5
```

Multiple samples are averaged into one target distribution. For important datasets, use multiple different teacher models and aggregate outside this first client rather than trusting one model family.

## Split discipline

Never tune calibration on the final test set.

```bash
my-jev-split \
  --input data/labeled.jsonl \
  --output-dir data/splits \
  --train-fraction 0.80 \
  --calibration-fraction 0.10
```

This creates:

- `train.jsonl`: gradient updates
- `calibration.jsonl`: temperature fitting only
- `test.jsonl`: final untouched measurement

For related events, users, incidents, documents, or time series, replace random splitting with group/time-aware splitting before trusting the benchmark.

## Baseline training

Start at 1,024-2,048 state tokens even though the backbone supports longer contexts. Increase only when truncation analysis proves useful signal is being lost.

```bash
my-jev-train \
  --train data/splits/train.jsonl \
  --valid data/splits/calibration.jsonl \
  --output runs/modernbert-base-v0 \
  --backbone answerdotai/ModernBERT-base \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 8 \
  --max-state-length 2048 \
  --max-candidate-length 192 \
  --gradient-checkpointing \
  --bf16
```

If memory is comfortable, first raise batch size. Raise context length only after measuring truncation.

The validation split used during model development should eventually be separate from the calibration split. For the first baseline, this command can use the calibration split for early checkpoint selection, but before claiming final numbers create four partitions: train / validation / calibration / test.

## Calibration

Fit one temperature on a held-out calibration set:

```bash
my-jev-calibrate \
  --checkpoint runs/modernbert-base-v0/best \
  --data data/splits/calibration.jsonl \
  --output runs/modernbert-base-v0/calibration.json
```

Then evaluate the untouched test set with exactly that artifact:

```bash
my-jev-eval \
  --checkpoint runs/modernbert-base-v0/best \
  --data data/splits/test.jsonl \
  --calibration runs/modernbert-base-v0/calibration.json
```

## Release metrics

Track at minimum:

- accuracy by primitive and schema
- negative log likelihood
- Brier score
- expected calibration error
- reliability by confidence bucket
- selective accuracy / risk-vs-coverage
- p50 and p95 latency
- decisions per second
- throughput as question count and option cardinality increase
- metrics by input-length bucket
- out-of-schema and schema-paraphrase generalization

Never promote a checkpoint on accuracy alone.

## Verifier-reward stage

Only add verifier reward after the supervised baseline is stable.

When every legal option can be scored, optimize expected reward directly. When only the selected action can be checked, use the sampled policy-gradient fallback.

Keep an anchoring supervised/calibration term during this stage. A reward-only objective can improve task reward while making probabilities unusable.

## First experiment matrix

Run these in order:

1. frozen ModernBERT + trainable decision head
2. full ModernBERT fine-tune
3. full fine-tune + soft teacher distributions
4. calibrated checkpoint
5. calibrated checkpoint + verifier-reward fine-tune
6. smaller distilled encoder if quality holds
7. fused state-K/V attention optimization after correctness is established

For each experiment, preserve config, git SHA, data-manifest hashes, seed, raw metrics, and calibration artifact.

## Success criterion for v0

v0 is successful when one shared model can answer previously unseen combinations of runtime-defined Noul/Choice/Score questions with:

- materially better than naive/fixed baselines
- stable calibration on untouched data
- useful selective behavior at high confidence
- latency substantially below an autoregressive structured-output LLM for the same decision bundle

That is the point where kernel optimization, distillation, and larger-scale verifier training become worth the engineering cost.
