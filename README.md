# my-jev

An open-source, independently implemented **System-One-style typed decision model** inspired by the public problem shape described for TypeSafe AI's Jev.

> This project is not Jev, is not affiliated with TypeSafe AI, and does not claim to reproduce TypeSafe's proprietary model architecture, parallel sampler, weights, datasets, or RLCD implementation.

## Goal

Build and train a model that maps:

```text
unstructured / structured application state
                 +
        typed decision schema
                 |
                 v
      bounded probabilities
```

The model never needs to generate prose. Its public primitives are:

- **Noul** — binary probability
- **Choice** — probability distribution over caller-provided options
- **Score** — ordered probability distribution plus an expected score

## v0 architecture

The first implementation uses a bidirectional encoder backbone (default: `answerdotai/ModernBERT-base`) and a dynamic decision head:

1. Encode the application **state once**.
2. Encode every question/option candidate in one batch.
3. Query the state token memory with all candidate representations in parallel.
4. Produce one scalar logit per legal option.
5. Normalize only across sibling options for each question.
6. Return typed probabilities; no vocabulary decoding and no autoregressive generation.

A dynamic option scorer is intentional: callers can define labels at runtime instead of retraining a fixed classifier head for every schema.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/TRAINING_PLAN.md](docs/TRAINING_PLAN.md).

## Training objective

The project optimizes decision quality and calibration together:

- categorical negative log likelihood / soft-target cross entropy
- Brier score
- ordinal earth-mover loss for Score questions
- confidence/correctness calibration regularization
- held-out temperature scaling
- optional verifier-reward fine-tuning for programmatically checkable decisions

Verifier-reward support includes exact expected-reward optimization when all legal options can be scored and a sampled policy-gradient fallback when only the chosen action can be verified.

## Install

```bash
git clone https://github.com/scottjoyner/my-jev.git
cd my-jev
git switch feature/system-one-v0
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Data format

One JSONL row contains one state, one or more typed questions, and optional hard or probabilistic targets:

```json
{
  "state": "Customers are seeing HTTP 500 errors after a deploy.",
  "questions": {
    "urgent": {
      "type": "noul",
      "instructions": "Does this need immediate attention?"
    },
    "severity": {
      "type": "score",
      "instructions": "How severe is the impact?",
      "options": ["minor", "moderate", "major", "critical"]
    }
  },
  "targets": {
    "urgent": {"index": 1},
    "severity": {"distribution": [0.0, 0.1, 0.3, 0.6]}
  }
}
```

Probability distributions are first-class labels so repeated human judgments, teacher consensus, or observed outcome frequencies do not need to be collapsed to fake certainty.

## Bootstrap soft labels

The teacher utility talks to an OpenAI-compatible chat-completions endpoint, which makes it usable with local gateways as well as hosted providers:

```bash
my-jev-label \
  --input data/unlabeled.jsonl \
  --output data/labeled.jsonl \
  --endpoint http://localhost:1234/v1/chat/completions \
  --model YOUR_TEACHER_MODEL \
  --samples 5
```

It asks only for bounded probability distributions, validates option cardinality, repeats the teacher call, and averages the returned distributions. Teacher labels are weak/bootstrap supervision; they should not be treated as empirical calibration truth.

## Split

```bash
my-jev-split \
  --input data/labeled.jsonl \
  --output-dir data/splits \
  --train-fraction 0.80 \
  --calibration-fraction 0.10
```

For production-quality evaluation, prefer train / validation / calibration / test partitions and use group- or time-aware splitting where related records could leak across splits.

## Train

Head-only smoke run:

```bash
my-jev-train \
  --train examples/train.jsonl \
  --valid examples/valid.jsonl \
  --output runs/smoke \
  --epochs 1 \
  --freeze-backbone
```

First full baseline:

```bash
my-jev-train \
  --train data/splits/train.jsonl \
  --valid data/splits/calibration.jsonl \
  --output runs/modernbert-base-v0 \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 8 \
  --max-state-length 2048 \
  --gradient-checkpointing \
  --bf16
```

Every run writes `run_config.json` alongside checkpoints so the effective device/mixed-precision/memory configuration is preserved.

## Calibrate

Fit temperature only on held-out calibration data:

```bash
my-jev-calibrate \
  --checkpoint runs/modernbert-base-v0/best \
  --data data/splits/calibration.jsonl \
  --output runs/modernbert-base-v0/calibration.json
```

The scaler accepts both hard labels and soft target distributions.

## Evaluate

Evaluate the untouched test split using the frozen calibration artifact:

```bash
my-jev-eval \
  --checkpoint runs/modernbert-base-v0/best \
  --data data/splits/test.jsonl \
  --calibration runs/modernbert-base-v0/calibration.json
```

Primary release metrics are accuracy, NLL, Brier score, expected calibration error (ECE), selective accuracy/coverage, and latency/decisions per second.

## Inference

```bash
my-jev decide \
  --checkpoint runs/modernbert-base-v0/best \
  --calibration runs/modernbert-base-v0/calibration.json \
  --input examples/support_ticket.json
```

The response is typed JSON containing legal options, probabilities, selected option, confidence, calibration temperature, and primitive-specific values such as `noul=P(true)` or a normalized `score`.

## Roadmap

The first milestone validates the learning problem before chasing exotic kernels:

1. build a real multi-domain state/question corpus
2. train and calibrate the ModernBERT-base baseline
3. benchmark hard labels vs. soft teacher distributions
4. add verifier-reward training after the supervised baseline is stable
5. compare with fixed classifiers and structured-output LLM baselines
6. fuse state K/V reuse and candidate attention for speed
7. distill into smaller encoders and test quantized inference
8. build workflow evals over multi-step application decision graphs

## Public references

- TypeSafe AI — Introducing System One Models and Jev: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- TypeSafe workflow evals: https://evals.typesafe.ai/
- LangChain — Building a Harness with Jev: https://www.langchain.com/blog/building-a-harness-with-jev
- ModernBERT: https://arxiv.org/abs/2412.13663

## Status

The trainable v0 implementation is on `feature/system-one-v0` in draft PR #1. The next milestone is the first non-toy dataset plus an exact-SHA training, calibration, and benchmark run.
