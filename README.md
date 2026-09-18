# my-jev

An open-source, independently implemented **System-One-style typed decision model** inspired by the public interface and research direction described for TypeSafe AI's Jev.

> This project is not Jev, is not affiliated with TypeSafe AI, and does not claim to reproduce TypeSafe's proprietary model architecture, sampler, weights, datasets, or RLCD implementation.

## Goal

Build and train a model that maps:

```text
unstructured/structured state + typed decision schema
                         ↓
              bounded probabilities
```

The model never needs to generate prose. Its public primitives are:

- **Noul** — binary probability
- **Choice** — probability distribution over a caller-provided option set
- **Score** — ordered probability distribution plus expected score/confidence

## v0 architecture

The first implementation uses a bidirectional encoder backbone (default: `answerdotai/ModernBERT-base`) and a dynamic decision head:

1. Encode the application **state once**.
2. Encode every question/option candidate in a single batch.
3. Query the state token memory with all candidate representations in parallel.
4. Produce one scalar logit per legal option.
5. Normalize only across sibling options for each question.
6. Return typed probabilities; no vocabulary decoding and no autoregressive generation.

A dynamic option scorer is intentional: the caller can define new labels at runtime instead of retraining a fixed classifier head for every schema.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

## Training objective

The project optimizes decision quality and calibration together:

- categorical negative log likelihood / soft-target cross entropy
- Brier score
- ordinal earth-mover loss for Score questions
- confidence/correctness calibration regularization
- held-out temperature scaling
- optional verifier-reward fine-tuning for programmatically checkable decisions

Verifier-reward support includes both exact expected-reward optimization when all legal actions can be scored and a sampled policy-gradient fallback when only the chosen action can be verified.

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

One JSONL row contains one state, one or more typed questions, and optional targets:

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

Hard labels and probability distributions are both first-class targets.

## Train

Start with the head-only smoke run:

```bash
python -m my_jev.train \
  --train examples/train.jsonl \
  --valid examples/valid.jsonl \
  --output runs/smoke \
  --epochs 1 \
  --freeze-backbone
```

Then fine-tune the full encoder:

```bash
python -m my_jev.train \
  --train data/train.jsonl \
  --valid data/valid.jsonl \
  --output runs/system-one-v0 \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 8 \
  --bf16
```

## Evaluate

```bash
python -m my_jev.evaluate \
  --checkpoint runs/system-one-v0/best \
  --data data/test.jsonl
```

Primary release metrics are accuracy, NLL, Brier score, expected calibration error (ECE), selective accuracy/coverage, and latency/decisions per second.

## Inference

```bash
my-jev decide \
  --checkpoint runs/system-one-v0/best \
  --input examples/support_ticket.json
```

The response is typed JSON containing the legal options, probabilities, selected option, confidence, and primitive-specific values such as `noul=P(true)` or normalized `score`.

## Roadmap

The first milestone is deliberately about validating the learning problem before chasing exotic kernels:

1. establish real train/calibration/test datasets
2. train ModernBERT-base baseline and measure calibration
3. add synthetic/teacher probability generation
4. run verifier-reward fine-tuning
5. benchmark against structured-output LLM baselines
6. fuse state K/V reuse and candidate attention for speed
7. distill into smaller encoders and test quantized inference
8. build workflow evals modeled after real application decision graphs

## Public references

- TypeSafe AI — Introducing System One Models and Jev: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- TypeSafe workflow evals: https://evals.typesafe.ai/
- LangChain — Building a Harness with Jev: https://www.langchain.com/blog/building-a-harness-with-jev
- ModernBERT: https://arxiv.org/abs/2412.13663

## Status

Trainable v0 scaffold is implemented on `feature/system-one-v0`. The next engineering slice is dataset generation + a reproducible baseline training/evaluation run.
