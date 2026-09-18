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
3. Fuse state and candidate representations with a learned decision scorer.
4. Normalize only across the options valid for each question.
5. Return typed probabilities; no vocabulary decoding and no autoregressive generation.

This is deliberately a practical open architecture, not a guess at TypeSafe's undisclosed internals.

## Training objective

The project will optimize decision quality and calibration together:

- categorical negative log likelihood
- Brier score
- soft-label KL divergence when probabilistic targets are available
- ordinal/earth-mover loss for Score questions
- calibration regularization
- post-hoc temperature scaling on held-out calibration data
- optional verifier-reward fine-tuning for programmatically checkable decisions

Primary evals:

- accuracy / macro F1
- NLL
- Brier score
- expected calibration error (ECE)
- calibration curves
- selective accuracy vs. coverage
- latency and decisions/sec

## Public references

- TypeSafe AI — Introducing System One Models and Jev: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- TypeSafe workflow evals: https://evals.typesafe.ai/
- LangChain — Building a Harness with Jev: https://www.langchain.com/blog/building-a-harness-with-jev
- ModernBERT: https://arxiv.org/abs/2412.13663

## Status

Bootstrapping the first trainable implementation on `feature/system-one-v0`.
