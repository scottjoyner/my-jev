# my-jev

An open-source, independently implemented **System-One-style typed decision model** for fast, calibrated decisions without autoregressive text generation.

The primary target is now the **Hermes / AssistX agent policy layer**: deciding whether a turn warrants chat, durable requirements/tasks, bounded tool action, clarification, cancellation, or abstention before a larger reasoning/tool loop takes over.

> This project is not Jev, is not affiliated with TypeSafe AI, and does not claim to reproduce TypeSafe's proprietary architecture, sampler, weights, datasets, or RLCD implementation.

## Hermes / AssistX target

One model call produces a policy vector such as:

```text
route = P(chat, create_tasks, act, clarify, cancel, abstain)
needs_tools = P(true)
needs_task_graph = P(true)
context_sufficient = P(true)
external_effect = P(true)
approval_likely = P(true)
action_scope = P(none, read_only, local_write, external_side_effect, privileged)
risk = P(low, moderate, high, critical)
delegation = P(none, self, single_agent, multi_agent)
response_depth = P(brief, normal, structured, project)
```

The learned model identifies intent, scope, risk, and work shape. It **never grants execution authority**. A deterministic resolver combines those probabilities with live Hermes/AssistX facts such as speaker verification, action permissions, approval availability, and active work.

Existing Hermes approval, tool-guardrail, claim/fencing, and AssistX mutation-authority systems remain authoritative.

See [docs/ASSISTX_POLICY.md](docs/ASSISTX_POLICY.md).

## Architecture

The default backbone is `answerdotai/ModernBERT-base`.

The default `option_query` decision head follows a useful pattern demonstrated by the independent MIT-licensed [vinnylarouge/jevlike](https://github.com/vinnylarouge/jevlike) project:

1. encode application state once;
2. encode every question/option candidate in one batch;
3. project each candidate into a query;
4. let all candidate queries attend over the shared state token memory;
5. produce one scalar logit per legal option;
6. normalize only across sibling options for each typed question.

The state tensor stays batched instead of being copied once per candidate. The older cross-attention head remains loadable as `legacy` for checkpoint compatibility.

The model never decodes vocabulary tokens during inference.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Typed primitives

- **Noul** — binary probability.
- **Choice** — probability distribution over caller-defined options.
- **Score** — ordered probability distribution plus an expected score.

Hard labels and probability distributions are both first-class training targets.

## Install

```bash
git clone https://github.com/scottjoyner/my-jev.git
cd my-jev
git switch feature/system-one-v0

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Agent-policy bootstrap

Generate deterministic, verifier-labeled policy states:

```bash
my-jev-agentic-synth \
  --output data/assistx-policy-v1.jsonl \
  --records 50000 \
  --seed 23
```

The synthetic generator contains paired **authority counterfactuals**: the same user intent is emitted once with runtime authority and once without it. The semantic labels remain identical. This prevents the learned policy from treating "permission exists" as "the user intended an action"; permission belongs to the deterministic resolver.

Create leakage-resistant train / validation / calibration / test splits:

```bash
my-jev-split \
  --input data/assistx-policy-v1.jsonl \
  --output-dir data/assistx-policy-v1 \
  --train-fraction 0.70 \
  --validation-fraction 0.10 \
  --calibration-fraction 0.10 \
  --group-key auto
```

Related counterfactual records share a `family_id` and therefore stay in the same split. Every dataset and split receives reproducibility metadata and SHA-256 hashes.

## Train

First head-only smoke run:

```bash
my-jev-train \
  --train data/assistx-policy-v1/train.jsonl \
  --valid data/assistx-policy-v1/validation.jsonl \
  --output runs/assistx-policy-head \
  --epochs 1 \
  --freeze-backbone
```

Then fine-tune the full encoder:

```bash
my-jev-train \
  --train data/assistx-policy-v1/train.jsonl \
  --valid data/assistx-policy-v1/validation.jsonl \
  --output runs/assistx-policy-modernbert \
  --head-kind option_query \
  --head-rank 256 \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 8 \
  --max-state-length 2048 \
  --gradient-checkpointing \
  --bf16
```

Every run preserves its effective configuration beside the checkpoint.

## Calibrate

Calibration is fit only on the held-out calibration partition:

```bash
my-jev-calibrate \
  --checkpoint runs/assistx-policy-modernbert/best \
  --data data/assistx-policy-v1/calibration.jsonl \
  --output runs/assistx-policy-modernbert/calibration.json
```

## Benchmark

The benchmark compares the model with both a uniform baseline and a **shuffled-state control**:

```bash
my-jev-benchmark \
  --checkpoint runs/assistx-policy-modernbert/best \
  --data data/assistx-policy-v1/test.jsonl \
  --calibration runs/assistx-policy-modernbert/calibration.json \
  --output runs/assistx-policy-modernbert/test-benchmark.json
```

It reports:

- accuracy, NLL, Brier score, and ECE;
- per-question, per-type, and per-domain metrics;
- uniform-baseline delta;
- shuffled-state accuracy delta;
- mean KL divergence between correct-state and wrong-state predictions;
- median and p95 batch latency;
- states/sec and decisions/sec;
- exact test-dataset manifest and SHA-256.

A router that stays accurate when its state is shuffled is not considered successful; it is probably learning priors or shortcuts.

## Generic synthetic data

The repo also retains a multi-domain operations/support/build generator for architecture experiments:

```bash
my-jev-synth \
  --output data/synthetic-v1.jsonl \
  --records 50000
```

## Teacher/bootstrap labels

An OpenAI-compatible teacher endpoint can produce bounded soft targets:

```bash
my-jev-label \
  --input data/unlabeled.jsonl \
  --output data/labeled.jsonl \
  --endpoint http://localhost:1234/v1/chat/completions \
  --model YOUR_TEACHER_MODEL \
  --samples 5
```

Teacher confidence is weak supervision, not empirical calibration truth. Prefer verifiable outcomes, observed real-world outcomes, adjudicated labels, and repeated user/operator corrections whenever available.

## Training objective

The supervised objective combines:

- categorical NLL / soft-target cross entropy;
- Brier loss;
- ordinal earth-mover loss for Score questions;
- a small confidence/correctness calibration regularizer.

A separate verifier-reward module supports exact expected-reward optimization when every legal option can be scored and a sampled policy-gradient fallback when only the selected action can be verified. This is **RLCD-inspired**, not an implementation of TypeSafe's undisclosed RLCD.

## Real Hermes trajectory loop

Synthetic data is only bootstrap supervision. The intended next stage is:

```text
real Hermes/AssistX turn
        |
        v
policy state + model distribution
        |
        v
deterministic resolver
        |
        v
chat / task graph / action / clarify
        |
        v
tools + approvals + outcome + user correction
        |
        v
trajectory dataset
        |
        +----> next supervised / DAgger-style policy iteration
```

Do not train on hidden chain-of-thought. Train on observable state, decision, action, approval, outcome, verification evidence, and user/operator correction.

## Public prior art and references

- TypeSafe AI — Introducing System One Models and Jev: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- TypeSafe workflow evals: https://evals.typesafe.ai/
- Jevlike — independent MIT-licensed one-pass option scorer: https://github.com/vinnylarouge/jevlike
- ModernBERT: https://arxiv.org/abs/2412.13663

Jevlike is used as prior art and architectural inspiration. `my-jev` keeps an independent typed multi-question implementation and does not require Jevlike at runtime.

## Status

Draft PR #1 contains the trainable v0 stack plus the Hermes/AssistX agent-policy contract. The next hard milestone is an exact-SHA agent-policy training run followed by shadow evaluation against real Hermes/AssistX traces.
