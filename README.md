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

The same calibrated typed-decision approach is being explored for fleet placement as an advisory layer over authoritative node eligibility, health, leases, and dispatch. See [docs/FLEET_POLICY.md](docs/FLEET_POLICY.md).

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

Prepare deterministic, verifier-labeled policy states and leakage-resistant
train / validation / calibration / test splits in one command:

```bash
my-jev-prepare \
  --output-dir data/assistx-policy-v1 \
  --records 50000 \
  --seed 23
```

The corpus contains paired **authority counterfactuals** and **conversation-context
counterfactuals**. Related records share a `family_id` and are kept in the same
partition. The command writes a source corpus, all four splits, SHA-256 manifests,
and a top-level preparation manifest; it refuses to overwrite an existing corpus
unless `--force` is explicit.

For lower-level experiments, `my-jev-agentic-synth` and `my-jev-split` remain
available independently.

## Fleet-placement bootstrap

The fleet lane is a separate observer-only task family. The encoder never sees
node IDs or provenance metadata; it sees workload requirements plus anonymous
health/capacity facts. A deterministic resolver filters health, freshness,
drain state, capability, RAM/VRAM capacity, and pinned locality before any
learned preference can rank a node.

Prepare verifier-labeled counterfactual data with group-safe splits:

```bash
my-jev-fleet-prepare \
  --output-dir data/fleet-placement-v1 \
  --records 12000 \
  --seed 31
```

The corpus contains three-record counterfactual families: a base state, a
node-permutation variant with identical targets, and a semantic variant such as
preferred-node drain, stale pinned health, VRAM capacity crossing, or checkpoint
support removal where the verified target deliberately changes.

Run the independent fleet experiment lane with:

```bash
my-jev-experiment \
  configs/experiments/fleet-modernbert.toml \
  --dry-run
```

The fleet resolver may return an observer-only recommended eligible node, but
always sets `dispatch_allowed=false`; scheduler leases, claims, health, and
dispatch remain outside model authority.

## Reproducible experiments

The preferred training path is now the in-repo experiment runner:

```bash
my-jev-doctor --require-gpu --require-bf16

my-jev-experiment \
  configs/experiments/assistx-modernbert.toml \
  --dry-run

my-jev-experiment \
  configs/experiments/assistx-modernbert.toml
```

For the Qwen3.5 causal scalar lane, install `.[causal]`, require the causal
preflight, and use `configs/experiments/assistx-qwen35-scalar.toml`.

The runner locks shared resources, hashes every dataset split, trains, calibrates,
benchmarks, applies absolute and parent-run regression gates, and records lineage
in the experiment registry. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

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

## Real shadow adjudication

Shadow predictions and the legacy router are evidence, not training truth. Convert
captured AssistX shadow rows to unlabeled policy states first:

```bash
my-jev-shadow-import \
  --input data/shadow-export.jsonl \
  --output data/shadow-unlabeled.jsonl
```

For a newly trained checkpoint, replay the captured states without dispatching
or mutating Hermes/AssistX, then rank the highest-value states for review:

```bash
my-jev-shadow-replay \
  --input data/shadow-export.jsonl \
  --output data/shadow-replay.jsonl \
  --checkpoint runs/assistx-policy-modernbert/best \
  --calibration runs/assistx-policy-modernbert/calibration.json

my-jev-review \
  --input data/shadow-replay.jsonl \
  --output data/shadow-review-queue.jsonl \
  --limit 200 \
  --min-priority 0.25
```

The review score prioritizes explicit correction evidence first, then policy
consistency violations, disagreement with the legacy AssistX routing evidence,
and calibrated model uncertainty. Replay rows preserve the original normalized
policy state and candidate probability vector so uncertainty is measured rather
than inferred from top-1 choices. The queue remains unlabeled and carries an
auditable `active_review` rank/reason block in metadata; it cannot enter the
training mix until it is adjudicated.

Then attach operator, outcome-verifier, or user-correction labels:

```json
{"source_intent_id":"intent-7","source":"operator","labels":{"route":"act","needs_tools":true,"risk":"moderate"}}
{"source_intent_id":"intent-7","source":"outcome_verifier","weight":2.0,"labels":{"route":"act","needs_tools":true}}
```

```bash
my-jev-adjudicate \
  --input data/shadow-unlabeled.jsonl \
  --annotations data/shadow-annotations.jsonl \
  --output data/shadow-adjudicated.jsonl
```

Annotations may label only the fields that the evidence actually supports.
Repeated independent labels are aggregated into soft probability targets, so
disagreement is preserved instead of being collapsed to a hard majority label.
Unknown questions, invalid options, and unmatched intent IDs fail closed by
default. The resulting corpus can be group-split and mixed into the next
supervised/DAgger-style policy iteration.

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

Draft PR #1 contains the typed Hermes/AssistX policy contract, ModernBERT and Qwen3.5 scalar model lanes, reproducible experiment/promotion infrastructure, shadow-import/adjudication tooling, and provenance-safe mixed-data preparation. The next hard milestone is the first exact-SHA GPU baseline run followed by shadow evaluation against real Hermes/AssistX trajectories.
