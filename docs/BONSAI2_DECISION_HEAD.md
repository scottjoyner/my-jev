# Bonsai 2 decision-attention lane

This lane adds a typed System-One decision head on top of the frozen ternary
Bonsai 2 27B language backbone.

It is a new provider lane. It does not replace the ModernBERT baseline and it
does not widen Hermes/AssistX runtime authority.

## Why this model is a good fit

Ternary Bonsai 2 27B is derived from Qwen3.8-27B with the language architecture
unchanged. The text backbone is 64 blocks with hidden width 5120.

The PrismML llama.cpp fork already contains staging APIs used by speculative
decoding to expose selected transformer layer input embeddings and the final
pre-norm hidden state.

That gives us a practical frozen-backbone path. We do not need gradients through
the ternary weights and we do not need to expand the GGUF weights back to FP16.

## v0 architecture

State path:

    typed DecisionRecord
      -> state prompt
      -> frozen Ternary Bonsai 2 27B / PrismML llama.cpp
      -> layer input 31 [tokens, 5120]
      -> layer input 47 [tokens, 5120]
      -> layer input 63 [tokens, 5120]
      -> final pre-norm [tokens, 5120]
      -> learned tap mixture
      -> token state memory

Option path:

    typed question + option
      -> same frozen Bonsai runtime
      -> same four activation taps
      -> masked-mean option vector per tap
      -> learned tap mixture

Both paths feed ExternalDecisionAttentionHead:

    5120 hidden -> rank-256 Q/K/V -> typed option logits/probabilities

The state path retains token-level memory. The option path is pooled to one
vector per runtime option at each tap.

The head owns only the trainable tap weights, normalization, Q/K/V projections,
and a small interaction gate. The Bonsai backbone remains frozen.

## Why activations are streamed

Persisting full-width token activations is too expensive for a serious training
corpus. At 5120 dimensions, several taps across a 2K-token state can consume
tens of megabytes per record even at FP16.

The v0 contract therefore uses a streaming activation provider:

1. llama.cpp evaluates a bounded batch of state/option prompts.
2. Selected hidden states are copied into host memory.
3. The Python head consumes the batch.
4. Gradients update only the decision head.
5. Full backbone activations are discarded after the step.

Small bounded caches may be added later for repeated validation/calibration
records, but the default contract does not persist the full corpus.

## Checked-in contract

configs/decision_backbones/bonsai2-27b.toml pins:

- model: prism-ml/Ternary-Bonsai-2-27B-gguf
- base architecture: Qwen3.8-27B / qwen35
- hidden size: 5120
- transformer blocks: 64
- taps: layer inputs 31, 47, 63 plus final pre-norm
- state representation: token memory
- option pooling: masked mean
- decision rank: 256
- frozen backbone
- evidence-only/non-dispatch authority boundary

src/my_jev/decision_attention.py contains the runtime-neutral trainable head.

src/my_jev/bonsai_contract.py contains the provider contract.

## Runtime capability probe

Before writing or compiling the native extractor, point the probe at the local
PrismML llama.cpp checkout:

    my-jev-bonsai-probe \
      --llama-root /path/to/PrismML-Eng/llama.cpp \
      --model-path /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
      --output runs/bonsai2/runtime-probe.json

The probe fails closed unless the checkout contains all four staging symbols:

    llama_set_embeddings_layer_inp
    llama_get_embeddings_layer_inp
    llama_set_embeddings_nextn
    llama_get_embeddings_nextn

The model path is optional during source-only development and required when
validating a real host.

## Native bridge: implemented path

The native bridge is now checked in under
`native/bonsai_bridge/bonsai_activation_bridge.cpp`.

It is built against the exact PrismML llama.cpp checkout selected by the
operator and performs the following sequence:

1. load the exact Bonsai 2 GGUF;
2. verify hidden width 5120 and 64 transformer blocks;
3. force causal attention for decision prompts;
4. enable layer-input extraction for 31, 47, and 63;
5. enable final pre-norm extraction;
6. tokenize/evaluate state prompts without autoregressive generation;
7. copy all selected state token activations;
8. evaluate every typed question/option prompt with the same tokenizer/runtime;
9. masked-mean the option-token activations per tap;
10. emit `my-jev-activation-frame-v1` with model/runtime/prompt provenance.

The bridge writes float32 activations because PrismML exposes the staging
buffers as float rows. The Python head may later autocast internally, but the
runtime transport remains explicit and lossless for the first physical
equivalence work.

`BonsaiNativeProvider` hashes the GGUF, resolves the exact PrismML git SHA,
constructs the same typed option prompts used by the ModernBERT lane, invokes
the bridge, and rejects any returned frame whose provider, model SHA, runtime
SHA, prompt contract, hidden shape, tap count, or typed question groups differ
from the expected record.

The bridge is compiled in CI with `g++ -fsyntax-only` against a stubbed version
of the PrismML staging API. Real runtime compatibility still has to be proven
against the actual PrismML checkout and Bonsai GGUF.

### One-command physical smoke

On a host with the PrismML checkout and Bonsai GGUF present:

```bash
bash scripts/run-bonsai-native-smoke.sh \
  /path/to/PrismML-Eng/llama.cpp \
  /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf
```

A specific single-record JSON/JSONL file and output directory may be supplied
as the third and fourth arguments. Without a record argument the script creates
one deterministic synthetic AssistX policy record.

The smoke path performs:

```text
runtime capability probe
        ->
native bridge build
        ->
exact GGUF / exact runtime activation capture
        ->
activation-frame validation
        ->
untrained ExternalDecisionAttentionHead forward pass
        ->
end-to-end smoke receipt
```

The final receipt is
`bonsai-native-smoke-receipt.json`. It proves transport/head compatibility
only; the head is intentionally random and makes no decision-quality claim.

The physical acceptance target remains:

- exact model SHA-256;
- exact PrismML git SHA;
- 5120-wide activations at all four configured taps;
- deterministic typed prompt contract;
- one complete typed decision batch through the external head;
- `dispatch_allowed=false`;
- `runtime_authority_changed=false`.

## Training progression

### Phase A — bridge equivalence

Use a tiny deterministic corpus and prove:

- repeated captures are numerically stable
- option order does not alter aligned option scores
- state shuffling materially changes scores
- activation tap shapes match the contract
- no state/option rows are silently truncated without evidence

### Phase B — frozen-head smoke run

Train only the decision head on 1K-5K synthetic policy states.

Success here means the head:

- learns above uniform
- reacts to shuffled state
- calibrates
- preserves typed policy consistency
- remains small enough to load beside Bonsai on the target machine

### Phase C — Bonsai policy baseline

Reuse the same group-safe AssistX train/validation/calibration/test contract as
the ModernBERT lane, but keep Bonsai frozen and compare:

- decision accuracy/NLL/Brier/ECE
- state sensitivity
- Choice-order invariance
- policy-consistency violation rate
- activation extraction throughput
- head latency and memory
- total resident model + head footprint

### Phase D — heterogeneous fleet

Once the Bonsai contract is stable, smaller machines can implement the same
provider protocol with smaller frozen backbones.

Every provider must emit the same typed probabilities and run through the same
deterministic resolver. Hardware/model differences are evidence and placement
inputs, not authority changes.

A future fleet can therefore contain:

    higher-capability nodes:
        Bonsai 2 27B + decision attention head

    mid-tier nodes:
        smaller causal/encoder decision backbone + same typed contract

    low-power nodes:
        compact distilled decision model + same typed contract

The resolver can choose which decision provider to query based on hardware,
latency, locality, and confidence, while Hermes/AssistX retains final execution
authority.

## Explicit non-goals for v0

Do not:

- fine-tune the ternary GGUF weights
- silently substitute stock llama.cpp if it cannot execute Bonsai correctly
- train against generated Bonsai answers as self-labels
- persist an unbounded full-activation corpus
- give the head direct tool or scheduler authority
- combine the Bonsai lane with fleet-placement training until the AssistX
  decision behavior is independently measured

## First physical milestone

The first real Bonsai milestone is not model training.

It is an exact-runtime activation receipt proving:

    exact Bonsai GGUF SHA
    + exact PrismML llama.cpp SHA
    + causal forward pass
    + 5120-wide captures at all four taps
    + deterministic state/option tokenization
    + one typed decision batch through ExternalDecisionAttentionHead
    + dispatch_allowed=false

Once that exists, the remaining work becomes ordinary head training,
calibration, benchmarking, shadow replay, and promotion using the infrastructure
already built in this repository.
