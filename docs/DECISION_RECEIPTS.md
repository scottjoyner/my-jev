# Provider-neutral decision receipts

`system-one-decision-receipt-v1` is the evidence envelope for comparing typed
decision providers without granting them runtime authority.

## Purpose

A local `my-jev` checkpoint, the Bonsai decision-attention lane, and future
reference/hosted providers should be comparable on the same decision state and
typed question schema.

The receipt binds:

- model-visible state hash;
- typed question-schema hash;
- legal candidate-set hash;
- provider/model identity;
- model artifact hash when available;
- inference latency;
- every per-question probability distribution;
- selected option and primitive-specific value;
- deterministic resolver disposition when one exists;
- literal-false authority fields.

It deliberately does **not** contain permission to act.

## Authority contract

Every valid receipt contains:

```json
{
  "evidence_only": true,
  "authority": {
    "dispatch_allowed": false,
    "approval_granted": false,
    "claim_acquired": false,
    "mutation_allowed": false,
    "routing_authority_changed": false
  }
}
```

The Pydantic schema uses literal false values and rejects extra fields. A caller
cannot turn a receipt into an authorization token by setting one of these fields
to true.

## Hash boundaries

`input_sha256`
: Hash of the model-visible state.

`question_schema_sha256`
: Hash of question names, primitive types, instructions, and options.

`candidate_set_sha256`
: Hash of question names and legal options only. This makes dynamic candidate
  changes easy to identify independently of instruction wording.

`response_sha256`
: Hash of normalized per-question decision results.

Training targets and `DecisionRecord.metadata` are excluded from model-input
hashes because they are not visible to inference.

## Policy-service integration

`/v1/agent-policy` remains `assistx-agent-policy-v1` and receives one new
additive field:

```text
decision_receipt
```

The current local runtime records:

- provider `my-jev`;
- policy-contract provider version;
- checkpoint identifier;
- checkpoint `model.pt` SHA-256 when available;
- device and temperature;
- measured decision-inference latency;
- resolver disposition.

The resolver disposition is evidence about what deterministic policy produced.
It does not change the receipt's authority fields.

## Comparison-provider rule

A future hosted/reference provider must emit the same receipt shape before its
outputs are compared with local candidates.

External providers remain opt-in and shadow/evidence-only:

- no silent provider switching;
- no automatic external fallback after local failure;
- no private state transmission without explicit approval;
- no routing/admission/dispatch/mutation changes;
- full probability vectors must remain available;
- provider/model/version identity must be recorded.

## Next fixtures

The next benchmark slice should cover:

1. multi-question routing in one request;
2. low-confidence review/abstention;
3. legal-candidate restriction;
4. question option-order changes;
5. overlapping or incomplete Choice schemas;
6. ambiguous Score rubrics;
7. reference-provider disagreement.

These fixtures should feed the same receipt contract so local ModernBERT,
Bonsai, and future providers can be compared without changing runtime authority.
