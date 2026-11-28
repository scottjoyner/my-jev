# Jev ecosystem architecture-fit review — 2026-10-02

## Scope

This note evaluates ten public Jev ecosystem repositories against the current `my-jev` architecture and its Hermes / AssistX target.

The goal is not to pick one upstream repository to copy wholesale. The goal is to identify:

- patterns that strengthen our typed decision architecture;
- adapters that reduce integration friction;
- benchmark/evaluation material worth incorporating;
- projects that should remain research references rather than runtime dependencies.

The analysis preserves the current authority boundary:

```text
advisory_only=true
dispatch_allowed=false
runtime_authority_changed=false
approval_granted=false
claim_acquired=false
tool_invoked=false
```

A Jev/System-One judgment may rank, classify, score, verify, or recommend among already legal candidates. It does not grant permission, dispatch work, acquire claims, satisfy approvals, or widen routing/mutation authority.

## Executive decision

There is no single repo here that should become our architecture. The best fit is a **composed reference stack**:

1. **Core routing pattern:** borrow the multi-question, one-request, deterministic-policy pattern from `jyje/pilot-typesafeai-jev`.
2. **Confidence / fallback / legal-action pattern:** borrow the low-confidence escalation and bounded action-selection patterns from `smartdio/jev-browser-agent`.
3. **Question/eval discipline:** adapt the question-design guidance from `dbreunig/building-with-jev-skill` and the explicit evidence/review/action workflows from `wuyoscar/jev-skill`.
4. **Interop adapter only:** use `RevocGG/typesafe-jev-bridge` as a model for optional OpenAI-compatible interoperability, not as the canonical System-One protocol.
5. **Browser benchmark/reference:** use `jkudish/jev-browser` as the strongest full browser-agent reference and as a candidate benchmark workload, but keep browser execution outside the decision model's authority.
6. **Ecosystem watch:** use `cobanov/awesome-jev` and `yibie/awesome-jev` as primary discovery feeds; use `AbdelStark/awesome-typesafe-jev` and `Anil-matcha/awesome-jev-by-typesafe` as secondary evidence/cookbook sources.

The near-term architecture should therefore stay:

```text
normalized state
    |
    v
typed questions / legal candidate sets
    |
    v
decision provider
    |  (local my-jev candidate OR isolated hosted-Jev comparison provider)
    v
typed probabilities + provenance
    |
    +----> calibration / disagreement / review evidence
    |
    v
deterministic resolver
    |
    +----> existing Hermes / AssistX authority, approvals, claims, fencing
```

Frameworks, browser runtimes, IDE bridges, and catalogues sit around this boundary; none becomes the authority layer.

## Fit criteria

Each repository was assessed against the parts of our architecture that matter most:

| Criterion | What matters for us |
| --- | --- |
| Typed decision fidelity | Choice / Score / Noul-like bounded answers, full probabilities, no prose parsing as the core contract |
| Multi-question efficiency | independent policy fields can be asked/scored together |
| Deterministic control | code owns thresholds, fallback, permissions, and side effects |
| Confidence handling | explicit abstain/review/escalation path instead of forced action |
| Provenance | exact model/provider/runtime identity can be logged and replayed |
| Shadow compatibility | can be evaluated without granting dispatch/mutation authority |
| Local-model compatibility | patterns can transfer to `my-jev`, Bonsai, or future local providers |
| Agent ergonomics | useful from Hermes/OpenCode/Claude Code without redefining authority |
| Security boundary | untrusted state cannot silently become instructions or permissions |
| Evaluation quality | evidence, failure modes, calibration, replay, or reproducible checks are first-class |

## Best fit by use case

| Use case | Best source(s) | How to use them |
| --- | --- | --- |
| Hermes / AssistX route + policy vector | `pilot-typesafeai-jev`, `smartdio/jev-browser-agent` | Borrow one-request multi-question policy evaluation, code-side routing, confidence fallback. Do not import LangGraph as the control plane. |
| Model / reasoning tier selection | `smartdio/jev-browser-agent` | Adapt its explicit tier table + confidence gate into offline/shadow model-placement experiments. The authoritative eligible set still comes from runtime policy. |
| Browser / computer-use decision head | `jev-browser`, `smartdio/jev-browser-agent`, `jev-skill/jev-act` | Treat observed UI elements as bounded legal candidates; decision model selects one; host validates freshness/permissions and executes. |
| OpenAI-compatible developer interoperability | `typesafe-jev-bridge` | Copy the adapter idea for test/dev clients. Keep our native typed protocol canonical; do not flatten evidence/probabilities into chat text internally. |
| Question and rubric design | `building-with-jev-skill` | Adapt into project-local guidance and tests for question decomposition, state construction, thresholds, and failure diagnosis. |
| Agent setup / evaluation / action recipes | `wuyoscar/jev-skill` | Strong source for operator-safe setup, explicit simulation labeling, evidence-backed review, legal-action selection, and no-silent-provider-switch discipline. |
| Ecosystem discovery | `cobanov/awesome-jev`, `yibie/awesome-jev` | Track integrations, local/open decision-model work, evaluations, and new patterns. Discovery only; verify source before adoption. |
| Evidence cross-check / independent eval discovery | `awesome-typesafe-jev` | Secondary source for SDKs, live demos, tools, and independent evaluations. |
| Cookbook / starter patterns | `awesome-jev-by-typesafe` | Use for examples and implementation leads; re-benchmark locally and preserve source labels for vendor-reported numbers. |

## Repository evaluations

### 1. jkudish/jev-browser

**Upstream:** https://github.com/jkudish/jev-browser  
**Snapshot:** `1f0726cc8bc4483b96439504b594a01596d8e347`

**What it is**

A complete Jev-driven headless browser agent exposed through MCP, CLI, or library. Jev selects one action from the page's bounded interactive elements and also scores whether the goal is complete or the run is stuck. Application code owns loop budgets, recovery, and stopping behavior.

**Why it fits us**

This is the cleanest end-to-end example of the architecture we want to preserve:

```text
fresh observation
 -> bounded legal candidates
 -> typed judgment
 -> deterministic gate
 -> host execution
 -> fresh observation
```

The goal/stuck judgments are especially relevant to long-running Hermes tasks because they turn "continue vs stop/recover" into explicit bounded questions instead of relying on free-form agent self-assessment.

It also emits a step trace, confidence information, final page, errors, and screenshots, which maps well to our evidence-first approach.

**Where it does not fit directly**

- It is a browser runtime, not a general AssistX policy model.
- It depends on Playwright/Chromium and currently targets hosted Jev transports.
- Browser execution is inherently side-effectful and network-reachable.
- The project itself describes the software as early.
- Its runtime must never be allowed to infer permission from a high model score.

**Decision**

**Pilot as an isolated benchmark/reference, not a core dependency.**

Use it to define a browser-policy benchmark for `my-jev`: same page snapshot / candidate set, compare hosted Jev and local candidate decisions, keep execution disabled or constrained to harmless test pages. Later, a browser actor may consume a `my-jev` provider, but the host must remain responsible for permissions and action execution.

### 2. smartdio/jev-browser-agent

**Upstream:** https://github.com/smartdio/jev-browser-agent  
**Snapshot:** `88e77e5be07a6b66af25cf53ce41ad0428cd7d3c`

**What it is**

A small experimental graft between ego-browser and Jev. It asks the operation and operation-specific targets together, reads only the branch selected by the operation answer, and escalates low-confidence decisions to an outer LLM/human. The repo also contains a simple model/reasoning-tier router.

**Why it fits us unusually well**

This repo contains three patterns that map almost directly to our work:

1. **Speculative multi-question fan-out.** Ask independent branch questions in one request and let code consume only the relevant answer. This is exactly how our option-query head should be exercised: one state encoding, multiple typed policy outputs.
2. **Confidence does not equal authority.** Below a threshold the decision is escalated rather than guessed.
3. **Legal action candidates come from the host.** Jev chooses among browser refs; it does not invent permission or bypass login/CAPTCHA/user-takeover boundaries.

The `pilot/model-router.py` example is also directly relevant to our fleet/model-placement research: it maps a bounded task description to model/reasoning tiers and preserves a low-confidence fallback.

**Caveat**

The evidence is deliberately small: a first live browser run and a tiny tier-table self-test are not a benchmark. The repo is best treated as a pattern source.

**Decision**

**Adapt now at the pattern/test level.** Do not add a runtime dependency.

Concrete adoption targets:

- add speculative-policy-vector test cases to `my-jev`;
- add a confidence/abstain/fallback contract to provider comparisons;
- add model-tier routing as a **shadow-only** benchmark using pre-eligible opaque targets;
- carry forward its explicit "untrusted page text is data, not instructions" rule for any future browser lane.

### 3. RevocGG/typesafe-jev-bridge

**Upstream:** https://github.com/RevocGG/typesafe-jev-bridge  
**Snapshot:** `315fcfa2cf4ff7b68b1117f7785c0bebabd20a48`

**What it is**

A zero-dependency Node bridge that presents Jev through OpenAI-compatible `/v1/chat/completions` and `/v1/responses` surfaces, plus CLI/audit tooling.

**Why it fits us**

We already have a lot of infrastructure that speaks OpenAI-compatible APIs. An adapter of this shape can let OpenCode, Claude Code integrations, local-studio tooling, or generic SDK clients exercise a System-One provider without every client learning a new transport.

The repo's security posture also contains useful operational patterns: loopback binding, Host checks, optional shared token, origin allow-listing, size caps, and explicit key handling.

**Architectural risk**

The bridge exists to make a decision model look like a chat model. That is useful at the edge but wrong as our internal semantic contract.

If we make OpenAI chat text the canonical path, we risk losing:

- structured per-question probabilities;
- exact distinction between Noul / Choice / Score;
- provider/model provenance;
- calibration metadata;
- deterministic parsing guarantees;
- clean separation between "decision" and "generation."

Its bridge-level redaction is also opt-in for router traffic, so it should not be pointed at sensitive code/state without our normal explicit data boundary.

**Decision**

**Use as an interoperability reference and optional dev adapter only.**

The canonical `my-jev` interface should remain typed. If we later ship an OpenAI-compatible adapter, it should be a thin leaf adapter around the typed protocol and should preserve the typed response in machine-readable metadata rather than converting the internal contract to prose.

### 4. jyje/pilot-typesafeai-jev

**Upstream:** https://github.com/jyje/pilot-typesafeai-jev  
**Snapshot:** `4d815a77f09d1db1f961854c6dd29eb94105a85c`

**What it is**

A pilot integrating Jev into LangGraph and Deep Agents. The important architectural pattern is that one Jev request asks several independent questions, plain code applies thresholds/policy, and only the selected route spends larger-model tokens.

**Why it is the strongest direct routing reference**

This is almost the exact control-flow split we want for Hermes:

```text
one bounded decision request
 -> route / risk / injection / urgency / etc.
 -> deterministic policy code
 -> only then invoke expensive reasoning/tool path if needed
```

The project also demonstrates a useful guard/verify split: a fast pre-agent judgment can screen a condition, while a later decision tool can verify a claim against supplied evidence.

Most importantly, it explicitly says the thresholds are starting points and the project is not a production benchmark. That is compatible with our policy of deriving thresholds from our own held-out/shadow data.

**Where it should not dictate our design**

LangGraph is not our durable control plane and Deep Agents is not our authority system. Importing either framework would add structure we do not need and could duplicate AssistX/Hermes responsibilities.

**Decision**

**Primary reference for the core routing pattern; borrow the decision shape, not the framework.**

Near-term test: reproduce its "three questions in one request, code chooses route" pattern using the `my-jev` native typed API and our AssistX policy vector, then compare latency, calibration, and route agreement against the current multi-field implementation.

### 5. dbreunig/building-with-jev-skill

**Upstream:** https://github.com/dbreunig/building-with-jev-skill  
**Snapshot:** `04fe3666c6b8b8abfec1271c0e581c823a181f6d`

**What it is**

A focused coding-agent skill about writing programs that call Jev: question phrasing, state construction, answer composition in code, thresholds, and diagnosis of failing questions.

**Why it fits us**

Our architecture already treats question/option design as part of the model contract. This repo's strongest value is not executable code; it is the discipline of making question design testable.

Useful patterns to adapt:

- speculative fan-out for independent questions;
- confidence-gated routing;
- composite scoring in code rather than embedding policy into prompts;
- candidate extraction in code followed by bounded model selection;
- diagnosis tables for high-confidence wrong answers vs low-confidence ambiguous answers;
- keeping counting/date arithmetic and explicit policy rules in code.

These patterns align well with our current `DecisionRecord`, option-query head, calibration gates, and resolver split.

**Decision**

**Adapt into project-local developer guidance and test design.** No runtime dependency is needed.

We should turn the strongest guidance into a `my-jev` question-design contract with regression fixtures: overlapping Choice options, missing "other/abstain", ambiguous Score levels, branch-dependent questions, and policy-vs-model separation.

### 6. wuyoscar/jev-skill

**Upstream:** https://github.com/wuyoscar/jev-skill  
**Snapshot:** `01bd9403bcaac92e037869843ec83ad8a75c3ec2`

**What it is**

A broader agent-skill collection covering Jev workflow design, triage, document evidence, evaluation, and legal-action selection.

**Why it is a strong fit**

This repo is the closest match to our operational philosophy:

- provider choice is explicit and cannot silently switch;
- simulation is explicitly labeled, with null probability/confidence rather than invented numbers;
- selection is not permission;
- unknown/review paths are first-class;
- evidence is supplied explicitly because Jev does not inherit hidden agent context;
- independent questions are batched; dependent steps wait for fresh evidence;
- action selection is limited to current legal candidates and the host executes/checks the result;
- eval outputs are review leads, not merge approval or authority.

That is almost the exact vocabulary we need around System-One use in Hermes/OpenCode.

**Important pinning caveat**

The repository's default branch is a moving source preview whose skill layout can differ from the published release. This reinforces our existing exact-head rule: never silently install `main` and treat it as a stable contract.

**Decision**

**Best fit for operator/agent workflow guidance and evaluation scaffolding.**

Adapt—not vendor—its concepts into our own project-local skill/runbooks:

- explicit provider identity;
- explicit simulation identity;
- no fabricated probabilities;
- dry-run/validation separated from a real judgment;
- evidence-backed eval mode;
- legal-candidate action mode;
- no silent installation or provider changes.

### 7. yibie/awesome-jev

**Upstream:** https://github.com/yibie/awesome-jev  
**Snapshot:** `052d7a7d426ed9c5e99c7e3e3ae3c8a4db9ad442`

**What it is**

A high-signal curated index of Jev projects, integrations, discussions, and limitations, organized by category.

**Fit**

This is useful to us as a **watchlist source**, not as architecture.

The value is curation: it can surface new router, guardrail, browser, evaluation, and local-model work for later source inspection. Because it is a directory, every interesting item still needs exact-source verification before entering our plans.

**Decision**

**Track regularly as a discovery input.** No runtime dependency.

### 8. AbdelStark/awesome-typesafe-jev

**Upstream:** https://github.com/AbdelStark/awesome-typesafe-jev  
**Snapshot:** `af429b407cfabd0f6bd9086d12e14efcb0ce20c2`

**What it is**

A source-backed field guide with SDKs, demos, agent tools, and independent evaluations.

**Fit**

Its strongest use for us is **evidence cross-checking**. When a new Jev integration or claim appears, this repo is useful as a second discovery/evidence source before we spend time reproducing it.

It is also useful for finding independent evaluations rather than relying only on vendor or project README claims.

**Decision**

**Secondary research/evaluation index.** No runtime dependency.

### 9. Anil-matcha/awesome-jev-by-typesafe

**Upstream:** https://github.com/Anil-matcha/awesome-jev-by-typesafe  
**Snapshot:** `68d934eb0b2f2e1c37ea1e701ffcfb4cc5221e7f`

**What it is**

An evidence-backed collection of use cases, patterns, prompts, and starter code.

**Fit**

This is a useful **cookbook** for generating new benchmark ideas and identifying decision-shaped tasks: classification, routing, scoring, verification, reranking, guardrails, etc.

A particularly good practice is the explicit distinction between measured results and vendor-reported latency/efficiency claims. We should retain that distinction in `my-jev` evidence.

**Decision**

**Use as a pattern/cookbook source.** Reproduce any performance or quality claim under our own exact model/runtime/data hashes before it informs architecture or promotion.

### 10. cobanov/awesome-jev

**Upstream:** https://github.com/cobanov/awesome-jev  
**Snapshot:** `e6d99fc3d9dc16183014250fecf8d581300616b4`

**What it is**

A broad, source-backed ecosystem catalogue covering official resources, provider/framework integrations, agent/browser tooling, local/open reproductions, and evaluation work.

**Why it matters more to us than a normal "awesome" list**

The breadth overlaps directly with our roadmap: provider adapters, LangChain/Pydantic/LiteLLM-style typed decision integrations, browser/computer use, guardrails, calibration, and—most importantly—local/open decision-model paths.

Its current catalogue also surfaces local Jev-compatible work such as Ollaya. That is not evidence that we should adopt it, but it is precisely the kind of development we should compare against the `my-jev` / Bonsai lane.

**Decision**

**Primary ecosystem research feed.**

Add a periodic research pass that checks this repo and then verifies only the items relevant to:

- local/open decision models;
- typed provider protocols;
- calibration/evaluation;
- browser/computer-use decision loops;
- framework adapters whose contract may improve our own provider boundary.

## Architecture implications

### 1. Keep a native typed provider protocol

The strongest common pattern across these projects is that the model should answer bounded questions and code should own policy.

Therefore the internal protocol should remain structurally close to:

```text
state
questions:
  route: Choice(...)
  needs_tools: Noul(...)
  risk: Choice/Score(...)
  ...
->
answers:
  typed values
  probabilities
  confidence where defined
provider provenance
model/runtime identity
timings/token/accounting metadata
```

OpenAI-compatible surfaces are useful only as adapters.

### 2. Formalize provider comparison

The ecosystem now gives us a clean way to compare hosted Jev against local `my-jev` candidates without conflating provider choice with routing authority.

Add a provider-neutral comparison receipt:

```text
input_hash
question_schema_hash
candidate_set_hash
provider_id
provider_version
model_id
response_hash
per-question probabilities
decision latency
resolver result
authority fields (all unchanged / false)
```

Run hosted Jev only on approved non-sensitive benchmark state, or on explicitly approved real evidence. The comparison itself remains shadow/advisory.

### 3. Add speculative-policy-vector benchmarks

Both `smartdio/jev-browser-agent` and `pilot-typesafeai-jev` validate the usefulness of asking independent questions together.

We should benchmark:

- one request with the full AssistX policy vector;
- split requests by question family;
- branch-dependent questions asked speculatively;
- local option-query head vs hosted Jev;
- latency, ECE/Brier/NLL, disagreement, and selective risk.

The benchmark should confirm that packing questions does not change semantics beyond an explicit tolerance.

### 4. Treat abstention/fallback as part of the contract

Several repos use confidence floors or review paths. Our equivalent should be explicit and per-decision-family.

Do not hard-code `0.6` because another project used it. Instead:

1. store the full distribution;
2. derive thresholds from held-out calibration/shadow traffic;
3. maintain action-cost-specific thresholds;
4. force review/abstain when evidence is insufficient;
5. keep hard policy independent of model confidence.

### 5. Add a browser/computer-use benchmark lane, not browser authority

The browser repos are valuable because they give a naturally dynamic candidate set.

Create an evidence-only benchmark:

```text
HTML/accessibility snapshot
 -> legal action candidates
 -> hosted Jev and my-jev both score
 -> compare selected action/probabilities
 -> no live execution in benchmark mode
```

Later, a host executor can perform harmless actions only after its own deterministic safety/permission checks. Login, CAPTCHA, private-network navigation, destructive actions, and user-takeover remain outside model authority.

### 6. Add a project-local System-One agent skill

Combine the best guidance from `building-with-jev-skill` and `wuyoscar/jev-skill`, but make it native to this repository.

It should teach an agent to:

- recognize a bounded decision-shaped problem;
- keep rules/arithmetic/permissions in code;
- construct minimal state and legal candidates;
- batch independent questions;
- distinguish probability from correctness;
- preserve unknown/abstain/review;
- never invent probabilities in simulation;
- pin provider/model/version;
- record receipts;
- never let a decision grant authority.

This is a much better fit than directly installing a moving third-party skill into every agent.

## Adoption priority

### P0 — incorporate into current design/tests

- speculative multi-question policy-vector test cases;
- confidence/abstain/fallback contract;
- provider-neutral decision receipt with exact provenance;
- question-design regression fixtures;
- explicit simulation/provider identity in agent workflows.

Sources: `pilot-typesafeai-jev`, `smartdio/jev-browser-agent`, `building-with-jev-skill`, `wuyoscar/jev-skill`.

### P1 — isolated comparison lanes

- hosted Jev comparison provider behind the native typed contract;
- optional OpenAI-compatible leaf adapter for developer tools;
- browser action-selection benchmark with execution disabled.

Sources: `typesafe-jev-bridge`, `jev-browser`.

### P2 — ongoing ecosystem watch

- monitor local/open System-One providers and reproductions;
- harvest independent evaluation ideas;
- add only source-verified, exact-head research leads to the plan.

Sources: `cobanov/awesome-jev`, `yibie/awesome-jev`, `awesome-typesafe-jev`, `awesome-jev-by-typesafe`.

## Explicit non-goals

This review does **not** authorize us to:

- replace AssistX/Hermes routing with hosted Jev;
- install a third-party router into the live control plane;
- import LangGraph/Deep Agents as our authority system;
- expose an OpenAI-compatible bridge on the network;
- send private repository/runtime state to an external provider automatically;
- copy third-party confidence thresholds into production;
- let browser/model selection bypass existing eligibility, approval, claim, fencing, or mutation checks;
- treat catalogue entries or README benchmarks as independently verified evidence;
- give a local `my-jev` candidate dispatch authority.

## Upstream snapshot

All repositories were reviewed from their default `main` branch on 2026-10-02 and pinned here for reproducibility:

| Repository | Exact reviewed head |
| --- | --- |
| `jkudish/jev-browser` | `1f0726cc8bc4483b96439504b594a01596d8e347` |
| `smartdio/jev-browser-agent` | `88e77e5be07a6b66af25cf53ce41ad0428cd7d3c` |
| `RevocGG/typesafe-jev-bridge` | `315fcfa2cf4ff7b68b1117f7785c0bebabd20a48` |
| `jyje/pilot-typesafeai-jev` | `4d815a77f09d1db1f961854c6dd29eb94105a85c` |
| `dbreunig/building-with-jev-skill` | `04fe3666c6b8b8abfec1271c0e581c823a181f6d` |
| `wuyoscar/jev-skill` | `01bd9403bcaac92e037869843ec83ad8a75c3ec2` |
| `yibie/awesome-jev` | `052d7a7d426ed9c5e99c7e3e3ae3c8a4db9ad442` |
| `AbdelStark/awesome-typesafe-jev` | `af429b407cfabd0f6bd9086d12e14efcb0ce20c2` |
| `Anil-matcha/awesome-jev-by-typesafe` | `68d934eb0b2f2e1c37ea1e701ffcfb4cc5221e7f` |
| `cobanov/awesome-jev` | `e6d99fc3d9dc16183014250fecf8d581300616b4` |

## Proposed next implementation slice

The smallest useful follow-up is **not** to integrate any external runtime.

Implement one read-only benchmark slice in `my-jev`:

1. define a provider-neutral typed `DecisionResponse` receipt with provider/model/version and per-question distributions;
2. add fixture cases modeled on the multi-question router and low-confidence fallback patterns;
3. run the existing local candidate against those fixtures;
4. optionally add an isolated hosted-Jev comparison provider behind an explicit feature flag and approved test data;
5. emit disagreement/calibration evidence only;
6. keep all authority fields false.

Only after that evidence exists should we decide whether an OpenAI adapter or browser benchmark deserves a separate implementation PR.
