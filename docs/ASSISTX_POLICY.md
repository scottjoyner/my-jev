# Hermes / AssistX Agent Policy Model

## Purpose

`my-jev` is intended to become the fast decision layer in front of the
Hermes reasoning/tool loop and the AssistX durable task control plane.

The first decision is not "what words should I generate?" It is:

```text
What kind of behavior does this turn warrant?
```

The v1 routes are:

- `chat` — answer conversationally without durable work.
- `create_tasks` — create requirements / epics / stories / tasks for durable work.
- `act` — perform a bounded read or mutation through Hermes tools.
- `clarify` — ask for missing information before material work.
- `cancel` — stop currently active work.
- `abstain` — do not take autonomous action; surface the turn for review.

The route is only one output. The model also scores whether tools are needed,
whether a durable task graph is warranted, whether context is sufficient,
whether an external effect is implied, likely approval need, action scope,
risk, delegation shape, and response depth.

## Authority boundary

The learned model is **advisory about intent and risk**. It is never the
authority that permits a side effect.

```text
utterance + conversation + live control-plane state
                    |
                    v
             my-jev policy vector
                    |
                    v
          deterministic resolver
          /                   \
 hard runtime facts          model probabilities
 speaker identity            route / scope / risk
 action permissions          context sufficiency
 approval availability       delegation / task need
          \                   /
                    v
              resolved directive
                    |
        +-----------+------------+
        |           |            |
      Hermes     AssistX      approval/review
      tools       task graph      gate
```

Existing Hermes controls remain authoritative:

- dangerous-command approval
- tool guardrails and loop caps
- write approval
- execution fencing / claims
- AssistX mutation authority
- recovery runbook signatures and allowlists

If the model predicts `act` but the runtime does not grant that authority,
the resolver returns `propose_action`, `clarify`, or an approval-gated
directive. A model prediction can never widen the runtime's permissions.

## Replacing the current intent split

AssistX currently has a regex classifier with categories such as query, task,
cancel and memory, then a separate LLM complexity call to choose simple vs
complex work.

The policy model is designed to collapse those fuzzy gates into one inference:

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

The deterministic adapter maps these results back onto AssistX's current
`classification` / `policy_action` fields so the integration can be
incremental rather than a flag-day rewrite.

## Training data from real Hermes trajectories

Synthetic policy data is only the bootstrap. The target dataset should come
from real Hermes/AssistX turns.

For each decision point, record:

- normalized policy state before the model/tool call
- all typed policy probabilities
- resolved directive after hard policy
- tool calls and their effect class
- whether an approval was requested, approved, denied or timed out
- durable tasks/requirements created
- user corrections or interruptions
- final success/failure and verification evidence
- whether the user immediately undid or contradicted the behavior

Do not train on hidden chain-of-thought. The useful supervision is observable
state, decision, action, outcome and correction.

## DAgger-style improvement loop

The Jevlike project is useful prior art here: its game experiments report that
imitation and DAgger reached state-dependent behavior sooner than reward-only
PPO, while policies trained on narrow states could collapse to action priors.

That maps directly to Hermes.

1. Train on existing high-quality traces plus verifier-generated synthetic cases.
2. Run the policy in shadow mode beside the current AssistX/Hermes behavior.
3. Collect states the new policy actually encounters.
4. Label disagreements using deterministic outcome checks, user corrections,
   operator review, and a stronger teacher only where necessary.
5. Add those visited states to the next training round.
6. Promote only on disjoint held-out sessions and explicit safety/calibration gates.

The key is to learn the distribution induced by the policy itself, not only
static prompts sampled before the policy existed.

## Controls against shortcut learning

The policy benchmark must include counterfactual controls:

- **authority counterfactual** — same utterance, different runtime authority;
  semantic intent targets remain the same while the resolver outcome changes.
- **shuffled state** — pair an utterance/schema with the wrong conversation
  state; quality should drop materially.
- **option permutation** — randomly reorder Choice options without changing the
  answer distribution.
- **context ablation** — remove conversation/task context and verify decisions
  that depend on it degrade.
- **speaker/source holdout** — hold out channels or speaker IDs from training.
- **task-family holdout** — hold out request templates/schema paraphrases.

A high score without sensitivity to the relevant state is not a successful
policy model.

## Jevlike prior art

`vinnylarouge/jevlike` is MIT-licensed and provides a compact independent
one-pass option scorer. Its useful design ideas for this project are:

- each option acts as a query over context tokens;
- one shared scorer handles a variable number of options;
- shuffled-context evaluation catches models that rely on option priors;
- related records are split together;
- imitation / DAgger on visited states can be more effective than reward-only
  training for policy behavior.

The implementation in `my-jev` remains independently structured around the
typed multi-question AssistX policy contract rather than importing Jevlike as a
runtime dependency.

Reference: https://github.com/vinnylarouge/jevlike
