from __future__ import annotations

from typing import Any

from .agent_policy import ResolvedAgentPolicy, ResolvedDisposition


def assistx_policy_payload(
    decision: ResolvedAgentPolicy,
) -> dict[str, Any]:
    """Map the learned policy contract onto AssistX's existing intent machinery.

    The mapping is deliberately conservative. Existing Hermes approvals,
    execution fencing, task claims, and tool guardrails remain authoritative.
    """
    mapping = {
        ResolvedDisposition.CHAT: ("query", "answer_inline"),
        ResolvedDisposition.CREATE_TASKS: ("task", "create_task_graph"),
        ResolvedDisposition.ACT: ("task", "direct_action"),
        ResolvedDisposition.ACT_WITH_APPROVAL: (
            "task",
            "direct_action_with_approval",
        ),
        ResolvedDisposition.PROPOSE_ACTION: ("task", "review_dispatch"),
        ResolvedDisposition.CLARIFY: ("unknown", "needs_clarification"),
        ResolvedDisposition.CANCEL: ("cancel", "cancel_active_work"),
        ResolvedDisposition.ABSTAIN: ("unknown", "review_dispatch"),
    }
    classification, policy_action = mapping[decision.disposition]
    return {
        "classification": classification,
        "policy_action": policy_action,
        "agent_policy": decision.model_dump(mode="json"),
    }


def hermes_turn_directive(
    decision: ResolvedAgentPolicy,
) -> dict[str, Any]:
    """Small transport-neutral directive for the Hermes harness."""
    return {
        "disposition": decision.disposition.value,
        "requires_approval": decision.approval_required,
        "needs_tools": decision.needs_tools,
        "needs_task_graph": decision.needs_task_graph,
        "action_scope": decision.action_scope.value,
        "risk": decision.risk.value,
        "delegation": decision.delegation.value,
        "reasons": list(decision.reasons),
    }
