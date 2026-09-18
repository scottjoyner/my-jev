from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from .schema import DecisionRecord, QuestionSpec, QuestionType, TargetSpec


class AgentRoute(StrEnum):
    CHAT = "chat"
    CREATE_TASKS = "create_tasks"
    ACT = "act"
    CLARIFY = "clarify"
    CANCEL = "cancel"
    ABSTAIN = "abstain"


class ActionScope(StrEnum):
    NONE = "none"
    READ_ONLY = "read_only"
    LOCAL_WRITE = "local_write"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"
    PRIVILEGED = "privileged"


class Delegation(StrEnum):
    NONE = "none"
    SELF = "self"
    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"


class RiskLevel(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ResolvedDisposition(StrEnum):
    CHAT = "chat"
    CREATE_TASKS = "create_tasks"
    ACT = "act"
    ACT_WITH_APPROVAL = "act_with_approval"
    PROPOSE_ACTION = "propose_action"
    CLARIFY = "clarify"
    CANCEL = "cancel"
    ABSTAIN = "abstain"


ROUTE_OPTIONS = [item.value for item in AgentRoute]
SCOPE_OPTIONS = [item.value for item in ActionScope]
DELEGATION_OPTIONS = [item.value for item in Delegation]
RISK_OPTIONS = [item.value for item in RiskLevel]
DEPTH_OPTIONS = ["brief", "normal", "structured", "project"]

# Primary behavior/authority-adjacent decisions matter more than presentation
# details during joint multi-question training. Generic datasets still default
# to weight 1.0 because the weight lives in question metadata.
POLICY_LOSS_WEIGHTS = {
    "route": 3.0,
    "needs_tools": 1.0,
    "needs_task_graph": 2.0,
    "context_sufficient": 2.0,
    "external_effect": 1.5,
    "approval_likely": 1.0,
    "action_scope": 2.0,
    "risk": 1.5,
    "delegation": 1.0,
    "response_depth": 1.0,
}


class AgentPolicyState(BaseModel):
    """Canonical policy state passed from Hermes/AssistX to the decision model.

    Authority facts are included for context and auditing, but they are never
    permission grants. The deterministic resolver remains the authority boundary.
    """

    utterance: str
    conversation_summary: str = ""
    source: str = "interactive"
    speaker_id: str = ""
    speaker_verified: bool = True
    foreground: bool = True
    active_work: list[str] = Field(default_factory=list)
    pending_approvals: list[str] = Field(default_factory=list)
    available_capabilities: list[str] = Field(default_factory=list)
    available_tools: list[str] = Field(default_factory=list)
    actions_allowed: bool = True
    external_actions_allowed: bool = False
    privileged_actions_allowed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    def as_model_state(self) -> str:
        """Stable structured text so the encoder sees explicit control-plane facts."""
        payload = {
            "utterance": self.utterance,
            "conversation_summary": self.conversation_summary,
            "source": self.source,
            "speaker": {
                "id": self.speaker_id,
                "verified": self.speaker_verified,
            },
            "foreground": self.foreground,
            "active_work": self.active_work,
            "pending_approvals": self.pending_approvals,
            "available_capabilities": sorted(self.available_capabilities),
            "available_tools": sorted(self.available_tools),
            "authority_context": {
                "actions_allowed": self.actions_allowed,
                "external_actions_allowed": self.external_actions_allowed,
                "privileged_actions_allowed": self.privileged_actions_allowed,
            },
            "metadata": self.metadata,
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


class AgentPolicyLabel(BaseModel):
    route: AgentRoute
    needs_tools: bool
    needs_task_graph: bool
    context_sufficient: bool
    external_effect: bool
    approval_likely: bool
    action_scope: ActionScope
    risk: RiskLevel
    delegation: Delegation
    response_depth: int = Field(ge=0, le=3)


def _question(
    question_type: QuestionType,
    instructions: str,
    *,
    options: list[str] | None = None,
    task_id: str,
    loss_weight: float = 1.0,
) -> QuestionSpec:
    return QuestionSpec(
        type=question_type,
        instructions=instructions,
        options=options,
        metadata={
            "task_id": task_id,
            "policy_contract": "assistx-agent-policy-v1",
            "loss_weight": loss_weight,
        },
    )


def agent_policy_questions() -> dict[str, QuestionSpec]:
    return {
        "route": _question(
            QuestionType.CHOICE,
            (
                "What should Hermes do next for this turn? Choose the behavioral route, "
                "not whether the runtime is authorized to execute it."
            ),
            options=ROUTE_OPTIONS,
            task_id="assistx.route",
            loss_weight=POLICY_LOSS_WEIGHTS["route"],
        ),
        "needs_tools": _question(
            QuestionType.NOUL,
            "Does satisfying the user's intent require reading or using any tool/system?",
            task_id="assistx.needs_tools",
            loss_weight=POLICY_LOSS_WEIGHTS["needs_tools"],
        ),
        "needs_task_graph": _question(
            QuestionType.NOUL,
            (
                "Does this request benefit from durable requirements and multiple tasks "
                "rather than a single conversational turn?"
            ),
            task_id="assistx.needs_task_graph",
            loss_weight=POLICY_LOSS_WEIGHTS["needs_task_graph"],
        ),
        "context_sufficient": _question(
            QuestionType.NOUL,
            (
                "Is there enough context to proceed with the intended behavior without "
                "asking the user a material clarifying question?"
            ),
            task_id="assistx.context_sufficient",
            loss_weight=POLICY_LOSS_WEIGHTS["context_sufficient"],
        ),
        "external_effect": _question(
            QuestionType.NOUL,
            (
                "Would fulfilling the intent change a system or communicate externally "
                "rather than only read information or chat?"
            ),
            task_id="assistx.external_effect",
            loss_weight=POLICY_LOSS_WEIGHTS["external_effect"],
        ),
        "approval_likely": _question(
            QuestionType.NOUL,
            (
                "Would a careful agent normally expect a user/runtime approval gate before "
                "performing the requested action? This is advisory only."
            ),
            task_id="assistx.approval_likely",
            loss_weight=POLICY_LOSS_WEIGHTS["approval_likely"],
        ),
        "action_scope": _question(
            QuestionType.CHOICE,
            "What is the highest-impact action scope implied by the request?",
            options=SCOPE_OPTIONS,
            task_id="assistx.action_scope",
            loss_weight=POLICY_LOSS_WEIGHTS["action_scope"],
        ),
        "risk": _question(
            QuestionType.SCORE,
            (
                "Rate the consequence of executing the intended action incorrectly. "
                "The levels are ordered from low to critical."
            ),
            options=RISK_OPTIONS,
            task_id="assistx.risk",
            loss_weight=POLICY_LOSS_WEIGHTS["risk"],
        ),
        "delegation": _question(
            QuestionType.CHOICE,
            "What delegation shape best matches the work implied by the request?",
            options=DELEGATION_OPTIONS,
            task_id="assistx.delegation",
            loss_weight=POLICY_LOSS_WEIGHTS["delegation"],
        ),
        "response_depth": _question(
            QuestionType.SCORE,
            (
                "How much structured work is warranted for this turn, from brief through "
                "project-scale?"
            ),
            options=DEPTH_OPTIONS,
            task_id="assistx.response_depth",
            loss_weight=POLICY_LOSS_WEIGHTS["response_depth"],
        ),
    }


def _index(options: list[str], value: StrEnum | str) -> int:
    raw = value.value if isinstance(value, StrEnum) else str(value)
    return options.index(raw)


def label_to_targets(label: AgentPolicyLabel) -> dict[str, TargetSpec]:
    return {
        "route": TargetSpec(index=_index(ROUTE_OPTIONS, label.route)),
        "needs_tools": TargetSpec(index=int(label.needs_tools)),
        "needs_task_graph": TargetSpec(index=int(label.needs_task_graph)),
        "context_sufficient": TargetSpec(index=int(label.context_sufficient)),
        "external_effect": TargetSpec(index=int(label.external_effect)),
        "approval_likely": TargetSpec(index=int(label.approval_likely)),
        "action_scope": TargetSpec(
            index=_index(SCOPE_OPTIONS, label.action_scope)
        ),
        "risk": TargetSpec(index=_index(RISK_OPTIONS, label.risk)),
        "delegation": TargetSpec(
            index=_index(DELEGATION_OPTIONS, label.delegation)
        ),
        "response_depth": TargetSpec(index=label.response_depth),
    }


def build_agent_policy_record(
    state: AgentPolicyState,
    *,
    label: AgentPolicyLabel | None = None,
) -> DecisionRecord:
    return DecisionRecord(
        state=state.as_model_state(),
        questions=agent_policy_questions(),
        targets=label_to_targets(label) if label is not None else None,
        metadata={
            "domain": "assistx_agent_policy",
            "policy_contract": "assistx-agent-policy-v1",
            "source": state.source,
            **state.metadata,
        },
    )


class AgentPolicyScores(BaseModel):
    route: dict[str, float]
    needs_tools: float
    needs_task_graph: float
    context_sufficient: float
    external_effect: float
    approval_likely: float
    action_scope: dict[str, float]
    risk: dict[str, float]
    delegation: dict[str, float]
    response_depth: dict[str, float]

    def choice(self, field: str) -> tuple[str, float]:
        distribution = getattr(self, field)
        if not isinstance(distribution, dict) or not distribution:
            raise ValueError(f"{field} is not a categorical distribution")
        name = max(distribution, key=distribution.get)
        return name, float(distribution[name])


def _distribution(prediction: dict[str, object]) -> dict[str, float]:
    options = prediction.get("options")
    probabilities = prediction.get("probabilities")
    if not isinstance(options, list) or not isinstance(probabilities, list):
        raise ValueError("prediction is missing options/probabilities")
    if len(options) != len(probabilities):
        raise ValueError("prediction option/probability length mismatch")
    return {
        str(option): float(probability)
        for option, probability in zip(options, probabilities, strict=True)
    }


def scores_from_predictions(
    predictions: list[dict[str, object]],
) -> AgentPolicyScores:
    by_name = {
        str(item.get("question")): item
        for item in predictions
    }
    missing = set(agent_policy_questions()) - set(by_name)
    if missing:
        raise ValueError(f"missing agent-policy predictions: {sorted(missing)}")

    def true_probability(name: str) -> float:
        distribution = _distribution(by_name[name])
        return float(distribution["true"])

    return AgentPolicyScores(
        route=_distribution(by_name["route"]),
        needs_tools=true_probability("needs_tools"),
        needs_task_graph=true_probability("needs_task_graph"),
        context_sufficient=true_probability("context_sufficient"),
        external_effect=true_probability("external_effect"),
        approval_likely=true_probability("approval_likely"),
        action_scope=_distribution(by_name["action_scope"]),
        risk=_distribution(by_name["risk"]),
        delegation=_distribution(by_name["delegation"]),
        response_depth=_distribution(by_name["response_depth"]),
    )


def policy_consistency_violations(
    scores: AgentPolicyScores,
) -> list[str]:
    """Audit relationships that should hold across independently scored fields.

    These are Hermes/AssistX contract invariants, not learned permissions.  The
    model still emits every field independently; this audit catches combinations
    that should never be trusted for direct routing.
    """
    route_name, _ = scores.choice("route")
    scope_name, _ = scores.choice("action_scope")
    route = AgentRoute(route_name)
    scope = ActionScope(scope_name)
    violations: list[str] = []

    needs_tools = scores.needs_tools >= 0.5
    needs_task_graph = (
        scores.needs_task_graph >= 0.5
    )
    external_effect = (
        scores.external_effect >= 0.5
    )

    if route == AgentRoute.ACT:
        if not needs_tools:
            violations.append(
                "act_without_tools"
            )
        if scope == ActionScope.NONE:
            violations.append(
                "act_with_none_scope"
            )

    if (
        route == AgentRoute.CREATE_TASKS
        and not needs_task_graph
    ):
        violations.append(
            "create_tasks_without_task_graph"
        )

    if route == AgentRoute.CHAT:
        if needs_task_graph:
            violations.append(
                "chat_with_task_graph"
            )
        if external_effect:
            violations.append(
                "chat_with_external_effect"
            )
        if scope != ActionScope.NONE:
            violations.append(
                "chat_with_action_scope"
            )

    if scope in {
        ActionScope.LOCAL_WRITE,
        ActionScope.EXTERNAL_SIDE_EFFECT,
        ActionScope.PRIVILEGED,
    } and not needs_tools:
        violations.append(
            "mutation_scope_without_tools"
        )

    if scope in {
        ActionScope.EXTERNAL_SIDE_EFFECT,
        ActionScope.PRIVILEGED,
    } and not external_effect:
        violations.append(
            "external_scope_without_external_effect"
        )

    return sorted(set(violations))


class PolicyConstraints(BaseModel):
    """Hard runtime facts. These always outrank the learned policy."""

    speaker_verified: bool = True
    actions_allowed: bool = True
    local_writes_allowed: bool = True
    external_actions_allowed: bool = False
    privileged_actions_allowed: bool = False
    approval_gate_available: bool = True
    active_work: bool = False


class PolicyThresholds(BaseModel):
    route_confidence: float = 0.45
    act_confidence: float = 0.55
    context_sufficient: float = 0.55
    approval_likely: float = 0.50
    high_risk_probability: float = 0.45


class ResolvedAgentPolicy(BaseModel):
    disposition: ResolvedDisposition
    model_route: AgentRoute
    model_route_confidence: float
    action_scope: ActionScope
    risk: RiskLevel
    delegation: Delegation
    needs_tools: bool
    needs_task_graph: bool
    approval_required: bool
    consistency_violations: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


def resolve_agent_policy(
    scores: AgentPolicyScores,
    constraints: PolicyConstraints,
    *,
    thresholds: PolicyThresholds = PolicyThresholds(),
) -> ResolvedAgentPolicy:
    route_name, route_confidence = scores.choice("route")
    scope_name, _ = scores.choice("action_scope")
    risk_name, _ = scores.choice("risk")
    delegation_name, _ = scores.choice("delegation")

    route = AgentRoute(route_name)
    scope = ActionScope(scope_name)
    risk = RiskLevel(risk_name)
    reasons: list[str] = []
    approval_required = False
    consistency_violations = policy_consistency_violations(
        scores
    )

    def resolved(disposition: ResolvedDisposition) -> ResolvedAgentPolicy:
        return ResolvedAgentPolicy(
            disposition=disposition,
            model_route=route,
            model_route_confidence=route_confidence,
            action_scope=scope,
            risk=risk,
            delegation=Delegation(delegation_name),
            needs_tools=scores.needs_tools >= 0.5,
            needs_task_graph=scores.needs_task_graph >= 0.5,
            approval_required=approval_required,
            consistency_violations=consistency_violations,
            reasons=reasons,
        )

    if route_confidence < thresholds.route_confidence:
        reasons.append("route confidence below resolver threshold")
        return resolved(ResolvedDisposition.CLARIFY)

    if route == AgentRoute.ACT and any(
        item in consistency_violations
        for item in (
            "act_without_tools",
            "act_with_none_scope",
            "mutation_scope_without_tools",
            "external_scope_without_external_effect",
        )
    ):
        reasons.append(
            "independent policy fields disagree on a safe action shape"
        )
        return resolved(
            ResolvedDisposition.PROPOSE_ACTION
        )

    if (
        route == AgentRoute.CREATE_TASKS
        and "create_tasks_without_task_graph"
        in consistency_violations
    ):
        reasons.append(
            "route and task-graph predictions disagree"
        )
        return resolved(
            ResolvedDisposition.CLARIFY
        )

    if route == AgentRoute.CANCEL:
        if constraints.active_work:
            return resolved(ResolvedDisposition.CANCEL)
        reasons.append("cancel requested but no active work is present")
        return resolved(ResolvedDisposition.CHAT)

    if route in {AgentRoute.ACT, AgentRoute.CREATE_TASKS}:
        if scores.context_sufficient < thresholds.context_sufficient:
            reasons.append("model reports insufficient context for material work")
            return resolved(ResolvedDisposition.CLARIFY)

    if route == AgentRoute.CHAT:
        return resolved(ResolvedDisposition.CHAT)
    if route == AgentRoute.CREATE_TASKS:
        return resolved(ResolvedDisposition.CREATE_TASKS)
    if route == AgentRoute.CLARIFY:
        return resolved(ResolvedDisposition.CLARIFY)
    if route == AgentRoute.ABSTAIN:
        return resolved(ResolvedDisposition.ABSTAIN)

    # ACT: the model may identify an action, but only runtime facts can permit it.
    if route_confidence < thresholds.act_confidence:
        reasons.append("action intent confidence below execution threshold")
        return resolved(ResolvedDisposition.PROPOSE_ACTION)
    if not constraints.speaker_verified:
        reasons.append("speaker identity is not verified for action execution")
        return resolved(ResolvedDisposition.CLARIFY)
    if not constraints.actions_allowed:
        reasons.append("runtime policy does not permit actions")
        return resolved(ResolvedDisposition.PROPOSE_ACTION)

    if scope == ActionScope.LOCAL_WRITE and not constraints.local_writes_allowed:
        reasons.append("local mutation is outside the current runtime authority")
        return resolved(ResolvedDisposition.PROPOSE_ACTION)
    if (
        scope == ActionScope.EXTERNAL_SIDE_EFFECT
        and not constraints.external_actions_allowed
    ):
        reasons.append("external side effect is outside the current runtime authority")
        return resolved(ResolvedDisposition.PROPOSE_ACTION)
    if scope == ActionScope.PRIVILEGED and not constraints.privileged_actions_allowed:
        reasons.append("privileged action is outside the current runtime authority")
        return resolved(ResolvedDisposition.PROPOSE_ACTION)

    high_risk_mass = sum(
        scores.risk.get(level, 0.0)
        for level in (RiskLevel.HIGH.value, RiskLevel.CRITICAL.value)
    )
    approval_required = (
        scores.approval_likely >= thresholds.approval_likely
        or scope == ActionScope.PRIVILEGED
        or high_risk_mass >= thresholds.high_risk_probability
    )

    if approval_required:
        if constraints.approval_gate_available:
            reasons.append("action is permitted but requires the existing approval gate")
            return resolved(ResolvedDisposition.ACT_WITH_APPROVAL)
        reasons.append("approval is required but no approval channel is available")
        return resolved(ResolvedDisposition.PROPOSE_ACTION)

    return resolved(ResolvedDisposition.ACT)
