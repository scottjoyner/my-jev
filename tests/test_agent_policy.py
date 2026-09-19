from my_jev.agent_policy import (
    ActionScope,
    AgentPolicyScores,
    AgentPolicyState,
    AgentRoute,
    Delegation,
    PolicyConstraints,
    ResolvedDisposition,
    build_agent_policy_record,
    policy_consistency_violations,
    resolve_agent_policy,
)
from my_jev.assistx_adapter import (
    assistx_policy_payload,
    hermes_turn_directive,
)


def scores(
    route: AgentRoute,
    *,
    route_confidence: float = 0.9,
    context_sufficient: float = 0.9,
    scope: ActionScope = ActionScope.NONE,
    risk: str = "low",
    approval: float = 0.0,
    external: float = 0.0,
    needs_tools: float | None = None,
    needs_task_graph: float | None = None,
) -> AgentPolicyScores:
    route_dist = {
        item.value: 0.02
        for item in AgentRoute
    }
    route_dist[route.value] = (
        route_confidence
    )
    scope_dist = {
        item.value: 0.02
        for item in ActionScope
    }
    scope_dist[scope.value] = 0.9
    risk_dist = {
        "low": 0.02,
        "moderate": 0.02,
        "high": 0.02,
        "critical": 0.02,
    }
    risk_dist[risk] = 0.9
    delegation = {
        item.value: 0.02
        for item in Delegation
    }
    delegation[
        Delegation.SELF.value
    ] = 0.9
    return AgentPolicyScores(
        route=route_dist,
        needs_tools=(
            needs_tools
            if needs_tools is not None
            else (
                0.9
                if route
                in {
                    AgentRoute.ACT,
                    AgentRoute.CANCEL,
                }
                else 0.1
            )
        ),
        needs_task_graph=(
            needs_task_graph
            if needs_task_graph is not None
            else (
                0.9
                if route
                == AgentRoute.CREATE_TASKS
                else 0.1
            )
        ),
        context_sufficient=(
            context_sufficient
        ),
        external_effect=external,
        approval_likely=approval,
        action_scope=scope_dist,
        risk=risk_dist,
        delegation=delegation,
        response_depth={
            "brief": 0.1,
            "normal": 0.7,
            "structured": 0.1,
            "project": 0.1,
        },
    )


def test_agent_policy_record_has_typed_questions():
    record = build_agent_policy_record(
        AgentPolicyState(
            utterance=(
                "Check CI and tell me "
                "what failed."
            )
        )
    )
    assert record.targets is None
    assert set(record.questions) == {
        "route",
        "needs_tools",
        "needs_task_graph",
        "context_sufficient",
        "external_effect",
        "approval_likely",
        "action_scope",
        "risk",
        "delegation",
        "response_depth",
    }
    assert (
        record.questions[
            "route"
        ].options[0]
        == "chat"
    )


def test_resolver_allows_low_risk_read_action():
    decision = resolve_agent_policy(
        scores(
            AgentRoute.ACT,
            scope=ActionScope.READ_ONLY,
        ),
        PolicyConstraints(
            speaker_verified=True,
            actions_allowed=True,
        ),
    )
    assert (
        decision.disposition
        == ResolvedDisposition.ACT
    )
    assert (
        decision.approval_required
        is False
    )
    assert (
        decision.consistency_violations
        == []
    )


def test_resolver_never_grants_external_authority():
    decision = resolve_agent_policy(
        scores(
            AgentRoute.ACT,
            scope=(
                ActionScope.EXTERNAL_SIDE_EFFECT
            ),
            risk="high",
            approval=0.9,
            external=0.9,
        ),
        PolicyConstraints(
            speaker_verified=True,
            actions_allowed=True,
            external_actions_allowed=False,
        ),
    )
    assert (
        decision.disposition
        == ResolvedDisposition.PROPOSE_ACTION
    )
    assert (
        "outside"
        in " ".join(
            decision.reasons
        )
    )


def test_resolver_requires_verified_speaker_for_actions():
    decision = resolve_agent_policy(
        scores(
            AgentRoute.ACT,
            scope=ActionScope.READ_ONLY,
        ),
        PolicyConstraints(
            speaker_verified=False,
            actions_allowed=True,
        ),
    )
    assert (
        decision.disposition
        == ResolvedDisposition.CLARIFY
    )


def test_permitted_low_risk_external_action_can_run_without_extra_policy_approval():
    decision = resolve_agent_policy(
        scores(
            AgentRoute.ACT,
            scope=(
                ActionScope.EXTERNAL_SIDE_EFFECT
            ),
            risk="moderate",
            approval=0.1,
            external=0.9,
        ),
        PolicyConstraints(
            speaker_verified=True,
            actions_allowed=True,
            external_actions_allowed=True,
            approval_gate_available=True,
        ),
    )
    assert (
        decision.disposition
        == ResolvedDisposition.ACT
    )
    assert (
        decision.approval_required
        is False
    )


def test_permitted_high_risk_action_still_hits_approval_gate():
    decision = resolve_agent_policy(
        scores(
            AgentRoute.ACT,
            scope=ActionScope.PRIVILEGED,
            risk="critical",
            approval=0.9,
            external=0.9,
        ),
        PolicyConstraints(
            speaker_verified=True,
            actions_allowed=True,
            privileged_actions_allowed=True,
            approval_gate_available=True,
        ),
    )
    assert (
        decision.disposition
        == (
            ResolvedDisposition
            .ACT_WITH_APPROVAL
        )
    )
    assert (
        decision.approval_required
        is True
    )


def test_inconsistent_act_is_not_directly_executable():
    inconsistent = scores(
        AgentRoute.ACT,
        scope=ActionScope.READ_ONLY,
        needs_tools=0.1,
    )
    violations = (
        policy_consistency_violations(
            inconsistent
        )
    )
    decision = resolve_agent_policy(
        inconsistent,
        PolicyConstraints(
            speaker_verified=True,
            actions_allowed=True,
        ),
    )

    assert (
        "act_without_tools"
        in violations
    )
    assert (
        decision.disposition
        == ResolvedDisposition.PROPOSE_ACTION
    )
    assert (
        "act_without_tools"
        in decision.consistency_violations
    )


def test_external_scope_without_external_effect_is_inconsistent():
    violations = (
        policy_consistency_violations(
            scores(
                AgentRoute.ACT,
                scope=(
                    ActionScope
                    .EXTERNAL_SIDE_EFFECT
                ),
                external=0.1,
            )
        )
    )
    assert (
        "external_scope_without_external_effect"
        in violations
    )


def test_create_tasks_requires_task_graph_signal():
    decision = resolve_agent_policy(
        scores(
            AgentRoute.CREATE_TASKS,
            needs_task_graph=0.1,
        ),
        PolicyConstraints(),
    )
    assert (
        decision.disposition
        == ResolvedDisposition.CLARIFY
    )
    assert (
        "create_tasks_without_task_graph"
        in decision.consistency_violations
    )


def test_assistx_and_hermes_adapters_preserve_gate_result():
    decision = resolve_agent_policy(
        scores(
            AgentRoute.CREATE_TASKS
        ),
        PolicyConstraints(),
    )
    assistx = assistx_policy_payload(
        decision
    )
    hermes = hermes_turn_directive(
        decision
    )

    assert (
        assistx["classification"]
        == "task"
    )
    assert (
        assistx["policy_action"]
        == "create_task_graph"
    )
    assert (
        hermes["disposition"]
        == "create_tasks"
    )


def test_low_route_confidence_forces_clarification():
    decision = resolve_agent_policy(
        scores(
            AgentRoute.CHAT,
            route_confidence=0.2,
        ),
        PolicyConstraints(),
    )
    assert (
        decision.disposition
        == ResolvedDisposition.CLARIFY
    )
