"""The HTTP surface: the only decision output in this repository without tests.

`grep -rl "my_jev.server" tests/` returned nothing. Its coverage was import
side-effect alone, and because `fastapi` is in the `serve` extra rather than a base
dependency, CI could not have reached it either. So the surface a decision escapes
through was untested in both places -- and its failure modes are the ones this
project exists to prevent: a refusal quietly dropped, or a response that reads as an
instruction.

The tests need `fastapi`, which CI does not install, so they skip rather than fail.
That is honest about the gap and keeps the rest of the suite green; CI reaching this
file needs a matrix job installing `.[serve]`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from my_jev.agent_policy import (  # noqa: E402
    AgentPolicyState,
)
from my_jev.server import PolicyRuntime, create_app  # noqa: E402


class _StubModel:
    """Stands in for the checkpoint's model.

    Mirrors the real `Model.predict` return shape exactly -- keyed on `question`,
    with `options` and `probabilities` -- because `scores_from_predictions` reads
    those exact keys and a plausible-looking stub that renames one produces
    "missing agent-policy predictions" for every question. An earlier version of this
    file used `name` and failed in a way that looked like a server bug.

    `peaks` names the option to concentrate probability on for each question, which
    is how a disposition is steered. Dispositions are *derived* by the resolver from
    the scores and the runtime constraints, so they cannot be set directly -- an
    earlier version of this file tried to pass one and was wrong.
    """

    def __init__(self, *, peaks: dict[str, str] | None = None, even: bool = False):
        self.peaks = peaks or {}
        self.even = even
        self.calls: list[tuple[int, float]] = []

    def predict(self, records, *, temperature):
        self.calls.append((len(records), temperature))
        out = []
        for record in records:
            for name, question in record.questions.items():
                options = list(question.options or ["false", "true"])
                if self.even:
                    share = 1.0 / len(options)
                    probabilities = [share] * len(options)
                else:
                    wanted = self.peaks.get(name)
                    if wanted is None or wanted not in options:
                        probabilities = [
                            1.0 if index == 0 else 0.0
                            for index in range(len(options))
                        ]
                    else:
                        probabilities = [
                            1.0 if option == wanted else 0.0
                            for option in options
                        ]
                out.append({
                    "question": name,
                    "options": options,
                    "probabilities": probabilities,
                })
        return out


#: A model that asks to act, writing locally, at low risk, with no approval expected.
#: This is the shape whose HTTP response reads most like an instruction, so it is the
#: one worth testing hardest.
def _acting_model(**overrides) -> _StubModel:
    peaks = {
        "route": "act",
        "action_scope": "local_write",
        "risk": "low",
        "delegation": "self",
        "response_depth": "brief",
        "needs_tools": "true",
        "needs_task_graph": "false",
        "context_sufficient": "true",
        "external_effect": "false",
        "approval_likely": "false",
    }
    peaks.update(overrides)
    return _StubModel(peaks=peaks)


def _state(**overrides) -> AgentPolicyState:
    """A minimal valid state.

    `utterance` is the only required field; `build_agent_policy_record` supplies the
    typed questions from it. Built through `AgentPolicyState(...)` the way the
    existing agent-policy tests do, rather than hand-rolling a payload that happens
    to validate.
    """
    payload = {
        "utterance": "Check CI and tell me what failed.",
        "foreground": True,
    }
    payload.update(overrides)
    return AgentPolicyState.model_validate(payload)


def _runtime(**kwargs) -> PolicyRuntime:
    return PolicyRuntime(
        model=kwargs.pop("model", _StubModel(even=True)),
        temperature=kwargs.pop("temperature", 0.7),
        checkpoint=kwargs.pop("checkpoint", "/models/policy.pt"),
        device=kwargs.pop("device", "cpu"),
    )


def _client(runtime=None) -> TestClient:
    return TestClient(create_app(runtime or _runtime()))


# --- the advisory marker ---------------------------------------------------


def test_the_decision_response_is_marked_advisory():
    """The gap this file exists to close.

    Eight other modules in this repository mark their output `advisory_only`. This
    response did not, and it is the only one that crosses HTTP -- so it is the output
    most likely to be read as an instruction rather than a recommendation.
    """
    client = _client()
    response = client.post(
        "/v1/agent-policy", json={"state": _state().model_dump(mode="json")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["advisory_only"] is True
    assert body["runtime_authority_changed"] is False


def test_no_authority_field_in_the_response_is_ever_true():
    """Belt and braces: nothing in the payload may claim authority.

    Walks the whole body rather than checking two named keys, so a field added later
    claiming authority is caught rather than grandfathered in.
    """

    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                lowered = key.lower()
                if any(word in lowered for word in
                       ("authority", "dispatch", "executed", "approved", "placed", "order")):
                    yield f"{path}.{key}", value
                yield from walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, item in enumerate(node):
                yield from walk(item, f"{path}[{index}]")

    response = _client().post(
        "/v1/agent-policy", json={"state": _state().model_dump(mode="json")}
    )
    claims = list(walk(response.json()))
    assert claims, "expected the response to mention authority at all"
    for path, value in claims:
        assert value in (False, "review_dispatch", "advisory_only"), f"{path}={value!r}"


def test_healthz_is_also_marked_advisory():
    body = _client().get("/healthz").json()
    assert body["advisory_only"] is True


# --- the refusal surface ---------------------------------------------------


def test_a_state_forbidding_privileged_actions_says_so_rather_than_acting():
    """The property the resolver exists to hold, observed over HTTP.

    `resolve_agent_policy` appends "privileged action is outside the current runtime
    authority" and similar reasons. What was untested is that those reasons survive
    the trip through the HTTP response intact.
    """
    state = _state(privileged_actions_allowed=False)
    body = _client().post(
        "/v1/agent-policy",
        json={"state": state.model_dump(mode="json"), "constraints": {}},
    ).json()

    assert body["advisory_only"] is True
    resolved = body["resolved"]
    # Whatever it decides, the refusal reasons must be present in the output rather
    # than swallowed on the way out.
    assert resolved["reasons"], "a constrained decision reported no reasons at all"
    assert isinstance(resolved["consistency_violations"], list)


def test_approval_required_is_consistent_across_the_two_payloads():
    """The same fact appears twice in the response; it must not disagree with itself.

    `resolved.approval_required` and `hermes.requires_approval` are separate fields.
    If they can diverge, a consumer reading the smaller one believes no approval is
    needed -- which is exactly the failure this project's discipline exists to
    prevent, and it would be invisible in either field alone.

    Dispositions are *derived* by the resolver rather than chosen, so the states are
    varied rather than the predictions.
    """
    states = [
        _state(),
        _state(actions_allowed=True),
        _state(external_actions_allowed=True, privileged_actions_allowed=True),
        _state(pending_approvals=["approve-the-deploy"]),
        _state(foreground=False),
    ]
    seen: set[tuple[str, bool]] = set()
    for confident in (False, True):
        for state in states:
            body = _client(_runtime(model=_StubModel(even=not confident))).post(
                "/v1/agent-policy",
                json={"state": state.model_dump(mode="json"), "constraints": {}},
            ).json()
            resolved = body["resolved"]
            assert (
                body["hermes"]["requires_approval"] == resolved["approval_required"]
            ), (confident, state.foreground)
            seen.add((resolved["disposition"], resolved["approval_required"]))

    assert len(seen) > 1, (
        f"every state produced the same disposition {seen}, so this test is not "
        "exercising more than one path"
    )


def test_the_directive_is_consistent_with_the_resolution():
    body = _client().post(
        "/v1/agent-policy", json={"state": _state().model_dump(mode="json")}
    ).json()
    hermes, resolved = body["hermes"], body["resolved"]
    for field in ("disposition", "action_scope", "risk", "delegation", "needs_tools",
                  "needs_task_graph", "reasons"):
        assert hermes[field] == resolved[field], field


# --- request handling ------------------------------------------------------


def test_a_malformed_state_is_rejected_rather_than_guessed():
    response = _client().post(
        "/v1/agent-policy", json={"state": {"unexpected": "shape"}}
    )
    assert response.status_code == 422


def test_a_missing_state_is_rejected():
    assert _client().post("/v1/agent-policy", json={}).status_code == 422


def test_constraints_default_rather_than_being_required():
    """A caller that omits constraints gets the documented default, not an error."""
    body = _client().post(
        "/v1/agent-policy", json={"state": _state().model_dump(mode="json")}
    ).json()
    assert body["resolved"]["disposition"]


def test_an_unknown_constraint_field_is_refused():
    """`extra="forbid"` discipline, asserted at the HTTP boundary."""
    response = _client().post(
        "/v1/agent-policy",
        json={
            "state": _state().model_dump(mode="json"),
            "constraints": {"not_a_real_constraint": True},
        },
    )
    assert response.status_code == 422


def test_the_response_is_json_serialisable_and_stable():
    """Re-derivable, like every other output here."""
    client = _client()
    payload = {"state": _state().model_dump(mode="json"), "constraints": {}}
    first = client.post("/v1/agent-policy", json=payload).json()
    second = client.post("/v1/agent-policy", json=payload).json()
    assert first == second


def test_healthz_reports_the_contract_and_the_runtime():
    body = _client(_runtime(temperature=0.25, device="cpu")).get("/healthz").json()
    assert body["ok"] is True
    assert body["contract"] == "assistx-agent-policy-v1"
    assert body["temperature"] == 0.25
    assert body["device"] == "cpu"


def test_an_unknown_route_is_a_404_not_a_crash():
    assert _client().get("/v1/not-a-route").status_code == 404


# --- construction ----------------------------------------------------------


def test_a_missing_fastapi_is_reported_as_an_actionable_message():
    """The lazy import exists so the base install stays usable; the error says so."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "fastapi" or name.startswith("fastapi."):
            raise ImportError("no fastapi")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = refuse
    try:
        with pytest.raises(RuntimeError, match=r"\[serve\]"):
            create_app(_runtime())
    finally:
        builtins.__import__ = real_import


def test_the_declared_bind_address_is_all_interfaces_by_default():
    """Documented rather than changed.

    `main()` defaults `--host` to `0.0.0.0` and `/healthz` is unauthenticated while
    disclosing the checkpoint path, temperature and device. That is a reasonable
    default for a sidecar on a private network and a poor one on a shared host, and
    which it is depends on deployment rather than on this code. So this test exists
    to make the default impossible to change by accident and to say where it is
    recorded.
    """
    import inspect

    from my_jev import server

    source = inspect.getsource(server.main)
    assert '"0.0.0.0"' in source, "the bind default moved; update this test and the docs"


# ===========================================================================
# The paths where the response could read as an instruction
# ===========================================================================


def _post(runtime, *, constraints=None):
    state = _state()
    return _client(runtime).post(
        "/v1/agent-policy",
        json={
            "state": state.model_dump(mode="json"),
            "constraints": constraints or {},
        },
    ).json()


def test_a_permitted_action_still_reports_itself_as_advisory_only():
    """The response that most looks like an instruction.

    This one reaches `direct_action` in the AssistX mapping, with nothing gating it
    but the runtime constraints the caller supplied. That is a legitimate answer, and
    it is exactly the answer most likely to be mistaken for an instruction -- so the
    advisory markers have to be on it, not just on the abstentions.
    """
    body = _post(_runtime(model=_acting_model()))
    assert body["resolved"]["disposition"] == "act"
    assert body["assistx"]["policy_action"] == "direct_action"
    assert body["advisory_only"] is True
    assert body["runtime_authority_changed"] is False


def test_an_action_needing_approval_says_so_in_both_payloads():
    """Same path, with the approval gate demanded.

    `requires_approval` and `approval_required` are separate fields; if a consumer
    reads only the smaller one it believes it may proceed.
    """
    body = _post(_runtime(model=_acting_model(approval_likely="true")))
    assert body["resolved"]["disposition"] == "act_with_approval"
    assert body["assistx"]["policy_action"] == "direct_action_with_approval"
    assert body["resolved"]["approval_required"] is True
    assert body["hermes"]["requires_approval"] is True


def test_a_privileged_scope_the_runtime_forbids_is_refused_with_its_reason():
    """The refusal has to survive the trip out through HTTP intact.

    `resolve_agent_policy` appends the reason and downgrades to `propose_action`. What
    was untested is that the reason is still present in the response body rather than
    swallowed on the way out -- a caller seeing only `propose_action` would not know
    why.
    """
    # `external_effect` has to agree with the scope, or the resolver refuses earlier
    # on `external_scope_without_external_effect` and the privileged-scope check is
    # never reached. My first version asserted the privileged reason and got the
    # consistency reason instead, which looked like the wrong refusal.
    body = _post(
        _runtime(
            model=_acting_model(action_scope="privileged", external_effect="true")
        )
    )
    assert body["resolved"]["disposition"] == "propose_action"
    assert body["assistx"]["policy_action"] == "review_dispatch"
    assert any(
        "privileged action" in reason for reason in body["resolved"]["reasons"]
    ), body["resolved"]["reasons"]
    assert body["advisory_only"] is True


def test_a_local_write_the_runtime_forbids_is_refused_too():
    body = _post(
        _runtime(model=_acting_model()),
        constraints={"local_writes_allowed": False},
    )
    assert body["resolved"]["disposition"] == "propose_action"
    assert any(
        "local mutation" in reason for reason in body["resolved"]["reasons"]
    ), body["resolved"]["reasons"]


def test_an_unverified_speaker_cannot_reach_an_action():
    """Identity is a runtime fact, and it outranks the model's confidence."""
    body = _post(
        _runtime(model=_acting_model()),
        constraints={"speaker_verified": False},
    )
    assert body["resolved"]["disposition"] == "clarify"
    assert any(
        "speaker identity" in reason for reason in body["resolved"]["reasons"]
    ), body["resolved"]["reasons"]


def test_an_action_needing_approval_with_no_gate_available_is_not_permitted():
    """Approval required but unobtainable falls back to proposing, not to acting."""
    body = _post(
        _runtime(model=_acting_model(approval_likely="true")),
        constraints={"approval_gate_available": False},
    )
    assert body["resolved"]["disposition"] == "propose_action"
    assert body["resolved"]["approval_required"] is True
    assert any(
        "no approval channel" in reason for reason in body["resolved"]["reasons"]
    ), body["resolved"]["reasons"]


@pytest.mark.parametrize(
    "constraints",
    [
        {},
        {"actions_allowed": False},
        {"speaker_verified": False},
        {"local_writes_allowed": False},
        {"external_actions_allowed": False},
        {"privileged_actions_allowed": False},
        {"approval_gate_available": False},
    ],
)
def test_no_constraint_combination_yields_an_unmarked_action(constraints):
    """Walk the constraint space rather than trusting the handful above.

    Any combination that reaches `direct_action` must still be marked advisory, and
    any that is refused must say why.
    """
    body = _post(_runtime(model=_acting_model()), constraints=constraints)
    assert body["advisory_only"] is True
    if body["assistx"]["policy_action"] == "direct_action":
        assert body["resolved"]["disposition"] in ("act", "act_with_approval")
        assert body["resolved"]["approval_required"] is (
            body["resolved"]["disposition"] == "act_with_approval"
        )
    else:
        # A refusal reaches the caller as one of two mappings: `propose_action` and
        # `abstain` both become `review_dispatch`, while `clarify` becomes
        # `needs_clarification`. Asserting the specific string assumed only the
        # first, and failed on the states that clarify.
        assert body["assistx"]["policy_action"] in (
            "review_dispatch",
            "needs_clarification",
        ), body["assistx"]["policy_action"]
        assert body["resolved"]["reasons"], (
            f"refused with no reason under {constraints}"
        )


def test_no_authority_claiming_field_is_true_on_any_action_path():
    """The recursive walk, on the response most likely to imply authority.

    The other version of this test used a flat prediction, which resolved to `chat`
    and never produced the interesting payload. This one runs it against the acting
    model, where `direct_action` is actually in the body.
    """
    claims = list(_walk_authority_claims(_post(_runtime(model=_acting_model()))))
    assert claims
    for path, value in claims:
        assert value in (False, "review_dispatch", "direct_action_with_approval"), (
            f"{path}={value!r}"
        )


def _walk_authority_claims(node, path=""):
    if isinstance(node, dict):
        for key, value in node.items():
            lowered = key.lower()
            if any(
                word in lowered
                for word in ("authority", "dispatch", "executed", "approved", "placed", "order")
            ):
                yield f"{path}.{key}", value
            yield from _walk_authority_claims(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _walk_authority_claims(item, f"{path}[{index}]")
