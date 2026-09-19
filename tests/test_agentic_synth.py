from collections import defaultdict

from my_jev.agentic_synth import (
    GENERATOR_VERSION,
    generate_agent_policy_records,
)


def _families(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[
            record.metadata["family_id"]
        ].append(record)
    return grouped


def test_agentic_synth_is_deterministic():
    first = generate_agent_policy_records(
        40,
        seed=29,
    )
    second = generate_agent_policy_records(
        40,
        seed=29,
    )
    assert [
        item.model_dump(mode="json")
        for item in first
    ] == [
        item.model_dump(mode="json")
        for item in second
    ]


def test_authority_counterfactuals_keep_semantic_targets_equal():
    records = generate_agent_policy_records(
        80,
        seed=31,
    )
    families = _families(records)
    authority_pairs = [
        items
        for items in families.values()
        if (
            len(items) == 2
            and items[0].metadata[
                "counterfactual_kind"
            ]
            == "authority"
        )
    ]
    assert authority_pairs
    for left, right in authority_pairs:
        assert left.targets == right.targets
        assert left.state != right.state
        assert (
            left.metadata["generator_version"]
            == GENERATOR_VERSION
        )


def test_context_counterfactuals_change_decision_when_referent_is_missing():
    records = generate_agent_policy_records(
        80,
        seed=37,
    )
    families = _families(records)
    context_pairs = [
        items
        for items in families.values()
        if (
            len(items) == 2
            and items[0].metadata[
                "counterfactual_kind"
            ]
            == "context"
        )
    ]
    assert context_pairs

    changed = 0
    for left, right in context_pairs:
        assert left.state != right.state
        if left.targets != right.targets:
            changed += 1
    assert changed == len(context_pairs)


def test_agentic_corpus_covers_real_hermes_behavior_families():
    records = generate_agent_policy_records(
        160,
        seed=41,
    )
    scenarios = {
        record.metadata.get("scenario")
        for record in records
    }
    assert {
        "chat_explain",
        "create_task_graph",
        "read_only_status",
        "local_repo_write",
        "memory_write",
        "schedule_reminder",
        "send_message",
        "remote_repo_mutation",
        "delegate_single_agent",
        "privileged_operation",
        "cancel_active_work",
        "authority_escalation",
    }.issubset(scenarios)
