from collections import defaultdict

from my_jev.agentic_synth import (
    GENERATOR_VERSION,
    generate_agent_policy_records,
)


def test_agentic_synth_is_deterministic():
    first = generate_agent_policy_records(
        20,
        seed=29,
    )
    second = generate_agent_policy_records(
        20,
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
        36,
        seed=31,
    )
    families = defaultdict(list)
    for record in records:
        families[record.metadata["family_id"]].append(record)

    paired = [
        items
        for items in families.values()
        if len(items) == 2
    ]
    assert paired
    for left, right in paired:
        assert left.targets == right.targets
        assert left.state != right.state
        assert (
            left.metadata["generator_version"]
            == GENERATOR_VERSION
        )
