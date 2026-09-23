import json

from my_jev.synth import (
    DOMAINS,
    GENERATOR_VERSION,
    generate_synthetic_records,
)


def test_synthetic_generation_is_deterministic_and_valid():
    first = generate_synthetic_records(12, seed=41)
    second = generate_synthetic_records(12, seed=41)

    first_json = [
        json.dumps(
            item.model_dump(mode="json"),
            sort_keys=True,
        )
        for item in first
    ]
    second_json = [
        json.dumps(
            item.model_dump(mode="json"),
            sort_keys=True,
        )
        for item in second
    ]

    assert first_json == second_json
    assert len(first) == 12
    assert {
        item.metadata["domain"]
        for item in first
    } == set(DOMAINS)
    assert all(
        item.metadata["generator_version"]
        == GENERATOR_VERSION
        for item in first
    )
    assert all(
        len(item.questions) == 4
        for item in first
    )
    assert all(
        item.targets
        and set(item.targets) == set(item.questions)
        for item in first
    )


def test_counterfactual_families_are_paired_and_change_state():
    records = generate_synthetic_records(10, seed=5)
    families: dict[str, list] = {}

    for record in records:
        families.setdefault(
            record.metadata["family_id"],
            [],
        ).append(record)

    assert all(
        1 <= len(items) <= 2
        for items in families.values()
    )
    paired = [
        items
        for items in families.values()
        if len(items) == 2
    ]
    assert paired
    assert all(
        items[0].state != items[1].state
        for items in paired
    )
