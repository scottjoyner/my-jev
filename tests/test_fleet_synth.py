from collections import defaultdict

from my_jev.fleet_synth import (
    GENERATOR_VERSION,
    generate_fleet_records,
)


def _targets(record):
    return {
        name: target.model_dump()
        for name, target in (
            record.targets or {}
        ).items()
    }


def test_fleet_synth_is_deterministic():
    first = [
        item.model_dump()
        for item in generate_fleet_records(
            24,
            seed=9,
        )
    ]
    second = [
        item.model_dump()
        for item in generate_fleet_records(
            24,
            seed=9,
        )
    ]
    assert first == second


def test_fleet_synth_never_exposes_identity_or_provenance_to_model():
    records = generate_fleet_records(
        24,
        seed=7,
    )
    assert records

    for record in records:
        assert (
            record.metadata[
                "dispatch_allowed"
            ]
            is False
        )
        assert "fleet-" not in record.state
        assert "family_id" not in record.state
        assert "scenario" not in record.state
        assert "variant" not in record.state
        assert (
            record.metadata[
                "generator_version"
            ]
            == GENERATOR_VERSION
        )
        assert (
            record.metadata[
                "label_source"
            ]
            == "programmatic_verifier"
        )


def test_node_permutation_preserves_state_and_targets():
    records = generate_fleet_records(
        36,
        seed=5,
    )
    families = defaultdict(list)
    for record in records:
        families[
            record.metadata["family_id"]
        ].append(record)

    checked = 0
    for items in families.values():
        by_variant = {
            item.metadata["variant"]: item
            for item in items
        }
        if (
            "node_permutation"
            not in by_variant
        ):
            continue

        permuted = by_variant[
            "node_permutation"
        ]
        base_candidates = [
            item
            for item in items
            if item.metadata[
                "variant"
            ]
            in {
                "base",
                "fits",
                "checkpointable",
            }
        ]
        assert len(base_candidates) == 1
        base = base_candidates[0]

        assert base.state == permuted.state
        assert (
            _targets(base)
            == _targets(permuted)
        )
        checked += 1

    assert checked > 0


def test_semantic_counterfactuals_change_targets_inside_family():
    records = generate_fleet_records(
        48,
        seed=13,
    )
    families = defaultdict(list)
    for record in records:
        families[
            record.metadata["family_id"]
        ].append(record)

    changed = 0
    for items in families.values():
        if len(items) < 3:
            continue

        unique_targets = {
            str(_targets(item))
            for item in items
        }
        if len(unique_targets) > 1:
            changed += 1

    assert changed > 0


def test_families_are_three_record_counterfactual_sets_when_complete():
    records = generate_fleet_records(
        30,
        seed=17,
    )
    families = defaultdict(list)
    for record in records:
        families[
            record.metadata["family_id"]
        ].append(record)

    assert families
    assert all(
        len(items) == 3
        for items in families.values()
    )
