from my_jev.fleet_synth import generate_fleet_records


def test_fleet_synth_is_deterministic():
    first = [x.model_dump() for x in generate_fleet_records(12, seed=9)]
    second = [x.model_dump() for x in generate_fleet_records(12, seed=9)]
    assert first == second


def test_fleet_synth_never_exposes_hostnames_or_dispatch_authority():
    records = generate_fleet_records(20, seed=7)
    assert records
    for record in records:
        assert record.metadata["dispatch_allowed"] is False
        assert "node-" not in record.state


def test_node_permutation_counterfactual_preserves_targets():
    records = generate_fleet_records(30, seed=5)
    by_family = {}
    for record in records:
        by_family.setdefault(record.metadata["family_id"], []).append(record)

    paired = [items for items in by_family.values() if len(items) > 1]
    assert paired
    for items in paired:
        target_dumps = {
            str(
                {
                    name: target.model_dump()
                    for name, target in record.targets.items()
                }
            )
            for record in items
        }
        assert len(target_dumps) == 1
