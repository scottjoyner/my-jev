from my_jev.split import split_records
from my_jev.synth import generate_synthetic_records


def test_group_safe_split_keeps_families_together():
    records = generate_synthetic_records(60, seed=13)
    splits, group_key = split_records(
        records,
        train_fraction=0.6,
        validation_fraction=0.1,
        calibration_fraction=0.1,
        seed=19,
        group_key="auto",
    )

    assert group_key == "family_id"
    assert sum(len(items) for items in splits.values()) == 60
    assert all(splits[name] for name in splits)

    family_sets = {
        name: {
            item.metadata["family_id"]
            for item in items
        }
        for name, items in splits.items()
    }
    names = list(family_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert family_sets[left].isdisjoint(
                family_sets[right]
            )


def test_record_level_split_still_available():
    records = generate_synthetic_records(20, seed=3)
    splits, group_key = split_records(
        records,
        train_fraction=0.5,
        validation_fraction=0.1,
        calibration_fraction=0.1,
        seed=7,
        group_key="none",
    )

    assert group_key is None
    assert sum(len(items) for items in splits.values()) == 20
