from collections import defaultdict

from my_jev.fleet_family_eval import (
    evaluate_fleet_families,
)
from my_jev.fleet_synth import (
    generate_fleet_records,
)


def _target_prediction(record):
    prediction = {}
    for name, question in (
        record.questions.items()
    ):
        options = question.options or []
        target = record.targets[name]
        if target.index is not None:
            index = target.index
        else:
            index = max(
                range(
                    len(
                        target.distribution
                    )
                ),
                key=target.distribution.__getitem__,
            )
        prediction[name] = {
            option: (
                1.0
                if offset == index
                else 0.0
            )
            for offset, option in enumerate(
                options
            )
        }
    return prediction


def test_verified_target_predictions_pass_family_controls():
    records = generate_fleet_records(
        24,
        seed=51,
    )
    predictions = [
        _target_prediction(record)
        for record in records
    ]

    metrics = evaluate_fleet_families(
        records,
        predictions,
    )

    assert (
        metrics[
            "complete_families"
        ]
        == 8
    )
    assert (
        metrics[
            "incomplete_families"
        ]
        == 0
    )
    assert (
        metrics[
            "invariant_top1_agreement"
        ]
        == 1.0
    )
    assert (
        metrics[
            "invariant_mean_abs_probability_delta"
        ]
        == 0.0
    )
    assert (
        metrics[
            "semantic_new_target_accuracy"
        ]
        == 1.0
    )
    assert (
        metrics[
            "semantic_probability_direction_rate"
        ]
        == 1.0
    )
    assert (
        metrics[
            "semantic_unchanged_top1_agreement"
        ]
        == 1.0
    )
    assert (
        metrics[
            "family_success_rate"
        ]
        == 1.0
    )
    assert (
        metrics["dispatch_allowed"]
        is False
    )


def test_static_family_predictions_fail_semantic_change_controls():
    records = generate_fleet_records(
        24,
        seed=53,
    )
    by_family = defaultdict(list)
    for index, record in enumerate(
        records
    ):
        by_family[
            record.metadata[
                "family_id"
            ]
        ].append(
            (index, record)
        )

    predictions = [
        None
        for _ in records
    ]
    for items in by_family.values():
        permutation = next(
            record
            for _, record in items
            if record.metadata[
                "variant"
            ]
            == "node_permutation"
        )
        base = next(
            record
            for _, record in items
            if (
                record.metadata[
                    "variant"
                ]
                != "node_permutation"
                and {
                    name: target.index
                    for name, target in (
                        record.targets
                        or {}
                    ).items()
                }
                == {
                    name: target.index
                    for name, target in (
                        permutation.targets
                        or {}
                    ).items()
                }
            )
        )
        frozen = _target_prediction(
            base
        )
        for index, _ in items:
            predictions[index] = frozen

    metrics = evaluate_fleet_families(
        records,
        predictions,
    )

    assert (
        metrics[
            "invariant_top1_agreement"
        ]
        == 1.0
    )
    assert (
        metrics[
            "semantic_new_target_accuracy"
        ]
        < 1.0
    )
    assert (
        metrics[
            "semantic_probability_direction_rate"
        ]
        == 0.0
    )
    assert (
        metrics[
            "family_success_rate"
        ]
        == 0.0
    )


def test_incomplete_family_is_reported_not_promoted_as_complete():
    records = generate_fleet_records(
        6,
        seed=59,
    )
    records = records[:-1]
    predictions = [
        _target_prediction(record)
        for record in records
    ]

    metrics = evaluate_fleet_families(
        records,
        predictions,
    )

    assert (
        metrics[
            "families"
        ]
        == 2
    )
    assert (
        metrics[
            "complete_families"
        ]
        == 1
    )
    assert (
        metrics[
            "incomplete_families"
        ]
        == 1
    )
