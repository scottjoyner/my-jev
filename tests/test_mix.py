from my_jev.agentic_synth import (
    generate_agent_policy_records,
)
from my_jev.data import (
    dump_jsonl,
)
from my_jev.mix import (
    MixSource,
    mix_records,
)
from my_jev.schema import (
    TargetSpec,
)


def _write(path, records):
    dump_jsonl(
        iter(records),
        path,
    )


def test_mix_is_deterministic_and_tags_source_hash(
    tmp_path,
):
    synthetic = (
        generate_agent_policy_records(
            20,
            seed=5,
        )
    )
    path = (
        tmp_path
        / "synthetic.jsonl"
    )
    _write(
        path,
        synthetic,
    )
    source = MixSource(
        "synthetic",
        path,
        max_records=8,
    )

    first, first_meta = mix_records(
        [source],
        seed=17,
    )
    second, second_meta = mix_records(
        [source],
        seed=17,
    )

    assert [
        record.model_dump(
            mode="json"
        )
        for record in first
    ] == [
        record.model_dump(
            mode="json"
        )
        for record in second
    ]
    assert first_meta == second_meta
    assert len(first) == 8
    assert all(
        record.metadata[
            "mix_source"
        ]
        == "synthetic"
        for record in first
    )
    assert all(
        len(
            record.metadata[
                "mix_source_sha256"
            ]
        )
        == 64
        for record in first
    )


def test_mix_rejects_unlabeled_shadow_records(
    tmp_path,
):
    record = (
        generate_agent_policy_records(
            1,
            seed=9,
        )[0]
        .model_copy(
            deep=True
        )
    )
    record.targets = None
    record.metadata[
        "label_status"
    ] = "unlabeled"
    record.metadata[
        "label_source"
    ] = None

    path = (
        tmp_path
        / "shadow.jsonl"
    )
    _write(
        path,
        [record],
    )

    try:
        mix_records(
            [
                MixSource(
                    "shadow",
                    path,
                )
            ]
        )
    except ValueError as exc:
        assert (
            "unlabeled record"
            in str(exc)
        )
    else:
        raise AssertionError(
            "unlabeled record "
            "should fail closed"
        )


def test_mix_allows_partial_adjudicated_targets(
    tmp_path,
):
    record = (
        generate_agent_policy_records(
            1,
            seed=11,
        )[0]
        .model_copy(
            deep=True
        )
    )
    route = (
        record.targets[
            "route"
        ]
    )
    record.targets = {
        "route": route,
    }
    record.metadata[
        "label_status"
    ] = "partially_adjudicated"
    record.metadata[
        "label_source"
    ] = "trajectory_adjudication"

    path = (
        tmp_path
        / "real.jsonl"
    )
    _write(
        path,
        [record],
    )
    mixed, _ = mix_records(
        [
            MixSource(
                "real",
                path,
            )
        ]
    )

    assert len(mixed) == 1
    assert set(
        mixed[0].targets
        or {}
    ) == {"route"}


def test_mix_deduplicates_exact_semantic_records(
    tmp_path,
):
    record = (
        generate_agent_policy_records(
            1,
            seed=13,
        )[0]
    )
    first = (
        tmp_path
        / "first.jsonl"
    )
    second = (
        tmp_path
        / "second.jsonl"
    )
    _write(
        first,
        [record],
    )
    _write(
        second,
        [
            record.model_copy(
                deep=True
            )
        ],
    )

    mixed, meta = mix_records(
        [
            MixSource(
                "first",
                first,
            ),
            MixSource(
                "second",
                second,
            ),
        ]
    )

    assert len(mixed) == 1
    assert (
        meta[
            "exact_duplicates_dropped"
        ]
        == 1
    )


def test_mix_rejects_conflicting_targets_for_identical_state(
    tmp_path,
):
    first_record = (
        generate_agent_policy_records(
            1,
            seed=15,
        )[0]
    )
    second_record = (
        first_record.model_copy(
            deep=True
        )
    )
    assert (
        second_record.targets
        is not None
    )
    current = (
        second_record.targets[
            "route"
        ]
    )
    current_index = (
        current.index
        if current.index is not None
        else 0
    )
    option_count = len(
        second_record.questions[
            "route"
        ].options
        or []
    )
    second_record.targets[
        "route"
    ] = TargetSpec(
        index=(
            current_index + 1
        )
        % option_count
    )

    first = (
        tmp_path
        / "first.jsonl"
    )
    second = (
        tmp_path
        / "second.jsonl"
    )
    _write(
        first,
        [first_record],
    )
    _write(
        second,
        [second_record],
    )

    try:
        mix_records(
            [
                MixSource(
                    "first",
                    first,
                ),
                MixSource(
                    "second",
                    second,
                ),
            ]
        )
    except ValueError as exc:
        assert (
            "conflicting targets"
            in str(exc)
        )
    else:
        raise AssertionError(
            "conflicting duplicate "
            "should fail closed"
        )


def test_mix_rejects_self_prediction_as_label_source(
    tmp_path,
):
    record = (
        generate_agent_policy_records(
            1,
            seed=19,
        )[0]
        .model_copy(
            deep=True
        )
    )
    record.metadata[
        "label_source"
    ] = "shadow_prediction"
    path = (
        tmp_path
        / "bad.jsonl"
    )
    _write(
        path,
        [record],
    )

    try:
        mix_records(
            [
                MixSource(
                    "bad",
                    path,
                )
            ]
        )
    except ValueError as exc:
        assert (
            "untrusted label_source"
            in str(exc)
        )
    else:
        raise AssertionError(
            "self-predicted labels "
            "must be rejected"
        )
