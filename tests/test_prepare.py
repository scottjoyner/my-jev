import json

import pytest

from my_jev.data import load_jsonl
from my_jev.prepare import (
    prepare_agent_policy_dataset,
)


def test_prepare_creates_group_safe_four_way_dataset(
    tmp_path,
):
    output = (
        tmp_path
        / "assistx-policy"
    )
    manifest = (
        prepare_agent_policy_dataset(
            output,
            records=80,
            seed=31,
        )
    )

    for name in (
        "source",
        "train",
        "validation",
        "calibration",
        "test",
    ):
        assert (
            output
            / f"{name}.jsonl"
        ).exists()

    assert (
        output
        / "prepare_manifest.json"
    ).exists()
    assert (
        manifest[
            "records_emitted"
        ]
        == 80
    )

    family_sets = {}
    for name in (
        "train",
        "validation",
        "calibration",
        "test",
    ):
        records = load_jsonl(
            output
            / f"{name}.jsonl"
        )
        assert records
        family_sets[name] = {
            str(
                record.metadata[
                    "family_id"
                ]
            )
            for record in records
        }

    names = list(
        family_sets
    )
    for index, left in enumerate(
        names
    ):
        for right in names[
            index + 1 :
        ]:
            assert family_sets[
                left
            ].isdisjoint(
                family_sets[
                    right
                ]
            )

    on_disk = json.loads(
        (
            output
            / "prepare_manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    assert (
        on_disk["source"]["sha256"]
        == manifest[
            "source"
        ]["sha256"]
    )


def test_prepare_refuses_to_overwrite_without_force(
    tmp_path,
):
    output = (
        tmp_path
        / "assistx-policy"
    )
    prepare_agent_policy_dataset(
        output,
        records=40,
        seed=7,
    )

    with pytest.raises(
        FileExistsError,
        match="use force=True",
    ):
        prepare_agent_policy_dataset(
            output,
            records=40,
            seed=7,
        )

    replaced = (
        prepare_agent_policy_dataset(
            output,
            records=40,
            seed=9,
            force=True,
        )
    )
    assert (
        replaced["seed"]
        == 9
    )


def test_prepare_rejects_dataset_too_small_for_requested_splits(
    tmp_path,
):
    with pytest.raises(
        ValueError,
        match="too small",
    ):
        prepare_agent_policy_dataset(
            tmp_path / "tiny",
            records=4,
            seed=3,
        )
