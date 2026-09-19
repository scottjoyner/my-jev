import json

import pytest

from my_jev.data import load_jsonl
from my_jev.fleet_prepare import (
    prepare_fleet_dataset,
)


def test_prepare_fleet_creates_group_safe_four_way_dataset(
    tmp_path,
):
    output = (
        tmp_path
        / "fleet-placement"
    )
    manifest = prepare_fleet_dataset(
        output,
        records=60,
        seed=31,
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
        manifest["records_emitted"]
        == 60
    )
    assert (
        manifest["group_key"]
        == "family_id"
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

    names = list(family_sets)
    for index, left in enumerate(
        names
    ):
        for right in names[
            index + 1 :
        ]:
            assert family_sets[
                left
            ].isdisjoint(
                family_sets[right]
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


def test_prepare_fleet_refuses_overwrite_without_force(
    tmp_path,
):
    output = (
        tmp_path
        / "fleet-placement"
    )
    prepare_fleet_dataset(
        output,
        records=30,
        seed=7,
    )

    with pytest.raises(
        FileExistsError,
        match="use force=True",
    ):
        prepare_fleet_dataset(
            output,
            records=30,
            seed=7,
        )

    replaced = prepare_fleet_dataset(
        output,
        records=30,
        seed=9,
        force=True,
    )
    assert replaced["seed"] == 9


def test_prepare_fleet_rejects_tiny_dataset(
    tmp_path,
):
    with pytest.raises(
        ValueError,
        match="records must be >= 12",
    ):
        prepare_fleet_dataset(
            tmp_path / "tiny",
            records=9,
        )
