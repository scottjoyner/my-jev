from pathlib import Path

import pytest

from my_jev.data import dump_jsonl
from my_jev.fleet_benchmark import (
    load_family_records,
    load_states,
)
from my_jev.fleet_synth import (
    generate_fleet_records,
)
from my_jev.fleet_policy import FleetNodeSnapshot, FleetPlacementState


def test_load_states_round_trip_and_requires_content(tmp_path: Path):
    path = tmp_path / "states.jsonl"
    state = FleetPlacementState(
        workload_id="bench",
        workload_type="gpu",
        required_capabilities=["gpu"],
        nodes=[
            FleetNodeSnapshot(
                node_id="opaque-a",
                capabilities=["gpu"],
                ram_free_gib=32,
                vram_free_gib=24,
            )
        ],
    )
    path.write_text(state.model_dump_json() + "\n", encoding="utf-8")
    loaded = load_states(path)
    assert loaded == [state]

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one state"):
        load_states(empty)


def test_load_family_records_requires_verified_fleet_family_data(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "families.jsonl"
    )
    records = generate_fleet_records(
        6,
        seed=71,
    )
    dump_jsonl(
        iter(records),
        path,
    )

    loaded = load_family_records(
        path
    )
    assert len(loaded) == 6
    assert all(
        record.metadata["family_id"]
        for record in loaded
    )
    assert all(
        record.targets
        for record in loaded
    )


def test_load_family_records_rejects_unlabeled_rows(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "families.jsonl"
    )
    record = generate_fleet_records(
        3,
        seed=73,
    )[0]
    record.targets = None
    dump_jsonl(
        iter([record]),
        path,
    )

    with pytest.raises(
        ValueError,
        match="unlabeled",
    ):
        load_family_records(
            path
        )
