from pathlib import Path

import pytest

from my_jev.fleet_benchmark import load_states
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
