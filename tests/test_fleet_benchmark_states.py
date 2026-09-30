from pathlib import Path

from my_jev.fleet_benchmark_states import generate_benchmark_states, write_benchmark_states
from my_jev.fleet_policy import eligible_nodes


def test_benchmark_states_are_deterministic_and_initially_eligible():
    left = generate_benchmark_states(12, seed=47)
    right = generate_benchmark_states(12, seed=47)
    assert [state.model_dump_json() for state in left] == [
        state.model_dump_json() for state in right
    ]
    assert all(eligible_nodes(state) for state in left)


def test_benchmark_state_manifest_matches_bytes(tmp_path: Path):
    path = tmp_path / "fleet-states.jsonl"
    manifest = write_benchmark_states(path, count=8, seed=47)
    assert manifest["records"] == 8
    assert len(manifest["sha256"]) == 64
    text = path.read_text(encoding="utf-8")
    assert "opaque-" in text
    assert '"source":"fleet_benchmark_v1"' in text
