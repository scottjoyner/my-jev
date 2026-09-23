from pathlib import Path

from my_jev.experiment import (
    _fleet_benchmark_command,
    load_experiment_spec,
)


def test_fleet_experiment_spec_is_separate_observer_lane():
    spec, digest = load_experiment_spec(
        Path(
            "configs/experiments/"
            "fleet-modernbert.toml"
        )
    )

    assert (
        spec.experiment.name
        == "fleet-placement-modernbert"
    )
    assert (
        spec.model.backend
        == "encoder_option_query"
    )
    assert (
        spec.data.train
        == (
            "data/fleet-placement-v1/"
            "train.jsonl"
        )
    )
    assert spec.train.seed == 31
    assert (
        spec.train.max_state_length
        == 1536
    )
    assert (
        "max_policy_consistency_"
        "violation_rate"
        not in spec.gates
    )
    assert (
        spec.benchmark.fleet_states
        == (
            "data/fleet-placement-v1/"
            "benchmark-states.jsonl"
        )
    )
    assert (
        spec.benchmark.fleet_family_data
        == (
            "data/fleet-placement-v1/"
            "test.jsonl"
        )
    )
    assert (
        spec.benchmark
        .fleet_min_family_semantic_probability_direction_rate
        == 0.80
    )
    assert len(digest) == 64


def test_fleet_experiment_command_includes_family_gates(
    tmp_path,
):
    spec, _ = load_experiment_spec(
        Path(
            "configs/experiments/"
            "fleet-modernbert.toml"
        )
    )
    command = _fleet_benchmark_command(
        spec,
        tmp_path,
        tmp_path / "run",
    )

    assert command is not None
    assert "--family-data" in command
    assert (
        "--min-family-semantic-"
        "probability-direction-rate"
        in command
    )
    assert (
        "--min-family-success-rate"
        in command
    )
    assert (
        "--max-family-invariant-"
        "mean-abs-probability-delta"
        in command
    )
