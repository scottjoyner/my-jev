from pathlib import Path

from my_jev.experiment import (
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
    assert len(digest) == 64
