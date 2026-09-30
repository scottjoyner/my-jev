from pathlib import Path

from my_jev.experiment import (
    load_experiment_spec,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = (
    ROOT
    / "configs"
    / "experiments"
    / "assistx-modernbert-r9700-scaleup.toml"
)


def test_r9700_scaleup_is_full_assistx_run_without_fleet_gate():
    spec, digest = (
        load_experiment_spec(
            SPEC
        )
    )

    assert len(digest) == 64
    assert (
        spec.experiment.name
        == (
            "assistx-policy-modernbert-"
            "r9700-scaleup"
        )
    )
    assert (
        spec.experiment.parent
        == ""
    )
    assert (
        spec.model.backbone
        == "answerdotai/ModernBERT-base"
    )
    assert (
        spec.train.epochs
        == 3
    )
    assert (
        spec.train.batch_size
        == 2
    )
    assert (
        spec.train.grad_accum
        == 8
    )
    assert (
        spec.train.bf16
        is True
    )
    assert (
        spec.train.freeze_backbone
        is False
    )
    assert (
        spec.train.gradient_checkpointing
        is True
    )
    assert (
        "assistx-policy-r9700-scaleup-v1"
        in spec.data.train
    )
    assert (
        spec.benchmark.fleet_states
        is None
    )
    assert (
        spec.benchmark.fleet_family_data
        is None
    )
    assert (
        spec.gates[
            "min_accuracy"
        ]
        == 0.80
    )
    assert (
        spec.gates[
            "max_ece"
        ]
        == 0.10
    )
