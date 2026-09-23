import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "decision_backbones"
    / "bonsai2-27b.toml"
)


def test_bonsai_provider_config_matches_runtime_contract():
    payload = tomllib.loads(
        CONFIG.read_text(
            encoding="utf-8"
        )
    )

    assert (
        payload[
            "provider"
        ][
            "model_repo"
        ]
        == (
            "prism-ml/"
            "Ternary-Bonsai-2-27B-gguf"
        )
    )
    assert (
        payload[
            "provider"
        ][
            "backbone_trainable"
        ]
        is False
    )
    assert (
        payload[
            "shape"
        ][
            "hidden_size"
        ]
        == 5120
    )
    assert (
        payload[
            "shape"
        ][
            "num_layers"
        ]
        == 64
    )
    assert (
        payload[
            "shape"
        ][
            "layer_taps"
        ]
        == [
            31,
            47,
            63,
        ]
    )
    assert (
        payload[
            "shape"
        ][
            "include_final_prenorm"
        ]
        is True
    )
    assert (
        payload[
            "head"
        ][
            "kind"
        ]
        == (
            "external_decision_attention"
        )
    )
    assert (
        payload[
            "head"
        ][
            "rank"
        ]
        == 256
    )
    assert (
        payload[
            "authority"
        ][
            "dispatch_allowed"
        ]
        is False
    )
