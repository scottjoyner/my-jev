from pathlib import Path

import torch

from my_jev.activation_protocol import (
    ActivationFrame,
    ActivationQuestionGroup,
    write_activation_frame,
)
from my_jev.bonsai_smoke import (
    run_bonsai_smoke,
)


def test_bonsai_smoke_runs_external_head_without_authority(
    tmp_path: Path,
):
    frame = ActivationFrame(
        provider=(
            "bonsai2-ternary-llama-cpp"
        ),
        model_sha256="a" * 64,
        runtime_revision="b" * 40,
        prompt_contract=(
            "my-jev-typed-decision-prompt-v1"
        ),
        state_taps=torch.randn(
            1,
            4,
            5,
            16,
        ),
        state_mask=torch.ones(
            1,
            5,
            dtype=torch.bool,
        ),
        option_taps=torch.randn(
            1,
            2,
            4,
            16,
        ),
        option_mask=torch.ones(
            1,
            2,
            dtype=torch.bool,
        ),
        groups=(
            ActivationQuestionGroup(
                record_index=0,
                name="route",
                type="choice",
                options=(
                    "chat",
                    "act",
                ),
                option_start=0,
                option_end=2,
            ),
        ),
    )

    path = (
        tmp_path
        / "frame.bin"
    )
    with path.open(
        "wb"
    ) as handle:
        write_activation_frame(
            handle,
            frame,
            dtype="float32",
        )

    result = run_bonsai_smoke(
        path,
        rank=8,
        seed=131,
    )

    assert (
        result["frame"][
            "tap_count"
        ]
        == 4
    )
    assert (
        result["head"][
            "trained"
        ]
        is False
    )
    assert (
        len(
            result[
                "predictions"
            ]
        )
        == 1
    )
    assert (
        result[
            "quality_claim"
        ][
            "validated"
        ]
        is False
    )
    assert (
        result[
            "authority_boundary"
        ][
            "dispatch_allowed"
        ]
        is False
    )
