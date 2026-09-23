import io

import pytest
import torch

from my_jev.activation_protocol import (
    ActivationFrame,
    read_activation_frame,
    roundtrip_activation_frame,
    write_activation_frame,
)


def _frame() -> ActivationFrame:
    return ActivationFrame(
        provider=(
            "bonsai2-ternary-llama-cpp"
        ),
        model_sha256="a" * 64,
        runtime_revision="b" * 40,
        prompt_contract=(
            "assistx-agent-policy-v1"
        ),
        state_taps=torch.randn(
            2,
            4,
            5,
            8,
        ),
        state_mask=torch.tensor(
            [
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                ],
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
            ],
            dtype=torch.bool,
        ),
        option_taps=torch.randn(
            2,
            3,
            4,
            8,
        ),
        option_mask=torch.tensor(
            [
                [
                    1,
                    1,
                    0,
                ],
                [
                    1,
                    1,
                    1,
                ],
            ],
            dtype=torch.bool,
        ),
    )


def test_activation_frame_roundtrip_preserves_shapes_and_provenance():
    original = _frame()
    restored = (
        roundtrip_activation_frame(
            original,
            dtype="float16",
        )
    )

    assert (
        restored.provider
        == original.provider
    )
    assert (
        restored.model_sha256
        == original.model_sha256
    )
    assert (
        restored.runtime_revision
        == original.runtime_revision
    )
    assert (
        restored.prompt_contract
        == original.prompt_contract
    )
    assert (
        restored.state_taps.shape
        == original.state_taps.shape
    )
    assert (
        restored.option_taps.shape
        == original.option_taps.shape
    )
    assert torch.equal(
        restored.state_mask,
        original.state_mask,
    )
    assert torch.equal(
        restored.option_mask,
        original.option_mask,
    )
    assert (
        restored.state_taps.dtype
        == torch.float16
    )


def test_activation_frame_rejects_bad_option_hidden_size():
    frame = _frame()
    broken = ActivationFrame(
        provider=frame.provider,
        model_sha256=(
            frame.model_sha256
        ),
        runtime_revision=(
            frame.runtime_revision
        ),
        prompt_contract=(
            frame.prompt_contract
        ),
        state_taps=(
            frame.state_taps
        ),
        state_mask=(
            frame.state_mask
        ),
        option_taps=torch.randn(
            2,
            3,
            4,
            7,
        ),
        option_mask=(
            frame.option_mask
        ),
    )

    with pytest.raises(
        ValueError,
        match="hidden size",
    ):
        broken.validate()


def test_activation_frame_rejects_unknown_dtype():
    with pytest.raises(
        ValueError,
        match="unsupported activation dtype",
    ):
        write_activation_frame(
            io.BytesIO(),
            _frame(),
            dtype="bfloat16",
        )


def test_activation_frame_rejects_truncated_payload():
    buffer = io.BytesIO()
    write_activation_frame(
        buffer,
        _frame(),
    )
    payload = (
        buffer.getvalue()
    )

    with pytest.raises(
        EOFError,
        match="truncated",
    ):
        read_activation_frame(
            io.BytesIO(
                payload[:-7]
            )
        )
