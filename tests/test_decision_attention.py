import pytest
import torch

from my_jev.decision_attention import (
    DecisionTapMixer,
    ExternalDecisionAttentionHead,
)


def test_tap_mixer_learns_convex_combination():
    mixer = DecisionTapMixer(
        3
    )
    values = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ]
    )

    mixed = mixer(
        values,
        tap_dim=1,
    )

    assert mixed.shape == (
        1,
        2,
    )
    assert torch.allclose(
        mixed,
        torch.tensor(
            [
                [
                    3.0,
                    4.0,
                ]
            ]
        ),
    )
    assert float(
        mixer.weights.sum()
    ) == pytest.approx(
        1.0
    )


def test_external_decision_attention_scores_options_and_backprops():
    torch.manual_seed(
        101
    )
    head = (
        ExternalDecisionAttentionHead(
            hidden_size=16,
            tap_count=4,
            rank=8,
            dropout=0.0,
        )
    )

    state = torch.randn(
        2,
        4,
        7,
        16,
    )
    options = torch.randn(
        2,
        3,
        4,
        16,
    )
    state_mask = torch.tensor(
        [
            [
                1,
                1,
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
                1,
                1,
            ],
        ],
        dtype=torch.bool,
    )
    option_mask = torch.tensor(
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
    )

    logits = head(
        state,
        state_mask,
        options,
        option_mask,
    )

    assert logits.shape == (
        2,
        3,
    )
    assert torch.isfinite(
        logits[
            option_mask
        ]
    ).all()
    assert (
        logits[
            0,
            2,
        ]
        < -1e20
    )

    logits[
        option_mask
    ].sum().backward()

    assert (
        head.query.weight.grad
        is not None
    )
    assert (
        head.state_taps.logits.grad
        is not None
    )
    assert (
        head.option_taps.logits.grad
        is not None
    )


def test_external_decision_attention_shuffle_control_changes_scores():
    torch.manual_seed(
        103
    )
    head = (
        ExternalDecisionAttentionHead(
            hidden_size=8,
            tap_count=2,
            rank=4,
            dropout=0.0,
        )
    )
    state = torch.randn(
        3,
        2,
        5,
        8,
    )
    options = torch.randn(
        3,
        2,
        2,
        8,
    )
    state_mask = torch.ones(
        3,
        5,
        dtype=torch.bool,
    )
    option_mask = torch.ones(
        3,
        2,
        dtype=torch.bool,
    )

    normal = head(
        state,
        state_mask,
        options,
        option_mask,
    )
    shuffled = head(
        state,
        state_mask,
        options,
        option_mask,
        shuffle_state=True,
    )

    assert not torch.allclose(
        normal,
        shuffled,
    )


def test_external_decision_attention_rejects_wrong_hidden_size():
    head = (
        ExternalDecisionAttentionHead(
            hidden_size=8,
            tap_count=2,
            rank=4,
        )
    )

    with pytest.raises(
        ValueError,
        match="hidden size",
    ):
        head(
            torch.randn(
                1,
                2,
                3,
                7,
            ),
            torch.ones(
                1,
                3,
                dtype=torch.bool,
            ),
            torch.randn(
                1,
                2,
                2,
                7,
            ),
            torch.ones(
                1,
                2,
                dtype=torch.bool,
            ),
        )
