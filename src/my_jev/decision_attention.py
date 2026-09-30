from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DecisionAttentionShape:
    hidden_size: int
    tap_count: int
    rank: int


class DecisionTapMixer(nn.Module):
    """Learn a convex mixture over frozen-backbone activation taps."""

    def __init__(
        self,
        tap_count: int,
    ) -> None:
        super().__init__()
        if tap_count < 1:
            raise ValueError(
                "tap_count must be >= 1"
            )
        self.tap_count = tap_count
        self.logits = nn.Parameter(
            torch.zeros(
                tap_count,
            )
        )

    @property
    def weights(
        self,
    ) -> Tensor:
        return torch.softmax(
            self.logits,
            dim=0,
        )

    def forward(
        self,
        values: Tensor,
        *,
        tap_dim: int,
    ) -> Tensor:
        if (
            values.shape[
                tap_dim
            ]
            != self.tap_count
        ):
            raise ValueError(
                "activation tap dimension "
                f"{values.shape[tap_dim]} "
                "does not match configured "
                f"tap_count={self.tap_count}"
            )

        weights = self.weights
        shape = [
            1
            for _ in range(
                values.ndim
            )
        ]
        shape[
            tap_dim
        ] = self.tap_count
        return (
            values
            * weights.view(
                *shape
            ).to(
                dtype=values.dtype,
                device=values.device,
            )
        ).sum(
            dim=tap_dim
        )


class ExternalDecisionAttentionHead(nn.Module):
    """Score typed options over hidden states from a frozen external model.

    State activations retain token memory:
      [batch, taps, state_tokens, hidden]

    Option activations are pooled by the activation provider:
      [batch, options, taps, hidden]

    The backbone is intentionally outside this module. This keeps the decision
    head trainable even when the provider is a non-PyTorch low-bit runtime such
    as PrismML's ternary llama.cpp path.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        tap_count: int = 1,
        rank: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError(
                "hidden_size must be >= 1"
            )
        if rank < 1:
            raise ValueError(
                "rank must be >= 1"
            )

        self.shape = (
            DecisionAttentionShape(
                hidden_size=(
                    hidden_size
                ),
                tap_count=tap_count,
                rank=rank,
            )
        )
        self.state_taps = (
            DecisionTapMixer(
                tap_count
            )
        )
        self.option_taps = (
            DecisionTapMixer(
                tap_count
            )
        )
        self.state_norm = (
            nn.LayerNorm(
                hidden_size
            )
        )
        self.option_norm = (
            nn.LayerNorm(
                hidden_size
            )
        )
        self.query = nn.Linear(
            hidden_size,
            rank,
            bias=False,
        )
        self.key = nn.Linear(
            hidden_size,
            rank,
            bias=False,
        )
        self.value = nn.Linear(
            hidden_size,
            rank,
            bias=False,
        )
        self.gate = nn.Sequential(
            nn.Linear(
                rank * 2,
                rank,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                rank,
                1,
            ),
        )
        self.dropout = (
            nn.Dropout(
                dropout
            )
        )

    def forward(
        self,
        state_taps: Tensor,
        state_mask: Tensor,
        option_taps: Tensor,
        option_mask: Tensor,
        *,
        shuffle_state: bool = False,
    ) -> Tensor:
        if state_taps.ndim != 4:
            raise ValueError(
                "state_taps must have shape "
                "[batch, taps, tokens, hidden]"
            )
        if option_taps.ndim != 4:
            raise ValueError(
                "option_taps must have shape "
                "[batch, options, taps, hidden]"
            )
        if state_mask.ndim != 2:
            raise ValueError(
                "state_mask must have shape "
                "[batch, tokens]"
            )
        if option_mask.ndim != 2:
            raise ValueError(
                "option_mask must have shape "
                "[batch, options]"
            )

        batch = state_taps.shape[0]
        if (
            option_taps.shape[0]
            != batch
            or state_mask.shape[0]
            != batch
            or option_mask.shape[0]
            != batch
        ):
            raise ValueError(
                "decision attention batch "
                "dimensions must match"
            )
        if (
            state_taps.shape[-1]
            != self.shape.hidden_size
            or option_taps.shape[-1]
            != self.shape.hidden_size
        ):
            raise ValueError(
                "activation hidden size does "
                "not match head configuration"
            )

        state = self.state_taps(
            state_taps,
            tap_dim=1,
        )
        options = self.option_taps(
            option_taps,
            tap_dim=2,
        )

        state = self.state_norm(
            state
        )
        options = self.option_norm(
            options
        )

        if (
            shuffle_state
            and batch > 1
        ):
            state = state.roll(
                1,
                dims=0,
            )
            state_mask = (
                state_mask.roll(
                    1,
                    dims=0,
                )
            )

        query = self.query(
            options
        )
        key = self.key(
            state
        )
        value = self.value(
            state
        )

        attention_logits = (
            torch.einsum(
                "bcr,blr->bcl",
                query,
                key,
            )
            / math.sqrt(
                self.shape.rank
            )
        )
        attention_logits = (
            attention_logits.masked_fill(
                ~state_mask[
                    :,
                    None,
                    :,
                ].bool(),
                torch.finfo(
                    attention_logits.dtype
                ).min,
            )
        )
        attention = torch.softmax(
            attention_logits.float(),
            dim=-1,
        ).to(
            value.dtype
        )
        attention = self.dropout(
            attention
        )

        attended = torch.einsum(
            "bcl,blr->bcr",
            attention,
            value,
        )
        interaction = torch.cat(
            [
                query,
                attended,
            ],
            dim=-1,
        )
        residual_score = (
            query
            * attended
        ).sum(
            dim=-1
        ) / math.sqrt(
            self.shape.rank
        )
        logits = (
            residual_score
            + self.gate(
                interaction
            ).squeeze(
                -1
            )
        )

        return logits.masked_fill(
            ~option_mask.bool(),
            torch.finfo(
                logits.dtype
            ).min,
        )
