"""Numeric multi-tap encoder that feeds :class:`ExternalDecisionAttentionHead`.

The head is deliberately provider-agnostic — it takes
``[batch, taps, tokens, hidden]`` state activations and
``[batch, options, taps, hidden]`` option activations and never learns about the
thing that produced them. This module is that provider for market data.

Shape contract
--------------
    market window [batch, tokens, features]
      -> value projection           [batch, tokens, hidden]
      -> causal encoder, taps read at T depths
      -> state_taps                 [batch, taps, tokens, hidden]
      -> per-option attention over tokens, per tap
      -> option_taps                [batch, options, taps, hidden]

Option representations
----------------------
Each option ("buy" / "sell" / "hold") owns a learned query vector. That query
attends over the token sequence *independently at every tap*, so each option gets
its own pooled summary of the same history and the head's tap mixer can weigh
taps differently per decision. That is the whole point of keeping token-level
memory: "hold" and "buy" should be able to attend to different parts of the
window rather than share one pooled vector.

Causality
---------
The encoder is a causal Transformer encoder over the window and the record for
bar ``t`` uses a window ending at ``t``, so nothing in the input path can see
bar ``t+1``. The label is the only place future information is used.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from . import OPTIONS

#: Taps are read at these fractions of the encoder depth. Four mirrors the
#: Bonsai lane's tap count so the two providers are interchangeable in shape.
TAP_FRACTIONS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)


class MarketTapEncoder(nn.Module):
    """Encode a market window into per-tap state and option activations."""

    def __init__(
        self,
        feature_count: int,
        *,
        hidden_size: int = 128,
        depth: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        options: tuple[str, ...] = OPTIONS,
        tap_fractions: tuple[float, ...] = TAP_FRACTIONS,
    ) -> None:
        super().__init__()
        if feature_count < 1:
            raise ValueError("feature_count must be >= 1")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if not tap_fractions:
            raise ValueError("tap_fractions must be non-empty")
        for fraction in tap_fractions:
            if not 0.0 < fraction <= 1.0:
                raise ValueError(f"tap fraction {fraction} out of range")

        self.feature_count = feature_count
        self.hidden_size = hidden_size
        self.depth = depth
        self.options = tuple(options)
        self.tap_fractions = tuple(tap_fractions)

        self.project = nn.Sequential(
            nn.Linear(feature_count, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor=False: the causal mask is passed per layer below,
        # and nested-tensor fast paths skip mask handling. torch also warns when
        # norm_first disables it, so set it explicitly rather than suppress.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(hidden_size)

        # One learned query per option, per tap. Distinct queries per tap let an
        # option reweight depth rather than sharing one pooled vector.
        self.option_queries = nn.Parameter(
            torch.randn(len(self.options), len(self.tap_fractions), hidden_size)
            * 0.02
        )
        self.option_attn = nn.MultiheadAttention(
            hidden_size, heads, dropout=dropout, batch_first=True
        )
        self.option_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _tap_layers(self) -> set[int]:
        """Encoder layer indices after which to read a tap."""
        reads = {
            min(self.depth, max(1, int(round(f * self.depth)))) for f in self.tap_fractions
        }
        # The final tap must see the last layer, or "final" is a lie.
        reads.add(self.depth)
        return reads

    def forward(self, window: Tensor, *, pad_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return ``(state_taps, option_taps)`` for one batch of windows.

        ``window``: ``[batch, tokens, features]``
        ``state_taps``: ``[batch, taps, tokens, hidden]``
        ``option_taps``: ``[batch, options, taps, hidden]``
        """
        if window.ndim != 3:
            raise ValueError("window must have shape [batch, tokens, features]")
        if window.shape[-1] != self.feature_count:
            raise ValueError(
                f"window feature width {window.shape[-1]} != expected {self.feature_count}"
            )
        batch, tokens, _ = window.shape
        taps = len(self.tap_fractions)

        hidden = self.project(window)
        positions = torch.arange(tokens, device=window.device).unsqueeze(0)
        hidden = hidden + positions.float().unsqueeze(-1) * 0.01

        causal = torch.triu(
            torch.ones(tokens, tokens, dtype=torch.bool, device=window.device),
            diagonal=1,
        )
        reads = self._tap_layers()
        collected: list[Tensor] = []
        collected_at: dict[int, Tensor] = {}
        for index, layer in enumerate(self.encoder.layers, start=1):
            hidden = layer(hidden, src_mask=causal, src_key_padding_mask=pad_mask)
            if index in reads:
                collected_at[index] = hidden
        missing = reads - collected_at.keys()
        if missing:
            raise AssertionError(f"no activation captured for layers {sorted(missing)}")
        for fraction in self.tap_fractions:
            index = min(self.depth, max(1, int(round(fraction * self.depth))))
            collected.append(collected_at[index])
        state_taps = self.norm(self.dropout(torch.stack(collected, dim=1)))
        # [batch, taps, tokens, hidden]

        option_vectors = []
        for tap_index in range(taps):
            memory = state_taps[:, tap_index]
            queries = self.option_queries[:, tap_index].unsqueeze(0).expand(batch, -1, -1)
            attended, _ = self.option_attn(queries, memory, memory)
            option_vectors.append(self.option_norm(attended))
        # [batch, options, taps, hidden]
        option_taps = torch.stack(option_vectors, dim=2)
        return state_taps, option_taps


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)