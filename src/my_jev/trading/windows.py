"""Windowed tensors for the trading decision head.

Standardisation is fitted on the **training rows only**. That is not a detail: a
scaler fitted on the full sample leaks the test period's mean and variance into
training features, and because the label is a threshold on the forward return,
that shifts which rows fall on each side of the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .dataset import LabeledSeries
from .features import feature_columns


@dataclass(frozen=True)
class WindowSet:
    """Flattened multi-asset windows plus the labels aligned to them."""

    features: np.ndarray  # [rows, features] standardised
    labels: np.ndarray  # [rows] option index
    net_forward: np.ndarray  # [rows] net edge, for PnL
    timestamps: np.ndarray  # [rows] datetime64
    product_ids: np.ndarray  # [rows] object
    asset_start: np.ndarray  # [rows] global index of each row's asset start
    position_in_asset: np.ndarray  # [rows]
    columns: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)


def stack_series(
    series: list[LabeledSeries], *, sequence_length: int
) -> WindowSet:
    """Concatenate assets into one flat, per-asset-contiguous block layout.

    Every row is kept, including each asset's warm-up prefix. Rows without enough
    history for a full window are excluded later, by
    :func:`window_indices` and by the caller when building splits.

    Keeping them here matters for correctness of the offsets: an earlier version
    dropped the warm-up rows at stack time and then advanced ``offset`` by the
    asset's *full* length, so ``asset_start`` drifted by ``sequence_length - 1``
    for every asset after the first and window extraction hit a spurious
    "crosses an asset boundary" assertion.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    columns = tuple(feature_columns(series[0].features))
    feat_blocks, label_blocks, net_blocks = [], [], []
    time_blocks, id_blocks, start_blocks, pos_blocks = [], [], [], []

    offset = 0
    for item in series:
        block = item.features[list(columns)].to_numpy(dtype=np.float64)
        n = len(block)
        if n == 0:
            continue
        feat_blocks.append(block)
        label_blocks.append(item.labels.to_numpy().astype(np.int64))
        net_blocks.append(item.net_forward_return.to_numpy())
        time_blocks.append(item.features["timestamp"].to_numpy())
        id_blocks.append(np.array([item.product_id] * n, dtype=object))
        # Every row of this asset starts at `offset`; position is 0..n-1.
        start_blocks.append(np.full(n, offset, dtype=np.int64))
        pos_blocks.append(np.arange(n, dtype=np.int64))
        offset += n

    if not feat_blocks:
        raise ValueError("no asset produced any rows")
    return WindowSet(
        features=np.concatenate(feat_blocks),
        labels=np.concatenate(label_blocks),
        net_forward=np.concatenate(net_blocks),
        timestamps=np.concatenate(time_blocks),
        product_ids=np.concatenate(id_blocks),
        asset_start=np.concatenate(start_blocks),
        position_in_asset=np.concatenate(pos_blocks),
        columns=columns,
        mean=np.zeros(len(columns)),
        scale=np.ones(len(columns)),
    )


def subset(data: WindowSet, index: np.ndarray) -> WindowSet:
    """Row-subset *data* and recompute the per-asset offsets.

    Asset offsets cannot be sliced: they are positions in the *original* frame, so
    carrying them through a subset leaves every asset pointing at a stale start
    and window extraction then refuses valid windows as boundary crossings. They
    are derived from the product ids instead, which is order-independent and
    cannot drift.
    """
    if len(index) == 0:
        raise ValueError("empty subset")
    product_ids = data.product_ids[index]
    # asset_start is a *global* position, so it must be recomputed after the
    # subset shifts everything. position_in_asset is asset-local and must be
    # carried through untouched: recomputing it here would renumber each asset
    # from zero and silently resurrect the warm-up rows that `warmed_up` just
    # filtered out, which then fail window extraction mid-epoch.
    starts: list[int] = []
    current = object()
    group_start = 0
    for row, pid in enumerate(product_ids):
        if pid != current:
            current = pid
            group_start = row
        starts.append(group_start)
    return WindowSet(
        features=data.features[index],
        labels=data.labels[index],
        net_forward=data.net_forward[index],
        timestamps=data.timestamps[index],
        product_ids=product_ids,
        asset_start=np.asarray(starts, dtype=np.int64),
        position_in_asset=data.position_in_asset[index],
        columns=data.columns,
        mean=data.mean,
        scale=data.scale,
    )


def warmed_up(data: WindowSet, sequence_length: int) -> np.ndarray:
    """Row indices with enough trailing history for a full window."""
    return np.flatnonzero(data.position_in_asset >= sequence_length - 1)


def fit_scaler(data: WindowSet, train_index: np.ndarray) -> WindowSet:
    """Return a copy standardised with statistics from *train_index* only."""
    if len(train_index) == 0:
        raise ValueError("empty training index")
    block = data.features[train_index]
    mean = block.mean(axis=0)
    scale = block.std(axis=0)
    # Constant columns (common in short warm-ups) would divide by zero; leave them
    # at zero rather than inventing a scale.
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardised = (data.features - mean) / scale
    return WindowSet(
        features=standardised,
        labels=data.labels,
        net_forward=data.net_forward,
        timestamps=data.timestamps,
        product_ids=data.product_ids,
        asset_start=data.asset_start,
        position_in_asset=data.position_in_asset,
        columns=data.columns,
        mean=mean,
        scale=scale,
    )


def window_indices(data: WindowSet, index: np.ndarray, sequence_length: int) -> np.ndarray:
    """Row indices for each window ending at each row in *index* -> [n, seq]."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    positions = data.position_in_asset[index]
    if np.any(positions < sequence_length - 1):
        raise ValueError("requested a window for a row without enough history")
    ends = index.astype(np.int64)[:, None]
    offsets = np.arange(-sequence_length + 1, 1, dtype=np.int64)[None, :]
    rows = ends + offsets
    # numpy silently treats a negative index as counting from the end, so a window
    # running off the front of the array would quietly read another asset's tail
    # instead of failing. Reject it explicitly.
    if np.any(rows < 0):
        raise ValueError(
            "window would run off the front of the array; keep each asset's "
            "warm-up rows in the buffer and exclude them from the split instead "
            "of subsetting them away"
        )
    # Never let a window straddle an asset boundary.
    earliest = (data.asset_start[index] + (positions - sequence_length + 1))[:, None]
    if np.any(rows < earliest):
        raise AssertionError("window would cross an asset boundary")
    return rows


def make_batch(
    data: WindowSet, index: np.ndarray, sequence_length: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(windows, labels, net_forward)`` for the given row indices."""
    rows = window_indices(data, index, sequence_length)
    windows = torch.from_numpy(data.features[rows].astype(np.float32))
    labels = torch.from_numpy(data.labels[index])
    net = torch.from_numpy(data.net_forward[index].astype(np.float32))
    return windows, labels, net


def temporal_index(
    timestamps: np.ndarray, boundary: np.datetime64
) -> np.ndarray:
    """Row indices at or after *boundary* (timestamps must be ascending)."""
    return np.flatnonzero(timestamps >= boundary)


def frame_for(data: WindowSet) -> pd.DataFrame:
    """A tidy frame, used by the leakage assertion and by reports."""
    return pd.DataFrame(
        {
            "timestamp": data.timestamps,
            "product_id": data.product_ids,
            "label": data.labels,
            "net_forward": data.net_forward,
        }
    )