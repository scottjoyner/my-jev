"""Temporal splitting with purging and embargo.

Why not a random split
----------------------
Adjacent hourly bars are almost perfectly correlated (lag-1 return correlation on
these assets is well above 0.9). A random split therefore puts bars from the same
hour in both train and test, and a model can score well by memorising local
volatility structure rather than by learning anything. Every split here is
temporal, on calendar boundaries shared across all assets.

Purging
-------
The label at bar ``t`` is defined by ``close[t+H]``. A training record at
``t = split_start - 1`` is therefore computed from a price that lies inside the
evaluation window. Those records are *purged*: dropped from the tail of train.

Embargo
-------
Serial correlation in returns means the bars immediately *after* an evaluation
window still carry information about it. An embargo of at least the horizon is
dropped on both sides of each boundary. Purging alone removes the direct label
overlap; the embargo removes the residual dependence that would otherwise make a
test number optimistic.

The embargo is expressed in bars and applied on the shared calendar index, so an
asset with gaps cannot silently shrink it.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Index arrays into a stacked, time-sorted frame."""

    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    boundaries: tuple[pd.Timestamp, pd.Timestamp]
    purged: int
    embargoed: int

    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "valid": len(self.valid),
            "test": len(self.test),
            "purged": self.purged,
            "embargoed": self.embargoed,
        }

    def summary(self) -> dict[str, object]:
        out: dict[str, object] = dict(self.sizes())
        out["valid_from"] = str(self.boundaries[0])
        out["test_from"] = str(self.boundaries[1])
        return out


def _boundary_index(stamps: pd.Series, boundary: pd.Timestamp) -> int:
    """Index of the first row at or after ``boundary``.

    Uses ``bisect`` over the Timestamp values rather than ``numpy.searchsorted``:
    coercing tz-aware timestamps to ``datetime64`` drops the offset, and mixing
    the naive result with tz-aware values raises. Timestamps compare correctly
    as objects, and the log-n cost is irrelevant at this scale.
    """
    return int(bisect.bisect_left(stamps.tolist(), boundary))


def temporal_split(
    timestamps: Sequence[pd.Timestamp] | pd.Series,
    *,
    valid_fraction: float = 0.2,
    test_fraction: float = 0.2,
    horizon: int,
    embargo: int | None = None,
) -> TemporalSplit:
    """Split by calendar time, purging label overlap and embargoing the seams.

    ``horizon`` must match the labelling horizon. ``embargo`` defaults to
    ``horizon``.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be in (0, 1)")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    if valid_fraction + test_fraction >= 1.0:
        raise ValueError("valid + test fractions must leave room for train")

    stamps = pd.Series(pd.to_datetime(pd.Series(timestamps), utc=True))
    if stamps.empty:
        raise ValueError("no timestamps supplied")

    order = np.argsort(stamps.to_numpy(), kind="stable")
    if not np.array_equal(order, np.arange(len(stamps))):
        raise ValueError(
            "timestamps must be sorted ascending; sort the frame before splitting"
        )
    unique = np.sort(stamps.unique())
    n = len(unique)
    n_valid = max(1, int(round(n * valid_fraction)))
    n_test = max(1, int(round(n * test_fraction)))
    valid_from = unique[n - n_valid - n_test]
    test_from = unique[n - n_test]
    gap = horizon if embargo is None else embargo
    if gap < 0:
        raise ValueError("embargo must be >= 0")

    # Stamps are monotonic, so a row's calendar position is its index.
    position = np.arange(len(stamps), dtype=np.int64)
    valid_start = _boundary_index(stamps, valid_from)

    # Purge: training rows whose label horizon reaches into valid.
    train_mask = position < valid_start
    horizon_end = position + horizon
    train_mask &= horizon_end < valid_start

    # Embargo on the valid/test seam, then the valid/test boundary itself.
    valid_mask = (position >= valid_start) & (position < valid_start + n_valid)
    valid_mask &= position >= valid_start + gap

    test_start = _boundary_index(stamps, test_from)
    test_mask = position >= test_start
    test_mask &= position >= test_start + gap

    train_idx = np.flatnonzero(train_mask)
    valid_idx = np.flatnonzero(valid_mask)
    test_idx = np.flatnonzero(test_mask)
    if not (len(train_idx) and len(valid_idx) and len(test_idx)):
        raise ValueError(
            "split produced an empty partition: "
            f"train={len(train_idx)} valid={len(valid_idx)} test={len(test_idx)}; "
            "reduce horizon/embargo or widen the series"
        )
    overlap = np.intersect1d(np.intersect1d(train_idx, valid_idx), test_idx)
    if overlap.size:
        raise AssertionError(f"split leaked {overlap.size} shared indices")

    return TemporalSplit(
        train=train_idx,
        valid=valid_idx,
        test=test_idx,
        boundaries=(pd.Timestamp(valid_from), pd.Timestamp(test_from)),
        purged=int((position < valid_start).sum() - train_mask.sum()),
        embargoed=int(gap),
    )


def calendar_boundaries(
    timestamps: Sequence[np.datetime64] | pd.Series,
    *,
    valid_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Shared calendar cut points covering the whole sample.

    Returned separately from the masks because a stacked multi-asset frame is
    ordered by asset, not by time, so it cannot be sorted into a single timeline
    without breaking the contiguity that window extraction depends on. Instead
    the boundaries are computed once here and every asset is cut at the same
    instants.
    """
    values = np.sort(pd.Series(pd.to_datetime(pd.Series(timestamps), utc=True)).unique())
    if len(values) < 10:
        raise ValueError("not enough distinct timestamps to split")
    n = len(values)
    if not 0.0 < valid_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("fractions must be in (0, 1)")
    if valid_fraction + test_fraction >= 1.0:
        raise ValueError("valid + test fractions must leave room for train")
    n_valid = max(1, int(round(n * valid_fraction)))
    n_test = max(1, int(round(n * test_fraction)))
    return (
        pd.Timestamp(values[n - n_valid - n_test]),
        pd.Timestamp(values[n - n_test]),
    )


def split_windows(
    timestamps: np.ndarray,
    *,
    valid_fraction: float = 0.2,
    test_fraction: float = 0.2,
    horizon: int,
    embargo: int | None = None,
    bar_delta: pd.Timedelta = pd.Timedelta(hours=1),
) -> TemporalSplit:
    """Split a time-ordered frame (one asset, or any single ordered series).

    For the stacked multi-asset case use :func:`calendar_boundaries` and mask
    each asset against the shared cut points; this function assumes one ordered
    series and is the single-asset path.
    """
    stamps = pd.Series(pd.to_datetime(pd.Series(timestamps), utc=True))
    if not stamps.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted ascending")
    valid_from, test_from = calendar_boundaries(
        stamps, valid_fraction=valid_fraction, test_fraction=test_fraction
    )
    gap = horizon if embargo is None else embargo
    horizon_delta = bar_delta * horizon
    embargo_delta = bar_delta * gap

    train_mask = stamps < valid_from
    # Purge: a train row whose label window reaches into valid.
    train_mask &= (stamps + horizon_delta) < valid_from

    valid_mask = (stamps >= valid_from) & (stamps < test_from)
    valid_mask &= stamps >= valid_from + embargo_delta

    test_mask = stamps >= test_from
    test_mask &= stamps >= test_from + embargo_delta

    train_idx = np.flatnonzero(train_mask.to_numpy())
    valid_idx = np.flatnonzero(valid_mask.to_numpy())
    test_idx = np.flatnonzero(test_mask.to_numpy())
    if not (len(train_idx) and len(valid_idx) and len(test_idx)):
        raise ValueError(
            "split produced an empty partition: "
            f"train={len(train_idx)} valid={len(valid_idx)} test={len(test_idx)}"
        )
    if len(np.intersect1d(np.intersect1d(train_idx, valid_idx), test_idx)):
        raise AssertionError("split leaked shared indices")
    return TemporalSplit(
        train=train_idx,
        valid=valid_idx,
        test=test_idx,
        boundaries=(valid_from, test_from),
        purged=int((stamps < valid_from).sum() - train_mask.sum()),
        embargoed=int(gap),
    )


def mask_windows(
    timestamps: np.ndarray,
    boundaries: tuple[pd.Timestamp, pd.Timestamp],
    *,
    horizon: int,
    embargo: int | None = None,
    bar_delta: pd.Timedelta = pd.Timedelta(hours=1),
) -> dict[str, np.ndarray]:
    """Cut an arbitrarily ordered (per-asset concatenated) frame at shared bounds.

    Returns disjoint index arrays. Assets may be concatenated in any order because
    every row is classified purely by its own timestamp, which is what lets window
    extraction keep each asset's rows contiguous.
    """
    stamps = pd.Series(pd.to_datetime(pd.Series(timestamps), utc=True))
    valid_from, test_from = boundaries
    gap = horizon if embargo is None else embargo
    horizon_delta = bar_delta * horizon
    embargo_delta = bar_delta * gap

    train = (stamps < valid_from) & ((stamps + horizon_delta) < valid_from)
    valid = (stamps >= valid_from) & (stamps < test_from) & (stamps >= valid_from + embargo_delta)
    test = (stamps >= test_from) & (stamps >= test_from + embargo_delta)
    out = {
        "train": np.flatnonzero(train.to_numpy()),
        "valid": np.flatnonzero(valid.to_numpy()),
        "test": np.flatnonzero(test.to_numpy()),
    }
    if not all(len(v) for v in out.values()):
        raise ValueError(f"empty partition in shared-boundary split: { {k: len(v) for k, v in out.items()} }")
    if len(np.intersect1d(np.intersect1d(out["train"], out["valid"]), out["test"])):
        raise AssertionError("shared-boundary split leaked indices")
    return out


def assert_no_forward_leakage(
    frame: pd.DataFrame, split: TemporalSplit, *, horizon: int
) -> None:
    """Fail if any train record's label window reaches into valid or test.

    This is the invariant that a random split violates and that no accuracy
    number would reveal on its own.
    """
    stamps = pd.to_datetime(frame["timestamp"], utc=True)
    train_end = stamps.iloc[split.train].max()
    valid_start = stamps.iloc[split.valid].min()
    test_start = stamps.iloc[split.test].min()
    if train_end + pd.Timedelta(hours=horizon) >= valid_start:
        raise AssertionError(
            "purging incomplete: last train bar "
            f"{train_end} + {horizon}h reaches into valid ({valid_start})"
        )
    valid_end = stamps.iloc[split.valid].max()
    if valid_end + pd.Timedelta(hours=horizon) >= test_start:
        raise AssertionError(
            "embargo incomplete: last valid bar "
            f"{valid_end} + {horizon}h reaches into test ({test_start})"
        )