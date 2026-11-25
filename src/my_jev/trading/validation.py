"""Out-of-sample validation: walk-forward folds, honest position accounting, CIs.

Why the single temporal split was not enough
---------------------------------------------
One train/valid/test split gives one number from one test window. If the market
did one thing in that window and something else later, the number means nothing.
Walk-forward folds with an expanding training window answer the only question
that matters for deployment: *retrained on everything up to time T, does the model
still work on the period after T?*

Why non-overlapping positions are mandatory
--------------------------------------------
The label at row ``t`` is the move over the next ``H`` bars, so consecutive rows
are overlapping claims on the same future. With ``H = 168`` and an hourly clock,
95% active rows means roughly 168 simultaneous positions per asset. That is a
signal-density measurement, not a strategy, and it inflates the apparent Sharpe
by treating highly correlated claims as independent ones.
:func:`non_overlapping` thins the evaluation set to at most one open position per
asset at a time, which is what a real deployment could actually hold.

Why the skill term, not the raw edge
------------------------------------
``mean(s * r)`` is not skill. See
``scripts/analyse_edge_decomposition.py``: it mixes ``mean(s) * mean(r)`` (a bet
size times an average return) with ``cov(s, r)`` (genuine anticipation). Only the
covariance term is evidence, so that is what gets a confidence interval here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd



@dataclass(frozen=True)
class Fold:
    """One walk-forward step."""

    index: int
    train: np.ndarray
    test: np.ndarray
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def summary(self) -> dict[str, object]:
        return {
            "fold": self.index,
            "train": len(self.train),
            "test": len(self.test),
            "train_end": str(self.train_end),
            "test_start": str(self.test_start),
            "test_end": str(self.test_end),
        }


def walk_forward(
    timestamps: np.ndarray,
    *,
    n_folds: int = 4,
    horizon: int,
    embargo: int | None = None,
    min_train_fraction: float = 0.4,
    bar_delta: pd.Timedelta = pd.Timedelta(hours=1),
) -> list[Fold]:
    """Expanding-window purged walk-forward folds over a calendar.

    ``horizon`` and ``embargo`` are in bars, converted with ``bar_delta``. Labels
    are defined on the *original* bar clock, so a row's label reaches ``horizon *
    bar_delta`` into the future and that is the span purged before each test
    window.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    gap = horizon if embargo is None else embargo
    horizon_delta = bar_delta * horizon
    embargo_delta = bar_delta * gap

    stamps = pd.Series(pd.to_datetime(pd.Series(timestamps), utc=True))
    calendar = np.sort(stamps.unique())
    n = len(calendar)
    if n < 50:
        raise ValueError(f"need at least 50 distinct timestamps, got {n}")

    # Reserve the final n_folds test windows; everything before is candidate train.
    first_test = int(n * min_train_fraction)
    remaining = n - first_test
    if remaining < n_folds * 2:
        raise ValueError("not enough calendar left for the requested folds")
    width = remaining // n_folds

    folds: list[Fold] = []
    for k in range(n_folds):
        lo = first_test + k * width
        hi = lo + width if k < n_folds - 1 else n
        test_start = pd.Timestamp(calendar[lo])
        # calendar has n entries with valid indices 0..n-1, so the final fold's
        # exclusive end index `n` is out of bounds; extend past the last bar.
        test_end = (
            pd.Timestamp(calendar[hi])
            if hi < n
            else pd.Timestamp(calendar[-1]) + bar_delta
        )

        train_mask = (stamps + horizon_delta + embargo_delta) < test_start
        test_mask = (stamps >= test_start + embargo_delta) & (stamps < test_end)
        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if not len(train_idx) or not len(test_idx):
            raise ValueError(f"fold {k} is empty: train={len(train_idx)} test={len(test_idx)}")
        # Compare timestamps, not indices: the stacked multi-asset frame is
        # ordered by asset, so index order says nothing about chronological order
        # and an index comparison here fails on every fold after the first.
        train_end = pd.Timestamp(stamps.iloc[train_idx].max())
        test_start = pd.Timestamp(stamps.iloc[test_idx].min())
        if train_end >= test_start:
            raise AssertionError(
                f"fold {k} train/test overlap in time: train ends {train_end}, "
                f"test starts {test_start}"
            )
        folds.append(
            Fold(
                index=k,
                train=train_idx,
                test=test_idx,
                train_end=train_end,
                test_start=test_start,
                test_end=pd.Timestamp(stamps.iloc[test_idx].max()),
            )
        )
    return folds


def non_overlapping(
    index: np.ndarray,
    *,
    product_ids: np.ndarray,
    timestamps: np.ndarray,
    spacing: pd.Timedelta,
) -> np.ndarray:
    """Thin *index* so no two kept rows are within *spacing* for the same asset.

    Greedy per asset in time order. Rows are assumed sorted within each asset,
    which the stacking layout guarantees.
    """
    if spacing <= pd.Timedelta(0):
        raise ValueError("spacing must be positive")
    keep: list[int] = []
    last: dict[object, pd.Timestamp] = {}
    for row in index:
        pid = product_ids[row]
        stamp = pd.Timestamp(timestamps[row])
        previous = last.get(pid)
        if previous is None or stamp - previous >= spacing:
            keep.append(int(row))
            last[pid] = stamp
    return np.asarray(keep, dtype=np.int64)


def skill_and_pnl(
    signed: np.ndarray,
    raw_forward: np.ndarray,
    *,
    cost_bps: float,
    active_threshold: float = 0.10,
) -> dict[str, float]:
    """Decompose into skill (covariance) and sizing terms, plus risk-adjusted P&L."""
    size = np.abs(signed)
    active = size >= active_threshold
    if active.sum() < 2:
        return {}
    s = signed[active]
    r = raw_forward[active]
    pnl = s * r - size[active] * (cost_bps / 10_000.0)
    cov = float(np.cov(s, r, ddof=0)[0, 1])
    return {
        "n_active": int(active.sum()),
        "hit_rate": float((np.sign(s) == np.sign(r)).mean()),
        "mean_size": float(size[active].mean()),
        "edge_bp": float((s * r).mean() * 1e4),
        "sizing_bp": float(s.mean() * r.mean() * 1e4),
        "skill_bp": cov * 1e4,
        "skill_per_unit_bp": float(cov / (size[active].mean() or 1.0) * 1e4),
        "net_bp": float(pnl.mean() * 1e4),
        "pnl_std_bp": float(pnl.std(ddof=0) * 1e4),
        "sharpe": float(pnl.mean() / (pnl.std(ddof=0) or 1e-12)),
        "total_net_bp": float(pnl.sum() * 1e4),
    }


def bootstrap_skill_ci(
    signed: np.ndarray,
    raw_forward: np.ndarray,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    active_threshold: float = 0.10,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI on the skill (covariance) term.

    Resamples *trades*, not bars, and only active ones. The statistic is the
    covariance between position and return, which is what distinguishes skill
    from sizing.
    """
    size = np.abs(signed)
    active = size >= active_threshold
    if active.sum() < 20:
        return {"low": float("nan"), "high": float("nan"), "point": float("nan"), "n": int(active.sum())}
    s = signed[active]
    r = raw_forward[active]
    rng = np.random.default_rng(seed)
    n = len(s)
    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        pick = rng.integers(0, n, size=n)
        stats[i] = float(np.cov(s[pick], r[pick], ddof=0)[0, 1]) * 1e4
    alpha = (1.0 - confidence) / 2.0
    return {
        "point": float(np.cov(s, r, ddof=0)[0, 1] * 1e4),
        "low": float(np.quantile(stats, alpha)),
        "high": float(np.quantile(stats, 1.0 - alpha)),
        "confidence": confidence,
        "n": int(n),
        "p_positive": float((stats > 0).mean()),
    }


def per_asset_breakdown(
    index: np.ndarray,
    signed: np.ndarray,
    raw_forward: np.ndarray,
    product_ids: np.ndarray,
    *,
    cost_bps: float,
) -> dict[str, dict[str, float]]:
    """Metrics for each asset separately, to expose a single-drift artifact.

    ``signed`` and ``raw_forward`` must already be aligned to ``index`` (the
    caller slices them the same way it slices ``index``); they are full-length
    arrays whose rows correspond positionally.
    """
    out: dict[str, dict[str, float]] = {}
    for pid in sorted(set(product_ids[index].tolist())):
        mask = product_ids[index] == pid
        out[pid] = skill_and_pnl(
            signed[mask], raw_forward[mask], cost_bps=cost_bps
        ) or {"n_active": 0}
    return out
