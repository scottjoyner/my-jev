"""Causal market-state features.

Every column here is a function of the current bar and *trailing* bars only. That
is the single most important property in this package and it is asserted, not
assumed: :func:`assert_causal` recomputes each feature from a truncated frame and
fails if any value changes when future bars are removed.

The failure this guards against is specific and easy to ship. A feature like
``(close - low.rolling(24).min())`` looks causal but is not if the rolling window
is centred, and a z-score computed over the whole sample silently encodes the
mean and variance of the *test* period into training rows. Both produce excellent
backtest numbers and nothing at all in production.

Normalisation is per-row and trailing for the same reason: a z-score over the
full sample is a leak, and a z-score over a trailing window is a legitimate
"unusual relative to recent history" feature.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

#: Trailing horizons, in bars. Deliberately short-to-medium: the horizon matches
#: the decision horizon, and very long windows on hourly data mostly encode the
#: slow drift that the forward-return label already rewards or punishes.
RETURN_WINDOWS: tuple[int, ...] = (1, 3, 6, 12, 24, 72)
VOL_WINDOWS: tuple[int, ...] = (6, 24, 72)
RANGE_WINDOWS: tuple[int, ...] = (24, 168)
LONGEST_LOOKBACK = 168


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Division that yields NaN instead of inf on a zero/absent denominator."""
    out = numerator / denominator.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def log_returns(close: pd.Series, periods: int) -> pd.Series:
    """Log return over ``periods`` bars, NaN until enough history exists."""
    if periods <= 0:
        raise ValueError("periods must be positive")
    return np.log(close / close.shift(periods))


def realised_volatility(close: pd.Series, window: int) -> pd.Series:
    """Trailing standard deviation of 1-bar log returns, scaled to a horizon."""
    one_bar = log_returns(close, 1)
    return one_bar.rolling(window, min_periods=max(2, window // 2)).std(ddof=0)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder-style RSI expressed in [0, 1] rather than [0, 100]."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing == EMA with alpha = 1/window.
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    strength = _safe_div(avg_gain, avg_loss)
    out = 1.0 - 1.0 / (1.0 + strength)
    # avg_loss == 0 with any gain is a pure up-streak -> RSI 1.0; the safe_div
    # NaN there would otherwise drop the strongest trend bars entirely.
    return out.where(avg_gain.notna() & avg_loss.notna(), np.nan).fillna(
        out.where((avg_loss == 0) & (avg_gain > 0), 1.0)
    )


def atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average true range, causal in the classic sense."""
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def trailing_zscore(series: pd.Series, window: int) -> pd.Series:
    """Deviation from a *trailing* mean, scaled by a *trailing* sigma."""
    mean = series.rolling(window, min_periods=max(4, window // 2)).mean()
    sigma = series.rolling(window, min_periods=max(4, window // 2)).std(ddof=0)
    return _safe_div(series - mean, sigma)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the causal feature block for one asset's OHLCV frame.

    The frame must be ordered oldest-first and carry ``timestamp, low, high,
    open, close, volume``.
    """
    for column in ("timestamp", "low", "high", "open", "close", "volume"):
        if column not in frame.columns:
            raise ValueError(f"missing column {column!r}")
    ordered = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    close = ordered["close"].astype(float)
    out = pd.DataFrame(index=ordered.index)

    # Momentum across several horizons.
    for period in RETURN_WINDOWS:
        out[f"ret_{period}"] = log_returns(close, period)

    # Volatility level and volatility-of-volatility.
    for window in VOL_WINDOWS:
        vol = realised_volatility(close, window)
        out[f"vol_{window}"] = vol
        out[f"vol_z_{window}"] = trailing_zscore(vol, window * 4)
    # Volatility term structure: short vs long. Regime information that a single
    # vol level throws away.
    out["vol_ratio"] = _safe_div(out["vol_6"], out["vol_72"])

    # Candle geometry, normalised by price so it is comparable across assets.
    out["body"] = _safe_div(ordered["close"] - ordered["open"], ordered["open"])
    out["range"] = _safe_div(ordered["high"] - ordered["low"], ordered["close"])
    upper = _safe_div(ordered["high"] - ordered[["open", "close"]].max(axis=1), ordered["close"])
    lower = _safe_div(ordered[["open", "close"]].min(axis=1) - ordered["low"], ordered["close"])
    out["upper_wick"] = upper
    out["lower_wick"] = lower
    atr14 = atr(ordered, 14)
    out["atr_pct"] = _safe_div(atr14, close)
    out["close_loc"] = _safe_div(close - ordered["low"], ordered["high"] - ordered["low"])

    # Oscillator and moving-average distance.
    out["rsi_14"] = rsi(close, 14)
    for span in (12, 48, 168):
        ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
        out[f"ema_gap_{span}"] = _safe_div(close - ema, close)

    # Where the price sits inside its recent range, and how stretched it is.
    for window in RANGE_WINDOWS:
        high_n = ordered["high"].rolling(window, min_periods=window // 2).max()
        low_n = ordered["low"].rolling(window, min_periods=window // 2).min()
        out[f"range_pos_{window}"] = _safe_div(close - low_n, high_n - low_n)
        out[f"drawdown_{window}"] = _safe_div(close - high_n, close)

    # Volume: level, relative level, and signed price/volume agreement.
    volume = ordered["volume"].astype(float)
    out["log_volume"] = np.log1p(volume)
    out["volume_z_24"] = trailing_zscore(np.log1p(volume), 24)
    out["volume_z_168"] = trailing_zscore(np.log1p(volume), 168)
    signed = np.sign(close.diff()) * np.log1p(volume)
    out["signed_volume_z"] = trailing_zscore(signed, 24)

    # Calendar effects, encoded cyclically so the model cannot treat 23:00 and
    # 00:00 as adjacent integers.
    hour = ordered["timestamp"].dt.hour
    dow = ordered["timestamp"].dt.dayofweek
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

    out.insert(0, "timestamp", ordered["timestamp"])
    out.insert(1, "product_id", ordered.get("product_id"))
    return out


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Numeric feature names, excluding keys."""
    return [
        column
        for column in features.columns
        if column not in {"timestamp", "product_id"}
    ]


def assert_causal(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    probe_at: int | None = None,
    tolerance: float = 1e-9,
) -> None:
    """Fail if any feature changes when future bars are removed.

    Truncating the frame after ``probe_at`` and recomputing must reproduce the
    original values at ``probe_at`` exactly. A centred rolling window, a
    full-sample z-score, or a ``shift(-n)`` anywhere in the feature code breaks
    this, which is the point: it breaks loudly here rather than silently in a
    backtest.
    """
    full = build_features(frame)
    if probe_at is None:
        # Default to a point with enough history for the longest lookback.
        probe_at = min(len(frame) - 1, LONGEST_LOOKBACK + 8)
    if probe_at <= LONGEST_LOOKBACK:
        raise ValueError(
            f"probe_at={probe_at} is inside the warm-up window "
            f"(need > {LONGEST_LOOKBACK} bars)"
        )
    truncated = build_features(frame.iloc[: probe_at + 1].copy())
    names = list(columns) if columns is not None else feature_columns(full)
    for name in names:
        a = full[name].iloc[probe_at]
        b = truncated[name].iloc[probe_at]
        if np.isnan(a) and np.isnan(b):
            continue
        if not np.isclose(a, b, rtol=0, atol=tolerance, equal_nan=True):
            raise AssertionError(
                f"feature {name!r} is not causal: value at bar {probe_at} changed from "
                f"{a!r} to {b!r} when future bars were removed"
            )