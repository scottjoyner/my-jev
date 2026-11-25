"""Labels and typed decision records for the trading lane.

Label definition
----------------
For a bar ``t`` and horizon ``H``, the realised forward return is
``log(close[t+H] / close[t])``. Subtracting a round-trip cost gives the *net*
edge that a trade could actually have captured. The label is then::

    buy   if net_edge >= +threshold
    sell  if net_edge <= -threshold
    hold  otherwise

Why the threshold defaults to the round-trip cost
-------------------------------------------------
Labelling every non-zero forward move as buy or sell teaches the model to act on
moves smaller than it could profitably capture. With a 40bp round trip, the
median hourly absolute return on these assets is well under 5bp, so an
unthresholded label is overwhelmingly noise and a model trained on it looks
sharp in-sample and does nothing out of sample. Tying the default threshold to the
cost makes "hold" the default answer and "buy"/"sell" mean *this move clears the
cost*.

The last ``H`` bars of a series have no future observation and are dropped rather
than padded, so no record can be built from a truncated horizon.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..schema import DecisionRecord, QuestionSpec, QuestionType, TargetSpec
from . import OPTIONS
from .features import build_features, feature_columns

#: Name of the single question every trading record asks.
QUESTION = "action"


@dataclass(frozen=True)
class LabelConfig:
    """Everything that defines the supervised target.

    Frozen because these values are part of the on-disk contract: changing any of
    them changes what the model is trained to predict, and silently reusing an old
    checkpoint against new labels is the kind of mistake that produces a confident
    model and a worthless one.
    """

    horizon_bars: int = 4
    round_trip_bps: float = 40.0
    threshold_bps: float | None = None

    @property
    def cost(self) -> float:
        return self.round_trip_bps / 10_000.0

    @property
    def threshold(self) -> float:
        # Default: only act when the move clears the cost.
        return self.cost if self.threshold_bps is None else self.threshold_bps / 10_000.0

    def as_metadata(self) -> dict[str, float | int]:
        return {
            "horizon_bars": self.horizon_bars,
            "round_trip_bps": self.round_trip_bps,
            "threshold_bps": self.threshold * 10_000.0,
        }


@dataclass(frozen=True)
class LabeledSeries:
    features: pd.DataFrame
    labels: pd.Series  # option index, aligned to features
    forward_return: pd.Series  # raw log return, aligned to features
    net_forward_return: pd.Series  # raw minus round-trip cost, for evaluation
    product_id: str


def forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    """Raw forward log return over ``horizon`` bars, NaN for the tail.

    Raw, not net. Cost is applied exactly once, at the labelling threshold — see
    :func:`label_indices`. Subtracting the cost here *and* comparing against it
    would demand +2x cost on the upside and -0 on the downside, which silently
    turns a symmetric three-way problem into a 13/49/38 skew toward selling.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return np.log(close.shift(-horizon) / close)


def label_indices(raw_forward: pd.Series, *, threshold: float) -> pd.Series:
    """Map raw forward return to an option index in :data:`OPTIONS` order."""
    out = pd.Series(
        np.full(len(raw_forward), OPTIONS.index("hold")),
        index=raw_forward.index,
        dtype="int8",
    )
    out[raw_forward >= threshold] = OPTIONS.index("buy")
    out[raw_forward <= -threshold] = OPTIONS.index("sell")
    return out


def build_labeled_series(
    frame: pd.DataFrame,
    config: LabelConfig,
    *,
    product_id: str | None = None,
) -> LabeledSeries:
    """Features plus labels for one asset, with unusable rows dropped."""
    ordered = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    features = build_features(ordered)
    close = ordered["close"].astype(float)

    raw = forward_returns(close, config.horizon_bars)
    labels = label_indices(raw, threshold=config.threshold)
    # Net is for evaluation only: what a trade would actually have kept.
    net = raw - config.cost

    usable = np.isfinite(features[feature_columns(features)].to_numpy()).all(axis=1)
    # Require the horizon to be fully inside the sample.
    usable &= np.isfinite(net.to_numpy())
    usable &= labels.notna().to_numpy()

    resolved = product_id
    if resolved is None and "product_id" in features.columns:
        first = features["product_id"].dropna()
        resolved = str(first.iloc[0]) if len(first) else "UNKNOWN"
    resolved = resolved or "UNKNOWN"

    return LabeledSeries(
        features=features.loc[usable].reset_index(drop=True),
        labels=labels.loc[usable].reset_index(drop=True),
        forward_return=raw.loc[usable].reset_index(drop=True),
        net_forward_return=net.loc[usable].reset_index(drop=True),
        product_id=resolved,
    )


def _state_text(row: pd.Series, product_id: str, columns: Sequence[str]) -> str:
    """Render one bar's state as the flat numeric block the encoder consumes.

    The encoder is numeric, so this is deliberately *not* prose. It exists only to
    satisfy the typed ``DecisionRecord.state: str`` field, which is what makes
    these records interchangeable with the System-One lane's records for
    splitting, manifesting and review tooling.
    """
    parts = [f"product={product_id}", f"t={row['timestamp']}"]
    parts.extend(f"{name}={row[name]:.6g}" for name in columns)
    return "\n".join(parts)


def to_decision_records(
    series: LabeledSeries, config: LabelConfig
) -> list[DecisionRecord]:
    """Emit typed :class:`DecisionRecord` objects for one labeled series.

    The target is a hard index rather than a distribution: here the ground truth
    is a deterministic function of future prices, not a teacher model's opinion,
    so there is no disagreement to represent.
    """
    columns = feature_columns(series.features)
    question = QuestionSpec(
        type=QuestionType.CHOICE,
        instructions=(
            f"Choose the position action for {series.product_id} over the next "
            f"{config.horizon_bars} bars, net of a {config.round_trip_bps:.0f}bp "
            f"round trip. 'hold' is correct unless the move clears that cost."
        ),
        options=list(OPTIONS),
    )
    records: list[DecisionRecord] = []
    for position in range(len(series.features)):
        row = series.features.iloc[position]
        records.append(
            DecisionRecord(
                state=_state_text(row, series.product_id, columns),
                questions={QUESTION: question},
                targets={
                    QUESTION: TargetSpec(index=int(series.labels.iloc[position]))
                },
                metadata={
                    **config.as_metadata(),
                    "product_id": series.product_id,
                    "timestamp": str(row["timestamp"]),
                    "forward_return": float(series.forward_return.iloc[position]),
                },
            )
        )
    return records


def load_universe(
    root,
    product_ids: Iterable[str] | None = None,
    *,
    config: LabelConfig | None = None,
) -> list[LabeledSeries]:
    """Build labeled series for every cached parquet asset under *root*."""
    from pathlib import Path

    cfg = config or LabelConfig()
    base = Path(root)
    frames: list[LabeledSeries] = []
    for path in sorted(base.glob("*__*.parquet")):
        stem = path.name.split("__", 1)[0]
        if product_ids is not None and stem not in set(product_ids):
            continue
        frame = pd.read_parquet(path)
        frames.append(build_labeled_series(frame, cfg, product_id=stem))
    if not frames:
        raise FileNotFoundError(f"no cached candles under {base}")
    return frames


def label_distribution(series: Sequence[LabeledSeries]) -> dict[str, float]:
    """Fraction of each label, for spotting degenerate targets before training."""
    counts = {name: 0 for name in OPTIONS}
    total = 0
    for item in series:
        for value in item.labels.to_numpy():
            counts[OPTIONS[int(value)]] += 1
            total += 1
    return {name: counts[name] / total for name in OPTIONS} if total else {}