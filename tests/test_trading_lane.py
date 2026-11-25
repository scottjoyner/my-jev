"""Tests for the trading lane.

The priority order here reflects where the real risks are. A wrong number in a
report is recoverable; a leak in the feature or split code produces excellent
metrics for a model that will lose money live, and nothing in the training output
would reveal it. So the leakage and causality tests come first and are written to
fail loudly if their subject regresses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from my_jev.trading import OPTIONS
from my_jev.trading.candles import COINBASE_FIELD_ORDER, _decode
from my_jev.trading.dataset import (
    LabelConfig,
    build_labeled_series,
    forward_returns,
    label_indices,
    to_decision_records,
)
from my_jev.trading.encoder import MarketTapEncoder, count_parameters
from my_jev.trading.features import assert_causal, build_features, feature_columns
from my_jev.trading.splits import calendar_boundaries, mask_windows
from my_jev.trading.windows import stack_series, warmed_up


def synthetic_candles(n: int = 900, seed: int = 7) -> pd.DataFrame:
    """Deterministic OHLCV with a mild upward drift and real ranges."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0002, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(steps))
    spread = np.abs(rng.normal(0.0, 0.004, size=n))
    open_ = close * (1.0 + rng.normal(0.0, 0.002, size=n))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC"),
            "low": low,
            "high": high,
            "open": open_,
            "close": close,
            "volume": rng.lognormal(3.0, 0.4, size=n),
        }
    )


# ---------------------------------------------------------------------------
# Candle decoding
# ---------------------------------------------------------------------------


def test_decode_uses_the_real_coinbase_field_order():
    """Coinbase returns [time, low, high, open, close, volume].

    Reading it as [time, open, high, low, close, volume] yields a plausible series
    with the wrong values, so the mapping is pinned here.
    """
    row = [1_700_000_000, 98.0, 102.0, 99.0, 101.0, 5.0]
    frame = _decode([row])
    assert set(COINBASE_FIELD_ORDER).issubset(frame.columns)
    assert frame["open"].iloc[0] == 99.0
    assert frame["close"].iloc[0] == 101.0
    assert frame["low"].iloc[0] == 98.0
    assert frame["high"].iloc[0] == 102.0


def test_decode_rejects_short_rows():
    with pytest.raises(Exception):
        _decode([[1_700_000_000, 1.0, 2.0]])


# ---------------------------------------------------------------------------
# Causality -- the highest-priority invariant
# ---------------------------------------------------------------------------


def test_features_are_causal_on_synthetic_data():
    assert_causal(synthetic_candles(), probe_at=400)


def test_planted_leaks_are_detected():
    """A guard that cannot fail is decoration.

    Each case is a classic lookahead bug: a centred window, a statistic computed
    over the whole sample, and an explicit negative shift.
    """
    import my_jev.trading.features as module

    frame = synthetic_candles()
    honest = module.build_features

    def with_extra(fn):
        def wrapped(inner):
            out = honest(inner)
            fn(out, inner)
            return out

        return wrapped

    # Each injected leak must derive from `inner`, the frame it is handed. An
    # earlier version closed over the full frame's close column, so truncating
    # the input changed nothing and the guard silently passed a real leak.
    def centred(out, inner):
        out["leak_centred"] = inner["close"].rolling(48, center=True, min_periods=1).mean()

    def full_sample(out, inner):
        column = inner["close"].astype(float)
        out["leak_full_sample"] = (column - column.mean()) / column.std()

    def shifted(out, inner):
        out["leak_shift"] = inner["close"].astype(float).shift(-1)

    for leak in (centred, full_sample, shifted):
        module.build_features = with_extra(leak)
        with pytest.raises(AssertionError, match="not causal"):
            module.assert_causal(frame, probe_at=400)

    module.build_features = honest
    module.assert_causal(frame, probe_at=400)  # restored, still passes


def test_encoder_never_lets_a_later_bar_change_earlier_activations():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    encoder = MarketTapEncoder(feature_count=6, hidden_size=32, depth=2).eval()
    window = torch.randn(2, 16, 6)
    base, _ = encoder(window)

    perturbed = window.clone()
    perturbed[:, -1, :] += 100.0
    after, _ = encoder(perturbed)

    # Tokens before the perturbed one must be bit-identical.
    assert torch.allclose(base[:, :, :-1, :], after[:, :, :-1, :], atol=1e-6)
    assert not torch.allclose(base[:, :, -1, :], after[:, :, -1, :])


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def test_forward_return_matches_an_independent_computation():
    frame = synthetic_candles(n=500)
    close = frame["close"].astype(float)
    got = forward_returns(close, 4)
    expected = np.log(close.shift(-4) / close)
    pd.testing.assert_series_equal(got, expected, check_names=False)


def test_forward_return_is_nan_in_the_tail():
    frame = synthetic_candles(n=200)
    got = forward_returns(frame["close"].astype(float), 4)
    assert got.iloc[-4:].isna().all()
    assert got.iloc[:-4].notna().all()


def test_labels_are_symmetric_around_zero():
    """A cost applied once must not shift the buy/sell balance.

    The first version subtracted the cost and *then* compared against it, which
    demanded +2x cost to buy and 0 to sell, producing a 13/49/38 skew.
    """
    raw = pd.Series([-0.10, -0.041, -0.0399, 0.0, 0.0399, 0.041, 0.10])
    labels = label_indices(raw, threshold=0.04)
    assert labels.tolist() == [
        OPTIONS.index("sell"),
        OPTIONS.index("sell"),
        OPTIONS.index("hold"),
        OPTIONS.index("hold"),
        OPTIONS.index("hold"),
        OPTIONS.index("buy"),
        OPTIONS.index("buy"),
    ]


def test_labels_are_balanced_on_synthetic_data():
    series = build_labeled_series(synthetic_candles(4000, seed=3), LabelConfig())
    counts = np.bincount(series.labels.to_numpy(), minlength=len(OPTIONS))
    shares = counts / counts.sum()
    assert shares.min() > 0.15, f"degenerate label distribution: {shares}"


def test_threshold_defaults_to_the_round_trip_cost():
    config = LabelConfig(round_trip_bps=40.0)
    assert config.threshold == pytest.approx(0.004)


def test_records_carry_the_typed_question_and_valid_target():
    series = build_labeled_series(synthetic_candles(600, seed=11), LabelConfig())
    records = to_decision_records(series, LabelConfig())
    assert records
    for record in records[:20]:
        question = record.questions["action"]
        assert question.options == list(OPTIONS)
        target = record.targets["action"]
        assert target.index is not None
        assert 0 <= target.index < len(OPTIONS)
        assert record.metadata["product_id"]
        assert "horizon_bars" in record.metadata


def test_records_are_rejected_when_the_label_index_is_out_of_range():
    from my_jev.schema import DecisionRecord, QuestionSpec, QuestionType, TargetSpec

    question = QuestionSpec(
        type=QuestionType.CHOICE, instructions="x", options=list(OPTIONS)
    )
    with pytest.raises(ValueError):
        DecisionRecord(
            state="s",
            questions={"action": question},
            targets={"action": TargetSpec(index=99)},
        )


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def test_random_split_would_leak_but_purged_split_does_not():
    n = 4000
    stamps = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC").to_numpy()
    bounds = calendar_boundaries(stamps, valid_fraction=0.2, test_fraction=0.2)
    parts = mask_windows(stamps, bounds, horizon=4)

    # Nothing is shared.
    assert not set(parts["train"]) & set(parts["valid"])
    assert not set(parts["valid"]) & set(parts["test"])
    assert not set(parts["train"]) & set(parts["test"])

    # No training row's label window reaches the validation period. Compared in
    # pandas: the raw numpy objects are tz-aware Timestamps, which timedelta64
    # arithmetic cannot combine with.
    valid_from = bounds[0]
    train_end = pd.Series(stamps[parts["train"]]).max()
    assert train_end + pd.Timedelta(hours=4) < valid_from


def test_purge_count_is_at_least_the_horizon():
    n = 4000
    stamps = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC").to_numpy()
    bounds = calendar_boundaries(stamps)
    parts = mask_windows(stamps, bounds, horizon=8)
    train_end = pd.Series(stamps[parts["train"]]).max()
    # Exactly the horizon rows nearest the boundary had to go.
    assert train_end + pd.Timedelta(hours=8) < bounds[0]
    assert len(parts["train"]) > 0


def test_split_boundaries_are_shared_across_assets():
    stamps = pd.date_range("2025-01-01", periods=3000, freq="h", tz="UTC").to_numpy()
    # Two assets stacked, as the real pipeline does.
    stacked = np.concatenate([stamps, stamps])
    bounds = calendar_boundaries(stacked)
    parts = mask_windows(stacked, bounds, horizon=4)
    test_rows = parts["test"]
    first_block = test_rows[test_rows < len(stamps)]
    second_block = test_rows[test_rows >= len(stamps)] - len(stamps)
    assert len(first_block) == len(second_block) > 0
    assert set(stamps[first_block]) == set(stamps[second_block])


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def test_windows_never_span_two_assets():
    series = [
        build_labeled_series(synthetic_candles(500, seed=s), LabelConfig(), product_id=f"A{s}")
        for s in (1, 2)
    ]
    data = stack_series(series, sequence_length=16)
    usable = warmed_up(data, 16)
    assert len(usable) > 0
    from my_jev.trading.windows import window_indices

    for row in usable[::37]:
        rows = window_indices(data, np.array([row]), 16).ravel()
        assert len(set(data.product_ids[rows].tolist())) == 1


def test_window_indices_rejects_running_off_the_front():
    series = [build_labeled_series(synthetic_candles(200, seed=5), LabelConfig())]
    data = stack_series(series, sequence_length=16)
    from my_jev.trading.windows import window_indices

    with pytest.raises(ValueError):
        window_indices(data, np.array([0]), 16)


def test_features_have_no_nan_after_warmup():
    frame = synthetic_candles(1200)
    features = build_features(frame)
    block = features[feature_columns(features)].iloc[200:]
    assert not block.isna().any().any()


def test_encoder_emits_the_shapes_the_head_expects():
    torch = pytest.importorskip("torch")
    from my_jev.decision_attention import ExternalDecisionAttentionHead

    encoder = MarketTapEncoder(feature_count=35, hidden_size=64, depth=4).eval()
    head = ExternalDecisionAttentionHead(hidden_size=64, tap_count=4, rank=16).eval()
    windows = torch.randn(3, 24, 35)
    state_taps, option_taps = encoder(windows)
    assert state_taps.shape == (3, 4, 24, 64)
    assert option_taps.shape == (3, 3, 4, 64)
    logits = head(
        state_taps,
        torch.ones(3, 24),
        option_taps,
        torch.ones(3, len(OPTIONS)),
    )
    assert logits.shape == (3, 3)
    assert count_parameters(encoder) > 0

# ---------------------------------------------------------------------------
# Advisory emission -- the authority boundary must be enforced, not documented
# ---------------------------------------------------------------------------


def test_advisory_can_never_assert_dispatch_or_approval(tmp_path):
    from my_jev.trading.advisory import AdvisoryLedger, TradingAdvisory

    ledger = AdvisoryLedger(tmp_path / "a.jsonl")
    good = TradingAdvisory(
        product_id="BTC-USD",
        asof="2026-01-01T00:00:00+00:00",
        schema_version=1,
        options=OPTIONS,
        probabilities={name: 1 / 3 for name in OPTIONS},
        top_option="hold",
        top_probability=1 / 3,
    )
    ledger.emit(good, "id-1")

    # The dataclass defaults are false, and the ledger refuses a record that
    # claims otherwise even if one is constructed by hand.
    assert good.dispatch_allowed is False
    assert good.approval_granted is False
    tampered = TradingAdvisory(
        product_id="BTC-USD",
        asof="2026-01-01T00:00:00+00:00",
        schema_version=1,
        options=OPTIONS,
        probabilities={name: 1 / 3 for name in OPTIONS},
        top_option="buy",
        top_probability=0.9,
        dispatch_allowed=True,
    )
    with pytest.raises(ValueError, match="never assert dispatch"):
        ledger.emit(tampered, "id-2")


def test_advisory_carries_no_order_shaped_field():
    from my_jev.trading.advisory import TradingAdvisory

    advisory = TradingAdvisory(
        product_id="BTC-USD",
        asof="2026-01-01T00:00:00+00:00",
        schema_version=1,
        options=OPTIONS,
        probabilities={name: 1 / 3 for name in OPTIONS},
        top_option="hold",
        top_probability=1 / 3,
    )
    payload = advisory.as_dict()
    banned = ("order", "size", "qty", "quantity", "venue", "notional", "limit")
    offenders = [k for k in payload if any(w in k.lower() for w in banned)]
    assert not offenders, f"advisory exposes order-shaped fields: {offenders}"


def test_advisory_defaults_to_unvalidated():
    """A missing validation verdict must never read as a passing one."""
    from my_jev.trading.advisory import TradingAdvisory

    advisory = TradingAdvisory(
        product_id="BTC-USD",
        asof="2026-01-01T00:00:00+00:00",
        schema_version=1,
        options=OPTIONS,
        probabilities={name: 1 / 3 for name in OPTIONS},
        top_option="hold",
        top_probability=1 / 3,
    )
    assert advisory.validation_state == "unvalidated"


def test_ledger_tracks_open_evidence_and_scores_only_resolved(tmp_path):
    from my_jev.trading.advisory import AdvisoryLedger, TradingAdvisory

    ledger = AdvisoryLedger(tmp_path / "a.jsonl")
    for i in range(3):
        ledger.emit(
            TradingAdvisory(
                product_id="BTC-USD",
                asof=f"2026-01-0{i + 1}T00:00:00+00:00",
                schema_version=1,
                options=OPTIONS,
                probabilities={"buy": 0.8, "sell": 0.1, "hold": 0.1},
                top_option="buy",
                top_probability=0.8,
            ),
            f"id-{i}",
        )
    assert ledger.unresolved() == ["id-0", "id-1", "id-2"]
    # Only the first resolves, and coverage must show that.
    ledger.resolve("id-0", {"forward_return": 0.01})
    stats = ledger.score(cost_bps=40.0)
    assert stats["emitted"] == 3
    assert stats["scored"] == 1
    assert stats["coverage"] == pytest.approx(1 / 3)
    assert ledger.unresolved() == ["id-1", "id-2"]


# ---------------------------------------------------------------------------
# Validation mechanics
# ---------------------------------------------------------------------------


def test_walk_forward_folds_expand_and_stay_disjoint():
    from my_jev.trading.validation import walk_forward

    stamps = pd.date_range("2025-01-01", periods=6000, freq="h", tz="UTC").to_numpy()
    folds = walk_forward(stamps, n_folds=4, horizon=168)
    assert len(folds) == 4
    sizes = [len(f.train) for f in folds]
    assert sizes == sorted(sizes), "training window must expand"
    for i in range(len(folds) - 1):
        assert folds[i].train_end < folds[i].test_start
        assert not set(folds[i].test) & set(folds[i + 1].test)
    for fold in folds:
        assert fold.train_end < fold.test_start


def test_non_overlapping_thins_to_one_open_position_per_asset():
    from my_jev.trading.validation import non_overlapping

    stamps = pd.date_range("2025-01-01", periods=2000, freq="h", tz="UTC").to_numpy()
    pids = np.array(["A"] * 2000 + ["B"] * 2000, dtype=object)
    times = np.concatenate([stamps, stamps])
    keep = non_overlapping(
        np.arange(4000),
        product_ids=pids,
        timestamps=times,
        spacing=pd.Timedelta(hours=168),
    )
    assert len(keep) < 4000 / 100
    for i in range(1, len(keep)):
        if pids[keep[i]] == pids[keep[i - 1]]:
            gap = pd.Timestamp(times[keep[i]]) - pd.Timestamp(times[keep[i - 1]])
            assert gap >= pd.Timedelta(hours=168)


def test_skill_term_separates_skill_from_sizing():
    """Edge and skill are different quantities, and only one is prediction.

    The pure-sizing case is a model that always says "buy": over a sample with a
    positive drift it books a large ``mean(s*r)`` while ``cov(s, r)`` is exactly
    zero, because a constant position carries no information about the move. That
    is the shape a confident-but-clueless model produces.
    """
    from my_jev.trading.validation import skill_and_pnl

    rng = np.random.default_rng(0)
    n = 4000
    drifting = rng.normal(0.002, 0.05, n)  # positive mean

    # Always long: big edge, no skill whatsoever.
    always_long = skill_and_pnl(np.full(n, 0.9), drifting, cost_bps=0.0)
    assert abs(always_long["skill_bp"]) < 1e-6, "constant position cannot be skill"
    assert always_long["sizing_bp"] > 0, "the edge here is entirely a bet-size product"
    # 0.2% drift at 0.9 size is ~18bp gross; assert the sign and rough scale
    # rather than a magic number, since the point is that edge > 0 with skill = 0.
    assert always_long["edge_bp"] > 5
    assert always_long["edge_bp"] == pytest.approx(always_long["sizing_bp"], rel=0.05)

    # Perfect direction: genuine skill.
    perfect = skill_and_pnl(np.sign(drifting) * 0.5, drifting, cost_bps=0.0)
    assert perfect["hit_rate"] == pytest.approx(1.0)
    assert perfect["skill_bp"] > 100, "a perfect forecaster must show real skill"
    assert perfect["sizing_bp"] == pytest.approx(0.0, abs=1.0), (
        "a perfect forecaster earns from direction, not from bet size"
    )

    # A coin flip: no skill, ~50% accuracy.
    coin = skill_and_pnl(
        rng.choice([-0.5, 0.5], size=n), drifting, cost_bps=0.0
    )
    # Sampling noise on cov(±0.5, r) with n=4000 and sigma(r)~500bp is ~4bp, so a
    # few bp of apparent skill is expected from a genuine null.
    assert abs(coin["skill_bp"]) < 10.0
    assert coin["hit_rate"] == pytest.approx(0.5, abs=0.05)


def test_bootstrap_ci_brackets_the_point_estimate():
    from my_jev.trading.validation import bootstrap_skill_ci

    rng = np.random.default_rng(1)
    raw = rng.normal(0.0, 0.05, 3000)
    signed = np.sign(raw) * rng.uniform(0.2, 0.9, 3000)
    ci = bootstrap_skill_ci(signed, raw, n_resamples=500, seed=3)
    assert ci["low"] <= ci["point"] <= ci["high"]
    assert ci["p_positive"] > 0.9, "a genuinely predictive signal should be flagged"

    noise = rng.normal(0.0, 1.0, 3000)
    null = bootstrap_skill_ci(noise, raw, n_resamples=500, seed=3)
    assert null["low"] < 0 < null["high"], "a null signal must not produce a tight positive CI"
