from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from my_jev.market_asset_crypto import BitcoinSpotPolicy, is_thin_liquidity
from my_jev.market_asset_index_fund import IndexFundPolicy
from my_jev.market_decision import (
    EvidenceRejection,
    MarketAction,
    MarketQuote,
    MarketSession,
    decide_market_action,
    decide_many,
)

POLICY = BitcoinSpotPolicy()
NOW = datetime(2026, 10, 7, 14, 0, tzinfo=UTC)  # Wednesday 14:00 UTC


def quote(**kwargs) -> MarketQuote:
    base = {
        "instrument_id": "opaque:bitcoin",
        "last": 95_000.0,
        "bid": 94_990.0,
        "ask": 95_010.0,
        "observed_at": NOW - timedelta(seconds=5),
    }
    base.update(kwargs)
    return MarketQuote(**base)


# --- 24/7 is the whole point ------------------------------------------------


@pytest.mark.parametrize(
    "moment",
    [
        datetime(2026, 10, 7, 14, tzinfo=UTC),  # Wednesday
        datetime(2026, 10, 10, 12, tzinfo=UTC),  # Saturday
        datetime(2026, 10, 11, 3, tzinfo=UTC),  # Sunday small hours
        datetime(2026, 12, 25, 12, tzinfo=UTC),  # Christmas
        datetime(2026, 1, 1, 12, tzinfo=UTC),  # New Year
    ],
    ids=["weekday", "saturday", "sunday", "christmas", "new-year"],
)
def test_every_moment_is_a_regular_session(moment):
    """Marking a live weekend quote closed would be the mistake this design avoids."""
    assert POLICY.session_at(moment) is MarketSession.regular


def test_the_freshness_window_never_moves():
    """A 24/7 market always has a current price, so no calendar widening is needed."""
    windows = {
        POLICY.freshness_window_seconds(moment)
        for moment in (
            datetime(2026, 10, 7, 14, tzinfo=UTC),
            datetime(2026, 10, 10, 12, tzinfo=UTC),
            datetime(2026, 12, 25, 12, tzinfo=UTC),
        )
    }
    assert windows == {POLICY.freshness_seconds}


def test_a_holiday_is_an_ordinary_trading_day_end_to_end():
    christmas = datetime(2026, 12, 25, 12, tzinfo=UTC)
    evidence = quote(observed_at=christmas - timedelta(seconds=5))
    advisory = decide_market_action(evidence, POLICY, now=christmas)
    assert advisory.session is MarketSession.regular
    assert advisory.rejected == {}


# --- thin liquidity is separate from staleness ------------------------------


@pytest.mark.parametrize(
    ("moment", "thin"),
    [
        (datetime(2026, 10, 9, 12, tzinfo=UTC), False),  # Friday midday
        (datetime(2026, 10, 9, 21, tzinfo=UTC), True),  # Friday evening
        (datetime(2026, 10, 10, 12, tzinfo=UTC), True),  # Saturday
        (datetime(2026, 10, 11, 2, tzinfo=UTC), True),  # Sunday night
        (datetime(2026, 10, 12, 0, tzinfo=UTC), True),  # Monday 00:00 UTC, still thin
        (datetime(2026, 10, 12, 2, tzinfo=UTC), False),  # Monday 02:00 UTC, Asia open
        (datetime(2026, 10, 12, 12, tzinfo=UTC), False),  # Monday midday
    ],
)
def test_the_thin_window_is_identified(moment, thin):
    assert is_thin_liquidity(moment) is thin


def test_a_thin_period_quote_abstains_from_proposing():
    weekend = datetime(2026, 10, 10, 12, tzinfo=UTC)
    evidence = quote(observed_at=weekend - timedelta(seconds=5))
    advisory = decide_market_action(evidence, POLICY, now=weekend)
    assert advisory.action is MarketAction.propose_hold
    assert advisory.actionable is False
    assert "thin" in advisory.reason


def test_thin_liquidity_does_not_make_a_stale_quote_acceptable():
    """Widening the window to excuse an old quote would blur stale into quiet."""
    stale = quote(
        observed_at=NOW - timedelta(hours=3),
    )
    advisory = decide_market_action(stale, POLICY, now=NOW)
    assert advisory.action is MarketAction.abstain
    assert advisory.rejected.get(EvidenceRejection.stale) == 1


# --- price basis ------------------------------------------------------------


def test_the_basis_is_the_trade_price():
    """There is no NAV and nothing to arbitrate against."""
    advisory = decide_market_action(quote(), POLICY, now=NOW)
    assert advisory.price_basis == "last"


def test_a_quote_with_no_trade_price_abstains():
    advisory = decide_market_action(quote(last=None, bid=None, ask=None), POLICY, now=NOW)
    assert advisory.action is MarketAction.abstain
    assert advisory.price_basis is None


# --- interpretation --------------------------------------------------------


def test_one_print_never_produces_an_actionable_proposal():
    """No consolidated tape, no official reference, no corroboration."""
    for last in (95_000.0, 90_000.0, 100_000.0, 60_000.0):
        evidence = quote(last=last, observed_at=NOW - timedelta(seconds=5))
        assert decide_market_action(evidence, POLICY, now=NOW).actionable is False, last


def test_bitcoin_never_proposes_from_a_normal_weekday_print():
    advisory = decide_market_action(quote(), POLICY, now=NOW)
    assert advisory.action is MarketAction.propose_hold
    assert "single" in advisory.reason
    assert advisory.decision_confidence <= 0.3


def test_bitcoin_confidence_is_below_the_index_funds():
    """It should hold its claims more loosely than a fund with a consolidated tape."""
    advisory = decide_market_action(quote(), POLICY, now=NOW)
    index_advisory = decide_market_action(
        MarketQuote(
            instrument_id="opaque:index",
            last=500.0,
            nav=499.9,
            observed_at=datetime(2026, 10, 6, 15, 0, tzinfo=UTC) - timedelta(seconds=5),
            session=MarketSession.regular,
        ),
        IndexFundPolicy(),
        now=datetime(2026, 10, 6, 15, 0, tzinfo=UTC),
    )
    assert advisory.decision_confidence < index_advisory.decision_confidence


# --- the abstraction, tested rather than asserted --------------------------


def test_one_core_decides_for_both_asset_classes():
    """`AssetPolicy` earns its keep if two very different assets share a path.

    Bitcoin has no calendar, no NAV, and a volatility an order of magnitude wider.
    If it needed its own decision function the extension point would be
    decoration, and that is the thing worth knowing.
    """
    session_now = datetime(2026, 10, 6, 15, 0, tzinfo=UTC)
    quotes = [
        MarketQuote(
            instrument_id="opaque:index",
            last=500.0,
            bid=499.9,
            ask=500.1,
            nav=499.8,
            observed_at=session_now - timedelta(seconds=5),
            session=MarketSession.regular,
        ),
        MarketQuote(
            instrument_id="opaque:bitcoin",
            last=95_000.0,
            bid=94_990.0,
            ask=95_010.0,
            observed_at=session_now - timedelta(seconds=5),
            session=MarketSession.regular,
        ),
    ]
    results = {
        advisory.asset_class: advisory
        for advisory in decide_many(
            quotes,
            {"opaque:index": IndexFundPolicy(), "opaque:bitcoin": POLICY},
            now=session_now,
        )
    }

    assert set(results) == {"us_equity_index_fund", "crypto_spot_bitcoin"}
    # Same action vocabulary, different confidence: the asset class is what differs.
    assert results["us_equity_index_fund"].action is MarketAction.propose_hold
    assert results["crypto_spot_bitcoin"].action is MarketAction.propose_hold
    assert (
        results["crypto_spot_bitcoin"].decision_confidence
        < results["us_equity_index_fund"].decision_confidence
    )


def test_the_decision_core_contains_no_asset_specific_branching():
    """Guard against the core quietly special-casing an asset later."""
    import inspect

    from my_jev import market_decision

    source = inspect.getsource(market_decision)
    for leak in ("IndexFund", "Bitcoin", "crypto_spot", "us_equity"):
        assert leak not in source, f"decision core references {leak}"


def test_both_policies_hold_no_authority():
    for policy, moment, evidence in (
        (
            IndexFundPolicy(),
            datetime(2026, 10, 6, 15, 0, tzinfo=UTC),
            MarketQuote(
                instrument_id="opaque:index",
                last=500.0,
                nav=499.9,
                observed_at=datetime(2026, 10, 6, 15, 0, tzinfo=UTC) - timedelta(seconds=5),
                session=MarketSession.regular,
            ),
        ),
        (
            POLICY,
            NOW,
            quote(),
        ),
    ):
        advisory = decide_market_action(evidence, policy, now=moment)
        assert set(advisory.authority.model_dump().values()) == {False}
        assert advisory.price_forecast is False


def test_the_move_threshold_is_reported_even_though_it_gates_nothing():
    """Honest about its own role.

    The threshold appears in the operator-facing reason so the rule that was
    applied is visible, but it does not currently change the *action*: one print
    is never enough regardless of how far it moved. A mutation narrowing the
    threshold to the index-fund scale passes every behavioural test, which is
    correct -- and worth pinned rather than left to be discovered later.
    """
    default = decide_market_action(quote(), POLICY, now=NOW)
    narrow = decide_market_action(
        quote(), BitcoinSpotPolicy(normal_move_threshold=0.01), now=NOW
    )

    assert "10.0% move threshold" in default.reason
    assert "1.0% move threshold" in narrow.reason
    # Same action either way: the threshold is descriptive, not a gate.
    assert default.action is narrow.action is MarketAction.propose_hold


def test_the_thin_liquidity_threshold_is_also_constructor_controlled():
    thin = decide_market_action(
        quote(observed_at=datetime(2026, 10, 10, 12, tzinfo=UTC) - timedelta(seconds=5)),
        BitcoinSpotPolicy(thin_liquidity_move_threshold=0.5),
        now=datetime(2026, 10, 10, 12, tzinfo=UTC),
    )
    assert "50%" in thin.reason or "50.0%" in thin.reason
