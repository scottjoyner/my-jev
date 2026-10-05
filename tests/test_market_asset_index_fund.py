from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from my_jev.market_asset_index_fund import IndexFundPolicy, is_trading_day
from my_jev.market_decision import (
    EvidenceRejection,
    MarketAction,
    MarketQuote,
    MarketSession,
    decide_market_action,
)

EASTERN = ZoneInfo("America/New_York")
POLICY = IndexFundPolicy()

# Tuesday 2026-10-06 is an ordinary trading day; Saturday 2026-10-10 is not.
IN_SESSION = datetime(2026, 10, 6, 15, 0, tzinfo=UTC)      # 11:00 ET
PRE_MARKET = datetime(2026, 10, 6, 8, 30, tzinfo=UTC)      # 04:30 ET
AFTER_HOURS = datetime(2026, 10, 6, 21, 0, tzinfo=UTC)     # 17:00 ET
WEEKEND = datetime(2026, 10, 10, 16, 0, tzinfo=UTC)        # Sat 12:00 ET
OVERNIGHT = datetime(2026, 10, 12, 7, 0, tzinfo=UTC)       # Mon 03:00 ET
LAST_CLOSE = datetime(2026, 10, 5, 20, 0, tzinfo=UTC)      # Mon 16:00 ET


def quote(**kwargs) -> MarketQuote:
    base = {
        "instrument_id": "opaque:index-fund",
        "last": 500.0,
        "observed_at": LAST_CLOSE - timedelta(seconds=30),
        "session": MarketSession.after_hours,
    }
    base.update(kwargs)
    return MarketQuote(**base)


# --- the session calendar --------------------------------------------------


def test_a_weekday_is_a_trading_day():
    assert is_trading_day(IN_SESSION) is True


def test_a_weekend_is_not():
    assert is_trading_day(WEEKEND) is False
    assert is_trading_day(OVERNIGHT - timedelta(days=1)) is False


def test_a_fixed_holiday_is_not():
    assert is_trading_day(datetime(2026, 12, 25, 18, 0, tzinfo=UTC)) is False


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (IN_SESSION, MarketSession.regular),
        (PRE_MARKET, MarketSession.pre_market),
        (AFTER_HOURS, MarketSession.after_hours),
        (WEEKEND, MarketSession.closed),
    ],
)
def test_sessions_are_reported(moment, expected):
    assert POLICY.session_at(moment) is expected


def test_the_boundary_times_are_where_they_are_claimed_to_be():
    """09:30 ET open, 16:00 ET close, both in Eastern regardless of DST."""
    opening = datetime(2026, 10, 6, 9, 30, tzinfo=EASTERN).astimezone(UTC)
    closing = datetime(2026, 10, 6, 16, 0, tzinfo=EASTERN).astimezone(UTC)
    assert POLICY.session_at(opening) is MarketSession.regular
    assert POLICY.session_at(opening - timedelta(seconds=1)) is MarketSession.pre_market
    assert POLICY.session_at(closing - timedelta(seconds=1)) is MarketSession.regular
    assert POLICY.session_at(closing) is MarketSession.after_hours


def test_dst_is_handled_by_using_eastern_rather_than_a_fixed_offset():
    """A fixed UTC-5 would put the winter open an hour out."""
    winter = datetime(2026, 1, 6, 11, 0, tzinfo=EASTERN).astimezone(UTC)
    assert POLICY.session_at(winter) is MarketSession.regular


# --- the freshness window, which is the whole point ------------------------


@pytest.mark.parametrize(
    "moment",
    [IN_SESSION, PRE_MARKET, AFTER_HOURS, WEEKEND, OVERNIGHT],
    ids=["session", "pre-market", "after-hours", "weekend", "overnight"],
)
def test_no_session_ever_produces_a_negative_window(moment):
    """A negative window makes every quote instantly stale -- the bug this avoids."""
    assert POLICY.freshness_window_seconds(moment) >= 0


def test_the_window_is_short_during_the_session():
    assert POLICY.freshness_window_seconds(IN_SESSION) == 300


def test_the_window_opens_up_once_the_session_closes():
    """Outside the session the newest possible evidence is the last close."""
    assert POLICY.freshness_window_seconds(OVERNIGHT) > POLICY.freshness_window_seconds(
        IN_SESSION
    )


def test_the_window_grows_across_a_weekend():
    assert POLICY.freshness_window_seconds(WEEKEND) > POLICY.freshness_window_seconds(
        AFTER_HOURS
    )


# --- last close is valid evidence outside the session ----------------------


def test_the_last_close_seen_overnight_is_not_stale():
    """This is the case a naive elapsed-time rule gets wrong."""
    advisory = decide_market_action(quote(), POLICY, now=LAST_CLOSE + timedelta(hours=2))
    assert advisory.action is not MarketAction.abstain
    assert advisory.rejected == {}


def test_the_last_close_seen_in_a_new_session_is_stale():
    """Once trading resumes, yesterday's close is no longer current."""
    advisory = decide_market_action(quote(), POLICY, now=IN_SESSION + timedelta(hours=16))
    assert advisory.action is MarketAction.abstain
    assert advisory.rejected.get(EvidenceRejection.stale) == 1


# --- price basis -----------------------------------------------------------


def test_intraday_the_trade_price_is_the_basis():
    advisory = decide_market_action(
        quote(session=MarketSession.regular, observed_at=IN_SESSION - timedelta(seconds=20)),
        POLICY,
        now=IN_SESSION,
    )
    assert advisory.price_basis == "last"


def test_outside_the_session_the_nav_is_preferred_over_a_last_trade():
    """The NAV is struck differently from a last trade, so they are not the same."""
    advisory = decide_market_action(quote(nav=499.0), POLICY, now=LAST_CLOSE + timedelta(hours=2))
    assert advisory.price_basis == "nav"


def test_a_quote_with_no_usable_price_abstains():
    advisory = decide_market_action(
        quote(last=None, nav=None, bid=None, ask=None), POLICY, now=LAST_CLOSE
    )
    assert advisory.action is MarketAction.abstain


# --- interpretation --------------------------------------------------------


def test_a_broad_index_fund_does_not_propose_a_change_on_ordinary_noise():
    """The bar is asymmetric on purpose: noise is not a reason to act."""
    advisory = decide_market_action(
        quote(
            last=500.2,
            nav=500.0,
            session=MarketSession.regular,
            observed_at=IN_SESSION - timedelta(seconds=20),
        ),
        POLICY,
        now=IN_SESSION,
    )
    assert advisory.action is MarketAction.propose_hold
    assert advisory.actionable is False


def test_a_trade_price_far_from_the_nav_reads_as_a_bad_feed():
    """A 25% gap between them points at a broken feed, not a real move."""
    advisory = decide_market_action(
        quote(
            last=625.0,
            nav=500.0,
            session=MarketSession.regular,
            observed_at=IN_SESSION - timedelta(seconds=20),
        ),
        POLICY,
        now=IN_SESSION,
    )
    assert advisory.action is MarketAction.abstain
    assert "bad feed" in advisory.reason


def test_a_move_without_a_usable_reference_says_so():
    """Comparing last against last would make every comparison 0%."""
    advisory = decide_market_action(
        quote(
            last=500.0,
            nav=None,
            session=MarketSession.regular,
            observed_at=IN_SESSION - timedelta(seconds=20),
        ),
        POLICY,
        now=IN_SESSION,
    )
    assert advisory.action is MarketAction.propose_hold
    assert "no prior reference" in advisory.reason


def test_a_move_against_a_real_reference_is_reported_with_its_size():
    advisory = decide_market_action(
        quote(
            last=515.0,
            nav=500.0,
            session=MarketSession.regular,
            observed_at=IN_SESSION - timedelta(seconds=20),
        ),
        POLICY,
        now=IN_SESSION,
    )
    assert advisory.action is MarketAction.propose_hold
    assert "3.0%" in advisory.reason


def test_the_threshold_is_a_constructor_field_not_a_constant():
    """So a caller can tighten it per instrument, visibly."""
    tight = IndexFundPolicy(intraday_move_threshold=0.0)
    loose = IndexFundPolicy(intraday_move_threshold=0.5)
    evidence = quote(
        last=515.0,
        nav=500.0,
        session=MarketSession.regular,
        observed_at=IN_SESSION - timedelta(seconds=20),
    )
    # The move exceeds the tight threshold and falls inside the loose one, so the
    # threshold demonstrably drives which finding is reported.
    assert "is within" not in decide_market_action(evidence, tight, now=IN_SESSION).reason
    assert "is within" in decide_market_action(evidence, loose, now=IN_SESSION).reason


def test_no_decision_from_this_policy_is_ever_actionable_by_default():
    """A broad index fund should rarely, if ever, warrant a proposal from one quote."""
    for last in (500.0, 505.0, 515.0, 480.0):
        advisory = decide_market_action(
            quote(
                last=last,
                nav=500.0,
                session=MarketSession.regular,
                observed_at=IN_SESSION - timedelta(seconds=20),
            ),
            POLICY,
            now=IN_SESSION,
        )
        assert advisory.actionable is False, last


def test_authority_is_all_false():
    advisory = decide_market_action(quote(), POLICY, now=LAST_CLOSE)
    assert set(advisory.authority.model_dump().values()) == {False}


def test_the_calendar_decides_the_session_not_the_feed():
    """A feed that declares itself `regular` when the market is shut is untrusted.

    Before this was covered, the basis came from `quote.session` -- the feed's own
    claim. A mislabelled quote at 07:00 ET would then get the intraday trade-price
    basis, which is the one the market is not currently offering. Mutation-checked:
    reading the feed's declaration instead of the calendar passes every other test.
    """
    quote_claiming_regular = quote(
        last=500.0,
        nav=499.0,
        session=MarketSession.regular,
        observed_at=PRE_MARKET - timedelta(seconds=30),
    )
    advisory = decide_market_action(
        quote_claiming_regular, POLICY, now=PRE_MARKET
    )

    assert POLICY.session_at(PRE_MARKET) is MarketSession.pre_market
    assert advisory.session is MarketSession.pre_market
    # Outside the session the NAV is preferred, despite the feed saying otherwise.
    assert advisory.price_basis == "nav"


def test_the_same_quote_in_session_does_get_the_trade_price():
    quote_in_session = quote(
        last=500.0,
        nav=499.0,
        session=MarketSession.regular,
        observed_at=IN_SESSION - timedelta(seconds=30),
    )
    advisory = decide_market_action(quote_in_session, POLICY, now=IN_SESSION)
    assert advisory.session is MarketSession.regular
    assert advisory.price_basis == "last"
