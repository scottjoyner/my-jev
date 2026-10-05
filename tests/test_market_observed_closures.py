"""Fixed-date holidays that land on a weekend are observed on a weekday.

The documentation used to name this as the gap it did not model:

> Independence Day 2026 falls on a Saturday and the market closes Friday 3 July,
> which this does not model.

That gap had a cost, and it was the expensive direction. On 3 July 2026 the policy
reported a *regular* session, so the freshness window was 30 seconds, so a quote
carried over from the previous session was judged stale and the advisory abstained
-- on a day the market was shut, for a reason the operator could not check.

Modelled now, with the standard rule: a holiday on Saturday is observed the
preceding Friday, and on Sunday the following Monday.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from my_jev.market_asset_index_fund import (
    IndexFundPolicy,
    _FIXED_CLOSURES,
    is_fixed_closure,
    is_trading_day,
)
from my_jev.market_decision import MarketSession

EASTERN = ZoneInfo("America/New_York")


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 7, 3, hour, minute, tzinfo=EASTERN)


# --- the rule --------------------------------------------------------------


@pytest.mark.parametrize(
    ("day", "expected", "why"),
    [
        # The documentation's own example.
        (date(2026, 7, 3), True, "4 July 2026 is a Saturday -> observed Friday"),
        (date(2026, 7, 4), True, "the nominal date, itself a Saturday"),
        (date(2026, 7, 6), False, "the Monday after is a trading day"),
        (date(2021, 7, 5), True, "4 July 2021 is a Sunday -> observed Monday"),
        (date(2021, 7, 6), False, "the Tuesday after is a trading day"),
        # New Year's Day is the case that makes the rule a function rather than a
        # table: 1 Jan 2022 was a Saturday, so the closure was Friday 31 December
        # 2021 -- a date whose own nominal holiday is nothing at all.
        (date(2021, 12, 31), True, "1 Jan 2022 is a Saturday -> observed prior Friday"),
        (date(2027, 12, 24), True, "Christmas 2027 is a Saturday -> observed Friday"),
        (date(2027, 12, 27), False, "the Monday after Christmas is a trading day"),
        (date(2022, 12, 26), True, "Christmas 2022 is a Sunday -> observed Monday"),
        # Nominal dates that need no adjustment.
        (date(2026, 1, 1), True, "New Year on a Thursday"),
        (date(2026, 12, 25), True, "Christmas on a Friday"),
        (date(2027, 1, 1), True, "New Year on a Friday"),
        # Ordinary days.
        (date(2026, 6, 3), False, "an ordinary Wednesday"),
        (date(2026, 1, 2), False, "the day after New Year is open"),
    ],
)
def test_observed_closure_dates(day, expected, why):
    assert is_fixed_closure(day) is expected, f"{day}: {why}"


def test_every_nominal_closure_is_still_recognised_on_its_own_date():
    """The observation rule must not have replaced the plain lookup."""
    for nominal in _FIXED_CLOSURES:
        month, day = (int(part) for part in nominal.split("-"))
        assert is_fixed_closure(date(2026, month, day)) is True, nominal


# --- the consequence, which is the part that matters -----------------------


def test_the_observed_closure_is_not_a_regular_session():
    policy = IndexFundPolicy()
    assert policy.session_at(_at(10)) is MarketSession.closed
    assert is_trading_day(_at(10)) is False


def test_the_observed_closure_gets_the_overnight_window_not_thirty_seconds():
    """The actual cost of not modelling it.

    A regular session means a 30-second freshness window, which judges a
    previous-session quote stale and abstains. On a day the market is shut that
    abstention is spurious, and spurious abstentions are what teach an operator to
    ignore the ones that matter.
    """
    policy = IndexFundPolicy()
    observed = policy.freshness_window_seconds(_at(10))
    ordinary = policy.freshness_window_seconds(
        datetime(2026, 7, 6, 10, 0, tzinfo=EASTERN)
    )
    assert ordinary == policy.session_freshness_seconds
    assert observed > 3600


def test_a_quote_from_the_previous_session_survives_the_observed_closure():
    """End to end: the evidence is valid and must not be called stale.

    Before this, the policy called a three-day-old Friday close stale on Friday 3
    July and abstained, when 3 July was a holiday and the close was two sessions
    earlier and perfectly good evidence.
    """
    policy = IndexFundPolicy()
    observed_at = datetime(2026, 7, 2, 16, 0, tzinfo=EASTERN)  # Thursday's close
    evaluated = _at(10)
    age = (evaluated.astimezone(UTC) - observed_at.astimezone(UTC)).total_seconds()
    assert age <= policy.freshness_window_seconds(evaluated)


def test_the_last_close_before_an_observed_closure_skips_it():
    """`_last_weekday_close` had its own copy of the closure test.

    Two copies of a rule this shape is how half of it ends up applied to one call
    site and forgotten in the other, so the second site is asserted directly rather
    than assumed to follow.
    """
    # Friday 3 July is a closure, so the most recent close before it is Thursday
    # 2 July -- not 3 July.
    from my_jev.market_asset_index_fund import _last_weekday_close

    close = _last_weekday_close(_at(10))
    assert close.date() == date(2026, 7, 2)
    assert close.timetz().replace(tzinfo=None) == time(16, 0)


def test_session_at_agrees_with_is_trading_day_across_a_full_year():
    """The two must not drift, and the sweep is what would catch it."""
    policy = IndexFundPolicy()
    day = date(2026, 1, 1)
    checked = 0
    while day < date(2027, 1, 1):
        probe = datetime.combine(day, time(10, 0), tzinfo=EASTERN)
        weekday_session = (
            policy.session_at(probe) is MarketSession.regular
        )
        trading = is_trading_day(probe)
        assert weekday_session is trading, day
        checked += 1
        day += timedelta(days=1)
    assert checked == 365


def test_no_weekday_in_2026_is_a_closure_unless_it_should_be():
    """The closure list is exactly the nominal dates plus their observations."""
    expected = {"2026-01-01", "2026-07-03", "2026-12-25"}
    found = set()
    day = date(2026, 1, 1)
    while day < date(2027, 1, 1):
        if is_fixed_closure(day) and day.weekday() < 5:
            found.add(str(day))
        day += timedelta(days=1)
    assert found == expected


def test_a_weekend_holiday_does_not_close_the_following_monday_by_accident():
    """One rule, one direction. A Saturday closure must not leak to the Monday."""
    # 4 July 2026 is Saturday: Friday 3rd is observed, Monday 6th is open.
    assert is_fixed_closure(date(2026, 7, 3)) is True
    assert is_fixed_closure(date(2026, 7, 6)) is False
    # 4 July 2021 is Sunday: Monday 5th is observed, Friday 2nd is open.
    assert is_fixed_closure(date(2021, 7, 5)) is True
    assert is_fixed_closure(date(2021, 7, 2)) is False
