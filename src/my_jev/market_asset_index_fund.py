"""Policy for a US equity index fund tracking the S&P 500.

Chosen as the first asset class for a concrete reason rather than as a
placeholder. Its failure mode is the least dramatic and the most reliably
measured: the harm that actually reaches people holding a broad index fund is
selling out during a drawdown, not a missed rally. A system that makes that
failure easier is worse than no system, so the interesting design question here
is not "will it go up" -- this module holds no view on that -- but **"is this
evidence a reason to change course at all?"**

## What differs from a continuously traded instrument

**Sessions.** The fund trades 09:30-16:00 America/New_York on weekdays, with
extended hours around it. That single fact changes the freshness rule:

* During the session, a quote minutes old is current, and the window is short.
* Outside it, the most recent trade is from the *previous session* and is
  entirely valid evidence -- there simply is no newer one. A naive elapsed-time
  rule would call every overnight quote stale, so the system would abstain
  constantly and correctly, which teaches an operator to ignore it.

So outside the session the window opens to cover the time since the last close,
and ``session_at`` reports which regime applies.

**NAV, not last trade.** The fund publishes an official NAV once daily. It is a
different quantity from the last trade -- it is struck on the underlying
basket's closing prices, not on when anyone happened to trade -- and treating one
as the other is a category error. During the session the trade price is what a
proposal should be judged on; outside it, the last close and the NAV are both
candidates and the policy says which it used, by name, in the output.

**Structure, not ticks.** An index fund of several hundred holdings has no
single-name event risk to trade on. A large move in one constituent is not a
reason to change a position in the fund that contains it. The policy is
deliberately unable to propose a trade on the strength of one constituent.

## What it will not do

It does not forecast the index, does not time drawdowns, and does not size a
position. Where evidence does not clearly support a change it proposes holding,
which is a real recommendation and is reported distinctly from abstention --
"nothing to do" and "not enough to say" are different findings.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from .market_decision import (
    AssetPolicy,
    MarketAction,
    MarketQuote,
    MarketSession,
)

#: US equity regular trading hours. The zone is fixed rather than inferred so a
#: daylight-saving transition cannot silently shift what "the close" means.
EASTERN = ZoneInfo("America/New_York")

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
PRE_MARKET_OPEN = time(4, 0)
AFTER_HOURS_CLOSE = time(20, 0)

#: Weekend, plus the fixed-date closures that actually move the calendar. A real
#: deployment needs a maintained calendar feed; this is honest about being partial
#: rather than pretending to know every holiday.
_FIXED_CLOSURES = frozenset(
    {
        # New Year's Day, observed
        "01-01",
        # Independence Day, observed
        "07-04",
        # Christmas, observed
        "12-25",
    }
)

#: Freshness during the session. A liquid index fund has negligible bid/ask
#: spread and trades constantly, so minutes-old evidence is current evidence.
SESSION_FRESHNESS_SECONDS = 300

#: How much of a move, intraday, is treated as noise rather than information.
#: Deliberately coarse: the aim is to abstain on ordinary volatility, not to
#: claim a threshold that would need justifying.
INTRADAY_MOVE_THRESHOLD = 0.01

#: How far past the close before "after hours" is treated as closed. Weekend and
#: overnight are the same case operationally: no live pricing.
AFTER_HOURS_LINGER_SECONDS = 4 * 3600

#: Tolerance between feeds quoting the same consolidated tape.
FEED_DISAGREEMENT_THRESHOLD = 0.001


def is_trading_day(moment: datetime) -> bool:
    """Whether the US equity market is open on this Eastern date."""
    eastern = moment.astimezone(EASTERN)
    if eastern.weekday() >= 5:
        return False
    return eastern.strftime("%m-%d") not in _FIXED_CLOSURES


@dataclass(frozen=True)
class IndexFundPolicy(AssetPolicy):
    """Session-aware policy for a US equity index fund.

    ``intraday_move_threshold`` and ``session_freshness_seconds`` are fields
    rather than constants so a caller can tighten or loosen them per instrument
    without subclassing, and so a threshold change is visible in a stored
    decision's inputs.
    """

    asset_class: str = "us_equity_index_fund"
    intraday_move_threshold: float = INTRADAY_MOVE_THRESHOLD
    session_freshness_seconds: int = SESSION_FRESHNESS_SECONDS
    #: Two feeds on one consolidated tape should agree to a tick or two. Wider
    #: than a real spread and this starts excusing a genuinely broken feed.
    feed_disagreement_threshold: float = FEED_DISAGREEMENT_THRESHOLD

    # --- session ------------------------------------------------------------

    def session_at(self, now: datetime) -> MarketSession:
        eastern = now.astimezone(EASTERN)
        if not is_trading_day(now):
            return MarketSession.closed
        local = eastern.time()
        if MARKET_OPEN <= local < MARKET_CLOSE:
            return MarketSession.regular
        if PREMARKET := (PRE_MARKET_OPEN <= local < MARKET_OPEN):
            return MarketSession.pre_market
        if MARKET_CLOSE <= local < time(AFTER_HOURS_CLOSE.hour, AFTER_HOURS_CLOSE.minute):
            return MarketSession.after_hours
        del PREMARKET
        return MarketSession.closed

    # --- freshness ----------------------------------------------------------

    def freshness_window_seconds(self, now: datetime) -> int:
        """Short during the session, long outside it.

        Outside the session the newest possible evidence is the last close, so a
        short window would reject the only quote that exists. That is the bug this
        override exists to prevent, and it is why ``now`` is a parameter.
        """
        session = self.session_at(now)
        if session is MarketSession.regular:
            return self.session_freshness_seconds
        window = self._non_session_window(now, session)
        # A negative window makes every quote stale on arrival, which looks like a
        # working safety check while being exactly the failure this design set out
        # to prevent. Clamped rather than trusted.
        return max(0, window)

    def _non_session_window(
        self,
        now: datetime,
        session: MarketSession,
    ) -> int:

        if not is_trading_day(now):
            # Weekend or holiday: the last close was the final session before it.
            return int(
                (now - _last_weekday_close(now)).total_seconds()
            ) + self.session_freshness_seconds
        if session in (MarketSession.pre_market, MarketSession.after_hours):
            return int(
                (now - _most_recent_session_edge(now)).total_seconds()
            ) + self.session_freshness_seconds
        # Closed on a trading day. Two distinct cases share this branch, and they
        # do not have the same answer: after the close, today's close is behind
        # us, but before today's open it is still ahead -- which is the overnight
        # gap, and measuring against a future close collapses the window to zero.
        close_today = _today_close(now)
        if now >= close_today:
            return (
                int((now - close_today).total_seconds())
                + self.session_freshness_seconds
            )
        return int(
            (now - _most_recent_session_edge(now)).total_seconds()
        ) + self.session_freshness_seconds

    # --- price basis --------------------------------------------------------

    def price_basis(self, quote: MarketQuote, *, session: MarketSession) -> str | None:
        """Which field a proposal should be judged on, by name.

        During the session the trade price is what an execution would meet. In
        extended hours a last trade is thin and unrepresentative, so the NAV or
        the close is used instead. The chosen field is reported in
        ``price_basis`` so a stored decision names what it reasoned about.
        """
        if session is MarketSession.regular:
            return "last" if quote.last is not None else None
        for candidate in ("nav", "last"):
            if candidate == "nav" and quote.nav is not None:
                return "nav"
            if candidate == "last" and quote.nav is None and quote.last is not None:
                return "last"
        return None

    def _basis_value(self, quote: MarketQuote, basis: str | None) -> float | None:
        if basis is None:
            return None
        return quote.prices().get(basis)

    # --- reconciliation -----------------------------------------------------

    def reconcile(
        self,
        observations: Sequence[MarketQuote],
        *,
        session: MarketSession,
    ) -> tuple[MarketQuote | None, str | None]:
        """Several feeds on one consolidated tape should agree closely.

        The opposite situation to bitcoin, and the reason ``reconcile`` is a hook
        rather than a fixed rule: an index fund has a single consolidated tape, so
        two feeds disagreeing means one of them is broken rather than that the
        asset has two prices. Declining is right, and the tolerance is tight
        because real disagreement on one tape is a fault, not a spread.

        The default implementation would have accepted the first observation and
        quietly ignored the contradiction.
        """
        if not observations:
            return None, "no usable observation survived gating"
        if len(observations) == 1:
            return observations[0], None

        prices = sorted(
            value
            for value in (observation.last for observation in observations)
            if value is not None and value > 0.0
        )
        if len(prices) < 2:
            return observations[0], None

        low, high = prices[0], prices[-1]
        spread = (high - low) / low
        if spread > self.feed_disagreement_threshold:
            return (
                None,
                f"{len(observations)} feeds on one consolidated tape disagree by "
                f"{round(spread * 100, 3)}%, beyond the "
                f"{round(self.feed_disagreement_threshold * 100, 3)}% tolerated; "
                "on a single tape that is a broken feed, not a spread",
            )

        centre = (low + high) / 2
        chosen = min(
            (o for o in observations if o.last is not None),
            key=lambda o: (abs(o.last - centre), o.source or ""),
        )
        note = None
        if spread > 0:
            note = (
                f"{len(observations)} feeds agreed within "
                f"{round(spread * 100, 3)}%; the price nearest the midpoint was "
                "reported rather than a synthesised average, since no feed quoted it"
            )
        return chosen, note

    # --- interpretation -----------------------------------------------------

    def interpret(
        self,
        quote: MarketQuote,
        *,
        session: MarketSession,
    ) -> tuple[MarketAction, float, str]:
        """Propose a change only on evidence that clearly warrants one.

        The bar is deliberately asymmetric. A fund of several hundred holdings has
        no single-name event to trade on, so ordinary movement does not justify
        changing course; the cases that do are an implausible price and a
        dislocation between the trade price and the official NAV.
        """
        basis = self.price_basis(quote, session=session)
        value = self._basis_value(quote, basis)
        if value is None:
            return (
                MarketAction.abstain,
                0.0,
                "no price field this policy can act on was present, so no "
                "proposal is possible",
            )

        if quote.nav is not None and quote.last is not None and quote.nav > 0.0:
            dislocation = abs(quote.last - quote.nav) / quote.nav
            if dislocation > INTRADAY_MOVE_THRESHOLD * 5:
                # A trade price far from the official NAV means one of them is
                # wrong, not that the fund moved. Proposing a trade on either
                # would be acting on a broken feed.
                return (
                    MarketAction.abstain,
                    0.0,
                    "trade price and official NAV disagree by "
                    f"{round(dislocation * 100, 2)}%, which points to a bad feed "
                    "rather than a real move; nothing is proposed until they agree",
                )

        reference = _session_reference(quote, basis=basis)
        if reference is not None and reference > 0.0:
            move = abs(value - reference) / reference
            if move <= self.intraday_move_threshold:
                return (
                    MarketAction.propose_hold,
                    0.6,
                    f"price is within {round(self.intraday_move_threshold * 100, 2)}% "
                    f"of its {session.value} reference, which is ordinary movement "
                    "for a broad index fund and not a reason to change course",
                )
            return (
                MarketAction.propose_hold,
                0.45,
                f"price has moved {round(move * 100, 2)}% against its "
                f"{session.value} reference; a broad index fund has no "
                "constituent-level catalyst for that, so a single observation is "
                "not sufficient grounds to propose a change",
            )

        return (
            MarketAction.propose_hold,
            0.35,
            "no prior reference is available in this observation, so there is "
            "nothing to have moved and no basis for proposing a change",
        )


def _session_reference(quote: MarketQuote, *, basis: str | None) -> float | None:
    """A price to compare ``basis`` against, excluding ``basis`` itself.

    Excluding it matters: the previous version returned whichever of nav/last it
    found first, which during a session is the very field being judged. Every
    comparison came out 0.0% and every quote landed on "within threshold" -- the
    interpret path looked like it was working while never once disagreeing.

    With no independent reference there is genuinely nothing to have moved, and
    the caller says so rather than inventing a comparison.
    """
    for candidate in ("nav", "last", "bid", "ask"):
        if candidate == basis:
            continue
        value = quote.prices().get(candidate)
        if value is not None and value > 0.0:
            return value
    return None


# --- calendar helpers ------------------------------------------------------


def _today_close(now: datetime) -> datetime:
    eastern = now.astimezone(EASTERN)
    return datetime.combine(eastern.date(), MARKET_CLOSE, tzinfo=EASTERN)


def _most_recent_session_edge(now: datetime) -> datetime:
    """The latest session boundary at or before ``now``.

    Only boundaries in the past count. Taking ``max(open, close)`` blindly
    returned *today's close* during pre-market, which is still ahead of us, and
    produced a negative freshness window -- at which point every quote is
    instantly stale and the policy abstains on everything. That is precisely the
    bug the session-aware window exists to avoid.
    """
    eastern = now.astimezone(EASTERN)
    candidates = [
        datetime.combine(eastern.date() - timedelta(days=offset), edge, tzinfo=EASTERN)
        for offset in (0, 1)
        for edge in (MARKET_CLOSE, PRE_MARKET_OPEN, MARKET_OPEN)
    ]
    behind = [moment for moment in candidates if moment <= eastern]
    return max(behind) if behind else _last_weekday_close(now)


def _last_weekday_close(now: datetime) -> datetime:
    """The close of the most recent weekday that is not a fixed holiday."""
    eastern = now.astimezone(EASTERN)
    day = eastern.date()
    for _ in range(10):
        if day.weekday() < 5 and day.strftime("%m-%d") not in _FIXED_CLOSURES:
            return datetime.combine(day, MARKET_CLOSE, tzinfo=EASTERN)
        day -= timedelta(days=1)
    # Unreachable for any real date; explicit rather than a silent wrong answer.
    return _today_close(now)


__all__ = [
    "IndexFundPolicy",
    "is_trading_day",
]