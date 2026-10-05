"""Policy for a single listed equity in a thinly traded name.

Added as the third asset class, and the choice is the point rather than an
accident of convenience. The first two policies were chosen where the evidence
tends to be *good*: bitcoin trades continuously, and a broad index fund has a
consolidated tape and a published NAV. Both can mostly be trusted, so the design
question in each was how much machinery the freshness rule needed.

A thinly traded single name is where that stops being true, and it is where this
framework's central obligation gets tested:

    > ``interpret`` must abstain rather than guess, and returning a confident
    > proposal from thin evidence is the one failure this design exists to prevent.

Every reason this policy refuses is a reason a naive policy would have spoken:

* **the spread is the evidence.** A thin name quotes wide, so a move measured
  mid-spread is mostly the spread moving. This policy reads the spread before it
  reads the price, and refuses a proposal whose move is not large relative to it.
* **one venue is not corroboration.** A thinly traded name often has exactly one
  feed. That feed's own series is that feed's story, and a proposal on it would be
  the module reporting its single input back to the caller as if it were a
  finding. Two independent venues are required, as for bitcoin.
* **there is no index to compare against.** For the index fund, a move with no
  constituent-level explanation is news. For a single name, *most* moves have a
  company-level explanation this layer knows nothing about, so the same move
  carries almost no information. The bar is set accordingly and the reason string
  says so rather than implying a catalyst was found.
* **a reversal earns nothing.** A failed rally in a thin name is often a single
  print that got hit. Reading it as a trend is how a system starts trading noise.

## What it will not do

It does not know what the company did, does not read news, and does not attempt
to identify a catalyst. There is no view here about any business. Where the
evidence does not clearly support a change it proposes holding, which is reported
distinctly from abstention -- "nothing to do" and "not enough to say" are
different findings and conflating them is how a system that never speaks up gets
trusted blindly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, time

from .market_asset_index_fund import EASTERN, is_trading_day
from .market_decision import (
    AssetPolicy,
    MarketAction,
    MarketQuote,
    MarketSession,
)

#: A move, as a fraction of the quoted spread, below which it is indistinguishable
#: from crossing the spread. The rule is deliberately expressed this way rather
#: than as a percentage move because a thin name's spread *is* its volatility: a
#: 5% move in a name quoting 8% wide is not a 5% move.
MINIMUM_MOVE_PERCENTAGE_POINTS_OF_SPREAD = 2.0

#: Sustained move required before a proposal, as a fraction. Much wider than the
#: index fund's, because a single name legitimately moves on its own news and this
#: layer cannot tell a catalyst from a fat finger.
PROPOSE_MOVE_THRESHOLD = 0.15

#: Half-spread above which the quote is too wide to act on. Wider than this and
#: the mid is not a price anyone can trade at.
MAX_HALF_SPREAD_FRACTION = 0.02


@dataclass(frozen=True)
class SingleNameEquityPolicy(AssetPolicy):
    """Session-aware policy for one thinly traded listed equity.

    ``requires_venue_liveness`` is true, unlike the index fund: there is no
    consolidated tape here, so each feed is a counterparty that can stop quoting
    mid-session. A wide-quoting single name is exactly where that matters, because
    the stale print is also the wide one.
    """

    asset_class: str = "single_name_equity"
    requires_venue_liveness: bool = True
    session_freshness_seconds: int = 30
    #: A thin name's history window is tighter than the core default, not looser.
    #: Two days covers a session and an overnight; a move from last month says
    #: nothing about today, and in a name this thin it is more likely to be a
    #: stale print still in someone's cache than a move anybody made.
    history_window_seconds_: int = 2 * 86400
    #: Two venues quoting the same single name should agree closely. They are
    #: quoting the same instrument, so a wide disagreement is a fault rather than
    #: a spread -- the same reasoning as the index fund's tape.
    venue_disagreement_threshold: float = 0.02
    propose_move_threshold: float = PROPOSE_MOVE_THRESHOLD
    max_half_spread_fraction: float = MAX_HALF_SPREAD_FRACTION

    # --- session ------------------------------------------------------------

    def session_at(self, now: datetime) -> MarketSession:
        eastern = now.astimezone(EASTERN)
        if not is_trading_day(now):
            return MarketSession.closed
        clock = eastern.time()
        if time(9, 30) <= clock < time(16, 0):
            return MarketSession.regular
        if time(4, 0) <= clock < time(9, 30):
            return MarketSession.pre_market
        if time(16, 0) <= clock < time(20, 0):
            return MarketSession.after_hours
        return MarketSession.closed

    def freshness_window_seconds(self, now: datetime) -> int:
        """Short in session, opening up overnight -- for the same reason as the fund.

        Outside the session the last trade is from the previous close and is
        perfectly good evidence. Marking it stale would abstain constantly and
        correctly, which teaches an operator to ignore abstention.
        """
        session = self.session_at(now)
        if session is MarketSession.regular:
            return self.session_freshness_seconds
        return 72 * 3600

    def history_window_seconds(self, now: datetime) -> int:
        return self.history_window_seconds_

    # --- price basis --------------------------------------------------------

    def price_basis(self, quote: MarketQuote, *, session: MarketSession) -> str | None:
        """The trade price during the session, the last trade outside it.

        There is no NAV to fall back on. That is the structural difference from the
        index fund: the fund publishes an official struck price once a day, and a
        single name does not, so outside the session the last trade is simply the
        best thing available -- which is a weaker statement than the fund's and is
        reflected in the freshness window above.
        """
        if session is MarketSession.regular:
            return "last" if quote.last is not None else None
        return "last" if quote.last is not None else None

    def reconcile(
        self,
        observations: Sequence[MarketQuote],
        *,
        session: MarketSession,
    ) -> tuple[MarketQuote | None, str | None]:
        """Two venues quoting one name should agree; a wide split means one is broken.

        Declining is right. Picking the "closest to consensus" quote would launder a
        broken feed into a valid reading, and the default implementation would have
        accepted the first observation and ignored the contradiction.
        """
        if len(observations) == 1:
            return observations[0], None

        prices = [
            float(observation.prices()["last"])
            for observation in observations
            if observation.last is not None
        ]
        if len(prices) < 2:
            return None, "venues did not all quote a last trade price"

        low, high = min(prices), max(prices)
        disagreement = (high - low) / low
        if disagreement > self.venue_disagreement_threshold:
            return None, (
                f"venues disagreed by {round(disagreement * 100, 1)}% on the same "
                "instrument, so at least one feed is broken; picking the price "
                "closest to consensus would launder a bad feed into a good reading"
            )
        return observations[0], (
            f"{len(observations)} venues agreed on the last trade to within "
            f"{round(disagreement * 100, 2)}%"
        )

    # --- the move -----------------------------------------------------------

    def consider_move(
        self,
        signal,
        *,
        quote,
        session,
    ):
        """Propose on a large sustained corroborated move. Rarely, and by design.

        Three independent bars, each of which can refuse on its own:

        * **two venues** -- a thin name often has one feed, and one feed's series is
          that feed's story;
        * **not a reversal** -- a move that did not hold has no established
          direction, and reading it as one is trading on a forecast;
        * **the move must exceed the spread** -- otherwise the finding is that the
          bid and ask moved apart, which is not a price change.

        That last bar is the one that distinguishes this policy. A move measured
        without reference to the spread is measured against the wrong yardstick
        precisely in the names where acting is most dangerous.
        """
        from .market_decision import SustainedMove

        if signal.venue_count < 2:
            return None
        if signal.is_reversal:
            return None
        if signal.magnitude < self.propose_move_threshold:
            return None

        spread = _fractional_spread(quote)
        if spread is None:
            return None
        if signal.magnitude < spread * MINIMUM_MOVE_PERCENTAGE_POINTS_OF_SPREAD:
            # The move is smaller than the cost of crossing the spread twice over.
            # Reporting a proposal here would be proposing a trade that cannot
            # profitably be made, which is worse than not proposing.
            return None

        direction = "risen" if signal.net_direction is SustainedMove.up else "fallen"
        action = (
            MarketAction.propose_increase
            if signal.net_direction is SustainedMove.up
            else MarketAction.propose_reduce
        )
        return (
            action,
            min(0.5, 0.2 + signal.magnitude),
            f"{direction} {round(signal.magnitude * 100, 1)}% and held across "
            f"{signal.sample_count} prints over "
            f"{round(signal.span_seconds / 60)} minutes, corroborated by "
            f"{signal.venue_count} venues, against a spread of "
            f"{round(spread * 100, 2)}%; a single name moves on its own news and "
            "this module knows nothing about the company, so this is a change "
            "large enough to be worth a human rather than a view on the business",
        )

    # --- interpretation -----------------------------------------------------

    def interpret(
        self,
        quote: MarketQuote,
        *,
        session: MarketSession,
    ) -> tuple[MarketAction, float, str]:
        """Propose a change only where a single name's evidence is unusually clear.

        The default reading of a thin name is a hold. A wide spread, a single
        venue, or a closed session all point the same way: there is not enough here
        to interrupt a human, and saying so is more useful than a confident reading
        built from one feed.
        """
        if session is not MarketSession.regular:
            return (
                MarketAction.propose_hold,
                0.2,
                "outside regular trading hours there is no live two-sided market in "
                "this name, so holding is the honest reading",
            )

        spread = _fractional_spread(quote)
        if spread is None:
            return (
                MarketAction.propose_hold,
                0.15,
                "the quote has no usable two-sided price, so there is nothing to "
                "act on",
            )
        if spread > self.max_half_spread_fraction:
            return (
                MarketAction.propose_hold,
                0.1,
                f"quoted half-spread is {round(spread * 100, 2)}%, too wide to act "
                "on; a thin market is not a reason to take the other side of it",
            )

        if quote.source is None:
            return (
                MarketAction.propose_hold,
                0.15,
                "the quote names no venue, so it cannot be attributed to a "
                "counterparty and cannot be relied on",
            )

        return (
            MarketAction.propose_hold,
            0.25,
            "a liquid-enough quote on a single name supports holding; this module "
            "has no view on the business, so a change here would need a reason it "
            "cannot supply",
        )


def _fractional_spread(quote: MarketQuote) -> float | None:
    """Half the bid-ask spread as a fraction of the mid, or ``None`` if unusable.

    Half-spread rather than full, because that is the cost of *crossing* to the
    other side -- what an actor actually pays to act on the quote.
    """
    if quote.bid is None or quote.ask is None or quote.bid <= 0 or quote.ask <= 0:
        return None
    mid = (quote.ask + quote.bid) / 2
    if mid <= 0:
        return None
    return (quote.ask - quote.bid) / 2 / mid


__all__ = [
    "MAX_HALF_SPREAD_FRACTION",
    "MINIMUM_MOVE_PERCENTAGE_POINTS_OF_SPREAD",
    "PROPOSE_MOVE_THRESHOLD",
    "SingleNameEquityPolicy",
]