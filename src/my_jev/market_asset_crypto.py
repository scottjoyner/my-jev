"""Policy for spot bitcoin.

Built to test the extension point rather than to broaden it. The index-fund
policy was written first and ``AssetPolicy`` was claimed to be a real abstraction
rather than a single-use shape; this is the test of that claim. If bitcoin needs
special cases the interface does not offer, the abstraction was decoration and
the honest thing is to say so.

## What genuinely differs, and it is mostly simpler

**No calendar.** Bitcoin trades continuously, including weekends. ``session_at``
returns ``regular`` unconditionally and the policy holds no holiday table at all.
Compare the index fund, which needs an Eastern-time zone, session boundaries, and
a partial holiday list that is wrong about observed holidays. All of that
complexity is absent here because the domain does not have it.

**Freshness really is elapsed time.** The index fund's window had to open
overnight to accept the last close, because outside its session no newer
observation can exist. That is a property of the *asset class*, not of freshness
as such. Here a current price always exists, so the window stays short and honest.

Thin overnight liquidity is a separate concern from staleness and is handled as
such: a stale weekend quote is rejected as stale, while a fresh-but-thin one is
reflected in the confidence rather than being called stale.

**No NAV, so no reference price to arbitrate.** An index fund publishes an
official NAV struck on its basket's closing prices, and choosing between it and a
last trade was a real decision. Bitcoin has no equivalent. The trade price *is*
the observation, and the policy names it as the basis without pretending a
sibling field exists to compare against.

**Volatility is an order of magnitude wider.** A one-percent move in a broad
index fund is notable; in bitcoin it is a quiet hour. Reusing the index fund's
threshold would abstain on almost nothing and propose on noise, so the
threshold is a policy field with a different default, not a shared constant.

## Where the abstraction strains, honestly

Two places, both recorded rather than worked around silently:

**There is no consolidated tape.** A bitcoin quote is from *one venue*, and two
venues can disagree about the same asset by more than an entire index-fund daily
move. ``AssetPolicy`` evaluates one ``MarketQuote`` and never sees a second one,
so it has no way to compare venues or notice a dislocation between them. The
index fund does not need this: it has a single consolidated tape and an official
NAV. Handling it properly means widening the interface to accept several
observations per instrument, which is a real change and is not smuggled in here.

**Custody and venue availability are not prices.** An exchange going down during
volatility is the characteristic operational failure, and nothing in this layer
sees an exchange's status. That belongs in a different kind of check — venue
liveness, not quote evidence — and pretending a price model covers it would be
worse than leaving it out.

Both gaps are why this policy abstains more readily than the index-fund one.
Understating what it knows is the cheaper error when the thing being under-asked
about is somebody's savings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from .market_decision import (
    AssetPolicy,
    MarketAction,
    MarketQuote,
    MarketSession,
)
from collections.abc import Sequence

#: Current-price freshness during continuous trading. Bitcoin quotes are
#: continuous and the market never closes, so this can be genuinely tight -- the
#: opposite of the index-fund policy, where the window had to stretch overnight
#: because no newer observation could exist.
CONTINUOUS_FRESHNESS_SECONDS = 30

#: Overnight and weekend liquidity thins. Applied from Friday close in UTC through
#: to Monday open, which for a 24/7 asset is the only period that genuinely
#: behaves differently.
THIN_HOURS_START_UTC = 20  # 20:00 UTC Friday
THIN_HOURS_END_UTC = 1  # 01:00 UTC Monday

#: A one-percent move is unremarkable here in a way it is not for an index fund,
#: so the default threshold is an order of magnitude wider.
THIN_LIQUIDITY_MOVE_THRESHOLD = 0.02
NORMAL_MOVE_THRESHOLD = 0.10

#: Beyond this, a single observation is not evidence of anything. Bitcoin can and
#: does move several percent in an hour, but a system that proposes on one print
#: is reacting to noise it cannot distinguish from news.
DISLOCATION_MOVE_THRESHOLD = 0.25


#: How far venues may disagree before the readings are treated as suspect.
#: Well below any real move: the point is catching a broken feed, not
#: arbitraging a genuine price difference.
VENUE_DISAGREEMENT_THRESHOLD = 0.005


def _midpoint(quote: MarketQuote) -> float | None:
    """Best single price for a quote, or None when it carries none."""
    if quote.last is not None:
        return quote.last
    if quote.bid is not None and quote.ask is not None:
        return (quote.bid + quote.ask) / 2
    return None


def _closest_to_midpoint(
    observations: Sequence[MarketQuote],
    low: float,
    high: float,
) -> MarketQuote:
    """The observation nearest the middle of where venues agree.

    Picking the median rather than the first observation keeps the choice
    deterministic and independent of input order.
    """
    centre = (low + high) / 2
    return min(
        (o for o in observations if _midpoint(o) is not None),
        key=lambda o: (abs(_midpoint(o) - centre), o.source or ""),
    )


def is_thin_liquidity(now: datetime) -> bool:
    """Whether ``now`` falls in the weekend/overnight liquidity trough.

    Expressed in UTC because a 24/7 market has no single domestic calendar, and
    Friday-evening-to-Monday-morning is the window in which order books are
    thinnest almost everywhere that matters.
    """
    moment = now.astimezone(UTC)
    weekday, hour = moment.weekday(), moment.hour
    if weekday == 4 and hour >= THIN_HOURS_START_UTC:  # Friday evening
        return True
    if weekday in (5, 6):  # Saturday, Sunday
        return True
    if weekday == 0 and hour < THIN_HOURS_END_UTC:  # Monday small hours
        return True
    return False


@dataclass(frozen=True)
class BitcoinSpotPolicy(AssetPolicy):
    """Continuously-traded spot policy, with no calendar of its own.

    Every threshold is a constructor field so a caller can tighten or loosen it per
    venue without subclassing, and so a stored decision records the ruleset that
    produced it.
    """

    asset_class: str = "crypto_spot_bitcoin"
    freshness_seconds: int = CONTINUOUS_FRESHNESS_SECONDS
    #: How far venues may disagree before the readings are treated as suspect.
    #: Sized against a normal cross-venue spread, not against a daily move.
    venue_disagreement_threshold: float = VENUE_DISAGREEMENT_THRESHOLD
    thin_liquidity_move_threshold: float = THIN_LIQUIDITY_MOVE_THRESHOLD
    normal_move_threshold: float = NORMAL_MOVE_THRESHOLD

    # --- session ------------------------------------------------------------

    def session_at(self, now: datetime) -> MarketSession:  # noqa: ARG002
        """Always ``regular``. There is no session to speak of.

        Returning ``closed`` for a weekend here would be the mistake this whole
        abstraction exists to avoid: it would mark live weekend quotes stale and
        leave the system abstaining when there is a perfectly good price to read.
        """
        return MarketSession.regular

    # --- freshness ----------------------------------------------------------

    def freshness_window_seconds(self, now: datetime) -> int:  # noqa: ARG002
        """Tight, because a current price always exists.

        No calendar means no overnight widening. Thinned liquidity is *not*
        handled here either -- widening the window to excuse an old quote would
        blur "stale" into "quiet", and the operator needs to know which it was.
        """
        return self.freshness_seconds

    # --- price basis --------------------------------------------------------

    def price_basis(self, quote: MarketQuote, *, session: MarketSession) -> str | None:
        """The trade price. There is no NAV and nothing to arbitrate against.

        ``session`` is ignored deliberately: a 24/7 market has no session in which
        a different field becomes more appropriate, and the interface takes the
        parameter so that policies which *do* have sessions can use it.
        """
        if quote.last is not None:
            return "last"
        return None

    # --- reconciliation -----------------------------------------------------

    def reconcile(
        self,
        observations: Sequence[MarketQuote],
        *,
        session: MarketSession,
    ) -> tuple[MarketQuote | None, str | None]:
        """Compare venues before interpreting, because there is no tape to trust.

        A single print is weak evidence here. Several venues agreeing is materially
        stronger, and several venues *disagreeing* is the finding: it means one of
        them is wrong, or none is liquid enough to price anything. Neither can be
        discovered by looking at quotes one at a time, which is why this exists
        rather than being folded into ``interpret``.
        """
        if not observations:
            return None, "no usable observation survived gating"

        by_venue: dict[str, list[MarketQuote]] = {}
        for observation in observations:
            # An unnamed feed cannot corroborate, so it is kept separate rather
            # than counted as another independent voice.
            by_venue.setdefault(observation.source or "<unspecified>", []).append(
                observation
            )

        # Repeated reports from one feed are not independent agreement.
        if len(by_venue) < 2:
            venue = next(iter(by_venue))
            count = len(by_venue[venue])
            return (
                by_venue[venue][0],
                None
                if count == 1
                else f"only one venue ({venue}) reported, {count} times; repeated "
                "reports from a single feed are not independent agreement",
            )

        midpoints = sorted(
            _midpoint(observation)
            for observation in observations
            if _midpoint(observation) is not None
        )
        if len(midpoints) < 2:
            return None, "venues reported no comparable mid price to reconcile"

        low, high = midpoints[0], midpoints[-1]
        disagreement = (high - low) / low
        if disagreement > self.venue_disagreement_threshold:
            return (
                None,
                f"{len(by_venue)} venues disagree by {round(disagreement * 100, 2)}%, "
                f"beyond the {round(self.venue_disagreement_threshold * 100, 2)}% "
                "this policy tolerates; that points to a stale or wrong feed rather "
                "than a tradeable price, and no venue is preferred over the others",
            )

        # Agreement is corroboration, so it raises confidence rather than
        # changing the action. One print and three agreeing venues still do not
        # support a proposal from this policy.
        #
        # The observation nearest the midpoint is chosen and its own price
        # reported. Averaging would publish a price no venue quoted, which is
        # exactly the sort of synthetic evidence this layer refuses elsewhere.
        return _closest_to_midpoint(observations, midpoints[0], midpoints[-1]), None

    # --- interpretation -----------------------------------------------------

    def interpret(
        self,
        quote: MarketQuote,
        *,
        session: MarketSession,
    ) -> tuple[MarketAction, float, str]:
        """Propose nothing from a single print.

        With no consolidated tape, no official reference, and no constituent
        structure to reason from, one observation supports almost nothing. The
        honest outcomes are hold and abstain; a proposal would require corroboration
        this layer does not have.
        """
        if self.price_basis(quote, session=session) is None:
            return (
                MarketAction.abstain,
                0.0,
                "no trade price present, so nothing can be said about this "
                "instrument's current value",
            )

        thin = is_thin_liquidity(quote.observed_at)
        threshold = (
            self.thin_liquidity_move_threshold
            if thin
            else self.normal_move_threshold
        )

        # A single print has no independent reference, by construction. Rather
        # than compare the trade price against itself -- which yields 0.0% and a
        # finding that looks meaningful but is not -- this states the limitation.
        #
        # Depth is the one signal a single book does carry, and a one-sided book
        # during thin liquidity is worth declining to reason about.
        if quote.bid is not None and quote.ask is not None:
            spread = (quote.ask - quote.bid) / quote.ask
            if spread > DISLOCATION_MOVE_THRESHOLD / 5:
                return (
                    MarketAction.abstain,
                    0.0,
                    f"quoted spread is {round(spread * 100, 2)}% of the ask, which "
                    "indicates a thin or broken book rather than a price; a "
                    "proposal from this quote would be reacting to an artefact",
                )

        if thin:
            return (
                MarketAction.propose_hold,
                0.3,
                f"observed during thin overnight or weekend liquidity, where a "
                f"single print can move several percent without news; the "
                f"{round(threshold * 100, 2)}% threshold for this period is not "
                "evidence of anything on its own",
            )

        if quote.bid is None or quote.ask is None:
            return (
                MarketAction.propose_hold,
                0.25,
                "only one side of the book was quoted, so there is no spread to "
                "judge depth by and no second venue to corroborate against",
            )

        return (
            MarketAction.propose_hold,
            0.2,
            "a single consolidated-free print supports no proposal; there is no "
            "official reference and no second observation to corroborate it, and "
            f"the {round(threshold * 100, 2)}% move threshold is untested without "
            "either",
        )


__all__ = [
    "BitcoinSpotPolicy",
    "is_thin_liquidity",
]
