"""Decide whether market evidence justifies proposing a trade. Never place one.

Built to the same discipline as the fleet advisory layer, because the failure
modes rhyme. There, a lane that had gone stale was still listed as qualified and
an operator reading the output could not tell. Here, a quote from last night's
close is still a quote, and the difference between "valid evidence from the
previous session" and "stale evidence" is the difference between abstaining for
a real reason and abstaining for a bogus one.

## Freshness is asset-class aware, and that is the whole design

A single "evidence older than N minutes is stale" rule is correct for a
continuously traded instrument and wrong for an exchange-traded fund:

* **BTC** trades 24/7. Any wall-clock instant is a plausible observation, so
  freshness is genuinely a function of elapsed time.

* **VOO** trades 09:30-16:00 America/New_York. Outside those hours the most
  recent trade is from the *previous session*, and it is perfectly good evidence
  -- it is simply from a different session than the one you are asking about.
  A naive elapsed-time rule marks every pre-market or after-hours quote stale, so
  the system abstains constantly and for the wrong reason. An operator learns to
  ignore it, which is how a correct abstention becomes a useless one.

So the freshness rule is not a constant here; it is supplied by the asset class's
policy. :class:`AssetPolicy` is the extension point, and the reason it exists is
narrower than "different assets need different parameters".

## What this will not do

It does not predict prices. There is no model here that claims to know where an
asset goes next, and adding one would not make it more useful -- it would make it
wrong in a way that is invisible. What this evaluates is narrower and, in
practice, more valuable: **whether the evidence in hand justifies proposing
anything at all.** Most of the work is in deciding when the honest answer is to
abstain, and in saying which of several mundane reasons applied.

The output is a proposal. Placing an order is an external side effect, so
:class:`TradingAuthority` is all-false and a proposal never becomes an
instruction. That is a contract property, not a convention: supplying ``True``
anywhere on it is a validation error.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Reason strings, one per rejection cause. Kept as constants because they are
#: compared against by operators and by tests; prose that drifts silently is
#: worse than no prose.
REASON_STALE = "quote older than this asset's freshness window"
REASON_UNDATED = "quote carried no observation time"
REASON_FUTURE_DATED = "quote dated in the future; clock skew, not a price"
REASON_NON_POSITIVE = "quote price was not positive"
REASON_CROSSED = "quote bid and ask were crossed or empty"
REASON_OUT_OF_SESSION = "no trading session open for this asset at evaluation time"
REASON_NO_QUOTE = "no quote supplied for this asset"
REASON_AMBIGUOUS_PRICE = "quote carried no unambiguous price to act on"


class MarketSession(StrEnum):
    """When an observation was taken, relative to the asset's trading calendar."""

    regular = "regular"
    pre_market = "pre_market"
    after_hours = "after_hours"
    closed = "closed"


class PolicyVerdict(StrEnum):
    """Why the asset policy declined, when it declined for its own reasons.

    Distinct from an evidence rejection: the quote may be perfectly good and the
    system still unable to say anything, because nobody supplied the rules for
    this instrument. Reporting that as "stale" sends an operator to fix a feed
    that was never broken.
    """

    no_policy = "no_policy"
    no_session = "no_session"
    no_basis = "no_basis"
    policy_abstained = "policy_abstained"


class MarketAction(StrEnum):
    """What to propose. Every one of these is a proposal, never an instruction."""

    #: Nothing in the evidence justifies a proposal. The common case, on purpose.
    abstain = "abstain"
    #: Evidence supports adding exposure. Requires human approval to act.
    propose_increase = "propose_increase"
    #: Evidence supports trimming exposure. Requires human approval to act.
    propose_reduce = "propose_reduce"
    #: Evidence supports no change; explicitly *not* the same as abstain, because
    #: "nothing to do" and "not enough to say anything" are different findings and
    #: collapsing them is how a system that never speaks up gets trusted blindly.
    propose_hold = "propose_hold"


class EvidenceRejection(StrEnum):
    """Why an observation was not usable. One per cause, never conflated."""

    absent = "absent"
    undated = "undated"
    future_dated = "future_dated"
    stale = "stale"
    non_positive = "non_positive"
    crossed = "crossed"
    ambiguous_price = "ambiguous_price"


class MarketQuote(BaseModel):
    """One observed price for an asset, addressed by instrument id.

    ``instrument_id`` is an opaque surrogate, not a ticker or an account. This
    layer never needs to know what it is holding, and a field that could carry a
    position would eventually carry one.
    """

    model_config = ConfigDict(extra="forbid")

    instrument_id: str = Field(min_length=1, max_length=64)
    #: Exactly one price should be present. Which one matters per asset class --
    #: an ETF has an official NAV that a last trade cannot substitute for -- so
    #: ambiguity is carried into the decision rather than resolved here.
    last: float | None = Field(default=None, gt=0.0)
    bid: float | None = Field(default=None, gt=0.0)
    ask: float | None = Field(default=None, gt=0.0)
    nav: float | None = Field(default=None, gt=0.0)
    #: When the observation was taken. Required and zone-aware: a quote with no
    #: time cannot be aged, and ageing it against an assumed zone widens the
    #: freshness window in exactly the direction that loses money.
    observed_at: datetime
    session: MarketSession = MarketSession.regular
    #: Where the number came from, recorded so a proposal can name its evidence.
    source: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def _zone_aware(self) -> MarketQuote:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        return self

    def prices(self) -> dict[str, float]:
        return {
            name: value
            for name, value in (
                ("last", self.last),
                ("bid", self.bid),
                ("ask", self.ask),
                ("nav", self.nav),
            )
            if value is not None
        }

    def crossed(self) -> bool:
        """True when bid and ask are present but not a valid pair."""
        if self.bid is None and self.ask is None:
            return not bool(self.prices())
        if self.bid is None or self.ask is None:
            # One-sided is not crossed; it is just thin. Treated separately so a
            # half-quoted book is not reported as a broken one.
            return False
        return self.bid > self.ask


class TradingAuthority(BaseModel):
    """All-false. A proposal is not an instruction."""

    model_config = ConfigDict(extra="forbid")

    order_placed: Literal[False] = False
    order_queued: Literal[False] = False
    capital_moved: Literal[False] = False
    position_opened: Literal[False] = False
    position_closed: Literal[False] = False
    approval_granted: Literal[False] = False
    approval_bypassed: Literal[False] = False
    runtime_authority_changed: Literal[False] = False


class MarketDecisionAdvisory(BaseModel):
    """Whether evidence justifies proposing a trade. Advisory only."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "my-jev-market-decision-v1"
    authority: TradingAuthority = TradingAuthority()
    advisory_only: Literal[True] = True

    instrument_id: str = Field(min_length=1, max_length=64)
    asset_class: str = Field(min_length=1, max_length=64)
    action: MarketAction = MarketAction.abstain
    #: English for a human. Composed from the structured fields rather than
    #: scraped, so rewording elsewhere cannot silently repoint it.
    reason: str = Field(default="", max_length=512)

    #: Confidence in the *recommendation to propose*, never in a price direction.
    #: Stated explicitly because a single "confidence" number invites the reading
    #: that this module forecasts, which it does not.
    decision_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Price field this module actually used, by name.
    price_basis: str | None = Field(default=None, max_length=64)
    #: The value it used and when it was observed. Without these a decision does
    #: not name its evidence, so two decisions made from different quotes produce
    #: identical documents and neither can be audited after the fact.
    evidence_price: float | None = Field(default=None, gt=0.0)
    evidence_observed_at: str | None = None
    evidence_source: str | None = Field(default=None, max_length=64)
    session: MarketSession = MarketSession.regular

    #: Freshness window actually applied, which depends on the asset class.
    freshness_window_seconds: int = Field(ge=0)
    evaluated_at: str | None = None

    #: Observations that were not usable, by cause.
    rejected: dict[EvidenceRejection, int] = Field(default_factory=dict)
    #: Always true. Present so a reader cannot mistake this for a signal.
    price_forecast: Literal[False] = False
    notes: list[str] = Field(default_factory=list)

    @property
    def actionable(self) -> bool:
        """Whether this proposes anything beyond holding.

        Abstain and hold are both non-actionable, and they are not the same
        finding, so this does not collapse them either.
        """
        return self.action in (MarketAction.propose_increase, MarketAction.propose_reduce)


class AssetPolicy(ABC):
    """What makes evidence valid for one asset class.

    Abstract because the freshness rule genuinely differs, not for symmetry. See
    the module docstring: an elapsed-time rule is correct for a 24/7 instrument
    and produces constant bogus abstentions for a session-traded fund.
    """

    #: Reported on every advisory so a stored decision names the ruleset it used.
    asset_class: str = "unknown"

    @abstractmethod
    def freshness_window_seconds(self, now: datetime) -> int:
        """How old an observation may be and still be usable, *at this moment*.

        Takes ``now`` because the answer can depend on where we are in the
        session: an overnight gap is not the same as an eight-hour-old quote
        during continuous trading.
        """

    @abstractmethod
    def session_at(self, now: datetime) -> MarketSession:
        """Which session ``now`` falls in, if the concept applies at all."""

    @abstractmethod
    def price_basis(self, quote: MarketQuote, *, session: MarketSession) -> str | None:
        """Which price field this asset class should be judged on.

        Takes the session computed at evaluation time rather than reading the
        quote's own ``session`` field. A feed that declares itself ``regular``
        when the market is shut would otherwise talk the system into the
        intraday basis, and the declaration is the untrusted input here.
        """

    @abstractmethod
    def interpret(
        self,
        quote: MarketQuote,
        *,
        session: MarketSession,
    ) -> tuple[MarketAction, float, str]:
        """Propose an action, a decision confidence, and a composed reason.

        The contract every asset policy owes the caller. Implementations must
        abstain rather than guess; returning a confident proposal from thin
        evidence is the one failure this framework exists to prevent.
        """


@dataclass
class _NullPolicy(AssetPolicy):
    """Used when no policy is supplied. Abstains, and says why."""

    asset_class: str = "unspecified"

    def freshness_window_seconds(self, now: datetime) -> int:  # noqa: ARG002
        return 0

    def session_at(self, now: datetime) -> MarketSession:  # noqa: ARG002
        return MarketSession.closed

    def price_basis(self, quote: MarketQuote, *, session: MarketSession) -> str | None:
        return None

    def interpret(
        self,
        quote: MarketQuote,
        *,
        session: MarketSession,
    ) -> tuple[MarketAction, float, str]:
        return (
            MarketAction.abstain,
            0.0,
            "no asset policy was supplied, so nothing can be said about this "
            "instrument; supplying a policy is what makes a proposal possible",
        )


def gate_evidence(
    quote: MarketQuote | None,
    policy: AssetPolicy,
    *,
    now: datetime,
) -> tuple[MarketQuote | None, list[EvidenceRejection], str | None]:
    """Decide whether one quote is usable, naming every reason it is not.

    Returns ``(usable_quote, rejections, basis)``. All rejections are collected
    rather than short-circuiting on the first, because an operator fixing a
    broken feed wants the whole list, not whichever check happened to run first.
    """
    if quote is None:
        return None, [EvidenceRejection.absent], None

    rejections: list[EvidenceRejection] = []
    if not quote.prices():
        rejections.append(EvidenceRejection.ambiguous_price)
    if quote.crossed():
        rejections.append(EvidenceRejection.crossed)

    observed = quote.observed_at.astimezone(UTC)
    moment = now.astimezone(UTC)
    age = (moment - observed).total_seconds()
    if age < 0:
        rejections.append(EvidenceRejection.future_dated)
    elif age > policy.freshness_window_seconds(moment):
        rejections.append(EvidenceRejection.stale)

    basis = policy.price_basis(quote, session=policy.session_at(now.astimezone(UTC)))
    # A policy declining to name a basis is not evidence being unusable, so it is
    # reported as a policy-level abstention elsewhere rather than folded in here.
    # Only a quote with genuinely no price field is `ambiguous_price`.

    if rejections:
        return None, rejections, basis
    return quote, [], basis


def rejection_summary(rejections: Sequence[EvidenceRejection]) -> str:
    """One operator-facing sentence naming each rejection cause and its count."""
    if not rejections:
        return ""
    counts: dict[str, int] = {}
    for rejection in rejections:
        counts[rejection.value] = counts.get(rejection.value, 0) + 1
    detail = ", ".join(f"{cause}={count}" for cause, count in sorted(counts.items()))
    return f"evidence not used ({detail})"


def advisory_digest(advisory: MarketDecisionAdvisory) -> str:
    """Stable id for a decision, so two identical decisions are recognisable."""
    payload = advisory.model_dump(mode="json", exclude={"evaluated_at"})
    encoded = "|".join(f"{key}={payload[key]}" for key in sorted(payload))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def decide_market_action(
    quote: MarketQuote | None,
    policy: AssetPolicy | None = None,
    *,
    now: datetime | None = None,
) -> MarketDecisionAdvisory:
    """Propose an action for one instrument, or abstain and say why.

    The pipeline is: check the clock, gate the evidence, then let the asset class
    interpret what survived. Abstention is a first-class outcome with a cause,
    not an error path.
    """
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    active = policy or _NullPolicy()
    notes: list[str] = []

    session = active.session_at(moment)
    window = active.freshness_window_seconds(moment)

    if quote is None:
        usable, rejections, basis = None, [EvidenceRejection.absent], None
    else:
        usable, rejections, basis = gate_evidence(quote, active, now=moment)

    if isinstance(active, _NullPolicy):
        # Before the staleness check: with no policy the freshness window is zero,
        # so every quote would be reported as stale, which is a different fault
        # with a different fix and would send an operator after the wrong thing.
        return MarketDecisionAdvisory(
            instrument_id=quote.instrument_id if quote is not None else "unknown",
            asset_class=active.asset_class,
            action=MarketAction.abstain,
            reason=(
                "no asset policy was supplied, so nothing can be said about this "
                "instrument; supplying a policy is what makes a proposal possible"
            )[:512],
            freshness_window_seconds=window,
            session=session,
            evaluated_at=moment.replace(microsecond=0).isoformat(),
            notes=notes,
        )

    if usable is None:
        cause = {
            EvidenceRejection.absent: REASON_NO_QUOTE,
            EvidenceRejection.undated: REASON_UNDATED,
            EvidenceRejection.future_dated: REASON_FUTURE_DATED,
            EvidenceRejection.stale: REASON_STALE,
            EvidenceRejection.non_positive: REASON_NON_POSITIVE,
            EvidenceRejection.crossed: REASON_CROSSED,
            EvidenceRejection.ambiguous_price: REASON_AMBIGUOUS_PRICE,
        }
        primary = rejections[0] if rejections else EvidenceRejection.absent
        reason = cause.get(primary, REASON_NO_QUOTE)
        summary = rejection_summary(rejections)
        if summary:
            reason = f"{reason}; {summary}"
        return MarketDecisionAdvisory(
            instrument_id=quote.instrument_id if quote is not None else "unknown",
            asset_class=active.asset_class,
            action=MarketAction.abstain,
            reason=reason[:512],
            freshness_window_seconds=window,
            session=session,
            rejected={rejection: rejections.count(rejection) for rejection in set(rejections)},
            evaluated_at=moment.replace(microsecond=0).isoformat(),
            notes=notes,
        )

    action, confidence, reason = active.interpret(usable, session=session)
    if basis is not None and basis not in reason:
        reason = f"{reason} (on {basis})"

    return MarketDecisionAdvisory(
        instrument_id=usable.instrument_id,
        asset_class=active.asset_class,
        action=action,
        reason=reason[:512],
        decision_confidence=max(0.0, min(1.0, float(confidence))),
        price_basis=basis,
        evidence_price=usable.prices().get(basis) if basis else None,
        evidence_observed_at=usable.observed_at.astimezone(UTC)
        .replace(microsecond=0)
        .isoformat(),
        evidence_source=usable.source,
        session=session,
        freshness_window_seconds=window,
        evaluated_at=moment.replace(microsecond=0).isoformat(),
        rejected={},
        notes=notes,
    )


def decide_many(
    quotes: Sequence[MarketQuote],
    policies: Mapping[str, AssetPolicy],
    *,
    now: datetime | None = None,
) -> list[MarketDecisionAdvisory]:
    """Decide across instruments, in a stable order.

    Sorted by instrument id so the output does not depend on input order, which
    would make two identical portfolios produce two different documents.
    """
    moment = now or datetime.now(UTC)
    out = [
        decide_market_action(
            quote,
            policies.get(quote.instrument_id),
            now=moment,
        )
        for quote in sorted(quotes, key=lambda item: item.instrument_id)
    ]
    return out


__all__ = [
    "AssetPolicy",
    "EvidenceRejection",
    "MarketAction",
    "MarketDecisionAdvisory",
    "MarketQuote",
    "MarketSession",
    "TradingAuthority",
    "advisory_digest",
    "decide_market_action",
    "decide_many",
    "gate_evidence",
    "rejection_summary",
]