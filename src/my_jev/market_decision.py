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
REASON_NO_BASIS = (
    "the asset policy declined to name a price field to act on, so the decision "
    "cannot say what it would have acted on"
)
REASON_VENUE_UNVERIFIED = (
    "no evidence the quoting venue is reachable; an unchecked venue is not an "
    "assumed-healthy one"
)
REASON_VENUE_LIVENESS_STALE = (
    "the quoting venue was last seen reachable too long ago to rely on now"
)
REASON_VENUE_DOWN = (
    "the quoting venue was observed unreachable, so its last price is a number "
    "with no counterparty behind it"
)


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
    #: The quote's venue could not be shown to be reachable. Kept distinct from
    #: `stale`: the price may be seconds old and still unusable, because there is
    #: nobody there to trade it against.
    venue_unverified = "venue_unverified"
    #: The venue was observed reachable at some point but has not been checked
    #: recently enough to rely on now.
    venue_liveness_stale = "venue_liveness_stale"
    #: The venue was observed not reachable.
    venue_down = "venue_down"
    #: There is no trading session open and the policy could not name a field to
    #: act on. Kept distinct from `absent`: a missing quote is a feed problem,
    #: whereas a closed market is a calendar fact, and reporting the latter as the
    #: former sends an operator to fix something that was never broken.
    out_of_session = "out_of_session"


#: The operator-facing reason for each rejection cause. Defined once, at module
#: scope, because it previously existed as two inline dictionaries and a silent
#: fallback to ``REASON_NO_QUOTE``: a cause nobody had remembered to name was
#: reported as "no quote", which is a different fault and points an operator at
#: the wrong repair. The tests assert this map covers every cause, and the call
#: sites index it directly so an unmapped cause fails loudly instead of lying.
_PRIMARY_REASON: dict[EvidenceRejection, str] = {
    EvidenceRejection.absent: REASON_NO_QUOTE,
    EvidenceRejection.undated: REASON_UNDATED,
    EvidenceRejection.future_dated: REASON_FUTURE_DATED,
    EvidenceRejection.stale: REASON_STALE,
    EvidenceRejection.non_positive: REASON_NON_POSITIVE,
    EvidenceRejection.crossed: REASON_CROSSED,
    EvidenceRejection.ambiguous_price: REASON_AMBIGUOUS_PRICE,
    EvidenceRejection.venue_unverified: REASON_VENUE_UNVERIFIED,
    EvidenceRejection.venue_liveness_stale: REASON_VENUE_LIVENESS_STALE,
    EvidenceRejection.venue_down: REASON_VENUE_DOWN,
    EvidenceRejection.out_of_session: REASON_OUT_OF_SESSION,
}


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


class VenueLiveness(BaseModel):
    """Evidence that a venue was actually reachable at a moment.

    Distinct from a price and gated separately, because a venue that has stopped
    quoting will happily serve its last price indefinitely. Acting on that is the
    characteristic operational failure for a continuously traded asset: the number
    looks fresh, the quote is well-formed, and the counterparty is gone.

    ``reachable=False`` is evidence too. A venue observed failing is *known* dead
    as of ``observed_at``, which is a firmer statement than knowing nothing, and
    the two are kept distinct because "we could not check" and "we checked and it
    is down" call for different responses.
    """

    model_config = ConfigDict(extra="forbid")

    venue: str = Field(min_length=1, max_length=64)
    reachable: bool
    observed_at: datetime
    #: Optional detail: an error string, a status code, whatever the probe saw.
    detail: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def _zone_aware(self) -> VenueLiveness:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        return self


#: Liveness is evidence about a *machine*, and machines fail faster than prices
#: go stale. Deliberately short: a liveness reading older than this is treated as
#: no reading at all.
DEFAULT_LIVENESS_MAX_AGE_SECONDS = 60


class InstrumentEvidence(BaseModel):
    """Every observation of one instrument, from however many venues.

    A single ``MarketQuote`` cannot express a disagreement between venues, and for
    a continuously traded, unconsolidated asset that disagreement *is* the
    information: two venues can differ by more than an entire day of index-fund
    movement, which means one of them is wrong or neither is liquid. Evaluating
    one quote at a time hid that behind a confident-sounding reading.

    ``source`` on each observation names the venue. Observations sharing a source
    are treated as the same feed reporting twice, which is not corroboration.
    """

    model_config = ConfigDict(extra="forbid")

    instrument_id: str = Field(min_length=1, max_length=64)
    observations: list[MarketQuote] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent_instrument(self) -> InstrumentEvidence:
        mismatched = sorted(
            {
                observation.instrument_id
                for observation in self.observations
                if observation.instrument_id != self.instrument_id
            }
        )
        if mismatched:
            raise ValueError(
                f"observations reference other instruments: {mismatched}"
            )
        return self

    def venues(self) -> list[str]:
        """Distinct venues, sorted, with unnamed feeds grouped as one.

        An observation with no ``source`` cannot be attributed to a venue, so it
        cannot corroborate anything. Counting it as its own venue would let a
        single anonymous feed masquerade as independent confirmation.
        """
        return sorted(
            {observation.source or "<unspecified>" for observation in self.observations}
        )

    def independent_venue_count(self) -> int:
        return len(self.venues())


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

    #: Why the *policy* declined, when it declined for its own reasons. Distinct
    #: from :attr:`rejected`, and the distinction is the point of the field: "the
    #: feed was broken" and "there are no rules, or no session, to reason with"
    #: are different findings that send an operator to different places.
    policy_verdict: PolicyVerdict | None = None

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

    #: Whether a quote's source must be shown reachable before it may be acted on.
    #:
    #: Defaults to True. A new asset class is therefore required to *opt out* of
    #: liveness checking rather than inheriting a silent pass, and the only
    #: sensible opt-out is a source that has no uptime of its own -- a consolidated
    #: tape, say, where the counterparty is an exchange rather than the feed. It is
    #: a property of the asset class, so it belongs here rather than on a caller
    #: flag that a caller could set carelessly.
    requires_venue_liveness: bool = True

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

    def reconcile(
        self,
        observations: Sequence[MarketQuote],
        *,
        session: MarketSession,
    ) -> tuple[MarketQuote | None, str | None]:
        """Choose which observation to act on, or decline and say why.

        The default accepts the observations as given and lets ``interpret`` see
        only the first usable one, which is the right behaviour for an asset with
        a consolidated tape where several feeds simply repeat the same price.

        An asset with *no* tape must override this: venues disagreeing is the
        finding, and reconciling them away before interpretation would discard
        exactly the evidence the decision needed.
        """
        return (observations[0] if observations else None), None

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


def gate_venue_liveness(
    venue: str | None,
    liveness: Sequence[VenueLiveness] | None,
    *,
    now: datetime,
    max_age_seconds: int = DEFAULT_LIVENESS_MAX_AGE_SECONDS,
) -> EvidenceRejection | None:
    """Whether a quote from this venue may be acted on at all.

    **Fails closed in every direction.** A venue with no liveness evidence is
    *unverified*, not assumed healthy -- "we could not check" and "it is fine" are
    different claims, and only the first is supported. A venue last seen reachable
    too long ago is treated as unknown rather than as a durable fact. A venue
    observed down is rejected outright.

    This runs before price interpretation on purpose: there is no reading of a
    price that rescues a counterparty that is not there.
    """
    if venue is None:
        # An unattributable quote cannot be attributed to a checked venue either.
        return EvidenceRejection.venue_unverified
    if not liveness:
        return EvidenceRejection.venue_unverified

    readings = [reading for reading in liveness if reading.venue == venue]
    if not readings:
        return EvidenceRejection.venue_unverified

    # Newest first, so a fresh "down" is not masked by an older "up".
    readings.sort(key=lambda reading: reading.observed_at, reverse=True)
    newest = readings[0]

    age = (now.astimezone(UTC) - newest.observed_at.astimezone(UTC)).total_seconds()
    if age < 0:
        # A future-dated liveness reading is clock skew, not a health claim.
        return EvidenceRejection.venue_unverified
    if not newest.reachable:
        return EvidenceRejection.venue_down
    if age > max_age_seconds:
        return EvidenceRejection.venue_liveness_stale
    return None


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

    session = policy.session_at(now.astimezone(UTC))
    basis = policy.price_basis(quote, session=session)
    # A policy declining to name a basis is a policy-level finding, not evidence
    # being unusable, so it is surfaced through the returned basis and turned into
    # an abstention by the caller rather than folded in here. The one exception is
    # a closed market: if there is no session open *and* the policy cannot name a
    # field to act on, there is nothing to reason about at all, and that is its own
    # cause rather than a missing quote.
    if basis is None and session is MarketSession.closed:
        rejections.append(EvidenceRejection.out_of_session)
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
    liveness: Sequence[VenueLiveness] | None = None,
    liveness_max_age_seconds: int = DEFAULT_LIVENESS_MAX_AGE_SECONDS,
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
        if usable is not None and active.requires_venue_liveness:
            venue_reason = gate_venue_liveness(
                usable.source,
                liveness,
                now=moment,
                max_age_seconds=liveness_max_age_seconds,
            )
            if venue_reason is not None:
                usable, rejections = None, [venue_reason]

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
            policy_verdict=PolicyVerdict.no_policy,
            evaluated_at=moment.replace(microsecond=0).isoformat(),
            notes=notes,
        )

    if usable is None:
        venue_reason = (
            gate_venue_liveness(
                quote.source if quote is not None else None,
                liveness,
                now=moment,
                max_age_seconds=liveness_max_age_seconds,
            )
            if active.requires_venue_liveness and quote is not None and quote.prices()
            else None
        )
        primary = venue_reason or (rejections[0] if rejections else EvidenceRejection.absent)
        # Indexed directly, not `.get`: a cause with no sentence here must fail
        # loudly rather than be silently reported as "no quote", which is a
        # different fault. `test_every_rejection_cause_has_an_operator_sentence`
        # asserts the map is complete.
        reason = _PRIMARY_REASON[primary]
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
            policy_verdict=(
                PolicyVerdict.no_session
                if primary is EvidenceRejection.out_of_session
                else None
            ),
            rejected={rejection: rejections.count(rejection) for rejection in set(rejections)},
            evaluated_at=moment.replace(microsecond=0).isoformat(),
            notes=notes,
        )

    if basis is None:
        # `gate_evidence` returned a basis and found it None. A policy that will
        # not name a price field cannot be acted on, and proceeding would produce
        # a document that names no evidence at all -- the failure the provenance
        # design exists to prevent. This is the "elsewhere" `gate_evidence`'s
        # comment promised and previously never reached.
        return MarketDecisionAdvisory(
            instrument_id=usable.instrument_id,
            asset_class=active.asset_class,
            action=MarketAction.abstain,
            reason=REASON_NO_BASIS[:512],
            freshness_window_seconds=window,
            session=session,
            policy_verdict=PolicyVerdict.no_basis,
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


def _corroborated(
    confidence: float,
    chosen: MarketQuote,
    usable: Sequence[MarketQuote],
) -> float:
    """Scale confidence by how many *independent* venues agreed.

    Applied by the core rather than by each policy, because independent
    corroboration raises confidence for any asset class; only the size of the
    effect is policy-specific. An unnamed feed is not a venue: two reports from
    ``source=None`` are one voice heard twice, and treating them as agreement
    would let a single feed manufacture its own confirmation.

    Confidence only ever moves upward, and never past 1.0. Corroboration can
    strengthen a reading but cannot make a policy more willing to *act*, which
    remains entirely ``interpret``'s call.
    """
    venues = {
        observation.source
        for observation in usable
        if observation.source is not None and observation is not chosen
    }
    if chosen.source is not None:
        venues.add(chosen.source)
    independent = len(venues)
    if independent <= 1:
        return confidence
    # Diminishing: a second voice is worth more than a fifth.
    lift = min(0.3, 0.15 * (independent - 1))
    return min(1.0, confidence + lift)


def decide_instrument(
    evidence: InstrumentEvidence | None,
    policy: AssetPolicy | None = None,
    *,
    now: datetime | None = None,
    liveness: Sequence[VenueLiveness] | None = None,
    liveness_max_age_seconds: int = DEFAULT_LIVENESS_MAX_AGE_SECONDS,
) -> MarketDecisionAdvisory:
    """Decide across every observation of one instrument.

    Each observation is gated on its own merits first, then the surviving ones are
    reconciled. Gating before reconciling matters: a stale quote from one venue
    must not be averaged into a fresh one from another and make a disagreement
    look like agreement.
    """
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    active = policy or _NullPolicy()
    notes: list[str] = []
    session = active.session_at(moment)
    window = active.freshness_window_seconds(moment)

    instrument_id = evidence.instrument_id if evidence is not None else "unknown"

    def abstain(
        reason: str,
        rejections: Sequence[EvidenceRejection] = (),
        verdict: PolicyVerdict | None = None,
    ) -> MarketDecisionAdvisory:
        summary = rejection_summary(rejections)
        return MarketDecisionAdvisory(
            instrument_id=instrument_id,
            asset_class=active.asset_class,
            action=MarketAction.abstain,
            reason=(f"{reason}; {summary}" if summary else reason)[:512],
            freshness_window_seconds=window,
            session=session,
            policy_verdict=verdict,
            rejected={r: rejections.count(r) for r in set(rejections)},
            evaluated_at=moment.replace(microsecond=0).isoformat(),
            notes=notes,
        )

    if isinstance(active, _NullPolicy):
        return abstain(
            "no asset policy was supplied, so nothing can be said about this "
            "instrument; supplying a policy is what makes a proposal possible",
            verdict=PolicyVerdict.no_policy,
        )

    observations = list(evidence.observations) if evidence is not None else []
    if not observations:
        return abstain("no quote supplied for this asset", [EvidenceRejection.absent])

    usable: list[MarketQuote] = []
    rejections: list[EvidenceRejection] = []
    for observation in observations:
        surviving, why, _basis = gate_evidence(observation, active, now=moment)
        # Liveness is checked after the quote is well-formed and fresh, but before
        # it can influence anything. A venue down makes its quote unusable however
        # recent the price is.
        if surviving is not None:
            venue_reason = (
                gate_venue_liveness(
                    surviving.source,
                    liveness,
                    now=moment,
                    max_age_seconds=liveness_max_age_seconds,
                )
                if active.requires_venue_liveness
                else None
            )
            if venue_reason is not None:
                why = [venue_reason]
                surviving = None
        if surviving is not None:
            usable.append(surviving)
        rejections.extend(why)
    dropped = len(observations) - len(usable)
    if dropped and usable:
        notes.append(
            f"{dropped} of {len(observations)} observation(s) were dropped by the "
            "evidence gate; the decision rests only on those that survived"
        )
    if not usable:
        primary = rejections[0] if rejections else EvidenceRejection.absent
        return abstain(
            _PRIMARY_REASON[primary],
            rejections,
            verdict=(
                PolicyVerdict.no_session
                if primary is EvidenceRejection.out_of_session
                else None
            ),
        )

    chosen, reconcile_note = active.reconcile(usable, session=session)
    if chosen is None:
        return abstain(reconcile_note or "observations could not be reconciled")
    if reconcile_note:
        # A caveat about the observations we chose to keep is still information;
        # dropping it because we proceeded anyway loses it entirely.
        notes.append(reconcile_note)

    basis = active.price_basis(chosen, session=session)
    if basis is None:
        # The policy will not name a price field to act on. Proceeding would name
        # no evidence at all, so this abstains through the basis `gate_evidence`
        # returned rather than letting `interpret` decide on a nameless reading.
        return abstain(REASON_NO_BASIS, verdict=PolicyVerdict.no_basis)

    action, confidence, reason = active.interpret(chosen, session=session)
    confidence = _corroborated(confidence, chosen, usable)
    if basis is not None and basis not in reason:
        reason = f"{reason} (on {basis})"
    if len(usable) > 1:
        notes.append(
            f"{len(usable)} observation(s) from {len({o.source or '<unspecified>' for o in usable})} "
            "venue(s) were gated before reconciling"
        )

    return MarketDecisionAdvisory(
        instrument_id=instrument_id,
        asset_class=active.asset_class,
        action=action,
        reason=reason[:512],
        decision_confidence=max(0.0, min(1.0, float(confidence))),
        price_basis=basis,
        evidence_price=chosen.prices().get(basis) if basis else None,
        evidence_observed_at=chosen.observed_at.astimezone(UTC)
        .replace(microsecond=0)
        .isoformat(),
        evidence_source=chosen.source,
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
    liveness: Sequence[VenueLiveness] | None = None,
    liveness_max_age_seconds: int = DEFAULT_LIVENESS_MAX_AGE_SECONDS,
) -> list[MarketDecisionAdvisory]:
    """Decide across instruments, in a stable order.

    Sorted by instrument id so the output does not depend on input order, which
    would make two identical portfolios produce two different documents.

    Liveness is passed through to every instrument: it describes venues, not
    instruments, so the same readings apply across the whole set.
    """
    moment = now or datetime.now(UTC)
    return [
        decide_market_action(
            quote,
            policies.get(quote.instrument_id),
            now=moment,
            liveness=liveness,
            liveness_max_age_seconds=liveness_max_age_seconds,
        )
        for quote in sorted(quotes, key=lambda item: item.instrument_id)
    ]


__all__ = [
    "AssetPolicy",
    "EvidenceRejection",
    "MarketAction",
    "MarketDecisionAdvisory",
    "MarketQuote",
    "MarketSession",
    "PolicyVerdict",
    "TradingAuthority",
    "advisory_digest",
    "InstrumentEvidence",
    "VenueLiveness",
    "DEFAULT_LIVENESS_MAX_AGE_SECONDS",
    "gate_venue_liveness",
    "decide_instrument",
    "decide_market_action",
    "decide_many",
    "gate_evidence",
    "rejection_summary",
]