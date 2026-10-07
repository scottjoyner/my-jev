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
REASON_FUTURE_DATED = "quote dated in the future; clock skew, not a price"
REASON_CROSSED = "quote bid and ask were crossed or empty"
#: Reported as a policy verdict rather than an evidence rejection: a calendar fact
#: about the market and the absence of rules for reasoning about it are different
#: findings and send an operator to different places.
REASON_NO_POLICY = (
    "no asset policy was supplied, so there are no rules to reason with; this is "
    "not a fault in the evidence"
)
REASON_NO_BASIS = (
    "the asset policy declined to name a price field to act on, so the decision "
    "cannot say what it would have acted on"
)
REASON_NO_QUOTE = "no quote supplied for this asset"
#: A series describing a period that has already ended. Not `REASON_STALE`: that is
#: an observation of *now* that has aged, and telling an operator "stale" here
#: points them at a feed that is working perfectly.
REASON_HISTORY_OUT_OF_WINDOW = (
    "the price series reaches further back than this asset's history window, so it "
    "describes a period that has already ended rather than the present"
)
#: History is not exempt from the liveness rule. Being old makes corroboration
#: weaker, not stronger: nothing recent says the feed is still reporting honestly.
REASON_HISTORY_VENUE_UNVERIFIED = (
    "part of the price series came from a venue that cannot be shown reachable, so "
    "the series cannot be corroborated by it"
)
REASON_AMBIGUOUS_PRICE = "quote carried no unambiguous price to act on"
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

    **This enum was dead for its whole life.** It was defined, documented at
    length, and exported -- and nothing ever set it, so the distinction its
    docstring argues for was not actually made anywhere. A consumer reading an
    abstention had a prose reason and an evidence-rejection count, and no way to
    tell "your feed is broken" from "nobody supplied the rules for this
    instrument". Both of the members that are now reachable are emitted by
    :func:`decide_instrument`.

    **The category is deliberately closed.** `no_session` and `policy_abstained`
    were removed rather than left as unused members, because an unreachable enum
    member is precisely the bug this layer has now found five times -- two dead
    action values, two dead history rejection causes by omission, and this. The
    reason they are not needed: a session-traded asset outside its session is not a
    policy abstention, it is a **hold**, and that is what the session policies
    return. If a future asset genuinely cannot be reasoned about outside a session
    and needs to say so, the honest change is to build that path and its test
    together rather than to inherit a label nobody emits.
    """

    #: No policy was supplied, so there are no rules to reason with at all.
    no_policy = "no_policy"
    #: The policy declined to name a price field to act on, so the decision cannot
    #: say what it would have acted on. Honoured since this enum was wired up: until
    #: then a ``None`` basis was computed and ignored, and the decision went ahead
    #: producing a document that named no evidence at all.
    no_basis = "no_basis"


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
    """Why an observation was not usable. One per cause, never conflated.

    **Two members were removed** rather than left in place. `undated` and
    `non_positive` were unreachable: `MarketQuote.observed_at` is a required,
    zone-aware ``datetime`` and every price field is ``gt=0.0``, so a quote that
    was undated or non-positive cannot be constructed at all. The type boundary
    enforces both, which is the stronger place to enforce them -- an unusable
    quote never enters the layer rather than being admitted and then rejected.

    Keeping them would have been the same bug as an unreachable enum member, which
    this layer has now found seven times. The consequence is that a *malformed
    document* is a different thing from a *rejected observation*: the first is an
    error the caller must fix, the second is a finding the advisory reports. The
    CLI draws that line explicitly and names the instrument and field.
    """

    absent = "absent"
    future_dated = "future_dated"
    stale = "stale"
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
    #: A historical observation older than the policy's history window. Kept
    #: distinct from `stale` because the two mean opposite things: a stale quote is
    #: an observation of *now* that has gone out of date, whereas an out-of-window
    #: history entry describes a period that has already ended and cannot describe
    #: the present no matter how good the evidence was then.
    history_out_of_window = "history_out_of_window"
    #: A historical observation from a venue that cannot be shown reachable now.
    #: History is not exempt from the liveness rule just because it is old.
    history_venue_unverified = "history_venue_unverified"


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


#: A move needs at least this much elapsed time to be a move rather than a tick.
DEFAULT_MOVE_SPAN_SECONDS = 300.0

#: A series that strayed this far from its opening price and then came back is a
#: reversal, not a flat series. Deliberately loose -- the point is that a round
#: trip gets *named*, not that a large one is tradeable. A policy that cares about
#: whether a reversal is worth acting on gates on magnitude or excursion ratio.
DEFAULT_REVERSAL_MINIMUM_EXCURSION = 0.02

#: How far back a history series may reach and still describe the present.
#: Deliberately much larger than a freshness window -- a quote goes stale in
#: seconds, a session's worth of prints is still evidence about today -- but still
#: bounded, because a series from last week is not a description of now however
#: well corroborated it was. Without this bound a decade-old series could qualify a
#: proposal today, which is the same class of error as treating a stale quote as
#: current: evidence about the wrong time presented as evidence about now.
DEFAULT_HISTORY_WINDOW_SECONDS = 3 * 86400

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
    #: Earlier observations of the same instrument, in any order. Deliberately a
    #: separate field from ``observations`` so a time series is never mistaken for
    #: a cross-venue sample set.
    history: list[MarketQuote] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent_instrument(self) -> InstrumentEvidence:
        mismatched = sorted(
            {
                observation.instrument_id
                for observation in (*self.observations, *self.history)
                if observation.instrument_id != self.instrument_id
            }
        )
        if mismatched:
            raise ValueError(
                "observations reference other instruments: "
                f"{mismatched}"
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


class SustainedMove(StrEnum):
    """Direction of a move that persisted, established from more than one instant.

    Separate from ``observations``, which are near-simultaneous samples across
    venues. Two venues quoting at one moment are a *reconciliation* problem; a
    series of prints over time is a *state change* problem, and conflating them
    would let a two-hour-old print be compared against a fresh one as though they
    were current.

    The layer sees a snapshot by default, and a snapshot cannot justify a
    proposal -- one print is noise whichever way it moved. This is what a proposal
    requires instead: evidence that something changed and stayed changed.
    """

    up = "up"
    down = "down"
    #: Price moved one way and then moved *back* -- a reversal, not a direction.
    #: Kept distinct from up/down because "up" would be a lie: the series ended
    #: where it began. Collapsing it into `flat` (no signal) or `up` (a rising
    #: trend) both destroy real information, and the second one is how a system
    #: starts reading reversals as strength.
    reversal_up = "reversal_up"
    reversal_down = "reversal_down"


class MoveSignal(BaseModel):
    """A move established across time, not from one print."""

    model_config = ConfigDict(extra="forbid")

    direction: SustainedMove
    #: Fractional move from the first to the last observation in the series.
    magnitude: float = Field(gt=0.0)
    #: Observations the series was computed from, so a caller can see how thin it was.
    sample_count: int = Field(ge=2)
    span_seconds: float = Field(ge=0.0)
    #: Venues that attested the move. One venue is a single feed's own story.
    venue_count: int = Field(ge=1)
    #: The venues themselves, so a caller can see which feeds the claim rests on.
    #:
    #: An ordered tuple, not a set. This field is serialised into the advisory
    #: report, and `set` iteration order depends on `PYTHONHASHSEED`, so a set here
    #: made the report bytes differ between two runs over identical evidence --
    #: which defeats the entire point of the document entry point. The existing
    #: reproducibility test did not catch it because it invoked the CLI twice in one
    #: process, where the hash seed is constant.
    #:
    #: Sorted and duplicate-free by validation, so the value is canonical on the way
    #: in rather than tidied on the way out.
    venues: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _canonical_venues(self) -> MoveSignal:
        names = list(self.venues)
        if names != sorted(set(names)):
            raise ValueError(
                "venues must be sorted and duplicate-free so the field serialises "
                f"identically every run; got {names}"
            )
        return self
    #: Largest move *away* from the opening price relative to the net move.
    #: 1.0 means the series went one way and came all the way back, ending flat;
    #: larger than 1.0 means it overshot before reversing. Only set for a
    #: reversal, because a clean trend has no meaningful excursion to report.
    excursion_ratio: float | None = Field(default=None, ge=1.0)

    @property
    def is_reversal(self) -> bool:
        return self.direction in (SustainedMove.reversal_up, SustainedMove.reversal_down)

    @property
    def net_direction(self) -> SustainedMove:
        """The trend this series traced, ignoring the reversal at its end.

        ``reversal_up`` -- rose then fell back -- is a *rise* that did not hold.
        That is the shape a proposal needs to know about, and no existing field
        carried it: `direction` said `reversal_up`, `magnitude` said the net move,
        and the word "up" appeared nowhere to say which way it had gone first.
        """
        if self.direction is SustainedMove.reversal_up:
            return SustainedMove.up
        if self.direction is SustainedMove.reversal_down:
            return SustainedMove.down
        return self.direction


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

    #: The move this decision leaned on, if any. Without it a proposal cannot be
    #: audited after the fact: the reason string says a move was corroborated but
    #: not *which* move, over what span, from how many prints, or from which feeds.
    #: Two proposals differing only in span and sample count would otherwise produce
    #: identical documents.
    move: MoveSignal | None = None

    #: Why the *policy* declined, when it declined for its own reasons.
    #:
    #: Distinct from :attr:`rejected`, and the distinction is the point of the
    #: field: "the feed was broken" and "nobody supplied the rules for this
    #: instrument" are different findings that send an operator to different places,
    #: and collapsing them means one of them gets misdiagnosed. This was a
    #: long-standing hole -- the enum existed and nothing set it.
    policy_verdict: PolicyVerdict | None = None

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

    def history_window_seconds(self, now: datetime) -> int:
        """How far back a history series may reach and still describe the present.

        Separate from :meth:`freshness_window_seconds` because the two answer
        different questions. Freshness asks "is this observation current?", which
        for a quote is seconds. History asks "is this series about the period I am
        deciding in?", which is hours or a session -- much longer, but still
        bounded.

        The default is deliberately generous, and a policy is expected to tighten
        it. What it must not do is remove the bound: an unbounded history window
        means a series from any point in the instrument's life can qualify a
        proposal now, which is presenting evidence about the wrong time as evidence
        about the present.
        """
        return DEFAULT_HISTORY_WINDOW_SECONDS

    def consider_move(
        self,
        signal: MoveSignal,
        *,
        quote: MarketQuote,
        session: MarketSession,
    ) -> tuple[MarketAction, float, str] | None:
        """Whether an established state change is grounds to propose anything.

        The default refuses. A move is evidence that the world looks different, not
        evidence about what happens next, and turning one into an instruction is
        the step this layer exists to decline. Policies opt in, and an opt-in has
        to be corroborated: :attr:`MoveSignal.venue_count` must exceed one, so a
        single feed's own story cannot propose on its own.

        Returning ``None`` leaves the caller's ``propose_hold`` in place, which is
        the right answer for "something changed and I have nothing to add".
        """
        return None

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


def sustained_move(
    history: Sequence[MarketQuote],
    *,
    minimum_span_seconds: float = DEFAULT_MOVE_SPAN_SECONDS,
    minimum_samples: int = 3,
    reversal_minimum_excursion: float = DEFAULT_REVERSAL_MINIMUM_EXCURSION,
) -> MoveSignal | None:
    """Establish that something changed *and stayed changed* -- or came back.

    Returns ``None`` unless the evidence genuinely supports a state change, which is
    most of the time. The requirements, each of which can independently refuse:

    * **a real span** -- two prints moments apart are one print sampled twice;
    * **enough samples** -- a series of three is thin evidence about a trend;
    * **some change to report** -- a series that ended exactly where it began with
      no material excursion genuinely moved nowhere.

    ## A reversal is a state change, and used to be reported as nothing

    A price that rose 10% and fell back to its opening level has not "moved
    nowhere". It went somewhere and came back, and for an operator that is one of
    the most consequential shapes a series can have -- it is the difference between
    a trend holding and a rally failing. Two wrong answers were available:

    * reporting ``None`` discards the excursion entirely, so the largest thing
      that happened in the window is invisible;
    * reporting the net direction alone would call a failed rally a rising trend,
      which is worse than silence because it is confidently wrong.

    So a reversal is its own direction. ``magnitude`` stays the fractional *net*
    change, and where that is nil the excursion is reported in its place rather than
    reporting a zero that would read as "nothing happened" -- a claim this module
    must never make when it has found a 10% round trip. :attr:`excursion_ratio`
    records how far the excursion ran relative to the net move, and
    :attr:`MoveSignal.net_direction` recovers the trend that was traced.

    The reversal threshold is deliberately loose. This is about *legibility*, not
    about deciding whether a reversal is worth acting on; a policy that cares gates
    on :attr:`excursion_ratio` or on magnitude, not on this constant.

    Deliberately *not* a momentum strategy. This asks whether the world looks
    different now than it did, which is a question about evidence rather than a
    prediction about the future. Nothing here says the move will continue, and the
    advisory it feeds places no orders.
    """
    priced = [
        observation
        for observation in history
        if observation.prices()
    ]
    if len(priced) < max(2, minimum_samples):
        return None

    ordered = sorted(
        priced,
        key=lambda observation: observation.observed_at,
    )
    # Collapse to one point per timestamp so a venue quoting repeatedly at the same
    # instant cannot inflate the sample count into looking like a series.
    by_time: dict[datetime, float] = {}
    for observation in ordered:
        basis = "last" if observation.last is not None else sorted(observation.prices())[0]
        by_time.setdefault(observation.observed_at.astimezone(UTC), float(observation.prices()[basis]))
    series = sorted(by_time.items())
    if len(series) < max(2, minimum_samples):
        return None

    span = (series[-1][0] - series[0][0]).total_seconds()
    if span < minimum_span_seconds:
        return None

    first, last = series[0][1], series[-1][1]
    if first <= 0.0 or last <= 0.0:
        return None

    net_change = (last - first) / first

    peak = max(price for _, price in series)
    trough = min(price for _, price in series)
    # Sorted and duplicate-free at construction, because this ends up serialised in
    # the report and set iteration order is not stable across processes.
    venues = tuple(sorted({o.source for o in priced if o.source is not None}))

    # How far the series strayed *against* its net direction, measured from where
    # it opened. `switched` means it reversed onto the other side of its opening
    # price; `failed` means it travelled one way and handed part of it back from the
    # extreme. Both are reversals and they need different measures -- measuring
    # only the giveback misses a series that rallied before it fell, and measuring
    # only the switch misses one that rose, fell, and recovered most of the rise.
    against_up = (peak - first) / first
    against_down = (first - trough) / first

    if net_change > 0.0:
        trend, reversal = SustainedMove.up, SustainedMove.reversal_up
        switched = against_down >= reversal_minimum_excursion
        failed = (peak - last) / peak >= reversal_minimum_excursion
        giveback = (peak - last) / peak
    elif net_change < 0.0:
        trend, reversal = SustainedMove.down, SustainedMove.reversal_down
        switched = against_up >= reversal_minimum_excursion
        failed = (last - trough) / trough >= reversal_minimum_excursion
        giveback = (last - trough) / trough
    else:
        # Ended exactly where it started, so whichever side strayed further is the
        # side that actually happened.
        excursion = max(against_up, against_down)
        if excursion == 0.0:
            # Flat all the way through. Inventing a move here is the failure to avoid.
            return None
        trend = SustainedMove.up if against_up >= against_down else SustainedMove.down
        reversal = (
            SustainedMove.reversal_up if trend is SustainedMove.up
            else SustainedMove.reversal_down
        )
        switched = excursion >= reversal_minimum_excursion
        failed = False
        giveback = 0.0
        excursion = excursion

    is_reversal = switched or failed
    direction = reversal if is_reversal else trend

    if net_change == 0.0 and not is_reversal:
        # Ended where it started and never strayed far enough to call it a
        # reversal. Genuinely flat. Reporting a zero move here would claim a move of
        # no size in the one module whose job is to know whether one happened.
        return None

    if not is_reversal:
        magnitude = abs(net_change)
        excursion_ratio = None
    elif net_change == 0.0:
        # No net change to report, and reporting 0.0 would assert the series went
        # nowhere. The excursion is the finding; the ratio is undefined because there
        # is no net move to be a fraction of.
        magnitude = excursion
        excursion_ratio = None
    else:
        magnitude = abs(net_change)
        # How many times over the counter-move was the size of the move that held.
        # Above 1.0 whenever a reversal exists, which is what makes it readable as
        # "this did not travel in one direction".
        against = against_down if net_change > 0.0 else against_up
        excursion_ratio = max(against, giveback) / abs(net_change)


    return MoveSignal(
        direction=direction,
        magnitude=magnitude,
        sample_count=len(series),
        span_seconds=span,
        venue_count=len(venues) or 1,
        venues=venues,
        excursion_ratio=excursion_ratio,
    )


def gate_history(
    history: Sequence[MarketQuote],
    policy: AssetPolicy,
    *,
    now: datetime,
    liveness: Sequence[VenueLiveness] | None = None,
    liveness_max_age_seconds: int = DEFAULT_LIVENESS_MAX_AGE_SECONDS,
) -> tuple[list[MarketQuote], list[EvidenceRejection]]:
    """Decide which historical observations may describe the present.

    History cannot go through :func:`gate_evidence`, and the reason is structural
    rather than a shortcut: freshness windows are short by design, so every
    historical entry would be rejected as stale and the feature could not exist.
    History needs its own bound, and it is looser -- but it is still a bound.

    Two failures are worth naming separately, because they are different mistakes:

    * **out of window** -- the entry describes a period that has ended. It may have
      been excellent evidence at the time and still say nothing about now;
    * **venue unverifiable now** -- the feed behind it cannot be shown reachable.
      Being old does not exempt a series from the liveness rule; if anything it
      makes corroboration *weaker*, since there is no recent sign the feed is
      still reporting honestly.

    All rejections are collected rather than short-circuited, for the same reason
    as :func:`gate_evidence`: an operator fixing a feed wants the whole list.
    """
    moment = now.astimezone(UTC)
    window = policy.history_window_seconds(moment)
    usable: list[MarketQuote] = []
    rejections: list[EvidenceRejection] = []

    for observation in history:
        if not observation.prices():
            rejections.append(EvidenceRejection.ambiguous_price)
            continue
        if observation.crossed():
            rejections.append(EvidenceRejection.crossed)
            continue

        observed = observation.observed_at.astimezone(UTC)
        age = (moment - observed).total_seconds()
        if age < 0:
            # A future-dated print is clock skew, not history.
            rejections.append(EvidenceRejection.future_dated)
            continue
        if age > window:
            rejections.append(EvidenceRejection.history_out_of_window)
            continue

        if policy.requires_venue_liveness and gate_venue_liveness(
            observation.source,
            liveness,
            now=moment,
            max_age_seconds=liveness_max_age_seconds,
        ) is not None:
            rejections.append(EvidenceRejection.history_venue_unverified)
            continue

        usable.append(observation)

    return usable, rejections


#: The operator-facing sentence for each rejection cause, used when an evidence set
#: has nothing usable in it at all and one cause has to be named as *the* reason.
#:
#: Defined once and looked up strictly. It used to be two inline copies that each
#: fell back to `REASON_NO_QUOTE` for an unmapped cause -- so adding a rejection
#: cause without remembering to edit two dictionaries produced "no quote supplied
#: for this asset" for something that was not that at all, silently. Both history
#: causes shipped that way. A missing key is now a loud `KeyError` rather than a
#: wrong sentence in production, and a test asserts the map covers every member.
_PRIMARY_REASON: dict[EvidenceRejection, str] = {
    EvidenceRejection.absent: REASON_NO_QUOTE,
    EvidenceRejection.future_dated: REASON_FUTURE_DATED,
    EvidenceRejection.stale: REASON_STALE,
    EvidenceRejection.crossed: REASON_CROSSED,
    EvidenceRejection.ambiguous_price: REASON_AMBIGUOUS_PRICE,
    EvidenceRejection.venue_unverified: REASON_VENUE_UNVERIFIED,
    EvidenceRejection.venue_liveness_stale: REASON_VENUE_LIVENESS_STALE,
    EvidenceRejection.venue_down: REASON_VENUE_DOWN,
    EvidenceRejection.history_out_of_window: REASON_HISTORY_OUT_OF_WINDOW,
    EvidenceRejection.history_venue_unverified: REASON_HISTORY_VENUE_UNVERIFIED,
}


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
            rejected={rejection: rejections.count(rejection) for rejection in set(rejections)},
            evaluated_at=moment.replace(microsecond=0).isoformat(),
            notes=notes,
        )

    if basis is None:
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
        return abstain(REASON_NO_POLICY, verdict=PolicyVerdict.no_policy)

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
        return abstain(_PRIMARY_REASON[rejections[0]], rejections)

    chosen, reconcile_note = active.reconcile(usable, session=session)
    if chosen is None:
        return abstain(reconcile_note or "observations could not be reconciled")

    basis = active.price_basis(chosen, session=session)
    if basis is None:
        # `gate_evidence` promises this is "reported as a policy-level abstention
        # elsewhere rather than folded in here", and until now there was no
        # elsewhere: the basis was computed, found to be None, and ignored. The
        # decision then proceeded and produced a document naming no evidence at
        # all -- which is the failure the whole evidence-provenance design exists
        # to prevent, reached by the back door.
        return abstain(REASON_NO_BASIS, verdict=PolicyVerdict.no_basis)
    if reconcile_note:
        # A caveat about the observations we chose to keep is still information;
        # dropping it because we proceeded anyway loses it entirely.
        notes.append(reconcile_note)

    action, confidence, reason = active.interpret(chosen, session=session)

    # A proposal requires an established state change, not a snapshot. Evaluated
    # after interpretation so an existing abstain or hold is never overridden --
    # the move can only *add* to a neutral reading, never rescue a rejected one.
    # History is gated before it is believed. `sustained_move` will happily compute
    # a clean, well-corroborated move from a series that ended last week; bounding
    # it to a period that can describe the present is this layer's job, not the
    # detector's.
    history_note = ""
    history, history_rejections = (
        gate_history(
            evidence.history,
            active,
            now=moment,
            liveness=liveness,
            liveness_max_age_seconds=liveness_max_age_seconds,
        )
        if evidence is not None
        else ([], [])
    )
    if history_rejections:
        causes = sorted({r.value for r in history_rejections})
        notes.append(
            f"{len(history_rejections)} historical observation(s) were dropped: "
            + ", ".join(causes)
        )
        # A dropped series changes what this decision could have been, so it belongs
        # in the reason and not only in a note. Previously an operator saw a
        # `propose_hold` whose reason talked about thin evidence and never mentioned
        # that the series which might have proposed had been discarded.
        history_note = (
            "part of the price series was not usable ("
            + ", ".join(causes)
            + "), so this rests on less history than was supplied"
        )
    signal = sustained_move(history)
    if signal is not None:
        # Corroboration counted from history alone could rest entirely on venues
        # that are down now -- a proposal whose every supporting feed is dead.
        # Require the move to be attested by at least one venue that is currently
        # usable.
        live_now = {observation.source for observation in usable if observation.source}
        attested = set(signal.venues) & live_now
        if not attested:
            signal = None
        else:
            signal = signal.model_copy(update={"venue_count": len(attested)})
    if signal is not None:
        proposal = active.consider_move(signal, quote=chosen, session=session)
        if proposal is not None:
            proposed_action, proposed_confidence, proposed_reason = proposal
            if proposed_action is not MarketAction.propose_hold:
                action = proposed_action
                confidence = proposed_confidence
                reason = proposed_reason
        # Recorded whether or not it produced a proposal: "there was a real move and
        # it was not enough" is a finding worth being able to see later, and it is
        # invisible if provenance is only attached to proposals.

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
        reason=(
            f"{reason}; {history_note}" if history_note else reason
        )[:512],
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
        move=signal,
        # History rejections are counted here too. They were previously only
        # summarised into a note, which meant an operator auditing a stored
        # decision had to parse English to learn that part of the series was
        # dropped -- and a series that was partly out of window looked identical to
        # one that was complete.
        rejected={
            rejection: history_rejections.count(rejection)
            for rejection in set(history_rejections)
        },
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
    "MoveSignal",
    "SustainedMove",
    "sustained_move",
    "TradingAuthority",
    "advisory_digest",
    "InstrumentEvidence",
    "VenueLiveness",
    "DEFAULT_HISTORY_WINDOW_SECONDS",
    "DEFAULT_LIVENESS_MAX_AGE_SECONDS",
    "DEFAULT_MOVE_SPAN_SECONDS",
    "DEFAULT_REVERSAL_MINIMUM_EXCURSION",
    "gate_venue_liveness",
    "decide_instrument",
    "decide_market_action",
    "decide_many",
    "gate_evidence",
    "gate_history",
    "rejection_summary",
]