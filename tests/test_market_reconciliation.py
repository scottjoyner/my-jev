from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from my_jev.market_asset_crypto import BitcoinSpotPolicy
from my_jev.market_asset_index_fund import IndexFundPolicy
from my_jev.market_decision import (
    EvidenceRejection,
    VenueLiveness,
    InstrumentEvidence,
    MarketAction,
    MarketQuote,
    MarketSession,
    decide_instrument,
)

NOW = datetime(2026, 10, 6, 15, 0, tzinfo=UTC)
INDEX = IndexFundPolicy()
BITCOIN = BitcoinSpotPolicy()


def index_quote(venue, last, **kwargs):
    base = {
        "instrument_id": "opaque:index",
        "last": last,
        "nav": 499.0,
        "bid": last - 0.1,
        "ask": last + 0.1,
        "observed_at": NOW - timedelta(seconds=5),
        "session": MarketSession.regular,
        "source": venue,
    }
    base.update(kwargs)
    return MarketQuote(**base)


def btc_quote(venue, last, **kwargs):
    base = {
        "instrument_id": "opaque:btcoin",
        "last": last,
        "bid": last - 5,
        "ask": last + 5,
        "observed_at": NOW - timedelta(seconds=5),
        "source": venue,
    }
    base.update(kwargs)
    return MarketQuote(**base)


def evidence(observations):
    return InstrumentEvidence(
        instrument_id=observations[0].instrument_id, observations=observations
    )


def decide(observations, policy=None, **kwargs):
    """``decide_instrument`` with liveness supplied for every venue present.

    These tests are about reconciliation. Liveness is a separate mechanism with
    its own suite, so deriving it here keeps a change to one from silently
    breaking the other.
    """
    liveness = [
        VenueLiveness(
            venue=venue,
            reachable=True,
            observed_at=NOW - timedelta(seconds=5),
        )
        for venue in {o.source for o in observations if o.source is not None}
    ]
    return decide_instrument(
        evidence(observations),
        policy if policy is not None else BITCOIN,
        now=NOW,
        liveness=liveness,
        **kwargs,
    )


# --- the evidence model ----------------------------------------------------


def test_observations_must_belong_to_one_instrument():
    with pytest.raises(ValueError, match="other instruments"):
        InstrumentEvidence(
            instrument_id="opaque:a",
            observations=[index_quote("x", 500.0), btc_quote("y", 95_000.0)],
        )


def test_venues_are_counted_distinctly():
    bundle = evidence([index_quote("a", 500.0), index_quote("b", 500.1)])
    assert bundle.venues() == ["a", "b"]
    assert bundle.independent_venue_count() == 2


def test_unnamed_feeds_group_as_one_venue():
    """Two anonymous reports are one voice heard twice."""
    bundle = evidence([index_quote(None, 500.0), index_quote(None, 500.1)])
    assert bundle.venues() == ["<unspecified>"]
    assert bundle.independent_venue_count() == 1


# --- gating happens before reconciling -------------------------------------


def test_a_stale_venue_is_dropped_before_reconciliation():
    """Averaging a stale quote into a fresh one would hide a disagreement."""
    advisory = decide(
        [
            btc_quote("a", 95_000.0, observed_at=NOW - timedelta(hours=3)),
            btc_quote("b", 95_010.0),
        ]
    )
    assert advisory.action is not MarketAction.abstain
    assert any("dropped by the evidence gate" in note for note in advisory.notes)


def test_every_venue_stale_abstains_with_the_cause():
    advisory = decide(
        [
            btc_quote("a", 95_000.0, observed_at=NOW - timedelta(hours=3)),
            btc_quote("b", 95_010.0, observed_at=NOW - timedelta(hours=3)),
        ]
    )
    assert advisory.action is MarketAction.abstain
    assert advisory.rejected.get(EvidenceRejection.stale) == 2


# --- corroboration raises confidence ---------------------------------------


def test_agreeing_venues_raise_confidence():
    single = decide([btc_quote("a", 95_000.0)])
    three = decide(
        [btc_quote("a", 95_000.0), btc_quote("b", 95_010.0), btc_quote("c", 94_990.0)]
    )
    assert three.decision_confidence > single.decision_confidence


def test_repeated_reports_from_one_feed_are_not_corroboration():
    single = decide([btc_quote("a", 95_000.0)])
    repeated = decide([btc_quote("a", 95_000.0), btc_quote("a", 95_005.0)])
    assert repeated.decision_confidence == pytest.approx(single.decision_confidence)


def test_two_anonymous_feeds_are_refused_before_corroboration_is_reached():
    """Reframed: liveness makes this case unreachable rather than merely low-confidence.

    An unattributable feed names no venue, so there is nothing to check and the
    quote is refused as `venue_unverified`. The corroboration rule still excludes
    anonymous feeds -- the mixed case below proves it -- but here the decision
    never gets far enough for it to matter.
    """
    advisory = decide([btc_quote(None, 95_000.0), btc_quote(None, 95_005.0)])
    assert advisory.action is MarketAction.abstain
    assert advisory.rejected.get(EvidenceRejection.venue_unverified) == 2


def test_corroboration_never_pushes_confidence_above_one():
    many = decide([btc_quote(str(index), 95_000.0) for index in range(12)])
    assert many.decision_confidence <= 1.0


def test_corroboration_never_makes_a_policy_actionable():
    """It can strengthen a reading; only `interpret` decides whether to act."""
    many = decide([btc_quote(str(index), 95_000.0) for index in range(6)])
    assert many.actionable is False


def test_the_reconcile_caveat_is_kept_even_when_proceeding():
    advisory = decide([btc_quote("a", 95_000.0), btc_quote("a", 95_005.0)])
    assert advisory.action is not MarketAction.abstain
    assert any("not independent agreement" in note for note in advisory.notes)


# --- bitcoin: no tape, so disagreement is the finding ----------------------


def test_venues_disagreeing_beyond_tolerance_abstains():
    advisory = decide([btc_quote("a", 95_000.0), btc_quote("b", 88_000.0)])
    assert advisory.action is MarketAction.abstain
    assert "disagree" in advisory.reason


def test_a_disagreement_names_no_winner():
    """Preferring the higher venue would be picking the feed that suits the answer."""
    advisory = decide([btc_quote("a", 95_000.0), btc_quote("b", 88_000.0)])
    assert "no venue is preferred" in advisory.reason
    assert advisory.evidence_price is None


def test_venues_agreeing_are_accepted_and_a_real_price_is_reported():
    """A venue's own price, not a synthesised midpoint nobody quoted."""
    advisory = decide([btc_quote("a", 95_000.0), btc_quote("b", 95_010.0)])
    assert advisory.action is not MarketAction.abstain
    # 95_005.0 is the midpoint of the two; reporting it would publish a price no
    # venue quoted. The nearest actual quote is reported instead.
    assert advisory.evidence_price in (95_000.0, 95_010.0)
    assert advisory.evidence_source in ("a", "b")


def test_the_chosen_venue_is_deterministic():
    """Input order must not decide which venue wins."""
    forward = decide([btc_quote("a", 95_000.0), btc_quote("b", 95_010.0)])
    backward = decide([btc_quote("b", 95_010.0), btc_quote("a", 95_000.0)])
    assert forward.evidence_price == backward.evidence_price
    assert forward.evidence_source == backward.evidence_source


def test_the_venue_tolerance_is_a_constructor_field():
    quotes = [btc_quote("a", 95_000.0), btc_quote("b", 94_800.0)]
    tight = decide(
        quotes, BitcoinSpotPolicy(venue_disagreement_threshold=0.0001)
    )
    assert tight.action is MarketAction.abstain
    relaxed = decide(quotes, BitcoinSpotPolicy(venue_disagreement_threshold=0.5))
    assert relaxed.action is not MarketAction.abstain


# --- index fund: one tape, so disagreement is a fault ----------------------


def test_feeds_disagreeing_on_one_tape_abstain():
    advisory = decide_instrument(
        evidence([index_quote("a", 500.00), index_quote("b", 520.00)]), INDEX, now=NOW
    )
    assert advisory.action is MarketAction.abstain
    assert "broken feed" in advisory.reason


def test_feeds_agreeing_are_accepted():
    advisory = decide_instrument(
        evidence([index_quote("a", 500.00), index_quote("b", 500.01)]), INDEX, now=NOW
    )
    assert advisory.action is MarketAction.propose_hold
    assert any("midpoint" in note for note in advisory.notes)
    assert any("no feed quoted it" in note for note in advisory.notes)


def test_the_two_asset_classes_reconcile_in_opposite_directions():
    """The same two observations mean different things on different tapes."""

    def with_tolerance(threshold_gap):
        return [index_quote("a", 500.00), index_quote("b", 500.0 + threshold_gap)]

    # A gap fine for a bitcoin venue spread is a broken feed on one tape.
    gap = 4.0
    crypto_like = decide(
        [btc_quote("a", 95_000.0), btc_quote("b", 95_000.0 + gap * 20)],
        BitcoinSpotPolicy(venue_disagreement_threshold=0.5),
    )
    index_like = decide_instrument(
        evidence(with_tolerance(gap)), INDEX, now=NOW
    )
    assert crypto_like.action is not MarketAction.abstain
    assert index_like.action is MarketAction.abstain


def test_a_single_observation_needs_no_reconciliation():
    assert (
        decide_instrument(evidence([index_quote("a", 500.0)]), INDEX, now=NOW).action
        is not MarketAction.abstain
    )
    assert decide([btc_quote("a", 95_000.0)]).action is not MarketAction.abstain


def test_no_observations_abstains_rather_than_crashing():
    advisory = decide_instrument(
        InstrumentEvidence(instrument_id="opaque:index"), INDEX, now=NOW
    )
    assert advisory.action is MarketAction.abstain
    assert advisory.rejected.get(EvidenceRejection.absent) == 1


def test_reconciling_never_grants_authority():
    advisory = decide([btc_quote("a", 95_000.0), btc_quote("b", 95_010.0)])
    assert set(advisory.authority.model_dump().values()) == {False}
    assert advisory.price_forecast is False


def test_an_anonymous_feed_cannot_corroborate_a_named_one():
    """The case that distinguishes "one venue" from "two sources".

    Two anonymous feeds collapse to one voice either way, so they cannot tell the
    two implementations apart. One named plus one anonymous is the discriminating
    case: an unattributable feed cannot independently confirm anything, so it must
    not add a second vote.
    """
    single = decide([btc_quote("a", 95_000.0)])
    mixed = decide([btc_quote("a", 95_000.0), btc_quote(None, 95_005.0)])
    assert mixed.decision_confidence == pytest.approx(single.decision_confidence)
