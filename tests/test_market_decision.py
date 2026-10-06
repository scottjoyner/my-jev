from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from my_jev.market_decision import (
    AssetPolicy,
    EvidenceRejection,
    InstrumentEvidence,
    MarketAction,
    MarketDecisionAdvisory,
    MarketQuote,
    MarketSession,
    PolicyVerdict,
    TradingAuthority,
    advisory_digest,
    decide_instrument,
    decide_market_action,
    decide_many,
    gate_evidence,
    rejection_summary,
)

NOW = datetime(2026, 10, 6, 15, 0, tzinfo=UTC)


class StubPolicy(AssetPolicy):
    """A minimal policy so the decision layer is tested without an asset class.

    Opts out of venue liveness: these tests are about the decision layer, and
    liveness has its own suite. The quotes here carry no source at all, so
    requiring it would mask every other behaviour behind an unverified venue.
    """

    asset_class = "stub"
    requires_venue_liveness = False

    def freshness_window_seconds(self, now: datetime) -> int:  # noqa: ARG002
        return 300

    def session_at(self, now: datetime) -> MarketSession:  # noqa: ARG002
        return MarketSession.regular

    def price_basis(self, quote: MarketQuote, *, session: MarketSession) -> str | None:
        return "last" if quote.last is not None else None

    def interpret(self, quote, *, session):
        return MarketAction.propose_hold, 0.5, "stub holds"


def quote(**kwargs) -> MarketQuote:
    base = {
        "instrument_id": "opaque:instrument",
        "last": 100.0,
        "observed_at": NOW - timedelta(seconds=30),
    }
    base.update(kwargs)
    return MarketQuote(**base)


# --- quotes ----------------------------------------------------------------


def test_a_quote_must_carry_a_timezone():
    """An unageable quote cannot be checked, and assuming UTC widens the window."""
    with pytest.raises(ValueError, match="timezone-aware"):
        MarketQuote(instrument_id="opaque:x", last=1.0, observed_at=datetime(2026, 1, 1))


def test_a_quote_carries_no_ticker_or_position_field():
    with pytest.raises(ValueError):
        MarketQuote(
            instrument_id="opaque:x",
            last=1.0,
            observed_at=NOW,
            ticker="VOO",
        )


def test_a_negative_price_is_refused_at_validation():
    with pytest.raises(ValueError):
        MarketQuote(instrument_id="opaque:x", last=-1.0, observed_at=NOW)


def test_crossed_quotes_are_detected():
    assert quote(bid=101.0, ask=100.0, last=None).crossed() is True
    assert quote(bid=99.0, ask=100.0, last=None).crossed() is False


def test_a_one_sided_book_is_not_called_crossed():
    """Thin is not broken, and the two deserve different treatment."""
    assert quote(bid=99.0, ask=None, last=None).crossed() is False


def test_a_quote_with_no_price_at_all_is_crossed():
    assert quote(bid=None, ask=None, last=None).crossed() is True


# --- the evidence gate -----------------------------------------------------


def test_a_fresh_quote_passes():
    usable, rejections, basis = gate_evidence(quote(), StubPolicy(), now=NOW)
    assert usable is not None
    assert rejections == []
    assert basis == "last"


def test_a_stale_quote_is_named_as_stale():
    usable, rejections, _ = gate_evidence(
        quote(observed_at=NOW - timedelta(hours=2)), StubPolicy(), now=NOW
    )
    assert usable is None
    assert EvidenceRejection.stale in rejections


def test_a_future_dated_quote_is_clock_skew_not_staleness():
    usable, rejections, _ = gate_evidence(
        quote(observed_at=NOW + timedelta(hours=2)), StubPolicy(), now=NOW
    )
    assert usable is None
    assert EvidenceRejection.future_dated in rejections
    assert EvidenceRejection.stale not in rejections


def test_an_absent_quote_is_its_own_cause():
    usable, rejections, _ = gate_evidence(None, StubPolicy(), now=NOW)
    assert usable is None
    assert rejections == [EvidenceRejection.absent]


def test_every_rejection_is_collected_not_just_the_first():
    """An operator fixing a feed wants the whole list."""
    _, rejections, _ = gate_evidence(
        quote(
            bid=101.0,
            ask=100.0,
            last=None,
            observed_at=NOW - timedelta(hours=2),
        ),
        StubPolicy(),
        now=NOW,
    )
    assert EvidenceRejection.crossed in rejections
    assert EvidenceRejection.stale in rejections


def test_a_quote_with_no_price_is_ambiguous_not_merely_unbased():
    _, rejections, _ = gate_evidence(
        quote(last=None, bid=None, ask=None), StubPolicy(), now=NOW
    )
    assert EvidenceRejection.ambiguous_price in rejections


def test_the_rejection_summary_names_each_cause_and_count():
    summary = rejection_summary(
        [EvidenceRejection.stale, EvidenceRejection.stale, EvidenceRejection.crossed]
    )
    assert "crossed=1" in summary
    assert "stale=2" in summary


def test_an_empty_rejection_summary_is_empty():
    assert rejection_summary([]) == ""


# --- abstention ------------------------------------------------------------


def test_without_a_policy_the_decision_is_abstain_for_that_reason():
    """Not "stale" -- that would send an operator to fix a feed that was fine."""
    advisory = decide_market_action(quote(), None, now=NOW)
    assert advisory.action is MarketAction.abstain
    assert "no asset policy" in advisory.reason
    assert advisory.rejected == {}


def test_stale_evidence_abstains_and_says_so():
    advisory = decide_market_action(
        quote(observed_at=NOW - timedelta(hours=9)), StubPolicy(), now=NOW
    )
    assert advisory.action is MarketAction.abstain
    assert advisory.rejected[EvidenceRejection.stale] == 1
    assert "freshness window" in advisory.reason


def test_absent_evidence_abstains():
    advisory = decide_market_action(None, StubPolicy(), now=NOW)
    assert advisory.action is MarketAction.abstain
    assert advisory.rejected[EvidenceRejection.absent] == 1


def test_future_dated_evidence_abstains_as_clock_skew():
    advisory = decide_market_action(
        quote(observed_at=NOW + timedelta(hours=1)), StubPolicy(), now=NOW
    )
    assert advisory.action is MarketAction.abstain
    assert "clock skew" in advisory.reason


def test_hold_and_abstain_are_different_findings():
    held = decide_market_action(quote(), StubPolicy(), now=NOW)
    abstained = decide_market_action(None, StubPolicy(), now=NOW)
    assert held.action is MarketAction.propose_hold
    assert abstained.action is MarketAction.abstain
    assert held.action is not abstained.action


def test_actionable_excludes_hold_and_abstain():
    assert decide_market_action(quote(), StubPolicy(), now=NOW).actionable is False
    assert decide_market_action(None, StubPolicy(), now=NOW).actionable is False


# --- authority -------------------------------------------------------------


def test_a_decision_places_no_order():
    assert set(
        decide_market_action(quote(), StubPolicy(), now=NOW).authority.model_dump().values()
    ) == {False}


@pytest.mark.parametrize("field", list(TradingAuthority.model_fields))
def test_no_authority_field_can_be_set_true(field):
    payload = TradingAuthority().model_dump()
    payload[field] = True
    with pytest.raises(ValueError):
        TradingAuthority.model_validate(payload)


def test_a_decision_is_advisory_only_and_not_a_forecast():
    advisory = decide_market_action(quote(), StubPolicy(), now=NOW)
    assert advisory.advisory_only is True
    assert advisory.price_forecast is False


def test_unknown_fields_are_refused():
    with pytest.raises(ValueError):
        MarketDecisionAdvisory.model_validate(
            decide_market_action(quote(), StubPolicy(), now=NOW)
            .model_dump()
            | {"target_weight": 0.5}
        )


# --- determinism -----------------------------------------------------------


def test_the_same_inputs_decide_identically():
    first = decide_market_action(quote(), StubPolicy(), now=NOW)
    second = decide_market_action(quote(), StubPolicy(), now=NOW)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_two_decisions_from_different_evidence_are_distinguishable():
    """Without recorded evidence these documents would be identical."""
    first = decide_market_action(quote(), StubPolicy(), now=NOW)
    other = decide_market_action(
        quote(observed_at=NOW - timedelta(seconds=90)), StubPolicy(), now=NOW
    )
    assert first.model_dump(mode="json") != other.model_dump(mode="json")
    assert advisory_digest(first) != advisory_digest(other)


def test_a_decision_names_the_evidence_it_used():
    advisory = decide_market_action(
        quote(last=123.5, source="feed-a"), StubPolicy(), now=NOW
    )
    assert advisory.price_basis == "last"
    assert advisory.evidence_price == 123.5
    assert advisory.evidence_observed_at is not None
    assert advisory.evidence_source == "feed-a"


def test_the_digest_is_stable_across_repeated_decisions():
    assert advisory_digest(
        decide_market_action(quote(), StubPolicy(), now=NOW)
    ) == advisory_digest(decide_market_action(quote(), StubPolicy(), now=NOW))


def test_confidence_is_clamped():
    class Greedy(StubPolicy):
        def interpret(self, quote, *, session):
            return MarketAction.propose_increase, 4.2, "too sure of itself"

    advisory = decide_market_action(quote(), Greedy(), now=NOW)
    assert advisory.decision_confidence == 1.0


def test_a_confidence_below_zero_is_clamped():
    class Negative(StubPolicy):
        def interpret(self, quote, *, session):
            return MarketAction.propose_increase, -1.0, "impossible"

    assert decide_market_action(quote(), Negative(), now=NOW).decision_confidence == 0.0


# --- many instruments ------------------------------------------------------


def test_output_order_does_not_depend_on_input_order():
    """Two identical portfolios must not produce two different documents."""
    quotes = [
        quote(instrument_id="opaque:b"),
        quote(instrument_id="opaque:a"),
    ]
    forward = decide_many(quotes, {"opaque:a": StubPolicy(), "opaque:b": StubPolicy()}, now=NOW)
    backward = decide_many(list(reversed(quotes)), {"opaque:a": StubPolicy(), "opaque:b": StubPolicy()}, now=NOW)
    assert [a.instrument_id for a in forward] == ["opaque:a", "opaque:b"]
    assert [a.model_dump(mode="json") for a in forward] == [
        a.model_dump(mode="json") for a in backward
    ]


def test_an_instrument_with_no_policy_is_still_answered():
    """It abstains for the missing-policy reason rather than crashing."""
    results = decide_many([quote(instrument_id="opaque:a")], {}, now=NOW)
    assert results[0].action is MarketAction.abstain
    assert "no asset policy" in results[0].reason


# --- policy verdicts, and the promise gate_evidence made -------------------


class NoSessionPolicy(StubPolicy):
    """A policy with no open session and no field it is willing to act on."""

    def session_at(self, now: datetime) -> MarketSession:  # noqa: ARG002
        return MarketSession.closed

    def price_basis(self, quote: MarketQuote, *, session: MarketSession) -> str | None:
        return None


class NamelessPolicy(StubPolicy):
    """A policy that is open for business but will not name a price field."""

    def price_basis(self, quote: MarketQuote, *, session: MarketSession) -> str | None:
        return None


def test_a_missing_session_is_named_and_abstains():
    """A closed market is a calendar fact, not a missing quote.

    Previously a policy that could not reason outside its session fell into the
    absent/REASON_NO_QUOTE path, which tells an operator to go and fix a feed
    that is working perfectly.
    """
    advisory = decide_market_action(quote(), NoSessionPolicy(), now=NOW)
    assert advisory.action is MarketAction.abstain
    assert advisory.policy_verdict is PolicyVerdict.no_session
    assert advisory.rejected[EvidenceRejection.out_of_session] == 1
    assert "no trading session" in advisory.reason
    assert "no quote" not in advisory.reason


def test_a_missing_session_is_named_whether_gated_one_or_reconciled():
    """The `decide_instrument` path must reach the same cause and verdict."""
    advisory = decide_instrument(
        InstrumentEvidence(instrument_id="opaque:instrument", observations=[quote()]),
        NoSessionPolicy(),
        now=NOW,
    )
    assert advisory.action is MarketAction.abstain
    assert advisory.policy_verdict is PolicyVerdict.no_session
    assert advisory.rejected[EvidenceRejection.out_of_session] == 1
    assert "no quote" not in advisory.reason


def test_gate_evidence_returns_a_declined_basis_rather_than_folding_it_in():
    """The basis is surfaced, not silently dropped, so the caller can abstain."""
    usable, rejections, basis = gate_evidence(quote(), NamelessPolicy(), now=NOW)
    assert usable is not None
    assert rejections == []
    assert basis is None


def test_a_policy_that_declines_a_basis_abstains_through_the_gate():
    """Fulfils the promise gate_evidence made in its own comment.

    The comment said a `None` basis is "reported as a policy-level abstention
    elsewhere". There was no elsewhere: the basis was computed, found to be None,
    and ignored, and the decision went on to produce a document naming no evidence
    at all. This is that elsewhere.
    """
    advisory = decide_instrument(
        InstrumentEvidence(instrument_id="opaque:instrument", observations=[quote()]),
        NamelessPolicy(),
        now=NOW,
    )
    assert advisory.action is MarketAction.abstain
    assert advisory.policy_verdict is PolicyVerdict.no_basis
    assert advisory.price_basis is None
    assert advisory.evidence_price is None
    assert "declined to name a price field" in advisory.reason


def test_the_market_action_path_also_honours_a_declined_basis():
    advisory = decide_market_action(quote(), NamelessPolicy(), now=NOW)
    assert advisory.action is MarketAction.abstain
    assert advisory.policy_verdict is PolicyVerdict.no_basis
    assert "declined to name a price field" in advisory.reason


def test_no_policy_is_a_verdict_not_a_broken_feed():
    advisory = decide_market_action(quote(), None, now=NOW)
    assert advisory.policy_verdict is PolicyVerdict.no_policy
    assert advisory.rejected == {}


def test_every_rejection_cause_has_an_operator_facing_sentence():
    """The cause map used to be two inline copies with a silent fallback.

    Adding a cause without editing both meant it was reported as "no quote",
    which is a different fault. Asserting completeness here is what makes the new
    out-of-session cause covered rather than merely present.
    """
    from my_jev.market_decision import _PRIMARY_REASON

    assert set(_PRIMARY_REASON) == set(EvidenceRejection)


def test_an_unmapped_cause_fails_loudly_at_the_call_site(monkeypatch):
    """Removing a mapping must break the decision, not quietly reword it."""
    from my_jev.market_decision import _PRIMARY_REASON

    monkeypatch.delitem(_PRIMARY_REASON, EvidenceRejection.out_of_session)
    with pytest.raises(KeyError):
        decide_market_action(quote(), NoSessionPolicy(), now=NOW)

