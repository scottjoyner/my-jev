from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from my_jev.market_asset_crypto import BitcoinSpotPolicy
from my_jev.market_asset_index_fund import IndexFundPolicy
from my_jev.market_decision import (
    DEFAULT_LIVENESS_MAX_AGE_SECONDS,
    EvidenceRejection,
    InstrumentEvidence,
    MarketAction,
    MarketQuote,
    MarketSession,
    VenueLiveness,
    decide_instrument,
    decide_market_action,
    gate_venue_liveness,
)

NOW = datetime(2026, 10, 6, 15, 0, tzinfo=UTC)
BITCOIN = BitcoinSpotPolicy()


def quote(venue="venue-a", **kwargs):
    base = {
        "instrument_id": "opaque:btc",
        "last": 95_000.0,
        "bid": 94_990.0,
        "ask": 95_010.0,
        "observed_at": NOW - timedelta(seconds=5),
        "source": venue,
    }
    base.update(kwargs)
    return MarketQuote(**base)


def reading(venue="venue-a", reachable=True, age=10):
    return VenueLiveness(
        venue=venue, reachable=reachable, observed_at=NOW - timedelta(seconds=age)
    )


def decide(observations=None, *, liveness=None, policy=BITCOIN, **kwargs):
    bundle = InstrumentEvidence(
        instrument_id="opaque:btc",
        observations=observations if observations is not None else [quote()],
    )
    return decide_instrument(bundle, policy, now=NOW, liveness=liveness, **kwargs)


# --- the gate --------------------------------------------------------------


def test_a_reachable_checked_venue_passes():
    assert gate_venue_liveness("venue-a", [reading()], now=NOW) is None


def test_no_liveness_data_is_unverified_not_assumed_healthy():
    """"We could not check" and "it is fine" are different claims."""
    assert (
        gate_venue_liveness("venue-a", None, now=NOW)
        is EvidenceRejection.venue_unverified
    )
    assert (
        gate_venue_liveness("venue-a", [], now=NOW)
        is EvidenceRejection.venue_unverified
    )


def test_a_venue_not_named_in_the_readings_is_unverified():
    assert (
        gate_venue_liveness("venue-a", [reading("venue-b")], now=NOW)
        is EvidenceRejection.venue_unverified
    )


def test_an_unattributable_quote_is_unverified():
    assert (
        gate_venue_liveness(None, [reading()], now=NOW)
        is EvidenceRejection.venue_unverified
    )


def test_an_observed_down_venue_is_rejected():
    assert (
        gate_venue_liveness("venue-a", [reading(reachable=False)], now=NOW)
        is EvidenceRejection.venue_down
    )


def test_liveness_too_old_is_treated_as_no_reading():
    stale = reading(age=DEFAULT_LIVENESS_MAX_AGE_SECONDS + 1)
    assert (
        gate_venue_liveness("venue-a", [stale], now=NOW)
        is EvidenceRejection.venue_liveness_stale
    )


def test_a_future_dated_reading_is_clock_skew_not_health():
    skewed = VenueLiveness(
        venue="venue-a", reachable=True, observed_at=NOW + timedelta(hours=1)
    )
    assert (
        gate_venue_liveness("venue-a", [skewed], now=NOW)
        is EvidenceRejection.venue_unverified
    )


def test_the_newest_reading_wins_over_an_older_one():
    """A fresh "down" must not be masked by an hour-old "up"."""
    readings = [reading(reachable=True, age=3600), reading(reachable=False, age=5)]
    assert (
        gate_venue_liveness("venue-a", readings, now=NOW)
        is EvidenceRejection.venue_down
    )


def test_a_recovery_is_seen_too():
    readings = [reading(reachable=False, age=3600), reading(reachable=True, age=5)]
    assert gate_venue_liveness("venue-a", readings, now=NOW) is None


def test_the_liveness_window_is_configurable():
    readings = [reading(age=100)]
    assert (
        gate_venue_liveness("venue-a", readings, now=NOW, max_age_seconds=30)
        is EvidenceRejection.venue_liveness_stale
    )
    assert (
        gate_venue_liveness("venue-a", readings, now=NOW, max_age_seconds=300) is None
    )


def test_liveness_must_carry_a_timezone():
    with pytest.raises(ValueError, match="timezone-aware"):
        VenueLiveness(venue="venue-a", reachable=True, observed_at=datetime(2026, 1, 1))


# --- the effect on the decision --------------------------------------------


def test_a_fresh_well_formed_price_from_a_dead_venue_is_refused():
    """The failure this gate exists for: a plausible number with no counterparty."""
    advisory = decide(liveness=[reading(reachable=False)])
    assert advisory.action is MarketAction.abstain
    assert advisory.rejected.get(EvidenceRejection.venue_down) == 1
    assert "no counterparty" in advisory.reason


def test_no_liveness_evidence_abstains_by_default():
    advisory = decide()
    assert advisory.action is MarketAction.abstain
    assert "not an assumed-healthy one" in advisory.reason


def test_a_checked_live_venue_proceeds_as_before():
    advisory = decide(liveness=[reading()])
    assert advisory.action is not MarketAction.abstain
    assert advisory.evidence_price == 95_000.0


def test_a_consolidated_tape_does_not_need_venue_liveness():
    """A tape subscription has no uptime of its own; an exchange behind it does.

    That is a property of the asset class, so it lives on the policy rather than on
    a caller flag a caller could set carelessly.
    """
    index_quote = MarketQuote(
        instrument_id="opaque:index",
        last=500.0,
        nav=499.0,
        observed_at=NOW - timedelta(seconds=5),
        session=MarketSession.regular,
        source="consolidated-tape",
    )
    assert IndexFundPolicy().requires_venue_liveness is False
    assert BitcoinSpotPolicy().requires_venue_liveness is True

    without = decide_instrument(
        InstrumentEvidence(instrument_id="opaque:index", observations=[index_quote]),
        IndexFundPolicy(),
        now=NOW,
    )
    assert without.action is MarketAction.propose_hold


def test_a_policy_may_require_liveness_and_be_declared_on_the_advisory():
    """The opt-out is per policy, so a stored decision names its own ruleset."""
    assert decide(liveness=[reading()]).asset_class == "crypto_spot_bitcoin"


def test_a_quote_with_no_source_is_always_unverified():
    advisory = decide(observations=[quote(venue=None)], liveness=[reading()])
    assert advisory.action is MarketAction.abstain


def test_a_dead_venue_does_not_mask_a_live_one():
    """One feed down should not silence a second that is checked and up."""
    advisory = decide(
        observations=[quote("venue-a"), quote(venue="venue-b", last=95_010.0)],
        liveness=[reading("venue-a", reachable=False), reading("venue-b", reachable=True)],
    )
    assert advisory.action is not MarketAction.abstain
    assert any("dropped by the evidence gate" in note for note in advisory.notes)


def test_liveness_never_grants_authority():
    advisory = decide(liveness=[reading()])
    assert set(advisory.authority.model_dump().values()) == {False}
    assert advisory.price_forecast is False


# --- the single-quote path ------------------------------------------------


def test_the_single_quote_path_gates_liveness_too():
    advisory = decide_market_action(
        quote(), BITCOIN, now=NOW, liveness=[reading(reachable=False)]
    )
    assert advisory.action is MarketAction.abstain
    assert "no counterparty" in advisory.reason
