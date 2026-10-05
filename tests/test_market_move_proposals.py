from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from my_jev.market_asset_crypto import BitcoinSpotPolicy
from my_jev.market_asset_index_fund import IndexFundPolicy
from my_jev.market_decision import (
    MarketAction,
    MarketQuote,
    MarketSession,
    SustainedMove,
    VenueLiveness,
    sustained_move,
)

NOW = datetime(2026, 10, 6, 15, 0, tzinfo=UTC)
BTC = BitcoinSpotPolicy()
INDEX = IndexFundPolicy()


def live(policy_prices, *, age_seconds=30, sessions=MarketSession.regular):
    return [
        MarketQuote(
            instrument_id="opaque:x",
            last=price,
            bid=price * 0.999,
            ask=price * 1.001,
            observed_at=NOW - timedelta(seconds=age_seconds),
            source=venue,
            session=sessions,
        )
        for venue, price in policy_prices.items()
    ]


def live_check(name="a", ok=True):
    return VenueLiveness(
        venue=name, reachable=ok, observed_at=NOW - timedelta(seconds=5)
    )


def series(base, *, up=True, days=0.10, samples=3, venues=("a", "b")):
    """A history that genuinely moved, ending at ``base`` so it lines up with live().

    The newest sample sits at NOW and quotes ``base``. The oldest is
    ``samples - 1`` intervals earlier and is ``days`` away from ``base``: *below* it
    for a rise, *above* it for a fall. Stated in those terms rather than in
    index arithmetic, which is how the previous two versions got the sign backwards.
    """
    span = max(1, samples - 1)
    out = []
    for index in range(samples):
        intervals_back = span - index          # 0 at the newest sample
        # Divided rather than subtracted, so the move from oldest to newest is
        # exactly `days` rather than `days` measured from a shifted base.
        factor = (
            1.0 / (1.0 + days * (intervals_back / span))
            if up
            else 1.0 + days * (intervals_back / span)
        )
        out.append(
            MarketQuote(
                instrument_id="opaque:x",
                last=base * factor,
                observed_at=NOW - timedelta(minutes=intervals_back * 30),
                source=venues[index % len(venues)],
            )
        )
    return out


# --- the detector ----------------------------------------------------------


def test_a_sustained_rise_and_fall_are_distinguished():
    assert sustained_move(series(95_000, up=True)).direction is SustainedMove.up
    assert sustained_move(series(95_000, up=False)).direction is SustainedMove.down


def test_two_prints_too_close_in_time_are_one_print_sampled_twice():
    assert (
        sustained_move(
            [
                MarketQuote(
                    instrument_id="opaque:x",
                    last=95_000.0,
                    observed_at=NOW - timedelta(seconds=30),
                    source="a",
                ),
                MarketQuote(
                    instrument_id="opaque:x",
                    last=95_100.0,
                    observed_at=NOW,
                    source="a",
                ),
            ]
        )
        is None
    )


def test_a_move_that_returns_to_flat_is_not_a_move():
    """Inventing a change that did not happen is the failure this guards."""
    assert (
        sustained_move(
            [
                MarketQuote(
                    instrument_id="opaque:x", last=95_000.0,
                    observed_at=NOW - timedelta(hours=2), source="a",
                ),
                MarketQuote(
                    instrument_id="opaque:x", last=94_000.0,
                    observed_at=NOW - timedelta(hours=1), source="b",
                ),
                MarketQuote(
                    instrument_id="opaque:x", last=95_000.0,
                    observed_at=NOW, source="a",
                ),
            ]
        )
        is None
    )


def test_repeated_prints_at_one_instant_do_not_inflate_the_sample_count():
    assert (
        sustained_move(
            [
                MarketQuote(
                    instrument_id="opaque:x", last=95_000.0 + index,
                    observed_at=NOW, source="a",
                )
                for index in range(5)
            ]
        )
        is None
    )


def test_too_few_samples_is_refused():
    assert sustained_move(series(95_000, samples=2)) is None


def test_a_signal_records_what_it_was_built_from():
    signal = sustained_move(series(95_000, days=0.10))
    assert signal.sample_count == 3
    assert signal.venue_count == 2
    assert signal.span_seconds == 3600.0
    assert signal.magnitude == pytest.approx(0.10, abs=1e-6)


# --- proposals -------------------------------------------------------------


def test_a_sustained_corroborated_move_proposes_on_bitcoin():
    from my_jev.market_decision import InstrumentEvidence, decide_instrument

    advisory = decide_instrument(
        InstrumentEvidence(
            instrument_id="opaque:x",
            observations=live({"a": 95_500.0, "b": 95_510.0}),
            history=series(95_000, days=0.10),
        ),
        BTC,
        now=NOW,
        liveness=[live_check("a"), live_check("b")],
    )
    assert advisory.action is MarketAction.propose_increase
    assert "corroborated by 2 venues" in advisory.reason
    assert set(advisory.authority.model_dump().values()) == {False}


def test_the_opposite_direction_proposes_reduce():
    from my_jev.market_decision import InstrumentEvidence, decide_instrument

    advisory = decide_instrument(
        InstrumentEvidence(
            instrument_id="opaque:x",
            observations=live({"a": 85_000.0, "b": 85_010.0}),
            history=series(95_000, up=False, days=0.10),
        ),
        BTC,
        now=NOW,
        liveness=[live_check("a"), live_check("b")],
    )
    assert advisory.action is MarketAction.propose_reduce


def test_one_venue_own_series_never_proposes():
    """Bitcoin has no tape, so one feed's series is that feed's story."""
    from my_jev.market_decision import InstrumentEvidence, decide_instrument

    advisory = decide_instrument(
        InstrumentEvidence(
            instrument_id="opaque:x",
            observations=live({"a": 95_500.0}),
            history=series(95_000, days=0.10, venues=("a",)),
        ),
        BTC,
        now=NOW,
        liveness=[live_check("a")],
    )
    assert advisory.action is not MarketAction.propose_increase
    assert advisory.actionable is False


def test_a_move_below_the_threshold_does_not_propose():
    from my_jev.market_decision import InstrumentEvidence, decide_instrument

    advisory = decide_instrument(
        InstrumentEvidence(
            instrument_id="opaque:x",
            observations=live({"a": 95_500.0, "b": 95_505.0}),
            history=series(95_000, days=0.02),
        ),
        BTC,
        now=NOW,
        liveness=[live_check("a"), live_check("b")],
    )
    assert advisory.action is MarketAction.propose_hold


def test_a_snapshot_alone_never_proposes_whatever_the_price():
    """The conservative behaviour that predates proposals must not have moved."""
    from my_jev.market_decision import InstrumentEvidence, decide_instrument

    for last in (50_000.0, 95_000.0, 500.0, 10.0):
        advisory = decide_instrument(
            InstrumentEvidence(
                instrument_id="opaque:x", observations=live({"a": last, "b": last * 1.001})
            ),
            BTC,
            now=NOW,
            liveness=[live_check("a"), live_check("b")],
        )
        assert advisory.action is MarketAction.propose_hold, last


def test_losing_one_corroborating_venue_removes_the_proposal_not_the_reading():
    """Losing a feed degrades a proposal to a hold; it does not kill the reading.

    Bitcoin needs two venues to propose. With one dead there is still a live venue
    and a real sustained move, so the honest answer is "something changed and I
    cannot corroborate it" -- a hold -- rather than abstaining as though there were
    nothing to say.
    """
    from my_jev.market_decision import InstrumentEvidence, decide_instrument

    advisory = decide_instrument(
        InstrumentEvidence(
            instrument_id="opaque:x",
            observations=live({"a": 95_500.0, "b": 95_510.0}),
            history=series(95_000, days=0.10),
        ),
        BTC,
        now=NOW,
        liveness=[live_check("a"), live_check("b", ok=False)],
    )
    assert advisory.action is MarketAction.propose_hold
    assert advisory.actionable is False
    assert any("dropped by the evidence gate" in note for note in advisory.notes)


def test_a_move_attested_only_by_dead_venues_never_proposes():
    """Corroboration must not rest on feeds that are down now.

    `venue_count` came from history, which includes venues that may no longer be
    reachable. A move every one of whose supporting feeds is dead cannot be acted
    on or verified, so it must not produce a proposal.
    """
    from my_jev.market_decision import InstrumentEvidence, decide_instrument

    advisory = decide_instrument(
        InstrumentEvidence(
            instrument_id="opaque:x",
            observations=live({"a": 95_500.0}),
            history=series(95_000, days=0.10, venues=("a",)),
        ),
        BTC,
        now=NOW,
        liveness=[live_check("a", ok=False), live_check("b", ok=False)],
    )
    assert advisory.action is MarketAction.abstain


def test_the_two_assets_propose_at_very_different_magnitudes():
    """The thresholds are not a tuning difference; they follow the asset."""
    assert BTC.propose_move_threshold > INDEX.propose_move_threshold * 2


# --- the enum is honest ----------------------------------------------------


def test_every_market_action_is_reachable():
    """Guards against a member no code path can emit.

    The third dead enum found in this layer. `propose_increase` and
    `propose_reduce` existed from the start with nothing able to return them, and
    a consumer branching on either had dead code. This sweeps a grid rather than
    asserting a handful of cases, so a future threshold change cannot quietly
    strand a value.
    """
    from my_jev.market_decision import InstrumentEvidence, decide_instrument

    seen: set[str] = set()
    for policy in (BTC, INDEX):
        base = 95_000.0 if policy is BTC else 500.0
        for up in (True, False):
            for venues in (1, 2, 3):
                for days in (0.02, 0.10, 0.50):
                    for samples in (3, 5):
                        for spread in (0.0, 0.02, 0.50):
                            drift = (1 + days) if up else (1 - days)
                            names = tuple(f"v{index}" for index in range(venues))
                            advisory = decide_instrument(
                                InstrumentEvidence(
                                    instrument_id="opaque:x",
                                    observations=live(
                                        {
                                            venue: base * drift * (1 + spread * index)
                                            for index, venue in enumerate(names)
                                        }
                                    ),
                                    history=series(
                                        base, up=up, days=days,
                                        samples=samples, venues=names,
                                    ),
                                ),
                                policy,
                                now=NOW,
                                liveness=[live_check(name) for name in names],
                            )
                            seen.add(advisory.action.value)
    assert seen == {action.value for action in MarketAction}


def test_the_move_policy_never_rescues_rejected_evidence():
    """A proposal is added to a neutral reading, never used to override a refusal."""
    from my_jev.market_decision import InstrumentEvidence, decide_instrument

    advisory = decide_instrument(
        InstrumentEvidence(
            instrument_id="opaque:x",
            observations=live({"a": 95_500.0, "b": 95_510.0}, age_seconds=100_000),
            history=series(95_000, days=0.10),
        ),
        BTC,
        now=NOW,
        liveness=[live_check("a"), live_check("b")],
    )
    assert advisory.action is MarketAction.abstain
