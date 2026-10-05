"""Reversals, history gating, move provenance, and a third asset class.

Four related additions, collected because each one exists to close a specific hole
and the holes are easier to see together than separately.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from my_jev.market_asset_crypto import BitcoinSpotPolicy
from my_jev.market_asset_equity import SingleNameEquityPolicy
from my_jev.market_asset_index_fund import IndexFundPolicy
from my_jev.market_decision import (
    DEFAULT_HISTORY_WINDOW_SECONDS,
    DEFAULT_MOVE_SPAN_SECONDS,
    DEFAULT_REVERSAL_MINIMUM_EXCURSION,
    EvidenceRejection,
    InstrumentEvidence,
    MarketAction,
    MarketQuote,
    MarketSession,
    MarketDecisionAdvisory,
    TradingAuthority,
    VenueLiveness,
    decide_instrument,
    sustained_move,
)

NOW = datetime(2026, 10, 6, 15, 0, tzinfo=UTC)  # Tuesday 11:00 America/New_York
BTC = BitcoinSpotPolicy()

#: A fall comfortably past bitcoin's 8% proposal bar. Written as a series of
#: prices rather than a percentage so the fixtures cannot silently drift under
#: the threshold -- an earlier version of this file used a 7.4% fall and asserted
#: it would propose.
BTC_FALL = [95_000.0, 88_000.0, 80_000.0]
EQUITY = SingleNameEquityPolicy()


def quote(*, age_seconds: float, last: float, source: str | None = "a",
          bid: float | None = None, ask: float | None = None,
          session: MarketSession = MarketSession.regular) -> MarketQuote:
    return MarketQuote(
        instrument_id="opaque:x",
        last=last,
        bid=bid if bid is not None else last * 0.999,
        ask=ask if ask is not None else last * 1.001,
        observed_at=NOW - timedelta(seconds=age_seconds),
        source=source,
        session=session,
    )


def live(venue: str = "a", ok: bool = True) -> VenueLiveness:
    return VenueLiveness(
        venue=venue, reachable=ok, observed_at=NOW - timedelta(seconds=5)
    )


def series(prices: list[float], *, step_minutes: int = 30,
           venues: tuple[str, ...] = ("a", "b")) -> list[MarketQuote]:
    """A series given in chronological order -- oldest price first.

    Written this way because three earlier versions of this file's fixtures
    inverted the direction and every one of them produced a passing test of the
    wrong thing. A list read left to right in time order cannot be flipped.
    """
    span = (len(prices) - 1) * step_minutes
    return [
        MarketQuote(
            instrument_id="opaque:x",
            last=price,
            observed_at=NOW - timedelta(minutes=span - index * step_minutes),
            source=venues[index % len(venues)],
        )
        for index, price in enumerate(prices)
    ]


# ===========================================================================
# A reversal is a state change, and used to be reported as nothing
# ===========================================================================


@pytest.mark.parametrize(
    ("prices", "expected"),
    [
        ([95_000, 98_000, 101_500], "up"),
        ([101_500, 98_000, 95_000], "down"),
        # rose 10.5% and came all the way back: the case that returned None before
        ([95_000, 105_000, 95_000], "reversal_up"),
        # fell 9.5% and came all the way back
        ([95_000, 86_000, 95_000], "reversal_down"),
        # rallied before collapsing: a switch, not a giveback
        ([95_000, 105_000, 88_000], "reversal_down"),
        # collapsed before bouncing: the mirror of the above
        ([105_000, 88_000, 101_500], "reversal_down"),
        # rose, fell, kept most of the rise: a giveback
        ([95_000, 105_000, 99_000], "reversal_up"),
    ],
)
def test_every_shape_of_series_is_named(prices, expected):
    assert sustained_move(series(prices)).direction.value == expected


@pytest.mark.parametrize(
    "prices",
    [
        [95_000, 95_000, 95_000],       # never moved
        [95_000, 95_100, 95_000],       # wobbled, but under the reversal threshold
        [95_000, 95_010, 95_000],
    ],
)
def test_a_series_that_really_went_nowhere_is_still_nothing(prices):
    """Reversal detection must not become a way to call noise a move.

    The threshold that makes a 10% round trip legible must not make a 0.1% one
    legible, or "reversal" stops meaning anything and every wobbling feed is
    suddenly reporting a failed move.
    """
    assert sustained_move(series(prices)) is None


def test_net_direction_recovers_the_trend_a_reversal_traced():
    """Without this, "reversal_up" never says which way it went first."""
    rose = sustained_move(series([95_000, 105_000, 95_000]))
    assert rose.direction.value == "reversal_up"
    assert rose.net_direction.value == "up"

    fell = sustained_move(series([95_000, 86_000, 95_000]))
    assert fell.direction.value == "reversal_down"
    assert fell.net_direction.value == "down"


def test_a_round_trip_reports_its_excursion_not_a_zero():
    """Reporting 0.0 would assert the series went nowhere.

    It went 10.5% and came back. `magnitude` is defined as the net change, which
    is nil here, so the excursion is reported in its place and `direction` marks
    why.
    """
    signal = sustained_move(series([95_000, 105_000, 95_000]))
    assert signal.magnitude == pytest.approx(0.10526, abs=1e-4)
    assert signal.excursion_ratio is None  # no net move to be a fraction of


def test_excursion_ratio_reads_above_one_when_a_move_failed_to_hold():
    clean = sustained_move(series([95_000, 98_000, 101_500]))
    assert clean.excursion_ratio is None

    failed = sustained_move(series([95_000, 105_000, 99_000]))
    assert failed.excursion_ratio > 1.0


def test_the_reversal_threshold_is_a_constant_not_a_literal():
    """It is referenced by name in the docstring and in the tests above."""
    assert DEFAULT_REVERSAL_MINIMUM_EXCURSION > 0.0
    # A move smaller than the reversal threshold must not be called a reversal
    # even though it is plainly a change.
    small = sustained_move(series([95_000, 95_150, 95_000]))
    assert small is None


# ===========================================================================
# A reversal earns no directional proposal
# ===========================================================================


def decide(history, *, policy=BTC, observations=None, liveness=None):
    observations = observations if observations is not None else [
        quote(age_seconds=30, last=95_500, source="a"),
        quote(age_seconds=30, last=95_510, source="b"),
    ]
    return decide_instrument(
        InstrumentEvidence(
            instrument_id="opaque:x", observations=observations, history=history
        ),
        policy,
        now=NOW,
        liveness=liveness if liveness is not None else [live("a"), live("b")],
    )


def test_a_reversal_proposes_no_direction_in_any_asset():
    """Reading a failed rally as a rise is the error this exists to prevent.

    The alternative was treating a reversal as the opposite direction, which is
    worse: that trades on the assumption a failed rally continues, which is a
    forecast, and this module holds no view on what happens next.

    All three policies are exercised rather than just the two that existed before,
    because a reversal guard added to a new policy and never fed a reversal is a
    guard that does not exist.

    Each series is sized to clear its own policy's *absolute* bar, so the reversal
    guard is what refuses. A smaller reversal is refused earlier by the magnitude
    threshold and would pass this test without ever reaching the guard -- which is
    how the first version of this test proved nothing.
    """
    equity_observations = [
        quote(age_seconds=30, last=100.0, bid=99.8, ask=100.2, source="a"),
        quote(age_seconds=30, last=100.02, bid=99.82, ask=100.22, source="b"),
    ]

    for policy, prices in (
        (BTC, [95_000.0, 110_000.0, 80_000.0]),
        (IndexFundPolicy(), [500.0, 580.0, 420.0]),
        (EQUITY, [100.0, 150.0, 84.0]),
    ):
        if policy is EQUITY:
            advisory = equity_decide(series(prices), observations=equity_observations)
        else:
            advisory = decide(series(prices), policy=policy)

        assert advisory.move.direction.value == "reversal_down", policy.asset_class
        # Fixture sanity: the reversal must be one the absolute bar would have let
        # through, or this test is measuring the magnitude threshold again.
        assert advisory.move.magnitude >= policy.propose_move_threshold, policy.asset_class
        assert advisory.action is MarketAction.propose_hold, policy.asset_class
        assert advisory.actionable is False


def test_the_move_is_still_recorded_when_it_proposes_nothing():
    """"A real move that was not enough" has to be visible afterwards."""
    advisory = decide(series([95_000, 105_000, 95_000]))
    assert advisory.action is MarketAction.propose_hold
    assert advisory.move is not None
    assert advisory.move.direction.value == "reversal_up"


# ===========================================================================
# History is bounded, and bounded history has its own rejection causes
# ===========================================================================


def test_a_series_from_last_week_cannot_propose_today():
    """Evidence about a period that ended is not evidence about the present.

    The detector will happily compute a clean, well-corroborated, 92% move from
    any series it is handed -- and without the window that series would be the one
    entry from ten days ago plus two current ones. The hazard is not that old
    evidence is useless; it is that it silently *inflates* the move, because a
    price from a week ago is a large distance from today's.
    """
    ancient_then_now = [
        MarketQuote(
            instrument_id="opaque:x", last=50_000,
            observed_at=NOW - timedelta(days=10), source="a",
        ),
        *series([95_000.0, 96_000.0], step_minutes=30),
    ]

    # Without the bound this is a spectacular 92% rise on two fresh prints.
    unbounded = sustained_move(ancient_then_now)
    assert unbounded is not None
    assert unbounded.magnitude > 0.9

    advisory = decide(ancient_then_now)
    assert advisory.action is MarketAction.propose_hold
    assert advisory.move is None
    assert EvidenceRejection.history_out_of_window in advisory.rejected


def test_out_of_window_is_not_conflated_with_stale():
    """They are opposite mistakes and send an operator to different places.

    A stale quote is an observation of *now* that has aged; an out-of-window
    history entry describes a period that has already ended. Telling an operator
    "stale" when the series is a week old points at the wrong feed entirely.
    """
    advisory = decide(
        [
            MarketQuote(
                instrument_id="opaque:x", last=50_000,
                observed_at=NOW - timedelta(days=10), source="a",
            ),
            *series([95_000.0, 96_000.0]),
        ]
    )
    assert EvidenceRejection.stale not in advisory.rejected
    assert EvidenceRejection.history_out_of_window in advisory.rejected


def test_history_dated_in_the_future_is_refused():
    """Clock skew is not history, and must not become a basis for a proposal.

    Same hazard as an out-of-window entry: an undated-ish print that lands in the
    wrong place would sort to the end of the series and turn a 1% drift into a
    47% fall.
    """
    future_then_now = [
        MarketQuote(
            instrument_id="opaque:x", last=50_000,
            observed_at=NOW + timedelta(minutes=5), source="a",
        ),
        *series([95_000.0, 96_000.0]),
    ]
    assert sustained_move(future_then_now).magnitude > 0.4

    advisory = decide(future_then_now)
    assert advisory.action is MarketAction.propose_hold
    assert EvidenceRejection.future_dated in advisory.rejected


def test_history_from_a_feed_that_cannot_be_checked_is_refused():
    """Being old does not exempt a series from the liveness rule.

    If anything it weakens corroboration: there is no recent sign the feed is
    still reporting honestly.
    """
    advisory = decide(
        series([95_000, 88_000], venues=("a", "zz")),
        liveness=[live("a"), live("b")],
    )
    assert advisory.action is MarketAction.propose_hold
    assert EvidenceRejection.history_venue_unverified in advisory.rejected


def test_the_history_window_is_bounded_by_default():
    """An unbounded window would let any series in the asset's life qualify."""
    assert DEFAULT_HISTORY_WINDOW_SECONDS > DEFAULT_MOVE_SPAN_SECONDS


def test_a_policy_may_tighten_its_history_window():
    assert EQUITY.history_window_seconds(NOW) <= DEFAULT_HISTORY_WINDOW_SECONDS


# ===========================================================================
# Move provenance on the advisory
# ===========================================================================


def test_a_proposal_names_the_move_it_leaned_on():
    """Otherwise two proposals differing only in span and sample count are
    indistinguishable documents, and neither can be audited."""
    advisory = decide(series(BTC_FALL))
    assert advisory.action is MarketAction.propose_reduce
    assert advisory.move is not None
    assert advisory.move.direction.value == "down"
    assert advisory.move.sample_count == 3
    assert advisory.move.span_seconds > 0
    assert advisory.move.venues


def test_two_proposals_from_different_evidence_are_different_documents():
    """The reason for having provenance at all."""
    thin = decide(series(BTC_FALL))
    thick = decide(series([95_000, 92_000, 89_000, 86_500]))
    assert thin.action is thick.action is MarketAction.propose_reduce
    assert thin.move.sample_count != thick.move.sample_count
    assert thin.model_dump_json() != thick.model_dump_json()


def test_no_move_means_no_provenance():
    advisory = decide_instrument(
        InstrumentEvidence(instrument_id="opaque:x",
                           observations=[quote(age_seconds=30, last=95_000)]),
        BTC, now=NOW, liveness=[live("a")],
    )
    assert advisory.move is None


def test_provenance_does_not_leak_into_the_authority():
    """A richer advisory must not become a more actionable one."""
    advisory = decide(series(BTC_FALL))
    assert set(advisory.authority.model_dump().values()) == {False}
    assert advisory.price_forecast is False
    assert advisory.advisory_only is True


def test_the_authority_still_rejects_a_true_anywhere():
    with pytest.raises(ValueError):
        TradingAuthority(order_placed=True)


# ===========================================================================
# A third asset class: the one where the framework's obligation is hardest
# ===========================================================================


def equity_decide(history, *, observations, liveness=None):
    return decide_instrument(
        InstrumentEvidence(
            instrument_id="opaque:x", observations=observations, history=history
        ),
        EQUITY,
        now=NOW,
        liveness=liveness if liveness is not None else [live("a"), live("b")],
    )


def test_a_single_name_proposes_on_a_large_corroborated_move():
    observations = [
        quote(age_seconds=30, last=130.0, bid=129.87, ask=130.13, source="a"),
        quote(age_seconds=30, last=130.02, bid=129.89, ask=130.15, source="b"),
    ]
    advisory = equity_decide(series([100.0, 112.0, 130.0]), observations=observations)
    assert advisory.action is MarketAction.propose_increase
    assert advisory.actionable is True  # it does propose a change
    assert set(advisory.authority.model_dump().values()) == {False}


def test_one_venue_is_not_corroboration_even_for_a_huge_move():
    """The failure this policy exists to prevent.

    A thin name often has exactly one feed. Reporting that feed's own series back
    as a finding is reporting the input as the conclusion.
    """
    observations = [quote(age_seconds=30, last=130.0, bid=129.87, ask=130.13, source="a")]
    advisory = equity_decide(
        series([100.0, 112.0, 130.0], venues=("a",)),
        observations=observations,
        liveness=[live("a")],
    )
    assert advisory.move is not None
    assert advisory.action is MarketAction.propose_hold


def test_a_move_much_smaller_than_the_spread_is_not_a_price_change():
    """The spread is the evidence in a thin name.

    A move measured mid-spread is mostly the bid and ask moving apart, and a
    proposal here would be proposing a trade that cannot profitably be made. The
    bar is deliberately expressed against the spread rather than as a fixed
    percentage, because in a name this thin the spread *is* the volatility.

    The numbers are deliberately extreme: the spread has to be wider than half the
    absolute 15% bar before this guard is the thing refusing, because below that
    the absolute bar speaks first. An earlier version of this test used a 2% move
    and a 4% spread and passed -- on the absolute threshold, never reaching the
    spread check at all, which is a test of the wrong thing that looks correct.
    """
    observations = [
        quote(age_seconds=30, last=100.0, bid=80.0, ask=120.0, source="a"),
        quote(age_seconds=30, last=100.0, bid=80.0, ask=120.0, source="b"),
    ]
    # A 20% fall: past the 15% absolute bar, but a fifth of the 40% round trip
    # through a 20% half-spread.
    advisory = equity_decide(series([120.0, 110.0, 96.0]), observations=observations)

    assert advisory.move is not None
    assert advisory.move.magnitude > 0.15, "fixture must clear the absolute bar"
    assert advisory.action is MarketAction.propose_hold


def test_a_wide_spread_alone_is_enough_to_refuse():
    observations = [
        quote(age_seconds=30, last=100.0, bid=94.0, ask=106.0, source="a"),
        quote(age_seconds=30, last=100.02, bid=94.0, ask=106.1, source="b"),
    ]
    advisory = equity_decide([], observations=observations)
    assert advisory.action is MarketAction.propose_hold
    assert "too wide to act on" in advisory.reason


def test_venues_that_disagree_about_one_instrument_are_refused():
    """They are quoting the same thing, so a wide split is a fault, not a spread."""
    observations = [
        quote(age_seconds=30, last=100.0, bid=99.9, ask=100.1, source="a"),
        quote(age_seconds=30, last=140.0, bid=139.8, ask=140.2, source="b"),
    ]
    advisory = equity_decide([], observations=observations)
    assert advisory.action is MarketAction.abstain
    assert "disagreed" in advisory.reason


def test_a_single_name_requires_liveness_and_the_other_two_do_not():
    """Each policy's answer is a consequence of its market structure, not a preference."""
    assert EQUITY.requires_venue_liveness is True
    assert BTC.requires_venue_liveness is True
    assert IndexFundPolicy().requires_venue_liveness is False


def test_the_three_policies_disagree_about_the_same_evidence():
    """Which is the reason to have a policy per asset class at all."""
    history = series([100.0, 112.0, 130.0])
    observations = [
        quote(age_seconds=30, last=130.0, bid=129.87, ask=130.13, source="a"),
        quote(age_seconds=30, last=130.02, bid=129.89, ask=130.15, source="b"),
    ]
    actions = {
        policy.asset_class: equity_decide(history, observations=observations)
        for policy in (EQUITY, BTC)
    }
    # A 30% move on a thin single name is proposal-worthy only because it dwarfs the
    # spread; for bitcoin it is ordinary, and bitcoin additionally demands two
    # venues that are both live.
    assert actions[EQUITY.asset_class].action is MarketAction.propose_increase
    assert all(a.action is not MarketAction.abstain for a in actions.values())
    for advisory in actions.values():
        assert set(advisory.authority.model_dump().values()) == {False}


def test_proposing_something_still_places_nothing():
    """`actionable` means "proposes a change", and is true by design for a proposal.

    What must be impossible is that a proposal becomes an instruction, so that is
    what this asserts -- on a real proposal, not a neutral reading.
    """
    advisory = decide(series(BTC_FALL))
    assert isinstance(advisory, MarketDecisionAdvisory)
    assert advisory.action is MarketAction.propose_reduce
    assert advisory.actionable is True  # it does propose a change
    # ...and proposes nothing else at all.
    assert set(advisory.authority.model_dump().values()) == {False}